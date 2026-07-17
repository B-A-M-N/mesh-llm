//! Run or probe an MLX partial-layer engine over Skippy's binary stage wire.

#[cfg(all(feature = "mlx", target_os = "macos"))]
mod real {
    use std::{
        io::Write,
        net::{SocketAddr, TcpStream},
        path::PathBuf,
        sync::Arc,
    };

    use anyhow::{Context, Result, ensure};
    use clap::{Parser, Subcommand, ValueEnum};
    use skippy_engine_mlx::{
        MlxComputeDtype, MlxStageEngine, MlxStageEngineConfig, MlxWeightQuantization,
    };
    use skippy_protocol::binary::{
        StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind, WireReplyKind,
        recv_ready, recv_reply, write_stage_message,
    };
    use skippy_server::engine_transport::{EngineStageServerOptions, serve_stage_engine};

    #[derive(Debug, Parser)]
    #[command(about = "Serve and prove partial SafeTensors MLX stages")]
    struct Cli {
        #[command(subcommand)]
        command: Command,
    }

    #[derive(Debug, Subcommand)]
    enum Command {
        /// Load one partial SafeTensors artifact and serve its layer range.
        Serve {
            #[arg(long)]
            model: PathBuf,
            #[arg(long, default_value = "mlx-stage-model")]
            model_id: String,
            #[arg(long)]
            stage_index: u32,
            #[arg(long)]
            layer_start: u32,
            #[arg(long)]
            layer_end: u32,
            #[arg(long)]
            bind: SocketAddr,
            #[arg(long)]
            downstream: Option<SocketAddr>,
            #[arg(long, value_enum, default_value_t = WireDtype::F16)]
            wire_dtype: WireDtype,
            #[arg(long, value_enum, default_value_t = ComputeDtype::Bf16)]
            compute_dtype: ComputeDtype,
            #[arg(long, value_enum)]
            weight_quantization: Option<WeightQuantization>,
        },
        /// Drive a stage chain and assert its greedy token sequence.
        Prove {
            #[arg(long)]
            connect: SocketAddr,
            #[arg(long, default_value = "1,1531,314,260,3575,28")]
            tokens: String,
            #[arg(long, default_value = "284,260,2240,314,1343,327,624,8685")]
            expected: String,
            #[arg(long, value_enum, default_value_t = WireDtype::F16)]
            wire_dtype: WireDtype,
        },
    }

    #[derive(Clone, Copy, Debug, ValueEnum)]
    enum WireDtype {
        F16,
        F32,
    }

    impl From<WireDtype> for WireActivationDType {
        fn from(value: WireDtype) -> Self {
            match value {
                WireDtype::F16 => Self::F16,
                WireDtype::F32 => Self::F32,
            }
        }
    }

    #[derive(Clone, Copy, Debug, ValueEnum)]
    enum ComputeDtype {
        F16,
        Bf16,
        F32,
    }

    impl From<ComputeDtype> for MlxComputeDtype {
        fn from(value: ComputeDtype) -> Self {
            match value {
                ComputeDtype::F16 => Self::F16,
                ComputeDtype::Bf16 => Self::Bf16,
                ComputeDtype::F32 => Self::F32,
            }
        }
    }

    #[derive(Clone, Copy, Debug, ValueEnum)]
    enum WeightQuantization {
        Affine4,
        Affine8,
        Mxfp4,
    }

    impl From<WeightQuantization> for MlxWeightQuantization {
        fn from(value: WeightQuantization) -> Self {
            match value {
                WeightQuantization::Affine4 => Self::Affine {
                    group_size: 64,
                    bits: 4,
                },
                WeightQuantization::Affine8 => Self::Affine {
                    group_size: 64,
                    bits: 8,
                },
                WeightQuantization::Mxfp4 => Self::MxFp4,
            }
        }
    }

    pub fn main() -> Result<()> {
        match Cli::parse().command {
            Command::Serve {
                model,
                model_id,
                stage_index,
                layer_start,
                layer_end,
                bind,
                downstream,
                wire_dtype,
                compute_dtype,
                weight_quantization,
            } => serve(
                MlxStageEngineConfig {
                    model_dir: model,
                    model_id,
                    stage_index,
                    layer_start,
                    layer_end,
                    compute_dtype: compute_dtype.into(),
                    weight_quantization: weight_quantization.map(Into::into),
                    ctx_size: None,
                },
                EngineStageServerOptions {
                    bind_addr: bind,
                    downstream_addr: downstream,
                    wire_dtype: wire_dtype.into(),
                },
            ),
            Command::Prove {
                connect,
                tokens,
                expected,
                wire_dtype,
            } => prove(connect, &tokens, &expected, wire_dtype.into()),
        }
    }

    fn serve(config: MlxStageEngineConfig, options: EngineStageServerOptions) -> Result<()> {
        let engine = Arc::new(MlxStageEngine::spawn(config)?);
        serve_stage_engine(engine, options)
    }

    fn prove(
        connect: SocketAddr,
        tokens: &str,
        expected: &str,
        wire_dtype: WireActivationDType,
    ) -> Result<()> {
        let prompt = parse_ids(tokens)?;
        let expected = parse_ids(expected)?;
        ensure!(
            !expected.is_empty(),
            "expected token sequence must not be empty"
        );
        let mut stream = TcpStream::connect(connect)
            .with_context(|| format!("connect first MLX stage at {connect}"))?;
        stream.set_nodelay(true).ok();
        recv_ready(&mut stream).context("first MLX stage did not become ready")?;

        let session_id = 1;
        let request_id = 1;
        let prefill = execution_message(
            WireMessageKind::PrefillFinalEmbd,
            &prompt,
            *prompt.last().context("prompt must not be empty")?,
            request_id,
            session_id,
            wire_dtype,
            0,
        );
        let mut generated = Vec::with_capacity(expected.len());
        generated.push(send_predicted(&mut stream, &prefill, wire_dtype)?);

        while generated.len() < expected.len() {
            let current = *generated.last().expect("generated has first token");
            let decode = execution_message(
                WireMessageKind::DecodeEmbd,
                &[],
                current,
                request_id,
                session_id,
                wire_dtype,
                i32::try_from(generated.len())?,
            );
            generated.push(send_predicted(&mut stream, &decode, wire_dtype)?);
        }

        let stop = StageWireMessage::stop_with_identity(wire_dtype, request_id, session_id);
        write_stage_message(&mut stream, &stop, wire_dtype)?;
        stream.flush().ok();
        let reply = recv_reply(&mut stream)?;
        ensure!(reply.kind == WireReplyKind::Ack, "stop did not return ACK");
        ensure!(
            generated == expected,
            "two-process stage tokens diverged: expected={expected:?} actual={generated:?}"
        );
        println!("PASS: two MLX stage processes matched the reference greedy tokens");
        println!("wire_dtype={wire_dtype:?}");
        println!("generated_tokens={generated:?}");
        Ok(())
    }

    fn execution_message(
        kind: WireMessageKind,
        tokens: &[i32],
        current_token: i32,
        request_id: u64,
        session_id: u64,
        wire_dtype: WireActivationDType,
        decode_step: i32,
    ) -> StageWireMessage {
        let mut state = StageStateHeader::new(kind, wire_dtype);
        state.current_token = current_token;
        state.prompt_token_count = i32::try_from(tokens.len()).unwrap_or_default();
        state.decode_step = decode_step;
        StageWireMessage {
            kind,
            pos_start: 0,
            token_count: if kind == WireMessageKind::DecodeEmbd {
                1
            } else {
                i32::try_from(tokens.len()).unwrap_or_default()
            },
            state,
            request_id,
            session_id,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: tokens.to_vec(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    fn send_predicted(
        stream: &mut TcpStream,
        message: &StageWireMessage,
        wire_dtype: WireActivationDType,
    ) -> Result<i32> {
        write_stage_message(&mut *stream, message, wire_dtype)?;
        stream.flush().ok();
        let reply = recv_reply(&mut *stream)?;
        ensure!(
            matches!(
                reply.kind,
                WireReplyKind::PredictedToken | WireReplyKind::PredictedTokens
            ),
            "stage chain did not return a predicted token"
        );
        Ok(reply.predicted)
    }

    fn parse_ids(value: &str) -> Result<Vec<i32>> {
        value
            .split(',')
            .map(|token| token.trim().parse().context("parse token ID"))
            .collect()
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    real::main()
}

#[cfg(not(all(feature = "mlx", target_os = "macos")))]
fn main() {
    eprintln!("mlx-stage requires macOS and `--features mlx`");
    std::process::exit(1);
}
