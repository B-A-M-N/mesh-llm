//! Minimal engine-neutral server for the existing Skippy binary stage wire.
//!
//! The mature llama.cpp lane in [`crate::binary_transport`] still owns KV-page
//! caching, telemetry, batching, MTP, and OpenAI orchestration. This module is
//! intentionally the smaller compatibility seam: it proves a `StageEngine`
//! can participate in a real multi-process Skippy chain without introducing a
//! second wire protocol. Advanced operations stay capability-gated.

use std::{
    io::{self, Write},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use skippy_engine::{
    StageActivation, StageEngine, StageExecutionKind, StageExecutionOutput, StageExecutionRequest,
};
use skippy_protocol::binary::{
    StageReply, StageWireMessage, WireActivationDType, WireMessageKind, WireReplyKind,
    encode_f32_activation_payload, read_stage_message, recv_ready, recv_reply, send_ready,
    send_reply_ack_with_stats, send_reply_predicted_tokens_with_stats,
    send_reply_predicted_with_tokens_and_stats, write_stage_message,
};

#[derive(Clone, Debug)]
pub struct EngineStageServerOptions {
    pub bind_addr: SocketAddr,
    pub downstream_addr: Option<SocketAddr>,
    pub wire_dtype: WireActivationDType,
}

pub fn serve_stage_engine(
    engine: Arc<dyn StageEngine>,
    options: EngineStageServerOptions,
) -> Result<()> {
    serve_stage_engine_until(engine, options, Arc::new(AtomicBool::new(false)))
}

pub fn serve_stage_engine_until(
    engine: Arc<dyn StageEngine>,
    options: EngineStageServerOptions,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
    let listener = prepare_stage_engine_listener(engine.as_ref(), &options)?;
    serve_prepared_stage_engine_until(engine, options, listener, shutdown)
}

pub(crate) fn prepare_stage_engine_listener(
    engine: &dyn StageEngine,
    options: &EngineStageServerOptions,
) -> Result<TcpListener> {
    engine.info().validate()?;
    validate_topology(engine, options)?;
    let listener = TcpListener::bind(options.bind_addr)
        .with_context(|| format!("bind engine stage at {}", options.bind_addr))?;
    listener.set_nonblocking(true)?;
    if let Some(downstream_addr) = options.downstream_addr {
        drop(
            connect_downstream(downstream_addr).with_context(|| {
                format!("preflight downstream engine stage at {downstream_addr}")
            })?,
        );
    }
    eprintln!(
        "skippy engine stage listening: engine={} model={} binary={} layers={}..{} width={} dtype={:?}",
        engine.info().engine,
        engine.info().model_id,
        listener.local_addr()?,
        engine.info().layer_start,
        engine.info().layer_end,
        engine.info().activation_width,
        options.wire_dtype,
    );
    Ok(listener)
}

pub(crate) fn serve_prepared_stage_engine_until(
    engine: Arc<dyn StageEngine>,
    options: EngineStageServerOptions,
    listener: TcpListener,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
    let mut connections = Vec::new();
    let result = (|| {
        while !shutdown.load(Ordering::SeqCst) {
            reap_finished_connections(&mut connections);
            let (upstream, peer_addr) = match listener.accept() {
                Ok(connection) => connection,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(25));
                    continue;
                }
                Err(error) => return Err(error).context("accept engine stage connection"),
            };
            upstream.set_nonblocking(false)?;
            upstream.set_nodelay(true).ok();
            let control = upstream.try_clone()?;
            let downstream_control = Arc::new(Mutex::new(None));
            let engine = engine.clone();
            let options = options.clone();
            let connection_downstream = Arc::clone(&downstream_control);
            let connection = thread::spawn(move || {
                if let Err(error) =
                    handle_connection(engine, options, upstream, connection_downstream)
                {
                    eprintln!("engine stage connection from {peer_addr} failed: {error:#}");
                }
            });
            connections.push(ActiveConnection {
                control,
                downstream_control,
                thread: connection,
            });
        }
        Ok(())
    })();
    stop_active_connections(connections);
    result
}

struct ActiveConnection {
    control: TcpStream,
    downstream_control: Arc<Mutex<Option<TcpStream>>>,
    thread: thread::JoinHandle<()>,
}

fn reap_finished_connections(connections: &mut Vec<ActiveConnection>) {
    let mut index = 0;
    while index < connections.len() {
        if connections[index].thread.is_finished() {
            let connection = connections.swap_remove(index);
            let _ = connection.thread.join();
        } else {
            index += 1;
        }
    }
}

fn stop_active_connections(connections: Vec<ActiveConnection>) {
    for connection in &connections {
        let _ = connection.control.shutdown(Shutdown::Both);
        if let Ok(downstream) = connection.downstream_control.lock()
            && let Some(downstream) = downstream.as_ref()
        {
            let _ = downstream.shutdown(Shutdown::Both);
        }
    }
    for connection in connections {
        let _ = connection.thread.join();
    }
}

fn validate_topology(engine: &dyn StageEngine, options: &EngineStageServerOptions) -> Result<()> {
    ensure!(
        engine.info().is_final() == options.downstream_addr.is_none(),
        "only the final stage may omit a downstream address"
    );
    Ok(())
}

fn handle_connection(
    engine: Arc<dyn StageEngine>,
    options: EngineStageServerOptions,
    mut upstream: TcpStream,
    downstream_control: Arc<Mutex<Option<TcpStream>>>,
) -> Result<()> {
    let mut downstream = options
        .downstream_addr
        .map(connect_downstream)
        .transpose()?;
    if let Some(stream) = downstream.as_ref() {
        *downstream_control
            .lock()
            .expect("downstream control lock poisoned") = Some(stream.try_clone()?);
    }
    send_ready(&mut upstream).context("send engine stage ready")?;
    upstream.flush().ok();
    let activation_width =
        i32::try_from(engine.info().activation_width).context("activation width exceeds i32")?;

    loop {
        let message = match read_stage_message(&mut upstream, activation_width) {
            Ok(message) => message,
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(error).context("read engine stage message"),
        };
        if message.kind == WireMessageKind::Stop {
            engine.reset_session(message.session_id)?;
            let downstream_reply =
                forward_control(downstream.as_mut(), &message, options.wire_dtype)?;
            send_ack(&mut upstream, downstream_reply)?;
            continue;
        }
        if message.kind.is_session_control() {
            execute_session_control(engine.as_ref(), &message)?;
            let downstream_reply =
                forward_control(downstream.as_mut(), &message, options.wire_dtype)?;
            send_ack(&mut upstream, downstream_reply)?;
            continue;
        }

        let request = execution_request(&message, activation_width)?;
        let output = engine.execute(request)?;
        match downstream.as_mut() {
            Some(downstream) => {
                let forwarded =
                    forwarded_message(engine.as_ref(), &message, output, options.wire_dtype)?;
                write_stage_message(&mut *downstream, &forwarded, options.wire_dtype)
                    .context("forward engine stage message")?;
                downstream.flush().ok();
                let reply = recv_reply(&mut *downstream).context("receive downstream reply")?;
                send_reply(&mut upstream, reply)?;
            }
            None => send_final_reply(&mut upstream, &message, output)?,
        }
    }
}

fn connect_downstream(addr: SocketAddr) -> Result<TcpStream> {
    const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
    let mut stream = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)
        .with_context(|| format!("connect downstream engine stage at {addr}"))?;
    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(CONNECT_TIMEOUT))?;
    stream.set_write_timeout(Some(CONNECT_TIMEOUT))?;
    recv_ready(&mut stream).context("downstream engine stage did not become ready")?;
    stream.set_read_timeout(None)?;
    stream.set_write_timeout(None)?;
    Ok(stream)
}

fn execute_session_control(engine: &dyn StageEngine, message: &StageWireMessage) -> Result<()> {
    match message.kind {
        WireMessageKind::CheckpointSession => engine.checkpoint_session(message.session_id),
        WireMessageKind::RestoreSession => engine.restore_session(message.session_id),
        WireMessageKind::TrimSession => {
            engine.trim_session(message.session_id, message.token_count.max(0) as u64)
        }
        _ => bail!("message is not session control"),
    }
}

fn forward_control(
    downstream: Option<&mut TcpStream>,
    message: &StageWireMessage,
    wire_dtype: WireActivationDType,
) -> Result<Option<StageReply>> {
    let Some(downstream) = downstream else {
        return Ok(None);
    };
    write_stage_message(&mut *downstream, message, wire_dtype)?;
    downstream.flush().ok();
    Ok(Some(recv_reply(&mut *downstream)?))
}

fn execution_request(
    message: &StageWireMessage,
    activation_width: i32,
) -> Result<StageExecutionRequest> {
    let kind = match message.kind {
        WireMessageKind::PrefillEmbd => StageExecutionKind::Prefill,
        WireMessageKind::PrefillFinalEmbd => StageExecutionKind::PrefillFinal,
        WireMessageKind::DecodeEmbd
        | WireMessageKind::DecodeReadout
        | WireMessageKind::DecodeLightCtx
        | WireMessageKind::DecodeReplayEmbd
        | WireMessageKind::DecodeReplayFinalEmbd => StageExecutionKind::Decode,
        WireMessageKind::VerifySpan => StageExecutionKind::Verify,
        other => bail!("engine stage does not execute {other:?}"),
    };
    let token_count = usize::try_from(message.token_count).context("negative token count")?;
    let token_ids = execution_tokens(message, kind, token_count)?;
    let input = if message.activation.is_empty() {
        None
    } else {
        let bytes = message
            .activation_f32_payload(activation_width)
            .context("decode input activation")?;
        Some(StageActivation::new(
            token_count,
            usize::try_from(activation_width)?,
            bytes,
        )?)
    };
    Ok(StageExecutionRequest {
        session_id: message.session_id,
        kind,
        token_ids,
        positions: message.positions.clone(),
        input,
        sampling: message.sampling.clone(),
    })
}

fn execution_tokens(
    message: &StageWireMessage,
    kind: StageExecutionKind,
    token_count: usize,
) -> Result<Vec<i32>> {
    if kind == StageExecutionKind::Decode {
        ensure!(token_count == 1, "decode requires one token");
        return Ok(vec![message.state.current_token]);
    }
    ensure!(
        message.tokens.len() == token_count,
        "token sideband length does not match token count"
    );
    Ok(message.tokens.clone())
}

fn forwarded_message(
    engine: &dyn StageEngine,
    incoming: &StageWireMessage,
    output: StageExecutionOutput,
    wire_dtype: WireActivationDType,
) -> Result<StageWireMessage> {
    let activation = output
        .activation
        .context("non-final engine stage returned no activation")?;
    ensure!(
        activation.width == engine.info().activation_width as usize,
        "engine output activation width mismatch"
    );
    let mut state = incoming.state;
    state.source_stage_index = i32::try_from(engine.info().stage_index)?;
    state.reserved = wire_dtype as i32;
    let activation = encode_f32_activation_payload(
        wire_dtype,
        incoming.token_count,
        i32::try_from(activation.width)?,
        &activation.f32_le_bytes,
    )?;
    Ok(StageWireMessage {
        kind: incoming.kind,
        pos_start: incoming.pos_start,
        token_count: incoming.token_count,
        state,
        request_id: incoming.request_id,
        session_id: incoming.session_id,
        sampling: incoming.sampling.clone(),
        chat_sampling_metadata: incoming.chat_sampling_metadata.clone(),
        tokens: incoming.tokens.clone(),
        positions: incoming.positions.clone(),
        activation,
        raw_bytes: Vec::new(),
    })
}

fn send_final_reply(
    upstream: &mut TcpStream,
    message: &StageWireMessage,
    output: StageExecutionOutput,
) -> Result<()> {
    if message.kind.requires_predicted_reply() {
        ensure!(
            !output.predicted_tokens.is_empty(),
            "final engine stage returned no prediction"
        );
        send_reply_predicted_with_tokens_and_stats(
            &mut *upstream,
            output.predicted().expect("checked non-empty"),
            &output.predicted_tokens,
            Default::default(),
        )?;
    } else {
        send_reply_ack_with_stats(&mut *upstream, Default::default())?;
    }
    upstream.flush().ok();
    Ok(())
}

fn send_ack(upstream: &mut TcpStream, downstream: Option<StageReply>) -> Result<()> {
    if let Some(reply) = downstream {
        ensure!(
            reply.kind == WireReplyKind::Ack,
            "control expected downstream ACK"
        );
        send_reply_ack_with_stats(&mut *upstream, reply.stats)?;
    } else {
        send_reply_ack_with_stats(&mut *upstream, Default::default())?;
    }
    upstream.flush().ok();
    Ok(())
}

fn send_reply(upstream: &mut TcpStream, reply: StageReply) -> Result<()> {
    match reply.kind {
        WireReplyKind::Ack => send_reply_ack_with_stats(&mut *upstream, reply.stats)?,
        WireReplyKind::PredictedToken => send_reply_predicted_with_tokens_and_stats(
            &mut *upstream,
            reply.predicted,
            &reply.predicted_tokens,
            reply.stats,
        )?,
        WireReplyKind::PredictedTokens => send_reply_predicted_tokens_with_stats(
            &mut *upstream,
            &reply.predicted_tokens,
            reply.stats,
        )?,
    }
    upstream.flush().ok();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::StageStateHeader;

    fn decode_message(tokens: Vec<i32>, current_token: i32) -> StageWireMessage {
        let kind = WireMessageKind::DecodeEmbd;
        let mut state = StageStateHeader::new(kind, WireActivationDType::F16);
        state.current_token = current_token;
        StageWireMessage {
            kind,
            pos_start: 0,
            token_count: 1,
            state,
            request_id: 1,
            session_id: 2,
            sampling: None,
            chat_sampling_metadata: None,
            tokens,
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    #[test]
    fn decode_uses_current_token_not_prompt_sideband() {
        let request = execution_request(&decode_message(vec![1, 2, 3], 7), 4).unwrap();
        assert_eq!(request.token_ids, vec![7]);
        assert_eq!(request.kind, StageExecutionKind::Decode);
    }
}
