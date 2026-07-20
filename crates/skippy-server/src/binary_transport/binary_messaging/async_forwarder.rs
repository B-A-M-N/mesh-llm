use crate::binary_transport::WireCondition;
use crate::binary_transport::stage_execution::elapsed_ms;
use crate::binary_transport::write_stage_message_after_propagation;
use crate::telemetry::Telemetry;
use crate::telemetry::now_unix_nanos;
use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use serde_json::Value;
use serde_json::json;
use skippy_protocol::binary::StageWireMessage;
use skippy_protocol::binary::WireActivationDType;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::net::TcpStream;
use std::sync::mpsc;
use std::sync::mpsc::TryRecvError;
use std::thread;
use std::time::Instant;

pub(crate) struct AsyncForwarder {
    sender: mpsc::SyncSender<AsyncForwardJob>,
    pending: VecDeque<AsyncForwardReceipt>,
}

pub(crate) struct AsyncForwardReceipt {
    receiver: mpsc::Receiver<Result<f64, String>>,
}

struct AsyncForwardJob {
    message: StageWireMessage,
    wire_dtype: WireActivationDType,
    condition: WireCondition,
    attrs: BTreeMap<String, Value>,
    done: mpsc::Sender<Result<f64, String>>,
    enqueued_at: Instant,
    enqueued_unix_nanos: u64,
}

impl AsyncForwarder {
    pub(crate) fn new(
        downstream: &TcpStream,
        telemetry: Telemetry,
        queue_capacity: usize,
    ) -> Result<Self> {
        let mut writer = downstream
            .try_clone()
            .context("clone downstream stream for async activation forwarding")?;
        let (sender, receiver) = mpsc::sync_channel::<AsyncForwardJob>(queue_capacity.max(1));
        thread::spawn(move || run_forwarder(&mut writer, &receiver, &telemetry));
        Ok(Self {
            sender,
            pending: VecDeque::new(),
        })
    }

    pub(crate) fn send(
        &mut self,
        message: StageWireMessage,
        wire_dtype: WireActivationDType,
        condition: WireCondition,
        attrs: BTreeMap<String, Value>,
    ) -> Result<()> {
        let receipt = self.send_tracked(message, wire_dtype, condition, attrs)?;
        self.pending.push_back(receipt);
        Ok(())
    }

    pub(crate) fn send_tracked(
        &mut self,
        message: StageWireMessage,
        wire_dtype: WireActivationDType,
        condition: WireCondition,
        attrs: BTreeMap<String, Value>,
    ) -> Result<AsyncForwardReceipt> {
        self.reap_completed()?;
        let (done, receiver) = mpsc::channel();
        self.sender
            .send(AsyncForwardJob {
                message,
                wire_dtype,
                condition,
                attrs,
                done,
                enqueued_at: Instant::now(),
                enqueued_unix_nanos: now_unix_nanos() as u64,
            })
            .map_err(|_| anyhow!("async activation forwarder stopped"))?;
        Ok(AsyncForwardReceipt { receiver })
    }

    fn reap_completed(&mut self) -> Result<()> {
        loop {
            let Some(receiver) = self.pending.front() else {
                return Ok(());
            };
            match receiver.try_finish() {
                Ok(Some(_write_ms)) => {
                    self.pending.pop_front();
                }
                Ok(None) => return Ok(()),
                Err(error) => {
                    self.pending.pop_front();
                    return Err(error);
                }
            }
        }
    }

    pub(super) fn flush(&mut self) -> Result<()> {
        while let Some(receiver) = self.pending.pop_front() {
            receiver.finish()?;
        }
        Ok(())
    }
}

fn run_forwarder(
    writer: &mut TcpStream,
    receiver: &mpsc::Receiver<AsyncForwardJob>,
    telemetry: &Telemetry,
) {
    while let Ok(job) = receiver.recv() {
        let wait = time_until_ready(&job);
        if !wait.is_zero() {
            thread::sleep(wait);
        }
        forward_job(writer, telemetry, job);
    }
}

fn time_until_ready(job: &AsyncForwardJob) -> std::time::Duration {
    let ready_at = job.enqueued_at + job.condition.propagation_delay();
    ready_at.saturating_duration_since(Instant::now())
}

fn forward_job(writer: &mut TcpStream, telemetry: &Telemetry, job: AsyncForwardJob) {
    let result =
        write_stage_message_after_propagation(writer, &job.message, job.wire_dtype, job.condition)
            .context("async forward activation frame downstream")
            .map(|()| elapsed_ms(job.enqueued_at))
            .map_err(|error| format!("{error:#}"));
    let write_end_unix_nanos = now_unix_nanos() as u64;
    let mut attrs = job.attrs;
    attrs.insert(
        "llama_stage.forward_write_ms".to_string(),
        json!(elapsed_ms(job.enqueued_at)),
    );
    telemetry.emit_debug_span(
        "stage.binary_downstream_write",
        attrs,
        job.enqueued_unix_nanos,
        write_end_unix_nanos,
    );
    let _ = job.done.send(result);
}

impl AsyncForwardReceipt {
    pub(crate) fn finish(self) -> Result<f64> {
        self.receiver
            .recv()
            .map_err(|_| anyhow!("async activation forwarder dropped result"))?
            .map_err(|error| anyhow!(error))
    }

    fn try_finish(&self) -> Result<Option<f64>> {
        match self.receiver.try_recv() {
            Ok(Ok(write_ms)) => Ok(Some(write_ms)),
            Ok(Err(error)) => Err(anyhow!(error)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                Err(anyhow!("async activation forwarder dropped result"))
            }
        }
    }
}
