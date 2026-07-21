use std::{
    collections::HashMap,
    io,
    net::{IpAddr, SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
        mpsc::TryRecvError,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use skippy_protocol::{
    StageConfig, StageTopology,
    binary::{
        StageReply, StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind,
        WireReplyKind, read_stage_message, recv_ready, recv_reply, send_ready, send_reply_message,
        write_stage_message,
    },
};

use super::socket::{connect_downstream_socket, downstream_source_ip, resolve_downstream_endpoint};
use super::stage_execution::{
    consume_optional_client_ready_hello, send_client_ready_hello_if_enabled,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PredictionReturnKey {
    request_id: u64,
    session_id: u64,
}

impl PredictionReturnKey {
    pub(crate) fn new(request_id: u64, session_id: u64) -> Self {
        Self {
            request_id,
            session_id,
        }
    }
}

pub struct PredictionReturnHub {
    waiters: Mutex<HashMap<PredictionReturnKey, mpsc::Sender<Result<StageReply, String>>>>,
}

#[derive(Default)]
pub(crate) struct PredictionReturnSinks {
    streams: Mutex<HashMap<PredictionReturnKey, TcpStream>>,
}

impl Default for PredictionReturnHub {
    fn default() -> Self {
        Self {
            waiters: Mutex::new(HashMap::new()),
        }
    }
}

pub struct PredictionReturnListener {
    shutdown: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    hub: Arc<PredictionReturnHub>,
}

impl PredictionReturnListener {
    pub fn start(bind_addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(bind_addr)
            .with_context(|| format!("bind direct prediction return listener {bind_addr}"))?;
        listener
            .set_nonblocking(true)
            .context("set direct prediction return listener nonblocking")?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let thread_shutdown = shutdown.clone();
        let hub = Arc::new(PredictionReturnHub::default());
        let thread_hub = hub.clone();
        let thread = thread::spawn(move || {
            while !thread_shutdown.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        if let Err(error) = stream.set_nonblocking(false) {
                            eprintln!(
                                "direct prediction return connection failed: set blocking: {error}"
                            );
                            continue;
                        }
                        let hub = thread_hub.clone();
                        thread::spawn(move || {
                            if let Err(error) = handle_prediction_return_connection(hub, stream) {
                                eprintln!("direct prediction return connection failed: {error:#}");
                            }
                        });
                    }
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(50));
                    }
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                    Err(error) => {
                        eprintln!("direct prediction return listener failed: {error}");
                        break;
                    }
                }
            }
        });
        Ok(Self {
            shutdown,
            thread: Some(thread),
            hub,
        })
    }

    pub fn hub(&self) -> Arc<PredictionReturnHub> {
        self.hub.clone()
    }
}

impl Drop for PredictionReturnListener {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn handle_prediction_return_connection(
    hub: Arc<PredictionReturnHub>,
    mut stream: TcpStream,
) -> Result<()> {
    consume_optional_client_ready_hello(&mut stream)
        .context("consume optional direct prediction return client ready hello")?;
    send_ready(&mut stream).context("send direct prediction return ready")?;
    let open = read_stage_message(&mut stream, 0).context("read direct prediction return open")?;
    hub.handle_return_connection(open, stream)
}

impl PredictionReturnHub {
    pub(crate) fn register(
        self: &Arc<Self>,
        request_id: u64,
        session_id: u64,
    ) -> Result<PredictionReturnReceiver> {
        let key = PredictionReturnKey::new(request_id, session_id);
        let (sender, receiver) = mpsc::channel();
        self.waiters
            .lock()
            .map_err(|_| anyhow!("prediction return hub lock poisoned"))?
            .insert(key, sender);
        Ok(PredictionReturnReceiver {
            key,
            hub: self.clone(),
            receiver,
            direct_reply_seen: Arc::new(AtomicBool::new(false)),
        })
    }

    pub(crate) fn unregister(&self, key: PredictionReturnKey) {
        if let Ok(mut waiters) = self.waiters.lock() {
            waiters.remove(&key);
        }
    }

    pub(crate) fn handle_return_connection(
        &self,
        open: StageWireMessage,
        stream: TcpStream,
    ) -> Result<()> {
        if open.kind != WireMessageKind::PredictionReturnOpen {
            bail!("expected prediction return open message");
        }
        let key = PredictionReturnKey::new(open.request_id, open.session_id);
        self.handle_return_stream(key, stream)
    }

    fn handle_return_stream(&self, key: PredictionReturnKey, mut stream: TcpStream) -> Result<()> {
        let sender = self
            .waiters
            .lock()
            .map_err(|_| anyhow!("prediction return hub lock poisoned"))?
            .get(&key)
            .cloned()
            .ok_or_else(|| anyhow!("no prediction return waiter for request {}", key.request_id))?;
        loop {
            match recv_reply(&mut stream) {
                Ok(reply) => {
                    if sender.send(Ok(reply)).is_err() {
                        return Ok(());
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(error) => {
                    let _ = sender.send(Err(error.to_string()));
                    return Err(error).context("read direct prediction return");
                }
            }
        }
    }
}

pub(crate) struct PredictionReturnReceiver {
    key: PredictionReturnKey,
    hub: Arc<PredictionReturnHub>,
    receiver: mpsc::Receiver<Result<StageReply, String>>,
    /// Trips the first time a reply is actually delivered over the direct route.
    ///
    /// This is the implicit "selection confirmation": the tail sends replies on
    /// this socket ONLY if it selected it as the return route for this request.
    /// A successful `PredictionReturnOpen` write does not prove selection (the
    /// tail may still reply on the forward lane), but an actual delivered direct
    /// reply does. Pipelining (depth > 1, direct-only completion) must gate on
    /// this to avoid a 300s hang waiting on a route the tail never chose.
    direct_reply_seen: Arc<AtomicBool>,
}

impl PredictionReturnReceiver {
    pub(crate) fn attach_opened_stream(&self, stream: TcpStream) {
        let hub = self.hub.clone();
        let key = self.key;
        thread::spawn(move || {
            if let Err(error) = hub.handle_return_stream(key, stream) {
                eprintln!("direct prediction return reader failed: {error:#}");
            }
        });
    }

    /// Whether a reply has been observed on the direct return route. Once true,
    /// the tail has committed to replying directly for this request (route
    /// selection is stable after generation configuration), so dependent
    /// (pipelined) direct-only windows are safe to dispatch.
    pub(crate) fn direct_confirmed(&self) -> bool {
        self.direct_reply_seen.load(Ordering::Acquire)
    }

    /// Clear the confirmation flag. Called once right after the generation-config
    /// ACK so that only POST-config direct replies count as route confirmation.
    ///
    /// Pre-config phases (prefix-cache restore, prefill) can send one-off direct
    /// replies over sockets that are NOT the persistent generation return route;
    /// counting those would falsely confirm the route and seed a direct-only
    /// pipeline that then hangs waiting on a route the tail did not actually
    /// select. Resetting here is safe because those pre-config replies are
    /// consumed synchronously before generation config completes.
    pub(crate) fn reset_direct_confirmation(&self) {
        self.direct_reply_seen.store(false, Ordering::Release);
    }

    pub(crate) fn try_recv_expected(&self, expected: WireReplyKind) -> Result<Option<StageReply>> {
        self.try_recv_one_of(std::slice::from_ref(&expected))
    }

    pub(crate) fn try_recv_one_of(&self, expected: &[WireReplyKind]) -> Result<Option<StageReply>> {
        let Some(reply) = self.try_recv()? else {
            return Ok(None);
        };
        if !expected.contains(&reply.kind) {
            bail!(
                "expected one of {expected:?} from direct prediction return, got {:?}",
                reply.kind
            );
        }
        Ok(Some(reply))
    }

    fn try_recv(&self) -> Result<Option<StageReply>> {
        match self.receiver.try_recv() {
            Ok(Ok(reply)) => {
                // A delivered direct reply confirms the tail selected this route.
                self.direct_reply_seen.store(true, Ordering::Release);
                Ok(Some(reply))
            }
            Ok(Err(error)) => Err(anyhow!(error)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                Err(anyhow!("prediction return channel disconnected"))
            }
        }
    }
}

impl Drop for PredictionReturnReceiver {
    fn drop(&mut self) {
        self.hub.unregister(self.key);
    }
}

impl PredictionReturnSinks {
    pub(crate) fn insert_opened_sink(
        &self,
        open: StageWireMessage,
        stream: TcpStream,
    ) -> Result<()> {
        if open.kind != WireMessageKind::PredictionReturnOpen {
            bail!("expected prediction return open message");
        }
        let key = PredictionReturnKey::new(open.request_id, open.session_id);
        self.streams
            .lock()
            .map_err(|_| anyhow!("prediction return sinks lock poisoned"))?
            .insert(key, stream);
        Ok(())
    }

    pub(crate) fn take_wait(
        &self,
        request_id: u64,
        session_id: u64,
        timeout: Duration,
    ) -> Result<Option<TcpStream>> {
        let key = PredictionReturnKey::new(request_id, session_id);
        let started = std::time::Instant::now();
        loop {
            if let Some(stream) = self
                .streams
                .lock()
                .map_err(|_| anyhow!("prediction return sinks lock poisoned"))?
                .remove(&key)
            {
                return Ok(Some(stream));
            }
            if started.elapsed() >= timeout {
                return Ok(None);
            }
            thread::sleep(Duration::from_millis(2));
        }
    }

    pub(crate) fn remove(&self, request_id: u64, session_id: u64) {
        let key = PredictionReturnKey::new(request_id, session_id);
        if let Ok(mut streams) = self.streams.lock() {
            streams.remove(&key);
        }
    }
}

/// Read timeout for the return-sink ready handshake. `recv_ready` is a blocking
/// `read_exact`; without this a stalled downstream connection hangs the open
/// forever, which mid-generation blocks the request from ever falling back to
/// the upstream reply. Cleared afterwards so the sink's normal reads stay
/// blocking.
///
/// Budget sizing (20s): over a WAN mesh the return sink connects to a LOCAL
/// bridge alias, but the remote `ready` byte only arrives after the bridge
/// COLD-establishes a fresh stage QUIC connection (up to ~10s) and the remote
/// inbound handler then dials its local binary server. A 5s budget timed out
/// during that cold setup (observed EAGAIN on a healthy ~26ms WAN split), even
/// though the pooled forward lanes — which get a 20s initial connect budget and
/// are then reused — succeeded on the same bridge. Matching the forward-lane
/// budget lets the cold return path complete instead of failing to the slower
/// upstream-reply fallback.
///
/// This is a *single bounded deadline*, not a retry budget: the sink is opened
/// on the generation hot path, and `connect_downstream_socket` already bounds
/// the connect itself, so wrapping this in an outer retry only compounds the
/// worst-case stall (see PR #1011 review).
const RETURN_SINK_READY_READ_TIMEOUT: Duration = Duration::from_secs(20);

/// Connect to `return_addr` and complete the ready handshake, WITHOUT sending a
/// `PredictionReturnOpen` message. This is the expensive, WAN-flaky half of
/// opening a return sink (fresh TCP connect through the local bridge alias +
/// QUIC bi-stream setup + blocking `recv_ready`). Splitting it out lets a pool
/// pre-warm return sockets off the generation hot path, exactly like the
/// forward `PersistentStageLanePool`.
///
/// A prepared-but-unbound socket is safe to park: the remote binary stage
/// server's first-message read is an untimed blocking read, so it simply waits
/// until we later send `PredictionReturnOpen` to bind the socket to a specific
/// `(request_id, session_id)`.
fn prepare_return_sink_socket(
    return_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    not_ready_context: &'static str,
) -> Result<TcpStream> {
    let mut stream = connect_downstream_socket(return_addr, source_ip, Duration::from_secs(2))
        .map_err(|error| anyhow!(error))?;
    stream.set_nodelay(true).ok();
    send_client_ready_hello_if_enabled(&mut stream)
        .context("send prediction return client ready hello")?;
    // Bound the ready handshake read. `recv_ready` is a blocking `read_exact`;
    // without a timeout a stalled downstream connection hangs the return-sink
    // open forever, blocking generation from falling back to the upstream reply.
    // A single short deadline (no outer retry) fails fast to that fallback.
    // Both the set and the clear are propagated: if the set fails, `recv_ready`
    // would be unbounded (defeating the fix); if the clear fails, the handshake
    // timeout would leak into the sink's later reads.
    stream
        .set_read_timeout(Some(RETURN_SINK_READY_READ_TIMEOUT))
        .context("set prediction return ready read timeout")?;
    let ready = recv_ready(&mut stream).context(not_ready_context);
    stream
        .set_read_timeout(None)
        .context("clear prediction return ready read timeout")?;
    ready?;
    Ok(stream)
}

/// Bind a prepared (connected + ready) return socket to a specific request by
/// writing the `PredictionReturnOpen` message. Cheap: a single write, safe to
/// run on the generation hot path.
fn bind_prepared_return_sink(
    stream: &mut TcpStream,
    request_id: u64,
    session_id: u64,
    wire_dtype: WireActivationDType,
) -> Result<()> {
    write_stage_message(
        stream,
        &prediction_return_open_message(request_id, session_id),
        wire_dtype,
    )
    .context("open prediction return stream")
}

/// Connect to `return_addr`, complete the ready handshake, and send the
/// prediction-return open message. Single bounded attempt — on failure the
/// caller falls back to the upstream reply path. This is the cold path used
/// when no pre-warmed socket is available.
fn open_return_sink_once(
    return_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    request_id: u64,
    session_id: u64,
    wire_dtype: WireActivationDType,
    not_ready_context: &'static str,
) -> Result<TcpStream> {
    let mut stream = prepare_return_sink_socket(return_addr, source_ip, not_ready_context)?;
    bind_prepared_return_sink(&mut stream, request_id, session_id, wire_dtype)?;
    Ok(stream)
}

pub(crate) fn open_prediction_return_stream(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    request_id: u64,
    session_id: u64,
    wire_dtype: WireActivationDType,
    _timeout_secs: u64,
) -> Result<TcpStream> {
    let endpoint = driver_stage_endpoint(config, topology)?;
    let return_addr = resolve_downstream_endpoint(endpoint)?;
    let source_ip = downstream_source_ip(config)?;
    open_return_sink_once(
        return_addr,
        source_ip,
        request_id,
        session_id,
        wire_dtype,
        "prediction return sink did not become ready",
    )
    .with_context(|| format!("connect direct prediction return sink at {endpoint}"))
}

pub(crate) fn open_downstream_prediction_return_stream(
    config: &StageConfig,
    request_id: u64,
    session_id: u64,
    wire_dtype: WireActivationDType,
) -> Result<TcpStream> {
    let downstream = config
        .downstream
        .as_ref()
        .ok_or_else(|| anyhow!("direct prediction return requires downstream stage"))?;
    let endpoint = strip_tcp_prefix(&downstream.endpoint);
    let return_addr = resolve_downstream_endpoint(endpoint)?;
    let source_ip = downstream_source_ip(config)?;
    open_return_sink_once(
        return_addr,
        source_ip,
        request_id,
        session_id,
        wire_dtype,
        "downstream prediction return sink did not become ready",
    )
    .with_context(|| format!("connect downstream prediction return sink at {endpoint}"))
}

/// Pre-warm a downstream return sink: perform the WAN-flaky connect + ready
/// handshake WITHOUT binding it to a request. The returned socket is parked and
/// can later be bound cheaply on the generation hot path via
/// [`bind_downstream_prediction_return_socket`]. Used by the prepared-return
/// pool so the cold, unreliable half of opening a return sink happens off the
/// request path (mirrors the forward `PersistentStageLanePool` pre-warm).
pub(crate) fn prepare_downstream_prediction_return_socket(
    config: &StageConfig,
) -> Result<TcpStream> {
    let downstream = config
        .downstream
        .as_ref()
        .ok_or_else(|| anyhow!("direct prediction return requires downstream stage"))?;
    let endpoint = strip_tcp_prefix(&downstream.endpoint);
    let return_addr = resolve_downstream_endpoint(endpoint)?;
    let source_ip = downstream_source_ip(config)?;
    prepare_return_sink_socket(
        return_addr,
        source_ip,
        "downstream prediction return sink did not become ready",
    )
    .with_context(|| format!("prewarm downstream prediction return sink at {endpoint}"))
}

/// Bind a pre-warmed downstream return socket to a specific request by writing
/// the `PredictionReturnOpen` message. Cheap; safe on the generation hot path.
pub(crate) fn bind_downstream_prediction_return_socket(
    stream: &mut TcpStream,
    request_id: u64,
    session_id: u64,
    wire_dtype: WireActivationDType,
) -> Result<()> {
    bind_prepared_return_sink(stream, request_id, session_id, wire_dtype)
}

pub(crate) fn send_direct_prediction_return(
    stream: &mut TcpStream,
    reply: StageReply,
) -> Result<()> {
    send_reply_message(stream, &reply).context("send direct prediction return")
}

fn driver_stage_endpoint<'a>(
    config: &'a StageConfig,
    topology: Option<&'a StageTopology>,
) -> Result<&'a str> {
    if let Some(topology) = topology {
        return driver_stage_endpoint_from_topology(topology);
    }
    if let Some(upstream) = config
        .upstream
        .as_ref()
        .filter(|upstream| upstream.stage_index == 0)
    {
        return Ok(strip_tcp_prefix(&upstream.endpoint));
    }
    Err(anyhow!("direct prediction return requires topology"))
}

fn driver_stage_endpoint_from_topology(topology: &StageTopology) -> Result<&str> {
    topology
        .stages
        .iter()
        .find(|stage| stage.stage_index == 0)
        .map(|stage| strip_tcp_prefix(&stage.endpoint))
        .ok_or_else(|| anyhow!("topology does not contain driver-facing stage 0"))
}

fn strip_tcp_prefix(endpoint: &str) -> &str {
    endpoint.strip_prefix("tcp://").unwrap_or(endpoint)
}

fn prediction_return_open_message(request_id: u64, session_id: u64) -> StageWireMessage {
    StageWireMessage {
        kind: WireMessageKind::PredictionReturnOpen,
        pos_start: 0,
        token_count: 0,
        state: StageStateHeader::new(
            WireMessageKind::PredictionReturnOpen,
            WireActivationDType::F32,
        ),
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    }
}

/// Classify a return-sink open/bind failure into a bounded, privacy-safe phase
/// label for telemetry. Raw endpoints and full error chains stay in local
/// stderr logs only; OTLP attributes must not carry them.
///
/// Phase meanings (see the return-sink open sequence in `open_return_sink_once`
/// and `prepare_return_sink_socket`):
/// - `connect_refused` — local bridge listener gone/wrong (no one accepted).
/// - `connect_timeout` — TCP connect to the (local bridge) endpoint timed out.
/// - `ready_timeout` — local connect ok; no `ready` handshake byte in time
///   (bridge/QUIC/remote-accept problem — the WAN leg).
/// - `remote_eof` — remote/bridge accepted then closed before handshake.
/// - `open_write_failed` — handshake ok but the open message write failed.
/// - `other` — anything else; inspect local logs.
pub(crate) fn classify_return_failure_phase(error_detail: &str) -> &'static str {
    let lower = error_detail.to_ascii_lowercase();
    if lower.contains("did not become ready") || lower.contains("ready read timeout") {
        "ready_timeout"
    } else if lower.contains("connection refused") {
        "connect_refused"
    } else if lower.contains("timed out") || lower.contains("timeout") {
        "connect_timeout"
    } else if lower.contains("unexpectedeof")
        || lower.contains("unexpected eof")
        || lower.contains("connection reset")
    {
        "remote_eof"
    } else if lower.contains("open prediction return stream") || lower.contains("broken pipe") {
        "open_write_failed"
    } else {
        "other"
    }
}

#[cfg(test)]
mod failure_phase_tests {
    use super::classify_return_failure_phase;

    #[test]
    fn maps_ready_timeout_the_wan_leg() {
        assert_eq!(
            classify_return_failure_phase("downstream prediction return sink did not become ready"),
            "ready_timeout"
        );
    }

    #[test]
    fn maps_connect_refused_local_bridge_gone() {
        assert_eq!(
            classify_return_failure_phase(
                "connect downstream prediction return sink at 127.0.0.1:54321: Connection refused (os error 61)"
            ),
            "connect_refused"
        );
    }

    #[test]
    fn maps_remote_eof() {
        assert_eq!(
            classify_return_failure_phase("read direct prediction return: UnexpectedEof"),
            "remote_eof"
        );
    }

    #[test]
    fn maps_open_write_failed() {
        assert_eq!(
            classify_return_failure_phase("open prediction return stream: broken pipe"),
            "open_write_failed"
        );
    }

    #[test]
    fn unknown_is_other() {
        assert_eq!(
            classify_return_failure_phase("some unexpected condition"),
            "other"
        );
    }

    #[test]
    fn connect_timeout_distinct_from_ready_timeout() {
        assert_eq!(
            classify_return_failure_phase("connect ...: operation timed out"),
            "connect_timeout"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::{recv_reply, send_reply_predicted_with_stats};

    #[test]
    fn handle_return_connection_delivers_reply_to_registered_waiter() {
        let request_id = 17;
        let session_id = 23;
        let hub = Arc::new(PredictionReturnHub::default());
        let receiver = hub.register(request_id, session_id).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        let open = prediction_return_open_message(request_id, session_id);
        let handle = {
            let hub = hub.clone();
            thread::spawn(move || hub.handle_return_connection(open, server))
        };

        send_reply_predicted_with_stats(&mut client, 42, Default::default()).unwrap();

        let reply = poll_test_reply(&receiver, WireReplyKind::PredictedToken);
        assert_eq!(reply.predicted, 42);
        drop(client);
        handle.join().unwrap().unwrap();
    }

    #[test]
    fn direct_prediction_return_preserves_typed_native_mtp_draft() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        let reply = StageReply {
            kind: WireReplyKind::PredictedToken,
            predicted: 42,
            predicted_tokens: vec![42],
            native_mtp_draft: Some(skippy_protocol::binary::StageNativeMtpDraft {
                token_ids: vec![43],
                proposal_compute_us: 123,
            }),
            window: skippy_protocol::binary::StageReplyWindow {
                window_id: 7,
                accepted_len: 2,
                correction_token: 123,
            },
            stats: Default::default(),
        };
        send_direct_prediction_return(&mut server, reply).unwrap();

        let received = recv_reply(&mut client).unwrap();
        assert_eq!(received.kind, WireReplyKind::PredictedToken);
        assert_eq!(received.predicted, 42);
        assert_eq!(received.predicted_tokens, vec![42]);
        assert_eq!(
            received.native_mtp_draft,
            Some(skippy_protocol::binary::StageNativeMtpDraft {
                token_ids: vec![43],
                proposal_compute_us: 123,
            })
        );
        assert_eq!(received.window.window_id, 7);
        assert_eq!(received.window.accepted_len, 2);
        assert_eq!(received.window.correction_token, 123);
    }

    #[test]
    fn prediction_return_sinks_store_upstream_opened_streams() {
        let request_id = 31;
        let session_id = 37;
        let sinks = PredictionReturnSinks::default();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();

        sinks
            .insert_opened_sink(
                prediction_return_open_message(request_id, session_id),
                server,
            )
            .unwrap();

        let stream = sinks
            .take_wait(request_id, session_id, Duration::from_millis(1))
            .unwrap()
            .expect("registered prediction return sink");
        assert_eq!(stream.peer_addr().unwrap(), client.local_addr().unwrap());
    }

    #[test]
    fn prediction_return_sinks_remove_abandoned_streams() {
        let request_id = 41;
        let session_id = 43;
        let sinks = PredictionReturnSinks::default();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();

        sinks
            .insert_opened_sink(
                prediction_return_open_message(request_id, session_id),
                server,
            )
            .unwrap();
        sinks.remove(request_id, session_id);

        assert!(
            sinks
                .take_wait(request_id, session_id, Duration::from_millis(1))
                .unwrap()
                .is_none()
        );
        drop(client);
    }

    fn poll_test_reply(receiver: &PredictionReturnReceiver, expected: WireReplyKind) -> StageReply {
        let started = std::time::Instant::now();
        loop {
            if let Some(reply) = receiver.try_recv_expected(expected).unwrap() {
                return reply;
            }
            assert!(
                started.elapsed() < Duration::from_secs(1),
                "timed out waiting for prediction return reply"
            );
            thread::sleep(Duration::from_millis(1));
        }
    }
}
