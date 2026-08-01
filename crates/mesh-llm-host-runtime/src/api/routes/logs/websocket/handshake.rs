use base64::Engine;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::handshake::derive_accept_key;

#[derive(Debug)]
pub(super) struct UpgradeHandshake {
    accept_key: String,
}

#[derive(Debug)]
pub(super) enum HandshakeError {
    Connection,
    Key,
    Method,
    Parse,
    Upgrade,
    Version,
}

impl UpgradeHandshake {
    pub(super) async fn write_response(&self, stream: &mut TcpStream) -> anyhow::Result<()> {
        let response = format!(
            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {}\r\n\r\n",
            self.accept_key
        );
        stream.write_all(response.as_bytes()).await?;
        Ok(())
    }
}

pub(super) fn validate(raw_request: &[u8]) -> Result<UpgradeHandshake, HandshakeError> {
    let mut headers = [httparse::EMPTY_HEADER; 32];
    let mut request = httparse::Request::new(&mut headers);
    match request
        .parse(raw_request)
        .map_err(|_| HandshakeError::Parse)?
    {
        httparse::Status::Complete(_) => {}
        httparse::Status::Partial => return Err(HandshakeError::Parse),
    }
    if request.method != Some("GET") {
        return Err(HandshakeError::Method);
    }
    let websocket_upgrade = header(&request, "upgrade")
        .map_err(|_| HandshakeError::Upgrade)?
        .is_some_and(|value| value.eq_ignore_ascii_case("websocket"));
    let connection_upgrade =
        connection_upgrades(&request).map_err(|_| HandshakeError::Connection)?;
    if !connection_upgrade || !websocket_upgrade {
        return Err(HandshakeError::Upgrade);
    }
    if header(&request, "sec-websocket-version").map_err(|_| HandshakeError::Version)? != Some("13")
    {
        return Err(HandshakeError::Version);
    }
    let key = header(&request, "sec-websocket-key")
        .map_err(|_| HandshakeError::Key)?
        .ok_or(HandshakeError::Key)?;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(key)
        .map_err(|_| HandshakeError::Key)?;
    if decoded.len() != 16 {
        return Err(HandshakeError::Key);
    }

    Ok(UpgradeHandshake {
        accept_key: derive_accept_key(key.as_bytes()),
    })
}

fn connection_upgrades<'headers, 'buffer>(
    request: &httparse::Request<'headers, 'buffer>,
) -> Result<bool, ()> {
    let Some(connection) = header(request, "connection")? else {
        return Ok(false);
    };
    Ok(connection
        .split(',')
        .any(|value| value.trim().eq_ignore_ascii_case("upgrade")))
}

fn header<'headers, 'buffer>(
    request: &httparse::Request<'headers, 'buffer>,
    name: &str,
) -> Result<Option<&'buffer str>, ()> {
    let mut values = request
        .headers
        .iter()
        .filter(|header| header.name.eq_ignore_ascii_case(name));
    let Some(value) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() {
        return Err(());
    }
    std::str::from_utf8(value.value).map(Some).map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use super::validate;

    #[test]
    fn accepts_rfc6455_upgrade_headers() {
        let request = b"GET /api/logs/stream HTTP/1.1\r\nHost: localhost\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n";
        let result = validate(request);
        assert!(result.is_ok(), "{result:?}");
    }
}
