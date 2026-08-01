mod related;
mod requests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuerySort {
    Ascending,
    Descending,
}

impl QuerySort {
    pub(crate) const fn sql_order(self) -> &'static str {
        match self {
            Self::Ascending => "ASC",
            Self::Descending => "DESC",
        }
    }

    pub(crate) const fn cursor_operator(self) -> &'static str {
        match self {
            Self::Ascending => ">",
            Self::Descending => "<",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestOutcome {
    Active,
    Completed,
    Failed,
    Rejected,
    Cancelled,
    Dropped,
}

impl RequestOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Rejected => "rejected",
            Self::Cancelled => "cancelled",
            Self::Dropped => "dropped",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RequestQuery {
    pub limit: usize,
    pub cursor: Option<String>,
    pub from: Option<String>,
    pub to: Option<String>,
    pub route: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub engine: Option<String>,
    pub status_code: Option<u16>,
    pub outcome: Option<RequestOutcome>,
    pub sort: QuerySort,
}

#[derive(Clone, Debug)]
pub struct PageQuery {
    pub limit: usize,
    pub cursor: Option<String>,
    pub sort: QuerySort,
}

#[derive(Clone, Debug)]
pub struct ProxyQuery {
    pub page: PageQuery,
    pub request_id: Option<String>,
    pub provider: Option<String>,
    pub engine: Option<String>,
    pub status_code: Option<u16>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestRecord {
    pub request_id: String,
    pub outcome: String,
    pub created_at: String,
    pub terminal_at: Option<String>,
    pub route: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub engine: Option<String>,
    pub status_code: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EventRecord {
    pub event_id: String,
    pub request_id: String,
    pub occurred_at: String,
    pub payload_json: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactRecord {
    pub artifact_id: String,
    pub request_id: String,
    pub occurred_at: String,
    pub kind: String,
    pub media_kind: Option<String>,
    pub checksum: Option<String>,
    pub bytes: i64,
    pub version: i32,
    pub redacted: bool,
    pub truncated: bool,
    pub missing: bool,
    pub corrupt: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProxyRecord {
    pub attempt_id: String,
    pub request_id: String,
    pub occurred_at: String,
    pub target: String,
    pub provider: Option<String>,
    pub engine: Option<String>,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub status_code: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueryPage<T> {
    pub items: Vec<T>,
    pub next_cursor: Option<String>,
}
