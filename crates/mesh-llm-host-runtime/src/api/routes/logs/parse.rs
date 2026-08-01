use chrono::{DateTime, SecondsFormat, Utc};
use mesh_llm_log_store::{PageQuery, ProxyQuery, QuerySort, RequestOutcome, RequestQuery};

use super::error::LogsError;

const DEFAULT_LIMIT: usize = 50;
pub(super) const MAX_LIMIT: usize = 100;
const MAX_FILTER_LENGTH: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SourceFilter {
    Active,
    Durable,
}

pub(super) struct RequestListQuery {
    pub store: RequestQuery,
    pub source: Option<SourceFilter>,
    pub cursor_boundary: Option<(String, String)>,
}

pub(super) fn request_query(path: &str) -> Result<RequestListQuery, LogsError> {
    let pairs = pairs(path)?;
    let mut query = RequestQuery {
        limit: DEFAULT_LIMIT,
        cursor: None,
        from: None,
        to: None,
        route: None,
        model: None,
        provider: None,
        engine: None,
        status_code: None,
        outcome: None,
        sort: QuerySort::Descending,
    };
    let mut source = None;
    for (key, value) in pairs {
        match key.as_str() {
            "cursor" => query.cursor = Some(nonempty(value)?),
            "limit" => query.limit = limit(&value)?,
            "from" => query.from = Some(timestamp(&value)?),
            "to" => query.to = Some(timestamp(&value)?),
            "route" => query.route = Some(filter(value)?),
            "model" => query.model = Some(filter(value)?),
            "provider" => query.provider = Some(filter(value)?),
            "engine" => query.engine = Some(filter(value)?),
            "status" => query.status_code = Some(status(&value)?),
            "outcome" => query.outcome = Some(outcome(&value)?),
            "source" => source = Some(source_filter(&value)?),
            "sort" => query.sort = sort(&value)?,
            _ => return Err(LogsError::InvalidQuery("unknown request filter")),
        }
    }
    if query.from > query.to && query.to.is_some() {
        return Err(LogsError::InvalidQuery("from must not be after to"));
    }
    let cursor_boundary = query
        .cursor
        .as_deref()
        .map(mesh_llm_log_store::decode_cursor)
        .transpose()
        .map_err(|_| LogsError::InvalidCursor)?;
    Ok(RequestListQuery {
        store: query,
        source,
        cursor_boundary,
    })
}

pub(super) fn page_query(path: &str) -> Result<PageQuery, LogsError> {
    let mut query = PageQuery {
        limit: DEFAULT_LIMIT,
        cursor: None,
        sort: QuerySort::Ascending,
    };
    for (key, value) in pairs(path)? {
        match key.as_str() {
            "cursor" => query.cursor = Some(nonempty(value)?),
            "limit" => query.limit = limit(&value)?,
            "sort" => query.sort = sort(&value)?,
            _ => return Err(LogsError::InvalidQuery("unknown page filter")),
        }
    }
    if let Some(cursor) = query.cursor.as_deref() {
        mesh_llm_log_store::decode_cursor(cursor).map_err(|_| LogsError::InvalidCursor)?;
    }
    Ok(query)
}

pub(super) fn proxy_query(path: &str) -> Result<ProxyQuery, LogsError> {
    let mut query = ProxyQuery {
        page: PageQuery {
            limit: DEFAULT_LIMIT,
            cursor: None,
            sort: QuerySort::Descending,
        },
        request_id: None,
        provider: None,
        engine: None,
        status_code: None,
    };
    for (key, value) in pairs(path)? {
        match key.as_str() {
            "cursor" => query.page.cursor = Some(nonempty(value)?),
            "limit" => query.page.limit = limit(&value)?,
            "sort" => query.page.sort = sort(&value)?,
            "request_id" => query.request_id = Some(id(&value)?),
            "provider" => query.provider = Some(filter(value)?),
            "engine" => query.engine = Some(filter(value)?),
            "status" => query.status_code = Some(status(&value)?),
            _ => return Err(LogsError::InvalidQuery("unknown proxy filter")),
        }
    }
    if let Some(cursor) = query.page.cursor.as_deref() {
        mesh_llm_log_store::decode_cursor(cursor).map_err(|_| LogsError::InvalidCursor)?;
    }
    Ok(query)
}

pub(super) fn id(value: &str) -> Result<String, LogsError> {
    uuid::Uuid::parse_str(value)
        .map(|id| id.to_string())
        .map_err(|_| LogsError::InvalidId)
}

fn pairs(path: &str) -> Result<Vec<(String, String)>, LogsError> {
    let Some(raw) = path.split_once('?').map(|(_, query)| query) else {
        return Ok(Vec::new());
    };
    valid_percent_encoding(raw)?;
    let pairs = url::form_urlencoded::parse(raw.as_bytes())
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    let mut keys = std::collections::HashSet::with_capacity(pairs.len());
    if pairs.iter().any(|(key, _)| !keys.insert(key.clone())) {
        return Err(LogsError::InvalidQuery("duplicate query parameter"));
    }
    Ok(pairs)
}

fn valid_percent_encoding(value: &str) -> Result<(), LogsError> {
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let valid = bytes.get(index + 1).is_some_and(u8::is_ascii_hexdigit)
                && bytes.get(index + 2).is_some_and(u8::is_ascii_hexdigit);
            if !valid {
                return Err(LogsError::InvalidQuery("query encoding is malformed"));
            }
            index += 3;
        } else {
            index += 1;
        }
    }
    Ok(())
}

fn nonempty(value: String) -> Result<String, LogsError> {
    if value.is_empty() {
        Err(LogsError::InvalidQuery("query value must not be empty"))
    } else {
        Ok(value)
    }
}

fn filter(value: String) -> Result<String, LogsError> {
    if value.is_empty() || value.len() > MAX_FILTER_LENGTH || value.chars().any(char::is_control) {
        Err(LogsError::InvalidQuery("filter value is invalid"))
    } else {
        Ok(value)
    }
}

fn limit(value: &str) -> Result<usize, LogsError> {
    let value = value
        .parse::<usize>()
        .map_err(|_| LogsError::InvalidQuery("limit must be an integer"))?;
    if (1..=MAX_LIMIT).contains(&value) {
        Ok(value)
    } else {
        Err(LogsError::InvalidQuery("limit must be between 1 and 100"))
    }
}

fn timestamp(value: &str) -> Result<String, LogsError> {
    DateTime::parse_from_rfc3339(value)
        .map(|timestamp| {
            timestamp
                .with_timezone(&Utc)
                .to_rfc3339_opts(SecondsFormat::Secs, true)
        })
        .map_err(|_| LogsError::InvalidQuery("time filter must be RFC 3339"))
}

fn status(value: &str) -> Result<u16, LogsError> {
    let status = value
        .parse::<u16>()
        .map_err(|_| LogsError::InvalidQuery("status must be an HTTP status code"))?;
    if (100..=599).contains(&status) {
        Ok(status)
    } else {
        Err(LogsError::InvalidQuery(
            "status must be between 100 and 599",
        ))
    }
}

fn outcome(value: &str) -> Result<RequestOutcome, LogsError> {
    match value {
        "active" => Ok(RequestOutcome::Active),
        "completed" => Ok(RequestOutcome::Completed),
        "failed" => Ok(RequestOutcome::Failed),
        "rejected" => Ok(RequestOutcome::Rejected),
        "cancelled" => Ok(RequestOutcome::Cancelled),
        "dropped" => Ok(RequestOutcome::Dropped),
        _ => Err(LogsError::InvalidQuery("outcome is invalid")),
    }
}

fn source_filter(value: &str) -> Result<SourceFilter, LogsError> {
    match value {
        "active" => Ok(SourceFilter::Active),
        "durable" => Ok(SourceFilter::Durable),
        _ => Err(LogsError::InvalidQuery("source is invalid")),
    }
}

fn sort(value: &str) -> Result<QuerySort, LogsError> {
    match value {
        "asc" => Ok(QuerySort::Ascending),
        "desc" => Ok(QuerySort::Descending),
        _ => Err(LogsError::InvalidQuery("sort must be asc or desc")),
    }
}
