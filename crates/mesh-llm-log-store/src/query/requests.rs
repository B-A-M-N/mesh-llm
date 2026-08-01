use rusqlite::types::Value;

use super::{QueryPage, RequestQuery, RequestRecord};
use crate::cursor::{decode_cursor, encode_cursor};
use crate::{LogStore, LogStoreError};

const REQUEST_COLUMNS: &str =
    "request_id, state, created_at, terminal_at, route, model, provider, engine, status_code";

impl LogStore {
    pub fn query_request(&self, request_id: &str) -> Result<Option<RequestRecord>, LogStoreError> {
        let connection = self.conn();
        let sql = format!("SELECT {REQUEST_COLUMNS} FROM summaries WHERE request_id = ?");
        match connection.query_row(&sql, [request_id], request_record) {
            Ok(record) => Ok(Some(record)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(error) => Err(LogStoreError::QueryFailed(error.to_string())),
        }
    }

    pub fn query_requests(
        &self,
        query: &RequestQuery,
    ) -> Result<QueryPage<RequestRecord>, LogStoreError> {
        let mut sql = format!("SELECT {REQUEST_COLUMNS} FROM summaries WHERE 1 = 1");
        let mut values = Vec::<Value>::new();
        if let Some(from) = &query.from {
            sql.push_str(" AND created_at >= ?");
            values.push(Value::Text(from.clone()));
        }
        if let Some(to) = &query.to {
            sql.push_str(" AND created_at <= ?");
            values.push(Value::Text(to.clone()));
        }
        for (column, value) in [
            ("route", &query.route),
            ("model", &query.model),
            ("provider", &query.provider),
            ("engine", &query.engine),
        ] {
            if let Some(value) = value {
                sql.push_str(&format!(" AND {column} = ?"));
                values.push(Value::Text(value.clone()));
            }
        }
        if let Some(status_code) = query.status_code {
            sql.push_str(" AND status_code = ?");
            values.push(Value::Integer(i64::from(status_code)));
        }
        if let Some(outcome) = query.outcome {
            sql.push_str(" AND state = ?");
            values.push(Value::Text(outcome.as_str().to_string()));
        }
        if let Some(cursor) = &query.cursor {
            let (timestamp, request_id) = decode_cursor(cursor)?;
            sql.push_str(&format!(
                " AND (created_at, request_id) {} (?, ?)",
                query.sort.cursor_operator()
            ));
            values.push(Value::Text(timestamp));
            values.push(Value::Text(request_id));
        }
        sql.push_str(&format!(
            " ORDER BY created_at {}, request_id {} LIMIT ?",
            query.sort.sql_order(),
            query.sort.sql_order()
        ));
        let fetch_limit = i64::try_from(query.limit.saturating_add(1))
            .map_err(|error| LogStoreError::QueryFailed(error.to_string()))?;
        values.push(Value::Integer(fetch_limit));

        let connection = self.conn();
        let mut statement = connection.prepare(&sql).map_err(LogStoreError::Sqlite)?;
        let rows = statement
            .query_map(rusqlite::params_from_iter(values.iter()), request_record)
            .map_err(LogStoreError::Sqlite)?;
        let mut items = rows
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| LogStoreError::QueryFailed(error.to_string()))?;
        let has_more = items.len() > query.limit;
        items.truncate(query.limit);
        let next_cursor = if has_more {
            items
                .last()
                .map(|record| encode_cursor(&record.created_at, &record.request_id))
        } else {
            None
        };
        Ok(QueryPage { items, next_cursor })
    }
}

fn request_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<RequestRecord> {
    Ok(RequestRecord {
        request_id: row.get(0)?,
        outcome: row.get(1)?,
        created_at: row.get(2)?,
        terminal_at: row.get(3)?,
        route: row.get(4)?,
        model: row.get(5)?,
        provider: row.get(6)?,
        engine: row.get(7)?,
        status_code: row.get(8)?,
    })
}
