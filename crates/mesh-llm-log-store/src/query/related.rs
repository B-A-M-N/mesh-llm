use rusqlite::types::Value;

use super::{ArtifactRecord, EventRecord, PageQuery, ProxyQuery, ProxyRecord, QueryPage};
use crate::cursor::{decode_cursor, encode_cursor};
use crate::{LogStore, LogStoreError};

impl LogStore {
    pub fn query_events(
        &self,
        request_id: &str,
        query: &PageQuery,
    ) -> Result<QueryPage<EventRecord>, LogStoreError> {
        let columns = "event_id, request_id, occurred_at, payload_json";
        self.query_related_page(
            RelatedQuery {
                table: "lifecycle_events",
                columns,
                id_column: "event_id",
                request_id,
                page: query,
            },
            RowProjection {
                map: event_record,
                cursor_fields: |record| (&record.occurred_at, &record.event_id),
            },
        )
    }

    pub fn query_artifacts(
        &self,
        request_id: &str,
        query: &PageQuery,
    ) -> Result<QueryPage<ArtifactRecord>, LogStoreError> {
        let columns = "artifact_id, request_id, occurred_at, kind, media_kind, checksum, bytes, version, redacted, truncated, missing, corrupt";
        self.query_related_page(
            RelatedQuery {
                table: "artifact_pointers",
                columns,
                id_column: "artifact_id",
                request_id,
                page: query,
            },
            RowProjection {
                map: artifact_record,
                cursor_fields: |record| (&record.occurred_at, &record.artifact_id),
            },
        )
    }

    pub fn query_artifact(
        &self,
        artifact_id: &str,
    ) -> Result<Option<ArtifactRecord>, LogStoreError> {
        let columns = "artifact_id, request_id, occurred_at, kind, media_kind, checksum, bytes, version, redacted, truncated, missing, corrupt";
        let sql = format!("SELECT {columns} FROM artifact_pointers WHERE artifact_id = ?");
        match self.conn().query_row(&sql, [artifact_id], artifact_record) {
            Ok(record) => Ok(Some(record)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(error) => Err(LogStoreError::QueryFailed(error.to_string())),
        }
    }

    pub fn query_proxy_records(
        &self,
        query: &ProxyQuery,
    ) -> Result<QueryPage<ProxyRecord>, LogStoreError> {
        let columns = "attempt_id, request_id, occurred_at, target, provider, engine, started_at, completed_at, status_code";
        let mut sql = format!("SELECT {columns} FROM proxy_records WHERE 1 = 1");
        let mut values = Vec::<Value>::new();
        for (column, value) in [
            ("request_id", &query.request_id),
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
        append_page_boundary(
            &mut sql,
            &mut values,
            PageBoundary {
                id_column: "attempt_id",
                query: &query.page,
            },
        )?;
        query_page(
            &self.conn(),
            PageExecution {
                sql,
                values,
                query: &query.page,
            },
            RowProjection {
                map: proxy_record,
                cursor_fields: |record| (&record.occurred_at, &record.attempt_id),
            },
        )
    }

    fn query_related_page<T>(
        &self,
        related: RelatedQuery<'_>,
        projection: RowProjection<T>,
    ) -> Result<QueryPage<T>, LogStoreError> {
        let mut sql = format!(
            "SELECT {} FROM {} WHERE request_id = ?",
            related.columns, related.table
        );
        let mut values = vec![Value::Text(related.request_id.to_string())];
        append_page_boundary(
            &mut sql,
            &mut values,
            PageBoundary {
                id_column: related.id_column,
                query: related.page,
            },
        )?;
        query_page(
            &self.conn(),
            PageExecution {
                sql,
                values,
                query: related.page,
            },
            projection,
        )
    }
}

struct RelatedQuery<'a> {
    table: &'static str,
    columns: &'static str,
    id_column: &'static str,
    request_id: &'a str,
    page: &'a PageQuery,
}

fn append_page_boundary(
    sql: &mut String,
    values: &mut Vec<Value>,
    boundary: PageBoundary<'_>,
) -> Result<(), LogStoreError> {
    if let Some(cursor) = &boundary.query.cursor {
        let (timestamp, id) = decode_cursor(cursor)?;
        let id_column = boundary.id_column;
        sql.push_str(&format!(
            " AND (occurred_at, {id_column}) {} (?, ?)",
            boundary.query.sort.cursor_operator()
        ));
        values.push(Value::Text(timestamp));
        values.push(Value::Text(id));
    }
    let id_column = boundary.id_column;
    sql.push_str(&format!(
        " ORDER BY occurred_at {}, {id_column} {} LIMIT ?",
        boundary.query.sort.sql_order(),
        boundary.query.sort.sql_order()
    ));
    let limit = i64::try_from(boundary.query.limit.saturating_add(1))
        .map_err(|error| LogStoreError::QueryFailed(error.to_string()))?;
    values.push(Value::Integer(limit));
    Ok(())
}

struct PageBoundary<'a> {
    id_column: &'a str,
    query: &'a PageQuery,
}

struct PageExecution<'a> {
    sql: String,
    values: Vec<Value>,
    query: &'a PageQuery,
}

struct RowProjection<T> {
    map: fn(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
    cursor_fields: fn(&T) -> (&str, &str),
}

fn query_page<T>(
    connection: &rusqlite::Connection,
    execution: PageExecution<'_>,
    projection: RowProjection<T>,
) -> Result<QueryPage<T>, LogStoreError> {
    let mut statement = connection
        .prepare(&execution.sql)
        .map_err(LogStoreError::Sqlite)?;
    let rows = statement
        .query_map(
            rusqlite::params_from_iter(execution.values.iter()),
            projection.map,
        )
        .map_err(LogStoreError::Sqlite)?;
    let mut items = rows
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| LogStoreError::QueryFailed(error.to_string()))?;
    let has_more = items.len() > execution.query.limit;
    items.truncate(execution.query.limit);
    let next_cursor = if has_more {
        items.last().map(|item| {
            let (timestamp, id) = (projection.cursor_fields)(item);
            encode_cursor(timestamp, id)
        })
    } else {
        None
    };
    Ok(QueryPage { items, next_cursor })
}

fn event_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<EventRecord> {
    Ok(EventRecord {
        event_id: row.get(0)?,
        request_id: row.get(1)?,
        occurred_at: row.get(2)?,
        payload_json: row.get(3)?,
    })
}

fn artifact_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<ArtifactRecord> {
    Ok(ArtifactRecord {
        artifact_id: row.get(0)?,
        request_id: row.get(1)?,
        occurred_at: row.get(2)?,
        kind: row.get(3)?,
        media_kind: row.get(4)?,
        checksum: row.get(5)?,
        bytes: row.get(6)?,
        version: row.get(7)?,
        redacted: row.get::<_, i32>(8)? != 0,
        truncated: row.get::<_, i32>(9)? != 0,
        missing: row.get::<_, i32>(10)? != 0,
        corrupt: row.get::<_, i32>(11)? != 0,
    })
}

fn proxy_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProxyRecord> {
    Ok(ProxyRecord {
        attempt_id: row.get(0)?,
        request_id: row.get(1)?,
        occurred_at: row.get(2)?,
        target: row.get(3)?,
        provider: row.get(4)?,
        engine: row.get(5)?,
        started_at: row.get(6)?,
        completed_at: row.get(7)?,
        status_code: row.get(8)?,
    })
}
