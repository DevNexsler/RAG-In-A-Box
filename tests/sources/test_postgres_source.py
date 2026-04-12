"""Live tests for PostgresSource against comm-data-store-postgres-1.

Skipped unless COMM_DATA_STORE_DSN is set. Can be run locally with:
    COMM_DATA_STORE_DSN=postgresql://comm_data_store:change-me@localhost:5433/comm_data_store \\
        PYTHONPATH=. pytest tests/sources/test_postgres_source.py -v
"""

import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("COMM_DATA_STORE_DSN"),
        reason="COMM_DATA_STORE_DSN not set — skipping live postgres tests",
    ),
]


@pytest.fixture
def pg_source():
    from sources.postgres import PostgresSource, TableSpec

    specs = [
        TableSpec(
            source_type="pg_message",
            query="""
                SELECT
                    m.source,
                    m.source_message_id,
                    m.sent_at,
                    m.updated_at,
                    m.body AS _text,
                    c.name AS channel,
                    p.display_name AS sender
                FROM messages m
                LEFT JOIN channels c ON c.id = m.channel_id
                LEFT JOIN participants p ON p.id = m.sender_participant_id
                WHERE m.body IS NOT NULL AND m.body <> ''
                ORDER BY m.id
                LIMIT 20
            """,
            id_template="{source}/{source_message_id}",
            text_column="_text",
            mtime_column="updated_at",
            metadata_columns=["channel", "sender", "source", "sent_at"],
        ),
    ]
    src = PostgresSource(
        name="comm_messages",
        dsn=os.environ["COMM_DATA_STORE_DSN"],
        tables=specs,
    )
    yield src
    src.close()


def test_scan_yields_records(pg_source):
    records = list(pg_source.scan())
    assert len(records) > 0
    assert len(records) <= 20  # respects LIMIT


def test_record_shape(pg_source):
    records = list(pg_source.scan())
    r = records[0]
    assert r.source_type == "pg_message"
    assert "/" in r.doc_id  # "{source}/{source_message_id}"
    assert r.mtime > 0
    assert r.size > 0
    assert "channel" in r.metadata
    assert "sender" in r.metadata
    assert "_text" in r.metadata  # cached text for zero-IO extract


def test_extract_returns_nonempty_text(pg_source):
    records = list(pg_source.scan())
    result = pg_source.extract(records[0])
    assert len(result.full_text) > 0
    # Frontmatter includes channel and sender
    assert "channel" in result.frontmatter or "sender" in result.frontmatter


def test_natural_key_matches_doc_id(pg_source):
    """For Postgres, natural_key and doc_id are the same (no rename layer)."""
    records = list(pg_source.scan())
    for r in records:
        assert r.doc_id == r.natural_key


def test_close_releases_connection(pg_source):
    # First close: connection was opened by scan() in fixture setup (if
    # any of the other tests ran), or still None (if this test runs alone).
    # Either way, close should leave _conn = None.
    pg_source.close()
    assert pg_source._conn is None
    # Second close is idempotent.
    pg_source.close()
    assert pg_source._conn is None


def test_scan_with_multiple_table_specs():
    """Multiple TableSpecs share one connection; both streams are readable."""
    import os
    if not os.environ.get("COMM_DATA_STORE_DSN"):
        pytest.skip("COMM_DATA_STORE_DSN not set")

    from sources.postgres import PostgresSource, TableSpec

    messages_spec = TableSpec(
        source_type="pg_message",
        query="""
            SELECT source, source_message_id, updated_at, body AS _text
            FROM messages
            WHERE body IS NOT NULL AND body <> ''
            ORDER BY id
            LIMIT 5
        """,
        id_template="{source}/{source_message_id}",
        text_column="_text",
        mtime_column="updated_at",
        metadata_columns=["source"],
    )
    transcripts_spec = TableSpec(
        source_type="pg_transcript",
        query="""
            SELECT id, updated_at, transcript_text AS _text
            FROM transcripts
            ORDER BY id
            LIMIT 5
        """,
        id_template="pg_transcript/{id}",
        text_column="_text",
        mtime_column="updated_at",
        metadata_columns=[],
    )

    src = PostgresSource(
        name="comm_multi",
        dsn=os.environ["COMM_DATA_STORE_DSN"],
        tables=[messages_spec, transcripts_spec],
    )
    try:
        records = list(src.scan())
        # Both specs should produce records
        source_types = {r.source_type for r in records}
        assert "pg_message" in source_types
        assert "pg_transcript" in source_types
        # LIMIT 5 on each = up to 10 total
        assert len(records) <= 10
    finally:
        src.close()
