"""Unit coverage for PostgreSQL source text normalization."""

from datetime import UTC, datetime
import hashlib

from sources.postgres import PostgresSource, TableSpec


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.itersize = 0
        self.query = ""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, query):
        self.query = query

    def __iter__(self):
        return iter(self.rows)


class _Connection:
    def __init__(self, message_rows, participant_rows):
        self.message_rows = message_rows
        self.participant_rows = participant_rows
        self.participant_queries = 0

    def cursor(self, name=None):
        if name is not None:
            return _Cursor(self.message_rows)
        self.participant_queries += 1
        return _Cursor(self.participant_rows)


def _source(message_rows, participant_rows):
    spec = TableSpec(
        source_type="pg_message",
        query="SELECT message rows",
        id_template="{source}/{source_message_id}",
        text_column="_text",
        mtime_column="updated_at",
        metadata_columns=["source"],
        text_normalizer="zoho_cliq_mentions",
    )
    source = PostgresSource("comm_messages", "postgresql://unused", [spec])
    connection = _Connection(message_rows, participant_rows)
    source._get_conn = lambda: connection
    return source, connection


def _message(message_id, text, *, source="zoho_cliq"):
    return {
        "source": source,
        "source_message_id": message_id,
        "updated_at": datetime(2026, 7, 14, tzinfo=UTC),
        "_text": text,
    }


def test_scan_expands_all_resolvable_cliq_mentions_and_forces_reindex():
    raw_text = "{@918334727} ask {@720844989} for the estimate"
    source, connection = _source(
        [_message("msg-1", raw_text)],
        [
            {"participant_key": "918334727", "display_name": "Nigel Pine"},
            {"participant_key": "720844989", "display_name": "Dan Park"},
        ],
    )

    records = list(source.scan())

    assert records[0].metadata["_text"] == "@Nigel Pine ask @Dan Park for the estimate"
    raw_hash = hashlib.blake2b(raw_text.encode(), digest_size=16).hexdigest()
    assert records[0].change_hash != raw_hash
    assert connection.participant_queries == 1


def test_scan_drops_cliq_messages_containing_only_mentions_and_punctuation():
    source, _ = _source(
        [
            _message("resolved", "  {@918334727}!!!  "),
            _message("unresolved", "{@000000000}, …"),
        ],
        [{"participant_key": "918334727", "display_name": "Nigel Pine"}],
    )

    assert list(source.scan()) == []


def test_scan_preserves_unresolved_mentions_in_substantive_cliq_messages():
    source, _ = _source(
        [_message("msg-2", "{@000000000} link here")],
        [],
    )

    [record] = list(source.scan())

    assert record.metadata["_text"] == "{@000000000} link here"


def test_scan_leaves_non_cliq_message_text_and_hash_unchanged():
    raw_text = "Literal {@918334727} syntax from another provider"
    source, _ = _source(
        [_message("msg-3", raw_text, source="quo")],
        [{"participant_key": "918334727", "display_name": "Nigel Pine"}],
    )

    [record] = list(source.scan())

    assert record.metadata["_text"] == raw_text
    expected_hash = hashlib.blake2b(raw_text.encode(), digest_size=16).hexdigest()
    assert record.change_hash == expected_hash


def test_scan_preserves_opaque_source_message_ids_verbatim():
    """Ticket #0618: the ledger holds keys like
    `zoho_cliq/1780327866430%2015958014910122`, which looked like an `_` that
    had been round-tripped through a URL encoder on our write path. It is not
    — those are the literal `source_message_id` values upstream stores, and
    the doc_id must reproduce them byte-for-byte. A doc_id that does not equal
    the id the store keys on can never be looked up, re-queued, or cleared, so
    this seam must never gain encoding, escaping, or normalization.
    """
    opaque = [
        "1780327866430%2015958014910122",   # upstream's own percent sequence
        "1780327866430_15958014910122",     # its canonical underscore twin
        "<7r3ekCixTQ6Uw7mj7RXcJQ@geopod-ismtpd-canary-0>",
        "id with spaces/and+plus",
    ]
    source, _ = _source(
        [_message(i, f"body for {i}", source="zoho_mail") for i in opaque], []
    )

    records = list(source.scan())

    assert [r.doc_id for r in records] == [f"zoho_mail/{i}" for i in opaque]
    assert [r.natural_key for r in records] == [r.doc_id for r in records]
