"""Tracking-heavy HTML mail stays useful and bounded through Lance readback."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from llama_index.core.node_parser import SentenceSplitter

import flow_index_vault as fiv
from lancedb_store import LanceDBStore
from sources.postgres import PostgresSource, TableSpec


class _Cursor:
    def __init__(self, rows):
        self._rows = rows
        self.itersize = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, _query):
        return None

    def __iter__(self):
        return iter(self._rows)


class _Connection:
    closed = False

    def __init__(self, messages):
        self._messages = messages

    def cursor(self, name=None):
        return _Cursor(self._messages if name else [])


class _Embed:
    def embed_texts(self, texts):
        return [[0.1] * 768 for _ in texts]


def _tracking_email() -> str:
    redirects = "\n".join(
        f'<a href="https://tracker.example/redirect?upn=u001{i}&target={'x' * 900}">'
        "Track delivery</a>"
        for i in range(18)
    )
    footer = """
      <footer>
        <a href="https://social.example/assets/icon.png?campaign=abc">Follow us</a>
        <p>Privacy settings | Unsubscribe | Terms of use</p>
      </footer>
    """
    return f"""
      Subject: Appliance delivery confirmation
      Body:
      <html><head><style>.hidden {{ display:none }}</style></head><body>
        <div hidden>{'&zwnj;&#847;' * 500}</div>
        <h1>Your delivery is scheduled</h1>
        <p>Order H0931-246565 includes a washer and dryer.</p>
        <p>Delivery: September 4, 8 AM–12 PM, 12 Oak Street.</p>
        {redirects}
        <img src="https://images.example/pixel.gif?campaign=abc&recipient=123">
        {footer}{footer}
      </body></html>
    """


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(fiv, "get_run_logger", lambda: MagicMock())
    store = LanceDBStore(tmp_path / "index", "chunks")
    fiv._RUNTIME.clear()
    fiv._RUNTIME.update(
        {
            "store": store,
            "embed_provider": _Embed(),
            "splitter": SentenceSplitter(chunk_size=512, chunk_overlap=20),
            "config": {"enrichment": {}},
        }
    )
    yield store
    fiv._RUNTIME.clear()


def test_tracking_html_is_cleaned_before_embedding_and_lance_write(pipeline):
    raw = _tracking_email()
    row = {
        "source": "zoho_mail",
        "source_message_id": "tracking-heavy",
        "updated_at": datetime(2026, 9, 1, tzinfo=UTC),
        "_text": raw,
    }
    source = PostgresSource(
        "comm_messages",
        "postgresql://unused",
        [
            TableSpec(
                source_type="pg_message",
                query="SELECT fixture",
                id_template="{source}/{source_message_id}",
                text_column="_text",
                text_normalizer="zoho_cliq_mentions",
                mtime_column="updated_at",
                metadata_columns=["source", "source_message_id"],
            )
        ],
    )
    source._get_conn = lambda: _Connection([row])
    [record] = list(source.scan())
    doc_id = f"comm_messages::{record.doc_id}"
    fiv._RUNTIME.update(
        {
            "sources_by_name": {source.name: source},
            "source_records_by_ns_doc_id": {doc_id: record},
        }
    )

    fiv.process_doc_task.fn(
        {
            "doc_id": doc_id,
            "rel_path": "postgres/comm_messages/tracking-heavy",
            "mtime": record.mtime,
            "size": record.size,
            "source_type": record.source_type,
            "source_name": source.name,
        }
    )

    chunks = pipeline.get_doc_chunks(doc_id)
    stored = "\n".join(chunk.text for chunk in chunks)
    assert 1 <= len(chunks) <= 3, len(chunks)
    assert "Order H0931-246565" in stored
    assert "washer and dryer" in stored
    assert "Track delivery" in stored
    assert "tracker.example" not in stored
    assert "upn=u001" not in stored
    assert "images.example" not in stored
    assert "social.example" not in stored
    assert "Privacy settings" not in stored
    assert "&#847;" not in stored
