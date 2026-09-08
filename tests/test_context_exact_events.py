"""Exact-event mode is an interface of the existing Context Builder tool."""
import datetime as dt

import cds_live
import mcp_server as srv
import pytest


def test_context_builder_reads_long_exact_event_in_stable_pages(monkeypatch):
    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql, params):
            self.offset = params[0] - 1
            assert params[-1] == "event-one"

        def fetchall(self):
            body = "a" * 12000 + "FINAL correction: not paid."
            import hashlib
            return [(1, "quo", "event-one", dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
                     "inbound", "Sender", None, body[self.offset:self.offset + 12000],
                     len(body), hashlib.sha256(body.encode()).hexdigest())]

    class Connection:
        def cursor(self):
            return Cursor()

        def rollback(self):
            pass

    monkeypatch.setattr(cds_live, "_get_readonly_conn", Connection)
    first = srv._context_builder_impl(event_refs=["event-one"])["cds"]["events"]
    assert first["has_more"] is True
    assert first["messages"][0]["body"] == "a" * 12000
    second = srv._context_builder_impl(event_refs=["event-one"], event_cursor=first["next_cursor"])["cds"]["events"]
    assert second["has_more"] is False
    assert second["messages"][0]["body"] == "FINAL correction: not paid."
    assert second["messages"][0]["body_offset"] == 12000
    assert "error" in srv._context_builder_impl(event_refs=["other"], event_cursor=first["next_cursor"])


@pytest.mark.parametrize("refs,cursor", [(None, "bad"), ([], None), (["a", "a"], None), (["a"], "W10="), (["a"], "bnVsbA==")])
def test_invalid_event_requests_do_not_read_database(refs, cursor, monkeypatch):
    monkeypatch.setattr(cds_live, "_get_readonly_conn", lambda: pytest.fail("invalid input queried database"))
    assert "error" in srv._context_builder_impl(event_refs=refs, event_cursor=cursor)


@pytest.mark.anyio
async def test_public_context_tool_exposes_event_pagination():
    result = await srv.context_builder(event_refs=[], include=["cds"])
    assert "event_refs" in result["error"]


def test_context_builder_resolves_exact_call_transcript_in_pages(monkeypatch):
    import hashlib
    ref = 'AC' + 'a' * 32
    body = 'Transcript line. ' * 1000
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, sql, params):
            self.call = 'FROM calls' in sql
            self.offset = params[0] - 1
        def fetchall(self):
            if not self.call: return []
            return [('call:1', 'quo', ref, dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
                     'inbound', None, 'Call transcript', body[self.offset:self.offset + 12000],
                     len(body), hashlib.sha256(body.encode()).hexdigest())]
    class Connection:
        def cursor(self): return Cursor()
        def rollback(self): pass
    monkeypatch.setattr(cds_live, '_get_readonly_conn', Connection)
    first = srv._context_builder_impl(event_refs=[ref])['cds']['events']
    assert first['missing_refs'] == []
    assert first['messages'][0]['event_kind'] == 'call_transcript'
    second = srv._context_builder_impl(event_refs=[ref], event_cursor=first['next_cursor'])['cds']['events']
    assert first['messages'][0]['body'] + second['messages'][0]['body'] == body


def test_context_builder_rejects_conflicting_call_transcripts(monkeypatch):
    import hashlib
    ref = 'AC' + 'b' * 32
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, sql, params): self.call = 'FROM calls' in sql
        def fetchall(self):
            if not self.call: return []
            return [('call:1', 'quo', ref, dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
                     'inbound', None, 'Call transcript', text, len(text), hashlib.sha256(text.encode()).hexdigest())
                    for text in ('Paid.', 'Not paid.')]
    class Connection:
        def cursor(self): return Cursor()
        def rollback(self): pass
    monkeypatch.setattr(cds_live, '_get_readonly_conn', Connection)
    page = srv._context_builder_impl(event_refs=[ref])['cds']['events']
    assert page['ambiguous_refs'] == [ref]
    assert page['messages'] == []
