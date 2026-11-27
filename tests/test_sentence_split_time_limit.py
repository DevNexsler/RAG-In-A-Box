"""Sentence splitting must never defer a document on NLTK's regex time limit."""

import pytest
from llama_index.core.node_parser import SentenceSplitter

import flow_index_vault as fiv
from extractors import begin_degradation_capture, collect_degradations


def _table(rows: int) -> str:
    return "\n".join(
        f"| {i} | Invoice {i}. | 12.50 | Paid. | 804-555-{i % 10000:04d} |" for i in range(rows)
    )


def test_short_text_is_one_segment_and_splits_exactly_as_before():
    splitter = SentenceSplitter(chunk_size=64, chunk_overlap=8)
    text = "First sentence here. Second sentence follows. " * 40
    assert fiv._bounded_text_segments(text, 200_000) == [text]
    assert fiv._split_text_bounded(splitter, text) == splitter.split_text(text)


def test_segments_are_bounded_lossless_and_cut_at_line_ends():
    text = _table(4000)
    segments = fiv._bounded_text_segments(text, 5_000)
    assert "".join(segments) == text
    assert all(0 < len(s) <= 5_000 for s in segments)
    assert all(s.endswith("\n") for s in segments[:-1])


def test_unbreakable_text_is_cut_at_the_limit():
    text = "x" * 2_500
    segments = fiv._bounded_text_segments(text, 1_000)
    assert [len(s) for s in segments] == [1_000, 1_000, 500]


def test_long_text_reaches_the_splitter_only_in_bounded_pieces(monkeypatch):
    monkeypatch.setattr(fiv, "_SENTENCE_SPLIT_SEGMENT_CHARS", 5_000)
    seen: list[int] = []

    class Recording:
        chunk_size, chunk_overlap = 64, 8

        def split_text(self, piece):
            seen.append(len(piece))
            return [piece]

    text = _table(2000)
    chunks = fiv._split_text_bounded(Recording(), text)
    assert len(seen) > 1 and max(seen) <= 5_000
    assert "".join(chunks) == text


def test_time_limit_on_one_piece_falls_back_instead_of_raising(monkeypatch):
    """The production failure: NLTK raised TimeoutError, the flow read it as a
    provider outage, and the document was deferred on every run, never indexed."""
    monkeypatch.setattr(fiv, "_SENTENCE_SPLIT_SEGMENT_CHARS", 5_000)

    class TimesOutOnce:
        chunk_size, chunk_overlap = 64, 8
        calls = 0

        def split_text(self, piece):
            type(self).calls += 1
            if type(self).calls == 1:
                raise TimeoutError(
                    "regular-expression match exceeded its 5.0s time limit"
                )
            return [piece]

    text = _table(400)
    begin_degradation_capture()
    chunks = fiv._split_text_bounded(TimesOutOnce(), text)
    assert chunks and all(c.strip() for c in chunks)
    assert "Invoice 0." in chunks[0] and "Invoice 399." in chunks[-1]
    noted = collect_degradations()
    assert [(d.reason, d.transient) for d in noted] == [
        ("sentence_split_time_limit", False)
    ]


def test_other_splitter_errors_still_propagate():
    class Broken:
        chunk_size, chunk_overlap = 64, 8

        def split_text(self, piece):
            raise ValueError("bad input")

    with pytest.raises(ValueError):
        fiv._split_text_bounded(Broken(), "some text")


def test_split_section_routes_through_the_bounded_splitter(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        fiv, "_split_text_bounded", lambda splitter, text: calls.append(text) or [text]
    )
    assert fiv._split_section("body text", object()) == ["body text"]
    assert calls == ["body text"]
