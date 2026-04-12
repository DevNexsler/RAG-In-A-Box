"""Contract tests for the Source protocol and SourceRecord dataclass."""

from dataclasses import FrozenInstanceError
from typing import Iterator

import pytest

from sources.base import Source, SourceRecord
from extractors import ExtractionResult


def test_source_record_is_frozen():
    """SourceRecord is immutable so the dispatch loop can't mutate it by accident."""
    r = SourceRecord(
        doc_id="x",
        source_type="md",
        natural_key="x.md",
        mtime=1.0,
        size=10,
        metadata={"k": "v"},
    )
    with pytest.raises(FrozenInstanceError):
        r.doc_id = "y"  # type: ignore[misc]


def test_source_record_metadata_defaults_to_empty_dict():
    r = SourceRecord(doc_id="x", source_type="md", natural_key="x.md", mtime=1.0, size=10)
    assert r.metadata == {}


def test_source_protocol_is_structural():
    """Any class with scan/extract/name/close satisfies the protocol — no subclassing required."""

    class Stub:
        name = "stub"

        def scan(self) -> Iterator[SourceRecord]:
            yield SourceRecord(doc_id="a", source_type="md", natural_key="a.md", mtime=1.0, size=5)

        def extract(self, record: SourceRecord) -> ExtractionResult:
            return ExtractionResult.from_text("hello")

        def close(self) -> None:
            pass

    s: Source = Stub()  # Would fail type-check if Protocol wasn't structural
    assert s.name == "stub"
    records = list(s.scan())
    assert len(records) == 1
    assert records[0].doc_id == "a"
