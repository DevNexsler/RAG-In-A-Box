"""Source protocol: uniform contract for heterogeneous data sources.

All sources (filesystem, postgres, slack, ...) implement this so the indexing
flow can iterate them uniformly. Downstream (chunk, embed, store, search) is
already source-agnostic and never sees this layer.
"""

from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable

from extractors import ExtractionResult


@dataclass(frozen=True)
class SourceRecord:
    """One indexable unit emitted by a Source. Shape is what diff_index_task
    and the flow's per-record loop expect.

    doc_id is unique *within* the source. The flow namespaces it globally
    by prefixing "{source_name}::" before upserting into LanceDB.
    """
    doc_id: str
    source_type: str
    natural_key: str
    mtime: float
    size: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Source(Protocol):
    """Structural protocol — any class with these four members is a Source."""

    name: str

    def scan(self) -> Iterator[SourceRecord]:
        """Yield every indexable record in this source. Should be streaming
        (not load everything into memory) so large sources don't OOM."""
        ...

    def extract(self, record: SourceRecord) -> ExtractionResult:
        """Convert a record to its extractable text + frontmatter. May do I/O
        (read file from disk, fetch row from DB) or may use cached data on
        the record if scan() already populated it."""
        ...

    def close(self) -> None:
        """Release any connections/handles. Called once after the last scan."""
        ...
