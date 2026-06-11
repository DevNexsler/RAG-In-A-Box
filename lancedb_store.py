"""LanceDB storage via LlamaIndex's LanceDBVectorStore. Implements our StorageInterface."""

import json
import logging
import re
import shutil
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.vector_stores.lancedb.base import TableNotFoundError

from core.storage import SearchHit
from doc_enrichment import CORE_ENRICHMENT_FIELDS

logger = logging.getLogger(__name__)

_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_CORRUPT_LANCE_MARKERS = (
    "lanceerror(io)",
    "invalid range 0..0",
    "manifest was not found",
)
_LANCE_CORRUPTION_MARKERS = _CORRUPT_LANCE_MARKERS
_STALE_READ_MARKERS = (
    ".lance/_versions/",
    ".lance/data/",
    ".lance/_deletions/",
    "manifest was not found",
)

_EXTRA_META_FIELDS = ("description", "author", "keywords", "custom_meta")
_ENRICHMENT_AUX_FIELDS = ("enr_importance_source",)
_DUPLICATE_META_FIELDS = (
    "dup_count",
    "dup_sources",
    "dup_locations",
    "dup_archive_paths",
    "dup_natural_keys",
)

# All metadata keys that map to explicit SearchHit attributes.
# Anything NOT in this set goes into SearchHit.extra_metadata.
_CORE_META_KEYS = {
    "doc_id", "source_type", "mtime", "size", "title", "tags", "folder",
    "status", "created", "loc", "snippet", "rel_path",
    "description", "author", "keywords", "custom_meta",
    *CORE_ENRICHMENT_FIELDS,
    *_ENRICHMENT_AUX_FIELDS,
}


def _extract_enrichment(meta: dict) -> dict[str, str]:
    """Pull enrichment + extra metadata fields from metadata, defaulting to empty strings."""
    result = {f: meta.get(f, "") or "" for f in CORE_ENRICHMENT_FIELDS}
    for f in _ENRICHMENT_AUX_FIELDS:
        result[f] = meta.get(f, "") or ""
    for f in _EXTRA_META_FIELDS:
        result[f] = meta.get(f, "") or ""
    return result


def _extract_extra_metadata(meta: dict) -> dict[str, str]:
    """Collect metadata fields not in the hardcoded core set (e.g. section, sentiment)."""
    return {k: str(v) for k, v in meta.items() if k not in _CORE_META_KEYS and v}


def _json_string_list(value: Any) -> list[str]:
    """Decode a JSON list string, preserving plain strings as single-item lists."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
        return [str(parsed)] if str(parsed).strip() else []
    text = str(value).strip()
    return [text] if text else []


def _append_unique(items: list[str], value: Any) -> None:
    """Append a non-empty string if it is not already present."""
    text = str(value).strip()
    if text and text not in items:
        items.append(text)


def _duplicate_natural_key(duplicate_ref: dict[str, Any]) -> str:
    """Resolve a natural key from explicit metadata or a source-qualified doc_id."""
    natural_key = str(duplicate_ref.get("natural_key") or "").strip()
    if natural_key:
        return natural_key
    source_name = str(duplicate_ref.get("source_name") or "").strip().lower()
    if source_name in {"documents", "filesystem"}:
        return ""
    doc_id = str(duplicate_ref.get("doc_id") or "").strip()
    if "::" not in doc_id:
        return ""
    return doc_id.split("::", 1)[1].strip()


class LanceDBStore:
    """Implements StorageInterface using LlamaIndex's LanceDBVectorStore."""

    def __init__(self, index_root: str | Path, table_name: str = "chunks") -> None:
        self.index_root = str(Path(index_root))
        self.table_name = table_name
        self._schema_lock = threading.Lock()
        self._vs = self._build_vector_store()
        self._ensure_scalar_index()
        try:
            self._probe_table_read()
        except Exception as exc:
            if not self._is_probable_corruption_error(exc) or not self._recover_corrupt_table():
                raise
            logger.warning(
                "Recovered probable LanceDB corruption while opening %s: %s",
                self.table_name,
                exc,
            )
            self._vs = self._build_vector_store()
            self._ensure_scalar_index()
            self._probe_table_read()

    def _build_vector_store(self) -> LanceDBVectorStore:
        return LanceDBVectorStore(
            uri=self.index_root,
            table_name=self.table_name,
            mode="create",  # "create" lets LanceDB create the table if missing, or open if exists
        )

    @staticmethod
    def _is_probable_corruption_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return _looks_like_corrupt_lance_error(exc) or "corrupt" in text

    def _probe_table_read(self) -> None:
        """Cheap read probe so init fails fast on unreadable tables, not first user query."""
        try:
            self._vs.table.to_lance().to_table(limit=1)
        except TableNotFoundError:
            return

    def _recover_corrupt_table(self) -> bool:
        """Rollback to the newest readable Lance version and rewrite the dataset."""
        import lance

        lance_path = Path(self.index_root) / f"{self.table_name}.lance"
        if not lance_path.exists():
            return False

        try:
            ds = lance.dataset(str(lance_path))
            versions = sorted(ds.versions(), key=lambda item: item["version"], reverse=True)
        except Exception as exc:
            logger.warning("Cannot inspect Lance versions for recovery: %s", exc)
            return False

        clean_version = None
        for version_info in versions:
            version = version_info["version"]
            try:
                lance.dataset(str(lance_path), version=version).to_table(limit=1)
                clean_version = version
                break
            except Exception:
                continue

        if clean_version is None:
            logger.warning("No readable Lance version found for %s", lance_path)
            return False

        clean_table = lance.dataset(str(lance_path), version=clean_version).to_table()
        corrupt_path = Path(f"{lance_path}.corrupt")
        if corrupt_path.exists():
            shutil.rmtree(corrupt_path)
        shutil.move(str(lance_path), str(corrupt_path))
        lance.write_dataset(clean_table, str(lance_path))
        return True

    def _table_path(self, table_name: str | None = None) -> Path:
        """Return on-disk path for a Lance table."""
        return Path(self.index_root) / f"{table_name or self.table_name}.lance"

    def _reconnect(self) -> None:
        """Reconnect the vector store to the current on-disk table path."""
        self._vs = LanceDBVectorStore(
            uri=self.index_root,
            table_name=self.table_name,
            mode="create",
        )
        self._ensure_scalar_index()

    def _ensure_scalar_index(self) -> None:
        """Create a BTREE scalar index on doc_id for O(log n) filtered lookups."""
        try:
            self._vs.table.create_scalar_index("doc_id", index_type="BTREE", replace=True)
        except TableNotFoundError:
            pass  # Table not created yet on first run
        except Exception as e:
            logger.warning("Failed to create scalar index: %s", e)

    def reset_table(self) -> None:
        """Drop all data for this table so a shadow rebuild starts from a clean slate."""
        table_path = self._table_path()
        if table_path.exists():
            shutil.rmtree(table_path)
        self._reconnect()

    def promote_table(self, shadow_table_name: str) -> None:
        """Atomically replace this table with a prepared shadow table."""
        shadow_path = self._table_path(shadow_table_name)
        active_path = self._table_path()
        backup_path = self._table_path(f"{self.table_name}__backup")

        if not shadow_path.exists():
            raise FileNotFoundError(f"Shadow table does not exist: {shadow_path}")

        if backup_path.exists():
            shutil.rmtree(backup_path)

        active_moved = False
        try:
            if active_path.exists():
                shutil.move(str(active_path), str(backup_path))
                active_moved = True
            shutil.move(str(shadow_path), str(active_path))
        except Exception:
            if active_moved and not active_path.exists() and backup_path.exists():
                shutil.move(str(backup_path), str(active_path))
            raise
        else:
            if backup_path.exists():
                shutil.rmtree(backup_path)
            self._reconnect()
            logger.info("Promoted shadow table %r into %r", shadow_table_name, self.table_name)

    @staticmethod
    def _is_stale_read_error(exc: Exception) -> bool:
        text = str(exc)
        text_lower = text.lower()
        if "not found" not in text_lower:
            return False
        return any(marker in text for marker in _STALE_READ_MARKERS)

    def _reopen_vector_store(self) -> None:
        self._vs = self._build_vector_store()

    def _run_read_with_recovery(self, operation, default_on_missing):
        try:
            return operation()
        except TableNotFoundError:
            return default_on_missing
        except Exception as exc:
            if not self._is_stale_read_error(exc):
                raise
            logger.warning("Refreshing LanceDB store after stale/corrupt read: %s", exc)
            self._reopen_vector_store()
            try:
                return operation()
            except TableNotFoundError:
                return default_on_missing

    @staticmethod
    def _sql_escape(value: str) -> str:
        """Escape single quotes for safe use in SQL WHERE clauses."""
        return value.replace("'", "''")

    @staticmethod
    def _escape_fts_query(query: str) -> str:
        """Escape natural-language tokens Tantivy misreads as field syntax."""
        return re.sub(r"(?<!\\):", r"\\:", query)

    @staticmethod
    def _is_phrase_query_without_positions(exc: Exception, query: str) -> bool:
        text = str(exc).lower()
        return '"' in query and "position is not found but required for phrase queries" in text

    @staticmethod
    def _strip_phrase_quotes(query: str) -> str:
        return " ".join(query.replace('"', " ").split())

    @staticmethod
    def _validate_identifier(key: str) -> None:
        """Raise ValueError if *key* is not a safe SQL identifier."""
        if not _SAFE_IDENTIFIER_RE.match(key):
            raise ValueError(f"Unsafe metadata filter key: {key!r}")

    def _metadata_field_sql(self, key: str) -> str:
        """Return SQL reference for a metadata field after identifier validation."""
        self._validate_identifier(key)
        return f"metadata.{key}"

    def _build_filter_ast_clause(self, node: dict) -> str:
        """Compile a structured filter AST to a LanceDB SQL WHERE fragment."""
        if not isinstance(node, dict) or len(node) != 1:
            raise ValueError("filter must be an object with exactly one operator")

        op, payload = next(iter(node.items()))
        if op in {"and", "or"}:
            if not isinstance(payload, list) or not payload:
                raise ValueError(f"{op} filter must be a non-empty list")
            joiner = f" {op.upper()} "
            return "(" + joiner.join(self._build_filter_ast_clause(item) for item in payload) + ")"

        if op == "not":
            if not isinstance(payload, dict):
                raise ValueError("not filter must contain one nested filter object")
            return f"NOT ({self._build_filter_ast_clause(payload)})"

        if op not in {"eq", "ne", "contains", "prefix", "in"}:
            raise ValueError(f"Unsupported filter operator: {op}")
        if not isinstance(payload, dict) or len(payload) != 1:
            raise ValueError(f"{op} filter must be an object with exactly one field")

        key, value = next(iter(payload.items()))
        field = self._metadata_field_sql(key)

        if op == "in":
            if not isinstance(value, list) or not value:
                raise ValueError("in filter value must be a non-empty list")
            values = ", ".join(
                f"'{self._sql_escape(str(item).lower())}'"
                for item in value
            )
            return f"lower({field}) IN ({values})"

        escaped = self._sql_escape(str(value))
        escaped_lower = self._sql_escape(str(value).lower())

        if op == "eq":
            return f"lower({field}) = '{escaped_lower}'"
        if op == "ne":
            return f"lower({field}) != '{escaped_lower}'"
        if op == "contains":
            return f"lower({field}) LIKE '%{escaped_lower}%'"
        if op == "prefix":
            if key == "rel_path":
                return f"{field} LIKE '{escaped}%'"
            return f"lower({field}) LIKE '{escaped_lower}%'"

        raise ValueError(f"Unsupported filter operator: {op}")

    def _metadata_subfields(self) -> set[str]:
        """Return the set of sub-field names inside the metadata struct column."""
        def _op():
            ds = self._vs.table.to_lance()
            for field in ds.schema:
                if field.name == "metadata":
                    return {sf.name for sf in field.type}
            return set()

        try:
            return self._run_read_with_recovery(_op, set())
        except Exception:
            return set()

    def _evolve_metadata_schema(self, new_fields: set[str]) -> None:
        """Add new string sub-fields to the metadata struct column.

        Reads the entire table as Arrow, adds empty-string columns for each
        new field, reconstructs the metadata struct, and replaces the table.
        """
        import lancedb as ldb

        table = self._vs.table
        arrow_table = table.to_arrow()

        # Extract existing metadata struct arrays
        meta_chunked = arrow_table.column("metadata")
        meta_col = meta_chunked.combine_chunks()  # StructArray (not ChunkedArray)
        meta_type = meta_col.type
        existing_names = [meta_type.field(i).name for i in range(meta_type.num_fields)]

        # Build new struct arrays: existing + new fields filled with ""
        n_rows = len(arrow_table)
        arrays = [meta_col.field(name) for name in existing_names]
        fields = [meta_type.field(i) for i in range(meta_type.num_fields)]

        for fname in sorted(new_fields):
            if fname not in existing_names:
                arrays.append(pa.array([""] * n_rows, type=pa.utf8()))
                fields.append(pa.field(fname, pa.utf8()))

        new_struct = pa.StructArray.from_arrays(arrays, fields=fields)

        # Replace metadata column in the table
        col_idx = arrow_table.schema.get_field_index("metadata")
        new_arrow = arrow_table.set_column(col_idx, pa.field("metadata", new_struct.type), new_struct)

        db = ldb.connect(self.index_root)
        temp_name = f"{self.table_name}__schema_tmp"
        backup_name = f"{self.table_name}__schema_backup"
        temp_path = Path(self.index_root) / f"{temp_name}.lance"
        table_path = Path(self.index_root) / f"{self.table_name}.lance"
        backup_path = Path(self.index_root) / f"{backup_name}.lance"

        def _table_names() -> set[str]:
            try:
                result = db.list_tables()
                names = getattr(result, "tables", result)
                return set(names)
            except AttributeError:
                return set(db.table_names())

        for stale in (temp_name, backup_name):
            if stale in _table_names():
                db.drop_table(stale)
        for stale_path in (temp_path, backup_path):
            if stale_path.exists():
                shutil.rmtree(stale_path)

        backup_created = False
        try:
            try:
                db.create_table(temp_name, new_arrow)
            except Exception:
                if temp_path.exists():
                    shutil.rmtree(temp_path)
                raise

            shutil.move(str(table_path), str(backup_path))
            backup_created = True
            shutil.move(str(temp_path), str(table_path))
        except Exception:
            if backup_created and not table_path.exists() and backup_path.exists():
                try:
                    shutil.move(str(backup_path), str(table_path))
                except Exception:
                    logger.exception("Failed to restore %s after schema evolution failure", self.table_name)
            raise
        else:
            if backup_path.exists():
                shutil.rmtree(backup_path)
        finally:
            if temp_path.exists():
                shutil.rmtree(temp_path)

        # Reconnect LanceDBVectorStore to the new table
        self._vs = LanceDBVectorStore(
            uri=self.index_root,
            table_name=self.table_name,
            mode="create",
        )
        self._ensure_scalar_index()
        logger.info("Schema evolved: added metadata fields %s", new_fields)

    # --- WHERE clause builder ---

    def _build_where_clause(
        self,
        doc_id_prefix: str | None = None,
        source_type: str | None = None,
        source_name: str | None = None,
        status: str | None = None,
        folder: str | None = None,
        tags: str | None = None,
        enr_doc_type: str | None = None,
        enr_topics: str | None = None,
        metadata_filters: dict[str, str] | None = None,
        filter_ast: dict | None = None,
    ) -> str | None:
        """Build a SQL WHERE clause from filter parameters for LanceDB prefilter.

        Returns None if no filters are active.
        """
        parts: list[str] = []

        # Exact match on metadata fields (case-insensitive)
        if source_type:
            parts.append(f"lower(metadata.source_type) = '{self._sql_escape(source_type.lower())}'")
        if source_name:
            parts.append(f"lower(metadata.source_name) = '{self._sql_escape(source_name.lower())}'")
        if status:
            parts.append(f"lower(metadata.status) = '{self._sql_escape(status.lower())}'")
        if folder:
            parts.append(f"lower(metadata.folder) = '{self._sql_escape(folder.lower())}'")

        # Prefix match on rel_path metadata field (case-sensitive — paths)
        # doc_id is now a persistent 5-char ID; path-based browsing uses rel_path
        if doc_id_prefix:
            parts.append(f"metadata.rel_path LIKE '{self._sql_escape(doc_id_prefix)}%'")

        # Comma-separated OR fields (tags, enr_doc_type, enr_topics) — case-insensitive
        for field, value in [
            ("tags", tags),
            ("enr_doc_type", enr_doc_type),
            ("enr_topics", enr_topics),
        ]:
            if value:
                items = [item.strip() for item in value.split(",") if item.strip()]
                if items:
                    or_parts = [
                        f"lower(metadata.{field}) LIKE '%{self._sql_escape(item.lower())}%'"
                        for item in items
                    ]
                    parts.append(f"({' OR '.join(or_parts)})")

        # Arbitrary metadata key=value pairs (case-insensitive)
        if metadata_filters:
            for key, val in metadata_filters.items():
                self._validate_identifier(key)
                parts.append(f"lower(metadata.{key}) = '{self._sql_escape(str(val).lower())}'")

        if filter_ast is not None:
            parts.append(self._build_filter_ast_clause(filter_ast))

        return " AND ".join(parts) if parts else None

    # --- Shared row-to-hit converter ---

    @staticmethod
    def _row_to_hit(row: dict) -> "SearchHit":
        """Convert a raw LanceDB row dict to a SearchHit.

        Handles both vector search (_distance → similarity) and FTS (_score).
        """
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        text = row.get("text", "") or ""

        doc_id = (
            meta.get("doc_id")
            or row.get("doc_id")
            or row.get("_node_ref_doc_id", "")
        )
        loc = meta.get("loc") or row.get("loc", "")
        snippet = meta.get("snippet") or text[:200]

        # Score: vector search returns _distance (cosine), FTS returns _score
        if "_distance" in row:
            score = 1.0 - float(row["_distance"])
        elif "_score" in row:
            score = float(row["_score"])
        elif "score" in row:
            score = float(row["score"])
        else:
            score = 0.0

        raw_mtime = meta.get("mtime") or row.get("mtime") or 0.0
        combined_meta = {**row, **meta} if meta else row

        return SearchHit(
            doc_id=doc_id,
            loc=loc,
            snippet=snippet,
            text=text,
            score=score,
            source_type=meta.get("source_type") or row.get("source_type"),
            title=meta.get("title") or row.get("title"),
            tags=meta.get("tags") or row.get("tags"),
            folder=meta.get("folder") or row.get("folder"),
            status=meta.get("status") or row.get("status"),
            created=meta.get("created") or row.get("created"),
            mtime=float(raw_mtime) if raw_mtime else 0.0,
            rel_path=meta.get("rel_path") or row.get("rel_path", ""),
            **_extract_enrichment(combined_meta),
            extra_metadata=_extract_extra_metadata(meta),
            vector=row.get("vector"),
        )

    # --- StorageInterface methods ---

    def upsert_nodes(self, nodes: list[TextNode]) -> None:
        """Delete existing nodes for each doc_id, then add new ones."""
        if not nodes:
            return

        # Detect new metadata fields and evolve schema if needed
        existing_subfields = self._metadata_subfields()
        if existing_subfields:  # table already has data
            incoming_keys: set[str] = set()
            for n in nodes:
                if n.metadata:
                    incoming_keys.update(n.metadata.keys())
            if incoming_keys:
                # Threaded indexing shares one store instance; re-check missing
                # fields under a lock so temp-table schema evolution is serialized.
                with self._schema_lock:
                    existing_subfields = self._metadata_subfields()
                    new_fields = incoming_keys - existing_subfields
                    if new_fields:
                        self._evolve_metadata_schema(new_fields)

        # Collect distinct doc_ids from this batch
        doc_ids = {n.ref_doc_id for n in nodes if n.ref_doc_id}
        # Delete old data for those doc_ids first
        for doc_id in doc_ids:
            try:
                self._vs.delete(doc_id)
            except TableNotFoundError:
                pass  # Table not created yet on first run
            except Exception as e:
                logger.warning("Failed to delete old data for %s: %s", doc_id, e)
        # Add new nodes
        try:
            self._vs.add(nodes)
        except Exception:
            logger.critical(
                "Failed to add %d nodes for doc_ids=%s after old chunks were deleted; "
                "these docs will self-heal on the next index run",
                len(nodes), sorted(doc_ids),
            )
            raise

    def update_canonical_duplicate_metadata(
        self,
        canonical_doc_id: str,
        duplicate_refs: list[dict[str, Any]],
    ) -> None:
        """Merge duplicate provenance into canonical chunk metadata and re-upsert it."""
        try:
            table = self._vs.table
        except TableNotFoundError as exc:
            raise LookupError(
                f"Canonical doc {canonical_doc_id} missing in LanceDB; refusing duplicate metadata update"
            ) from exc

        rows = (
            table.search(None)
            .where(f"doc_id = '{self._sql_escape(canonical_doc_id)}'", prefilter=True)
            .select(["id", "doc_id", "text", "vector", "metadata"])
            .to_list()
        )
        if not rows:
            raise LookupError(
                f"Canonical doc {canonical_doc_id} missing in LanceDB; refusing duplicate metadata update"
            )

        first_meta = rows[0].get("metadata") if isinstance(rows[0].get("metadata"), dict) else {}
        dup_sources = _json_string_list(first_meta.get("dup_sources"))
        dup_locations = _json_string_list(first_meta.get("dup_locations"))
        dup_archive_paths = _json_string_list(first_meta.get("dup_archive_paths"))
        dup_natural_keys = _json_string_list(first_meta.get("dup_natural_keys"))

        seen_duplicates: set[tuple[str, str, str, str, str]] = set()
        for duplicate_ref in duplicate_refs:
            natural_key = _duplicate_natural_key(duplicate_ref)
            duplicate_key = (
                str(duplicate_ref.get("doc_id") or "").strip(),
                str(duplicate_ref.get("source_name") or "").strip(),
                str(duplicate_ref.get("rel_path") or "").strip(),
                str(duplicate_ref.get("archive_path") or "").strip(),
                natural_key,
            )
            if duplicate_key in seen_duplicates:
                continue
            seen_duplicates.add(duplicate_key)
            _append_unique(dup_sources, duplicate_ref.get("source_name"))
            _append_unique(dup_locations, duplicate_ref.get("rel_path"))
            _append_unique(dup_archive_paths, duplicate_ref.get("archive_path"))
            _append_unique(dup_natural_keys, natural_key)

        existing_dup_count = str(first_meta.get("dup_count") or "").strip()
        try:
            prior_dup_count = int(existing_dup_count)
        except ValueError:
            prior_dup_count = 0
        merged_dup_count = max(
            prior_dup_count,
            len(seen_duplicates),
            len(dup_locations),
            len(dup_archive_paths),
            len(dup_natural_keys),
        )
        duplicate_metadata = {
            "dup_count": str(merged_dup_count),
            "dup_sources": json.dumps(dup_sources),
            "dup_locations": json.dumps(dup_locations),
            "dup_archive_paths": json.dumps(dup_archive_paths),
            "dup_natural_keys": json.dumps(dup_natural_keys),
        }

        canonical_nodes: list[TextNode] = []
        for row in rows:
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            vector = row.get("vector")
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            elif vector is None:
                vector = []
            else:
                vector = list(vector)
            loc = metadata.get("loc") or ""
            node = TextNode(
                text=row.get("text", "") or "",
                id_=row.get("id") or f"{canonical_doc_id}::{loc}",
                embedding=vector,
                metadata={**metadata, **duplicate_metadata},
            )
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=canonical_doc_id)
            canonical_nodes.append(node)

        self.upsert_nodes(canonical_nodes)

    def delete_by_doc_ids(self, doc_ids: list[str]) -> None:
        """Remove all nodes for the given doc_ids."""
        for doc_id in doc_ids:
            try:
                self._vs.delete(doc_id)
            except TableNotFoundError:
                pass  # Table not created yet — nothing to delete
            except Exception as e:
                logger.warning("Failed to delete doc %s: %s", doc_id, e)

    def list_doc_ids(self) -> list[str]:
        """Return all distinct doc_ids in the store.

        Uses Lance SQL DISTINCT — reads only the doc_id column, no vectors or text.
        Raises on failure so callers get a proper error (not empty results).
        """
        def _op():
            ds = self._vs.table.to_lance()
            batches = ds.sql("SELECT DISTINCT doc_id FROM dataset WHERE doc_id IS NOT NULL").build().to_batch_records()
            if not batches:
                return []
            return pa.Table.from_batches(batches)["doc_id"].to_pylist()

        return self._run_read_with_recovery(_op, [])

    def list_doc_mtimes(self) -> dict[str, float]:
        """Return {doc_id: mtime} for all docs in the store.

        Uses Lance SQL GROUP BY — reads only doc_id + metadata.mtime, no vectors or text.
        Raises on failure so callers get a proper error (not empty results).
        """
        def _op():
            ds = self._vs.table.to_lance()
            batches = ds.sql(
                "SELECT doc_id, MAX(metadata.mtime) AS mtime "
                "FROM dataset WHERE doc_id IS NOT NULL GROUP BY doc_id"
            ).build().to_batch_records()
            if not batches:
                return {}
            t = pa.Table.from_batches(batches)
            doc_ids = t["doc_id"].to_pylist()
            mtimes = t["mtime"].to_pylist()
            return {d: float(m) if m is not None else 0.0 for d, m in zip(doc_ids, mtimes)}

        return self._run_read_with_recovery(_op, {})

    def count_chunks(self) -> int:
        """Return total number of chunks (rows) in the store. O(1) via LanceDB.

        Raises on failure so callers get a proper error (not zero).
        """
        return self._run_read_with_recovery(lambda: self._vs.table.count_rows(), 0)

    def vector_search(
        self,
        query_vector: list[float],
        top_k: int,
        where: str | None = None,
        include_vector: bool = True,
    ) -> list[SearchHit]:
        """Vector search via direct LanceDB query with optional prefilter.

        Raises on failure so the caller (hybrid_search) can track degradation.
        """
        def _op():
            q = self._vs.table.search(query_vector, query_type="vector")
            if where:
                q = q.where(where, prefilter=True)
            rows = q.limit(top_k).to_list()
            if not include_vector:
                for row in rows:
                    row.pop("vector", None)
            return [self._row_to_hit(row) for row in rows]

        return self._run_read_with_recovery(_op, [])

    # --- Full-Text Search (BM25, Lance-native inverted index) ---

    def create_fts_index(self) -> None:
        """Create or fully rebuild the native FTS index on the text column.

        The Lance-native index updates incrementally — rows added after index
        creation are searchable immediately and deletes are respected without
        a rebuild. A full rebuild is only needed for first-time creation and
        recovery paths; routine indexing runs should call ensure_fts_index().
        Raises on failure so the caller can track the error.
        """
        table = self._vs.table
        text_key = getattr(self._vs, "text_key", "text")
        table.create_fts_index(text_key, use_tantivy=False, replace=True)
        logger.info("FTS index created/rebuilt on column %r", text_key)

    def ensure_fts_index(self) -> None:
        """Make sure the native FTS index exists and is optimized.

        Creates the index if missing; otherwise merges newly written rows into
        the existing index via optimize(). Unindexed rows are still searchable
        before the merge, so this is a performance step, not a correctness one.
        Raises on failure so the caller can track the error.
        """
        table = self._vs.table
        text_key = getattr(self._vs, "text_key", "text")
        has_fts = any(
            str(getattr(idx, "index_type", "")).upper() == "FTS"
            for idx in table.list_indices()
        )
        if not has_fts:
            table.create_fts_index(text_key, use_tantivy=False)
            logger.info("FTS index created on column %r", text_key)
            return
        table.optimize()
        logger.info("FTS index optimized (incremental merge)")

    def fts_available(self) -> bool:
        """Check if the FTS/tantivy index is operational (health check for file_status)."""
        try:
            self._run_read_with_recovery(
                lambda: self._vs.table.search("test", query_type="fts").limit(1).to_list(),
                [],
            )
            return True
        except Exception:
            return False

    def keyword_search(
        self,
        query: str,
        top_k: int = 50,
        where: str | None = None,
        include_vector: bool = True,
    ) -> list[SearchHit]:
        """BM25/FTS keyword search via LanceDB tantivy index with optional prefilter.

        Returns SearchHit list ranked by BM25 relevance. Raises on FTS failure
        so the caller (hybrid_search) can track degradation in diagnostics.
        """
        if not query.strip():
            return []

        def _search_rows(fts_query: str):
            q = self._vs.table.search(fts_query, query_type="fts")
            if where:
                q = q.where(where, prefilter=True)
            return q.limit(top_k).to_list()

        def _op():
            try:
                rows = _search_rows(self._escape_fts_query(query))
            except Exception as exc:
                if not self._is_phrase_query_without_positions(exc, query):
                    raise
                fallback_query = self._strip_phrase_quotes(query)
                if not fallback_query:
                    raise
                logger.warning(
                    "FTS phrase query unsupported without positions; retrying without quotes: %r",
                    query,
                )
                rows = _search_rows(self._escape_fts_query(fallback_query))
            if not include_vector:
                for row in rows:
                    row.pop("vector", None)
            return [self._row_to_hit(row) for row in rows]

        return self._run_read_with_recovery(_op, [])

    _RECENT_DOC_FIELDS = (
        "title",
        "source_type",
        "folder",
        "tags",
        "status",
        "created",
        "keywords",
        *_DUPLICATE_META_FIELDS,
    )

    def list_recent_docs(
        self,
        limit: int = 20,
        source_type: str | None = None,
        folder: str | None = None,
    ) -> list[dict]:
        """Return recently modified docs sorted by mtime descending.

        Uses Lance SQL with GROUP BY, ORDER BY, and LIMIT — all server-side.
        No vectors or text loaded. Raises on failure so callers get a proper error.
        """
        def _op():
            ds = self._vs.table.to_lance()
            available = self._metadata_subfields()

            where_parts: list[str] = ["doc_id IS NOT NULL"]
            if source_type and "source_type" in available:
                where_parts.append(f"metadata.source_type = '{self._sql_escape(source_type)}'")
            if folder and "folder" in available:
                where_parts.append(f"metadata.folder = '{self._sql_escape(folder)}'")
            where_clause = " AND ".join(where_parts)

            # Build SELECT with only fields that exist in the metadata struct
            select_parts = ["doc_id"]
            output_fields = ["doc_id"]
            if "mtime" in available:
                select_parts.append("MAX(metadata.mtime) AS mtime")
                output_fields.append("mtime")
            for f in self._RECENT_DOC_FIELDS:
                if f in available:
                    select_parts.append(f"MAX(metadata.{f}) AS {f}")
                    output_fields.append(f)

            sql = (
                f"SELECT {', '.join(select_parts)} "
                f"FROM dataset WHERE {where_clause} "
                "GROUP BY doc_id "
                f"ORDER BY mtime DESC LIMIT {limit}"
            )
            batches = ds.sql(sql).build().to_batch_records()
            if not batches:
                return []
            t = pa.Table.from_batches(batches)
            result: list[dict] = []
            for i in range(len(t)):
                rec: dict = {}
                for f in output_fields:
                    try:
                        rec[f] = t[f][i].as_py()
                    except (KeyError, IndexError):
                        rec[f] = None
                result.append(rec)
            return result

        return self._run_read_with_recovery(_op, [])

    _FACET_KEY_MAP = {
        "folder": "folders",
        "source_type": "source_types",
        "status": "statuses",
        "enr_doc_type": "doc_types",
        "author": "authors",
        "enr_topics": "topics",
        "enr_keywords": "keywords",
        "enr_entities_people": "entities_people",
        "enr_entities_places": "entities_places",
        "enr_entities_orgs": "entities_orgs",
    }

    # Metadata fields that should NOT be faceted (structural/numeric, not categorical)
    _NON_FACET_FIELDS = {"doc_id", "loc", "snippet", "mtime", "size", "title", "created"}
    _SAFE_CONTEXT_FACET_FIELDS = {"enr_context_confidence"}

    @classmethod
    def _dynamic_facet_fields(cls, available: set[str]) -> set[str]:
        """Return dynamic metadata fields safe to comma-split into facets."""
        all_facet_fields = set(cls._FACET_KEY_MAP.keys())
        candidates = available - all_facet_fields - cls._NON_FACET_FIELDS - {"tags"}
        return {
            field
            for field in candidates
            if not field.startswith("enr_context_") or field in cls._SAFE_CONTEXT_FACET_FIELDS
        }

    def facets(self) -> dict:
        """Return distinct values and doc counts for all filterable fields.

        Uses column projection to load only metadata fields — no vectors or text.
        Deduplicates by doc_id so counts reflect documents, not chunks.
        Tags are split on commas and counted individually.
        Dynamic fields (not in _FACET_KEY_MAP) are automatically included.

        TODO(perf): This does a full table scan — O(chunks). Fine at ~15K chunks
        (~200ms) but will get slow at 50K+. When that happens, write a
        facets_cache.json at end of index_vault_flow and read from it here.
        Cache invalidation is trivial: indexing is the only writer.
        """
        total_chunks = self.count_chunks()
        if total_chunks == 0:
            return {"total_docs": 0, "total_chunks": 0}
        def _op():
            available = self._metadata_subfields()

            # Discover dynamic fields, excluding context narratives/JSON that
            # would be corrupted by the comma-splitting facet logic.
            dynamic_fields = self._dynamic_facet_fields(available)

            projection = {"doc_id": "doc_id"}
            for field in self._FACET_KEY_MAP:
                if field in available:
                    projection[field] = f"metadata.{field}"
            for field in dynamic_fields:
                projection[field] = f"metadata.{field}"
            if "tags" in available:
                projection["tags"] = "metadata.tags"

            t = self._vs.table.search(None).select(projection).to_arrow()

            if len(t) == 0:
                return {"total_docs": 0, "total_chunks": total_chunks}

            # Deduplicate by doc_id and count facet values
            seen: set[str] = set()
            all_counted_fields = set(self._FACET_KEY_MAP.keys()) | dynamic_fields
            counters: dict[str, Counter] = {f: Counter() for f in all_counted_fields}
            tag_counter: Counter = Counter()

            doc_ids = t["doc_id"].to_pylist()
            field_columns: dict[str, list] = {}
            for f in all_counted_fields:
                try:
                    field_columns[f] = t[f].to_pylist()
                except KeyError:
                    field_columns[f] = [None] * len(t)
            try:
                tags_col = t["tags"].to_pylist()
            except KeyError:
                tags_col = [None] * len(t)

            for i, did in enumerate(doc_ids):
                if not did or did in seen:
                    continue
                seen.add(did)

                for f in all_counted_fields:
                    val = field_columns[f][i]
                    if val and str(val).strip():
                        for item in str(val).split(","):
                            item = item.strip()
                            if item:
                                counters[f][item] += 1

                raw_tags = tags_col[i]
                if raw_tags and str(raw_tags).strip():
                    for tag in str(raw_tags).split(","):
                        tag = tag.strip()
                        if tag:
                            tag_counter[tag] += 1

            def _to_list(counter: Counter) -> list[dict]:
                return [{"value": v, "count": c} for v, c in counter.most_common()]

            result: dict = {
                "tags": _to_list(tag_counter),
                "total_docs": len(seen),
                "total_chunks": total_chunks,
            }
            for f, key in self._FACET_KEY_MAP.items():
                result[key] = _to_list(counters[f])
            # Dynamic fields use their own name as the result key
            for f in dynamic_fields:
                facet_list = _to_list(counters[f])
                if facet_list:  # only include if there are non-empty values
                    result[f] = facet_list

            return result

        return self._run_read_with_recovery(_op, {"total_docs": 0, "total_chunks": 0})

    def get_vector(self, chunk_uid: str) -> list[float] | None:
        """Retrieve the stored embedding vector for a chunk by its UID (doc_id::loc).

        Used by MMR diversity and cosine fallback reranking. Returns None if
        the chunk is not found or the table doesn't exist yet.
        """
        def _op():
            rows = (
                self._vs.table.search(None)
                .where(f"id = '{self._sql_escape(chunk_uid)}'", prefilter=True)
                .select(["vector"])
                .limit(1)
                .to_list()
            )
            if not rows:
                return None
            vec = rows[0].get("vector")
            if vec is None:
                return None
            # Convert from Arrow/numpy to plain list if needed
            if hasattr(vec, "tolist"):
                return vec.tolist()
            return list(vec)

        return self._run_read_with_recovery(_op, None)

    def get_vectors(self, chunk_uids: list[str]) -> dict[str, list[float]]:
        """Batch-load stored vectors for chunk UIDs missing from search hits."""
        if not chunk_uids:
            return {}
        try:
            table = self._vs.table
        except TableNotFoundError:
            return {}

        unique_uids = list(dict.fromkeys(uid for uid in chunk_uids if uid))
        if not unique_uids:
            return {}
        in_values = ", ".join(f"'{self._sql_escape(uid)}'" for uid in unique_uids)
        rows = (
            table.search(None)
            .where(f"id IN ({in_values})", prefilter=True)
            .select(["id", "vector"])
            .limit(len(unique_uids))
            .to_list()
        )

        vectors: dict[str, list[float]] = {}
        for row in rows:
            uid = row.get("id")
            vec = row.get("vector")
            if not uid or vec is None:
                continue
            vectors[str(uid)] = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        return vectors

    def get_chunk(self, doc_id: str, loc: str) -> SearchHit | None:
        """Get a single chunk by doc_id and loc.

        Uses predicate pushdown with BTREE index — O(log n), no full scan.
        Loads text + metadata for the matching row only; no vectors.
        """
        def _op():
            esc_doc = self._sql_escape(doc_id)
            esc_loc = self._sql_escape(loc)
            rows = (
                self._vs.table.search(None)
                .where(f"doc_id = '{esc_doc}' AND metadata.loc = '{esc_loc}'", prefilter=True)
                .select(["doc_id", "text", "metadata"])
                .limit(1)
                .to_list()
            )
            if not rows:
                return None
            return self._row_to_hit(rows[0])

        return self._run_read_with_recovery(_op, None)

    def get_doc_chunks(self, doc_id: str) -> list[SearchHit]:
        """Get all chunks for a document, sorted by loc.

        Uses predicate pushdown with BTREE index — O(log n + k), no full scan.
        Loads text + metadata for matching rows only; no vectors.
        """
        def _op():
            esc_doc = self._sql_escape(doc_id)
            rows = (
                self._vs.table.search(None)
                .where(f"doc_id = '{esc_doc}'", prefilter=True)
                .select(["doc_id", "text", "metadata"])
                .to_list()
            )
            hits = [self._row_to_hit(row) for row in rows]
            hits.sort(key=lambda h: h.loc)
            return hits

        return self._run_read_with_recovery(_op, [])


def _looks_like_corrupt_lance_error(exc: Exception) -> bool:
    """Return True when a store-open failure matches known Lance corruption signatures."""
    message = str(exc).lower()
    return any(marker in message for marker in _CORRUPT_LANCE_MARKERS)


def recover_corrupt_table(
    index_root: str | Path,
    table_name: str,
    logger_obj: logging.Logger | None = None,
) -> LanceDBStore | None:
    """Walk Lance versions backward, rebuild from last readable snapshot, return fresh store."""
    import lance

    active_logger = logger_obj or logger
    lance_path = Path(index_root) / f"{table_name}.lance"
    if not lance_path.exists():
        return None

    active_logger.warning("Corruption detected — attempting auto-recovery for %s", lance_path)

    try:
        dataset = lance.dataset(str(lance_path))
        versions = sorted(dataset.versions(), key=lambda item: item["version"], reverse=True)
    except Exception as exc:
        active_logger.error("Cannot read dataset versions: %s", exc)
        return None

    clean_version = None
    for version_info in versions:
        version = version_info["version"]
        try:
            clean_dataset = lance.dataset(str(lance_path), version=version)
            clean_dataset.to_table(limit=1)
            clean_version = version
            break
        except Exception:
            continue

    if clean_version is None:
        active_logger.error("No readable version found — manual recovery needed")
        return None

    clean_table = lance.dataset(str(lance_path), version=clean_version).to_table()
    active_logger.info(
        "Found clean version %d with %d rows — rebuilding table",
        clean_version,
        clean_table.num_rows,
    )

    corrupt_path = Path(f"{lance_path}.corrupt")
    if corrupt_path.exists():
        shutil.rmtree(corrupt_path)
    shutil.move(str(lance_path), str(corrupt_path))
    lance.write_dataset(clean_table, str(lance_path))

    rebuilt = lance.dataset(str(lance_path))
    rebuilt.to_table(limit=1)
    active_logger.info(
        "Recovery complete: %d rows restored. Corrupt backup at %s",
        rebuilt.count_rows(),
        corrupt_path,
    )
    return LanceDBStore(index_root, table_name)


def open_store_with_recovery(
    index_root: str | Path,
    table_name: str = "chunks",
    *,
    logger_obj: logging.Logger | None = None,
    auto_recover: bool = True,
) -> LanceDBStore:
    """Open LanceDBStore and auto-recover from known corruption signatures when possible."""
    try:
        return LanceDBStore(index_root, table_name)
    except Exception as exc:
        if not auto_recover or not _looks_like_corrupt_lance_error(exc):
            raise
        recovered = recover_corrupt_table(index_root, table_name, logger_obj=logger_obj)
        if recovered is not None:
            return recovered
        raise
