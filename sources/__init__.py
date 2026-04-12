"""Source protocol package. See sources/base.py for the contract."""

from sources.base import Source, SourceRecord
from sources.filesystem import FilesystemSource
from sources.postgres import PostgresSource, TableSpec

__all__ = [
    "Source",
    "SourceRecord",
    "FilesystemSource",
    "PostgresSource",
    "TableSpec",
    "build_source",
]


def build_source(config: dict, *, registry=None, pdf_config=None) -> Source:
    """Build a Source from a config dict (one entry of the sources: list).

    Args:
        config: {type, name, ...type-specific keys}
        registry: DocIDStore to pass to filesystem sources. Required for
            type='filesystem', ignored for type='postgres'.
        pdf_config: PDF extraction config to pass to filesystem sources.

    Raises:
        ValueError: Unknown source type.
        KeyError: Missing required config key.
    """
    name = config["name"]
    src_type = config.get("type")

    if src_type == "filesystem":
        if registry is None:
            raise ValueError(f"filesystem source '{name}' requires a DocIDStore registry")
        return FilesystemSource(
            name=name,
            root=config["root"],
            scan_config=config.get("scan", {}),
            registry=registry,
            pdf_config=pdf_config,
        )

    if src_type == "postgres":
        tables = [TableSpec(**t) for t in config.get("tables", [])]
        return PostgresSource(
            name=name,
            dsn=config["dsn"],
            tables=tables,
        )

    raise ValueError(f"Unknown source type: {src_type!r}")
