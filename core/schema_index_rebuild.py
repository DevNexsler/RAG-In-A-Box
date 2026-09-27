"""Recreate a table's indexes on its unpublished schema replacement.

Run in a memory-guarded worker: native index training allocations disappear
on exit, and failure leaves the serving table untouched. Never promote here.
"""
from __future__ import annotations

import argparse


def rebuild_indexes(source_path: str, replacement_path: str) -> None:
    import lance

    source = lance.dataset(source_path)
    replacement = lance.dataset(replacement_path)
    # One logical index can have several delta segments under the same name.
    indexes = {index["name"]: index for index in source.list_indices()}
    for name, index in indexes.items():
        fields = index["fields"]
        if len(fields) != 1:
            raise ValueError(f"Cannot preserve multi-column index {name!r}")
        stats = source.stats.index_stats(name)
        kind = stats["index_type"].upper()
        if kind.startswith("IVF"):
            details = stats["indices"][0]
            sub_index = details["sub_index"]
            options = {}
            if "PQ" in kind:
                options["num_sub_vectors"] = sub_index["num_sub_vectors"]
                options["num_bits"] = sub_index.get("nbits", 8)
            if "HNSW" in kind:
                params = sub_index["params"]
                options.update({key: params[key] for key in ("m", "ef_construction", "max_level")})
            replacement.create_index(
                fields[0], kind, name=name, metric=details["metric_type"],
                num_partitions=details["num_partitions"], **options,
            )
        elif kind == "INVERTED":
            params = dict(stats["indices"][0]["params"])
            # Statistics include the internal tokenizer implementation name;
            # Lance 10 rejects it as a create-index option. Tokenization is
            # configured by base_tokenizer and the remaining persisted options.
            params.pop("lance_tokenizer", None)
            replacement.create_scalar_index(fields[0], "INVERTED", name=name, **params)
        elif kind in {"BTREE", "BITMAP", "LABEL_LIST"}:
            replacement.create_scalar_index(fields[0], kind, name=name)
        else:
            raise ValueError(f"Cannot preserve index {name!r} of type {kind!r}")
    actual = {index["name"] for index in replacement.list_indices()}
    if actual != set(indexes):
        raise RuntimeError("Replacement index inventory does not match serving table")
    for name in actual:
        if replacement.stats.index_stats(name)["num_unindexed_rows"]:
            raise RuntimeError(f"Replacement index {name!r} has unindexed rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_path")
    parser.add_argument("replacement_path")
    args = parser.parse_args()
    rebuild_indexes(args.source_path, args.replacement_path)


if __name__ == "__main__":
    main()
