from pathlib import Path

from doc_id_store import DocIDStore
from flow_index_vault import scan_filesystem_records
from markdown_link_repair import rewrite_markdown_links


def _scan(root: Path, registry: DocIDStore):
    return scan_filesystem_records(
        root,
        ["**/*.md"],
        ["registry.db*"],
        doc_id_store=registry,
    )


def test_scan_rewrites_links_when_doc_ids_rename_targets(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "index.md").write_text(
        "[Roadmap](roadmap.md#next)\n"
        "[[roadmap]]\n"
        "![Diagram](../assets/plan.png)\n",
        encoding="utf-8",
    )
    (notes / "roadmap.md").write_text("# Roadmap\n", encoding="utf-8")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "plan.png").write_bytes(b"png")
    registry = DocIDStore(tmp_path / "registry.db")

    records = scan_filesystem_records(
        tmp_path,
        ["**/*.md", "**/*.png"],
        ["registry.db*"],
        doc_id_store=registry,
    )

    index_record = next(record for record in records if "index@" in record["rel_path"])
    roadmap_record = next(record for record in records if "roadmap@" in record["rel_path"])
    plan_record = next(record for record in records if "plan@" in record["rel_path"])
    text = Path(index_record["abs_path"]).read_text(encoding="utf-8")
    assert f"[Roadmap]({Path(roadmap_record['rel_path']).name}#next)" in text
    assert f"[[{Path(roadmap_record['rel_path']).stem}]]" in text
    assert f"![Diagram](../assets/{Path(plan_record['rel_path']).name})" in text
    assert index_record["size"] == len(text.encode())


def test_full_scan_self_heals_links_broken_before_upgrade(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "Guide@abc12@.md"
    target.write_text("# Guide\n", encoding="utf-8")
    source = docs / "Home@abc13@.md"
    source.write_text("[Guide](Guide.md) and [[Guide#start|read it]]\n", encoding="utf-8")
    registry = DocIDStore(tmp_path / "registry.db")
    registry.register("abc12", "docs/Guide@abc12@.md")
    registry.register("abc13", "docs/Home@abc13@.md")

    _scan(tmp_path, registry)

    assert source.read_text(encoding="utf-8") == (
        "[Guide](Guide@abc12@.md) and [[Guide@abc12@#start|read it]]\n"
    )


def test_external_links_and_code_examples_are_untouched():
    aliases = {"docs/Guide.md": "docs/Guide@abc12@.md"}
    original = (
        "[web](https://example.com/Guide.md) [anchor](#local)\n"
        "`[example](Guide.md)`\n"
        "```md\n[example](Guide.md)\n[[Guide]]\n```\n"
        "[real](Guide.md)\n"
    )

    repaired, count = rewrite_markdown_links(original, "docs/Home.md", aliases)

    assert count == 1
    assert repaired.endswith("[real](Guide@abc12@.md)\n")
    assert "[web](https://example.com/Guide.md)" in repaired
    assert "`[example](Guide.md)`" in repaired
    assert "```md\n[example](Guide.md)\n[[Guide]]\n```" in repaired


def test_obsidian_embed_repairs_non_markdown_attachment():
    aliases = {"assets/diagram.png": "assets/diagram@abc12@.png"}

    repaired, count = rewrite_markdown_links(
        "![[diagram.png]]\n", "notes/Home.md", aliases
    )

    assert repaired == "![[diagram@abc12@.png]]\n"
    assert count == 1


def test_reference_links_angle_paths_and_idempotence():
    aliases = {"folder/My Guide.md": "folder/My Guide@FAWW42@.md"}
    original = "[guide][g]\n\n[g]: <My Guide.md#top> \"Title\"\n"

    repaired, count = rewrite_markdown_links(original, "folder/Home.md", aliases)
    second, second_count = rewrite_markdown_links(repaired, "folder/Home.md", aliases)

    assert repaired == "[guide][g]\n\n[g]: <My Guide@FAWW42@.md#top> \"Title\"\n"
    assert count == 1
    assert second == repaired
    assert second_count == 0
