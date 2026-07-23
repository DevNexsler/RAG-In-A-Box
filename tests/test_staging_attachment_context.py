import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_staging_message_source_exposes_channel_id_used_by_attachment_sidecar():
    config = yaml.safe_load((ROOT / "config.staging.yaml").read_text())
    postgres = next(source for source in config["sources"] if source["type"] == "postgres")
    table = postgres["tables"][0]
    assert "channel_name AS source_channel_id" in table["query"]
    assert "source_channel_id" in table["metadata_columns"]


def test_real_media_staging_message_source_keeps_attachment_channel_mapping():
    config = yaml.safe_load((ROOT / "config.staging.realmedia.yaml").read_text())
    postgres = next(source for source in config["sources"] if source["type"] == "postgres")
    table = postgres["tables"][0]
    assert "channel_name AS source_channel_id" in table["query"]
    assert "source_channel_id" in table["metadata_columns"]


def test_staging_video_sidecar_brackets_attachment_between_seed_messages():
    sidecar = json.loads((ROOT / "tests/fixtures/e2e/clip.json").read_text())
    assert sidecar["source"] == "quo"
    assert sidecar["channel"]["source_channel_id"] == "ops"
    assert sidecar["message"]["sent_at"] == "2026-06-01T10:00:30Z"
