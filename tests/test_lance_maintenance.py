"""Short-lived Lance maintenance subprocess contract."""

from unittest.mock import MagicMock, patch


def test_compact_dataset_uses_streaming_binary_copy():
    from core.lance_maintenance import compact_dataset

    dataset = MagicMock()
    with patch("lance.dataset", return_value=dataset) as open_dataset:
        compact_dataset("/data/index/chunks.lance")

    open_dataset.assert_called_once_with("/data/index/chunks.lance")
    dataset.optimize.compact_files.assert_called_once_with(
        compaction_mode="try_binary_copy"
    )
