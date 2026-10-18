from pathlib import Path
import faiss
import pytest

from scripts.build_liar_index import build_index, load_liar_records


from unittest.mock import MagicMock, patch
import numpy as np


def test_build_liar_index_with_local_tsv(tmp_path):
    """Verify build_index creates liar_index.faiss and liar_metadata.json from local TSV."""
    sample_tsv = tmp_path / "train.tsv"
    tsv_content = (
        "2635.json\tpants-fire\tSays the price of gas doubled yesterday.\tenergy\tblogger\tactivist\tTexas\tRepublican\t0\t5\t0\t0\t1\tsocial media post\n"
        "10540.json\ttrue\tUnemployment dropped by 0.5 percent last month.\teconomy\tgovernor\tpolitician\tOhio\tDemocrat\t0\t1\t2\t3\t0\tpress conference\n"
    )
    sample_tsv.write_text(tsv_content, encoding="utf-8")

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((2, 384), dtype=np.float32)

    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        success = build_index(data_dir=tmp_path)

    assert success is True

    index_file = tmp_path / "liar_index.faiss"
    meta_file = tmp_path / "liar_metadata.json"

    assert index_file.exists()
    assert meta_file.exists()

    index = faiss.read_index(str(index_file))
    assert index.ntotal == 2


def test_build_liar_index_instructions_when_missing(tmp_path):
    """When no dataset is present and HF fails, instructions are provided."""
    with patch("datasets.load_dataset", side_effect=RuntimeError("HF datasets offline")):
        records, instructions = load_liar_records(tmp_path)
    assert records == []
    assert instructions is not None
    assert "train.tsv" in instructions

