"""Script to build and save a FAISS index over the LIAR dataset (Phase 3).

Saves:
- data/liar/liar_index.faiss (FAISS Inner-Product cosine index)
- data/liar/liar_metadata.json (Claims statements, labels, context)
"""

import csv
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/liar")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# LIAR TSV column definitions
TSV_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]


def load_from_local_tsv(data_dir: Path) -> List[Dict[str, Any]]:
    """Load claims from local TSV files (train.tsv, test.tsv, valid.tsv)."""
    records: List[Dict[str, Any]] = []
    tsv_files = [data_dir / name for name in ("train.tsv", "valid.tsv", "test.tsv") if (data_dir / name).exists()]

    if not tsv_files:
        # Check any .tsv file in data_dir
        tsv_files = list(data_dir.glob("*.tsv"))

    for tsv_path in tsv_files:
        logger.info(f"Loading local LIAR records from {tsv_path}...")
        try:
            with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) >= 3:
                        label = row[1].strip() if len(row) > 1 else "unknown"
                        statement = row[2].strip() if len(row) > 2 else ""
                        speaker = row[4].strip() if len(row) > 4 else ""
                        context = row[13].strip() if len(row) > 13 else ""
                        if statement:
                            records.append({
                                "statement": statement,
                                "label": label,
                                "speaker": speaker,
                                "context": context,
                            })
        except Exception as exc:
            logger.warning(f"Error reading {tsv_path}: {exc}")

    return records


def load_liar_records(data_dir: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Try loading records via Hugging Face datasets library, then fallback to local TSV."""
    # 1. First, check if local TSV already exists
    local_records = load_from_local_tsv(data_dir)
    if local_records:
        logger.info(f"Found {len(local_records)} records in local TSV files.")
        return local_records, None

    # 2. Try Hugging Face datasets library
    try:
        from datasets import load_dataset
        logger.info("Loading LIAR dataset from Hugging Face datasets library...")
        ds = load_dataset("liar", trust_remote_code=True)
        records = []
        label_names = ["false", "half-true", "mostly-true", "true", "barely-true", "pants-fire"]

        for split_name in ds.keys():
            for row in ds[split_name]:
                stmt = row.get("statement", "").strip()
                if not stmt:
                    continue
                raw_label = row.get("label")
                if isinstance(raw_label, int) and 0 <= raw_label < len(label_names):
                    label = label_names[raw_label]
                else:
                    label = str(raw_label)
                records.append({
                    "statement": stmt,
                    "label": label,
                    "speaker": row.get("speaker", ""),
                    "context": row.get("context", ""),
                })
        if records:
            logger.info(f"Loaded {len(records)} records from Hugging Face datasets.")
            return records, None
    except Exception as exc:
        err_msg = str(exc)
        logger.warning(f"Hugging Face dataset load failed: {err_msg}")

    # 3. If both failed, provide clear instructions
    instructions = (
        "Could not load the LIAR dataset automatically via Hugging Face.\n"
        "To build the index, please download the LIAR dataset manually:\n"
        "1. Download 'train.tsv' (and optionally 'test.tsv', 'valid.tsv') from:\n"
        "   https://raw.githubusercontent.com/thiagorainmaker92/liar-dataset/master/train.tsv\n"
        "   or from the original archive: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip\n"
        f"2. Place the file(s) into: {data_dir.resolve() / 'train.tsv'}\n"
        "3. Run this script again: python scripts/build_liar_index.py"
    )
    return [], instructions


def build_index(data_dir: Optional[Path] = None, max_samples: Optional[int] = None) -> bool:
    """Build and save FAISS index and metadata into data/liar/."""
    target_dir = data_dir or DEFAULT_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    records, instructions = load_liar_records(target_dir)
    if not records:
        print("\n" + "=" * 80)
        print("MANUAL DOWNLOAD REQUIRED:")
        print("=" * 80)
        print(instructions)
        print("=" * 80 + "\n")
        return False

    if max_samples and len(records) > max_samples:
        records = records[:max_samples]

    logger.info(f"Building index for {len(records)} claims...")

    from sentence_transformers import SentenceTransformer
    import faiss

    logger.info(f"Loading SentenceTransformer '{EMBEDDING_MODEL_NAME}' on CPU...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

    statements = [r["statement"] for r in records]
    logger.info("Computing normalized statement embeddings...")
    embeddings = model.encode(statements, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = target_dir / "liar_index.faiss"
    metadata_path = target_dir / "liar_metadata.json"

    logger.info(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, str(index_path))

    logger.info(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info(f"Successfully built LIAR index with {index.ntotal} vectors at {target_dir}.")
    return True


if __name__ == "__main__":
    success = build_index()
    sys.exit(0 if success else 1)
