# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Compute multiple-choice accuracy (A-D) from one or more JSONL result files.

Usage examples
--------------

Single file::

    python compute_accuracy.py --files path/to/results.jsonl

Multiple files::

    python compute_accuracy.py --files zs_full.jsonl zs_noctx.jsonl rag_topk.jsonl

When multiple files are provided, accuracy is computed *for each file* **only on
those instance IDs that appear in the smallest file** (the typical “intersection
baseline” used when different prompts skip different rows).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import jsonlines
from loguru import logger

# ---------------------------------------------------------------------------
# Prediction extraction helpers
# ---------------------------------------------------------------------------

EXTRACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"The\s+correct\s+answer\s+is\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"\b(?:answer|option)\s*(?:is|:)\s*\(?([A-D])\)?", re.IGNORECASE),
]


def _extract_prediction(text: str | None) -> str | None:
    """Return the predicted choice letter (``A``-``D``) or ``None``."""
    if not text:
        return None
    for pat in EXTRACTION_PATTERNS:
        if m := pat.search(text):
            return m.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _load_results(path: Path) -> dict[str, tuple[str, str | None]]:
    """Load a JSONL file keyed by *instance id* → (gold, pred)."""
    records: dict[str, tuple[str, str | None]] = {}
    with jsonlines.open(path, "r") as rdr:
        for row in rdr:
            iid = str(row.get("id", row.get("_id")))
            gold = str(row.get("answer", "")).strip().upper()
            pred = _extract_prediction(row.get("model_output"))
            records[iid] = (gold, pred)
    return records


def _compute_accuracy(pairs: list[tuple[str, str | None]]) -> tuple[int, int, int, float]:
    """Return *(total, correct, missing, acc%)* for *pairs* of (gold, pred)."""
    total = len(pairs)
    correct = sum(1 for g, p in pairs if p is not None and p == g)
    missing = sum(1 for _g, p in pairs if p is None)
    acc = 100.0 * correct / total if total else 0.0
    return total, correct, missing, acc


def evaluate(paths: list[Path]) -> None:
    """Compute and log accuracy for one or more JSONL files."""
    if not paths:
        msg = "No input files supplied."
        raise ValueError(msg)

    logger.info("Evaluating {} file(s)…", len(paths))

    # 1. Load all files into dicts keyed by instance id
    data = {p: _load_results(p) for p in paths}

    # 2. Determine the *reference ID set* - IDs in the smallest file
    ref_path = min(paths, key=lambda p: len(data[p]))
    ref_ids = set(data[ref_path].keys())
    logger.info("Reference set → '{}' with {} instances.", ref_path.name, len(ref_ids))

    # 3. For each file, compute accuracy restricted to *ref_ids*
    for path in paths:
        pairs = [data[path][iid] for iid in ref_ids if iid in data[path]]
        total, correct, missing, acc = _compute_accuracy(pairs)

        logger.info("\nResults - {}", path)
        logger.info("-----------------------")
        logger.info("Total (evaluated)  : {}", total)
        logger.info("Correct            : {}", correct)
        logger.info("Missing prediction : {}", missing)
        logger.info("Accuracy           : {:.2f}%", acc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute accuracy from JSONL result files.")
    ap.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="One or more JSONL files (space-separated).",
        type=Path,
    )
    cli_args = ap.parse_args()
    evaluate(cli_args.files)
