# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Compute accuracy using **xFinder** answer extraction.

This script relies on an xFinder model to parse the model outputs.
It can evaluate **one or many** JSONL result files:

Single file::

    python xfinder_eval.py --files zs_full.jsonl

Multiple files (intersection evaluation)::

    python xfinder_eval.py --files zs_full.jsonl rag_topk.jsonl zs_noctx.jsonl

When several files are provided, accuracy for every file is computed **only on
instance IDs present in the smallest file** (so all systems are compared on
the same subset).
"""

from __future__ import annotations

import json
import argparse
from collections.abc import Sequence
from pathlib import Path

import jsonlines
from loguru import logger
from tqdm import tqdm
from transformers import modeling_utils
from xfinder.eval import Evaluator

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

# ---------------------------------------------------------------------------
# xFinder initialisation (lazy-singleton)
# ---------------------------------------------------------------------------


class _LazyEvaluator:
    """Instantiate the *Evaluator* exactly once (GPU-heavy)."""

    _instance: Evaluator | None = None

    @classmethod
    def get(cls, model_name: str = "xFinder-llama38it") -> Evaluator:
        if cls._instance is None:
            logger.info("Loading xFinder model '{}' …", model_name)
            cls._instance = Evaluator(
                model_name=model_name,
                inference_mode="local",
                model_path_or_url=f"IAAR-Shanghai/{model_name}",
            )
        return cls._instance


# ---------------------------------------------------------------------------
# Helpers to prepare the xFinder call per dataset row
# ---------------------------------------------------------------------------

LETTERS: list[str] = ["A", "B", "C", "D", "E", "F"]


def _xfinder_inputs(sample: dict) -> tuple[str, list[list[str]], str, str]:
    r"""Return ``question, answer_range, answer_type, correct_answer`` for xFinder.

    Supported schemas
    -----------------
    1. *choice_\** fields (e.g. ``choice_A``, ``choice_B`` …)
    2. ``options`` list (``{"options": ["Paris", "London", …]}``)

    The correct answer letter must be in ``sample['answer']``.
    """

    answer_range: list[list[str]] = []

    # Schema 1: explicit ``choice_X`` keys
    if any(k.startswith("choice_") for k in sample):
        for letter in LETTERS:
            lower_case_key = f"choice_{letter.lower()}"
            upper_case_key = f"choice_{letter}"
            if lower_case_key in sample:
                answer_range.append([letter, str(sample[lower_case_key])])
            elif upper_case_key in sample:
                answer_range.append([letter, str(sample[upper_case_key])])

    # Schema 2: list stored under "options"
    elif "options" in sample and isinstance(sample["options"], list):
        for letter, option in zip(LETTERS, sample["options"], strict=False):
            answer_range.append([letter, str(option)])

    else:
        msg = "Sample lacks 'choice_*' fields and an 'options' list; cannot build answer range."
        raise KeyError(msg)

    # Assemble the prompt for xFinder
    joined = " ".join(f"({label}) {text}" for label, text in answer_range)

    question = None
    if "input" in sample:
        question = f"{sample['input']} Answer Choices: {joined}"
    elif "question" in sample: 
        question = f"{sample['question']} Answer Choices: {joined}"

    answer_type = "alphabet_option"

    correct_answer = None
    if isinstance(sample["answer"], str):
        correct_answer = sample["answer"].strip()
    elif isinstance(sample["answer"], list):
        correct_answer = sample["answer"][0].strip()

    # If the answer is represented with the text, we need to convert it into the corresponding letter.
    if correct_answer not in LETTERS: 
        correct_answer = next(pair[0] for pair in answer_range if pair[1].strip() == correct_answer)

    return question, answer_range, answer_type, correct_answer


# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------


def _load_file(path: Path, evaluator: Evaluator, maximum_input_length: int | list) -> dict[str, tuple[str, str | None]]:
    """Load *path* into a mapping ``id -> (gold, pred)`` using xFinder."""
    records: dict[str, tuple[str, str | None]] = {}
    with jsonlines.open(path, "r") as rdr:
        for row in tqdm(rdr, desc=f"xfinder→{path.name}"):
            original_ctx_length = row.get("original_ctx_length", None)
            if isinstance(maximum_input_length, int):
                if original_ctx_length > maximum_input_length:
                    continue
            if isinstance(maximum_input_length, list):
                if original_ctx_length < maximum_input_length[0] or original_ctx_length > maximum_input_length[1]:
                    continue
            iid = str(row.get("id", row.get("_id")))
            question, rng, typ, gold = _xfinder_inputs(row)
            pred = evaluator.evaluate_single_item(
                question=question,
                llm_output=row.get("model_output", ""),
                answer_range=rng,
                answer_type=typ,
                correct_answer=gold,
            )[-3]
            records[iid] = (gold.upper(), pred.upper() if pred else None)
    return records


def _compute_accuracy(pairs: Sequence[tuple[str, str | None]]) -> tuple[int, int, int, float]:
    total = len(pairs)
    correct = sum(1 for g, p in pairs if p == g)
    missing = sum(1 for _g, p in pairs if p is None)
    acc = 100.0 * correct / total if total else 0.0
    return total, correct, missing, acc


def evaluate(paths: list[Path], model_name: str = "xFinder-llama38it", output_dir: str = "data/evaluation", maximum_input_length: int | list = None) -> None:
    if not paths:
        msg = "No --files provided."
        raise ValueError(msg)

    if maximum_input_length and len(maximum_input_length) == 1:
        maximum_input_length = maximum_input_length[0]
    
    output_dir = Path(output_dir)

    evaluator = _LazyEvaluator.get(model_name)

    logger.info("Evaluating {} file(s) with xFinder …", len(paths))

    data = {p: _load_file(p, evaluator, maximum_input_length) for p in paths}

    ref_path = min(paths, key=lambda p: len(data[p]))
    ref_ids = set(data[ref_path].keys())
    logger.info("Reference set = '{}' ({} instances).", ref_path.name, len(ref_ids))

    for path in paths:
        subset = [data[path][iid] for iid in ref_ids if iid in data[path]]
        total, correct, missing, acc = _compute_accuracy(subset)

        out_subdir = output_dir / Path("/".join(path.parts[2:-1]))
        out_subdir.mkdir(parents=True, exist_ok=True)

        if maximum_input_length:
            if isinstance(maximum_input_length, int):
                out_path = out_subdir / f"{path.stem}_maximum_input_length={maximum_input_length}.json"
            if isinstance(maximum_input_length, list):
                out_path = out_subdir / f"{path.stem}_maximum_input_length=[{maximum_input_length[0]},{maximum_input_length[1]}].json"    
        else:
            out_path = out_subdir / f"{path.stem}.json"

        with open(out_path, "w") as fout:
            json.dump(
                {
                    "instances": total, 
                    "correct": correct, 
                    "missing": missing, 
                    "accuracy": acc,
                    "maximum_input_length": maximum_input_length
                }, 
                fout, 
                indent=2
            )

        logger.info("\nResults - {}", path.name)
        logger.info("-----------------------")
        logger.info("Instances          : {}", total)
        logger.info("Max Input Length   : {}", maximum_input_length)
        logger.info("Correct            : {}", correct)
        logger.info("Missing prediction : {}", missing)
        logger.info("Accuracy           : {:.2f}%", acc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="xFinder-based accuracy computation")
    ap.add_argument("--files", nargs="+", type=Path, required=True, help="JSONL result files")
    ap.add_argument("--model_name", type=str, default="xFinder-llama38it", help="xFinder model to load")
    ap.add_argument("--output_dir", type=str, default="data/evaluation", help="Output directory to write the evaluation results")
    ap.add_argument('--maximum_input_length', type=int, nargs='+', help="The maximum number of context tokens. Used to filer out instances.")
    cli = ap.parse_args()
    evaluate(cli.files, model_name=cli.model_name, output_dir=cli.output_dir, maximum_input_length=cli.maximum_input_length)
