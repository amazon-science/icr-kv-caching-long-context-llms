# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Promote distractor documents that are near-duplicates of any gold document
"""
from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path
from typing import List, Set
from rapidfuzz import fuzz
from tqdm import tqdm

from loguru import logger
import jsonlines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean training JSONL from near-duplicate docs.")
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True, 
        help="Path to the original JSONL file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Where the cleaned JSONL will be written.",
    )
    parser.add_argument(
        "--promotion_thresh",
        type=float,
        default=0.6, # TODO: manually validate on a subset of the training data.
        help="Similarity threshold for promoting documents to gold status (0-1).",
    )
    parser.add_argument(
        "--max_promotions_per_example",
        type=int,
        default=10,
        help="Maximum number of documents to promote to gold per example.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_DOC_REGEX = re.compile(r"\[DOC (\d+)]\s*(.*?)(?=\s*\[DOC \d+]|$)", re.DOTALL)


def _split_docs(training_context: str) -> List[str]:
    """Return list of document strings in the order they appear."""
    return [m.group(2).strip() for m in _DOC_REGEX.finditer(training_context)]


def _extract_gold_idxs(gold_doc_ids: Iterable[str]) -> Set[int]:
    """Extract integer indices from '[DOC k]' strings."""
    idxs: Set[int] = set()
    for tag in gold_doc_ids:
        m = re.match(r"\[DOC (\d+)]", tag)
        if m:
            idxs.add(int(m.group(1)))
    return idxs


def similarity_score(a: str, b: str) -> float:
    """
    Calculate similarity score between two documents using multiple metrics.
    Returns a score between 0 and 1.
    """
    # Use token_sort_ratio for better semantic similarity detection
    token_sort = fuzz.token_sort_ratio(a, b, processor=str.lower) / 100.0
    
    # Use partial_ratio to detect if one document contains relevant parts of another
    partial = max(
        fuzz.partial_ratio(a, b, processor=str.lower),
        fuzz.partial_ratio(b, a, processor=str.lower)
    ) / 100.0
    
    # Use token_set_ratio to handle documents with different order but similar content
    token_set = fuzz.token_set_ratio(a, b, processor=str.lower) / 100.0
    
    # Return weighted average, giving more weight to token_sort and token_set
    # as they're better for semantic similarity
    return (token_sort * 0.4 + partial * 0.3 + token_set * 0.3)


def _process_docs_with_promotion(
    docs: list[str],
    gold_idxs: set[int],
    *,
    promotion_thresh: float = 0.7,
    max_promotions_per_example: int = 5,
) -> tuple[list[str], list[int], set[int]]:
    """
    Process documents with optional promotion of similar documents to gold status.
    Returns (kept_docs, kept_old_idxs, new_gold_idxs).
    """
    
    # When promoting, keep ALL documents in original order
    gold_docs = [docs[i] for i in sorted(gold_idxs)]
    new_gold_idxs = set(gold_idxs)  # Start with original gold indices
    promotion_candidates = []  # (similarity_score, idx)
    
    # Check all non-gold documents for promotion
    for idx, doc in enumerate(docs):
        if idx in gold_idxs:
            continue  # Skip already gold documents
            
        # Calculate similarity to gold documents
        if gold_docs:
            max_sim = max(similarity_score(doc, g) for g in gold_docs)
            if max_sim >= promotion_thresh:
                promotion_candidates.append((max_sim, idx))
    
    # Sort by similarity score (highest first) and promote top candidates
    promotion_candidates.sort(reverse=True, key=lambda x: x[0])
    promoted_count = 0
    
    for sim_score, idx in promotion_candidates:
        if promoted_count >= max_promotions_per_example:
            break
            
        # Promote to gold
        new_gold_idxs.add(idx)
        promoted_count += 1
        logger.debug(f"Promoted document {idx} to gold (similarity: {sim_score:.3f})")
    
    # Return ALL documents in original order with original indices
    kept_docs = docs[:]  # Keep all documents
    kept_old_idxs = list(range(len(docs)))  # Keep original indices
    
    return kept_docs, kept_old_idxs, new_gold_idxs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def promote_hard_negatives(
    input_path: Path,
    output_path: Path,
    promotion_thresh: float,
    max_promotions_per_example: int,
) -> None:
    n_in, n_out = 0, 0
    total_promotions = 0
    original_gold_counts = 0
    final_gold_counts = 0
    with jsonlines.open(input_path, "r") as reader, jsonlines.open(output_path, "w") as writer:
        for example in tqdm(reader):
            n_in += 1
            
            original_gold_count = len(example["gold_doc_ids"])

            docs = _split_docs(example["training_context"])

            gold_idxs = _extract_gold_idxs(example["gold_doc_ids"])

            cleaned_docs, kept_old_idxs, new_gold_idxs = _process_docs_with_promotion(
                docs,
                gold_idxs,
                promotion_thresh=promotion_thresh,
                max_promotions_per_example=max_promotions_per_example,
            )

            # Only update gold_doc_ids, keep original training_context
            example["promoted_gold_doc_ids"] = [f"[DOC {idx}]" for idx in sorted(new_gold_idxs)]
            promotions_in_example = len(new_gold_idxs) - len(gold_idxs)
            total_promotions += promotions_in_example
            
            if promotions_in_example > 0:
                logger.debug(f"Example {n_in}: promoted {promotions_in_example} documents to gold")

            final_gold_count = len(example["promoted_gold_doc_ids"])
            original_gold_counts += original_gold_count
            final_gold_counts += final_gold_count
            writer.write(example)
            n_out += 1

    logger.success("Finished! Wrote {} examples ({} processed) to '{}'.", n_out, n_in, output_path)
    logger.info("Average number of gold documents per example (before): {}", original_gold_counts / n_out)
    logger.info("Average number of gold documents per example (after): {}", final_gold_counts / n_out)
    logger.info("Total documents promoted to gold status: {}", total_promotions)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    logger.info("Input JSONL:  {}", input_path)
    logger.info("Output JSONL: {}", output_path)
    logger.info("Promotion threshold: {}", args.promotion_thresh)
    logger.info("Max promotions per example: {}", args.max_promotions_per_example)

    promote_hard_negatives(
        input_path=input_path,
        output_path=output_path,
        promotion_thresh=args.promotion_thresh,
        max_promotions_per_example=args.max_promotions_per_example,
    )


if __name__ == "__main__":
    main()