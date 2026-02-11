# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import re
import argparse
import jsonlines
from pathlib import Path
from tqdm import tqdm
from loguru import logger


def _parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter training dataset by word count and reformat documents (always keeping gold docs).",
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file to filter.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output filtered JSONL file.",
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=10,
        help="Minimum number of words required for a document to be kept (default: 10).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Maximum number of training samples to clean (default: -1).",
    )

    return parser.parse_args()


def count_words(text: str) -> int:
    """Count the number of words in a text string."""
    if text is None or text.strip() == "":
        return 0
    return len(text.split())


def parse_training_context(training_context: str) -> list[tuple[int, str]]:
    """
    Parse training_context to extract document ID and content pairs robustly.

    Returns:
        List of (doc_id, content) tuples in order of appearance.
    """
    # Find all [DOC N] markers and capture content until the next marker or end
    marker_re = re.compile(r"\[DOC (\d+)\]\s*")
    matches = list(marker_re.finditer(training_context))

    documents: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        doc_id = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(training_context)
        content = training_context[start:end].strip()
        # include all docs (even if content is empty) — gold docs must not be lost
        documents.append((doc_id, content))

    return documents


def filter_and_reformat_context(
    training_context: str,
    gold_doc_ids: list[str],
    min_words: int
) -> tuple[str, list[str]]:
    """
    Filter documents by word count and reformat with better demarcation.
    Gold documents are always preserved. If a gold doc is missing from parsing,
    we attempt to recover it from the raw string and append it.
    """
    # Parse original documents (robustly)
    documents = parse_training_context(training_context)
    original_doc_map = {doc_id: content for doc_id, content in documents}

    # Extract gold IDs as integers for easier checking and keep original gold order
    gold_ids_int = []
    for gold_id in gold_doc_ids:
        match = re.match(r"\[DOC (\d+)\]", gold_id)
        if match:
            gold_ids_int.append(int(match.group(1)))

    filtered_docs: list[str] = []
    old_to_new_id_map: dict[int, int] = {}

    # Keep docs if they meet the min_words threshold OR are gold
    for old_doc_id, content in documents:
        word_count = count_words(content)
        keep_doc = (word_count >= min_words) or (old_doc_id in gold_ids_int)
        if keep_doc:
            new_doc_id = len(filtered_docs)
            old_to_new_id_map[old_doc_id] = new_doc_id
            filtered_docs.append(content)

    # Ensure every gold doc is present — if not, attempt to recover and append it
    for old_gold in gold_ids_int:
        if old_gold not in old_to_new_id_map:
            # try to get content from parsed map first
            content = original_doc_map.get(old_gold, None)
            if content is None:
                # fallback: robust regex search using DOTALL
                search_re = re.compile(rf"\[DOC {old_gold}\]\s*(.*?)(?=\[DOC \d+\]|\Z)", re.DOTALL)
                m = search_re.search(training_context)
                content = m.group(1).strip() if m else ""
            # append it to the end and map it
            new_doc_id = len(filtered_docs)
            old_to_new_id_map[old_gold] = new_doc_id
            filtered_docs.append(content)
            logger.warning("Recovered and appended missing gold doc [DOC {}].", old_gold)

    # Reformat with clearer boundaries
    new_training_context = ""
    for new_doc_id, content in enumerate(filtered_docs):
        new_training_context += f"[DOC {new_doc_id}]\n{content}\n\n"
    new_training_context = new_training_context.rstrip()

    # Remap gold_doc_ids preserving original gold order (and duplicates if any)
    new_gold_doc_ids: list[str] = []
    for gold_id in gold_doc_ids:
        match = re.match(r"\[DOC (\d+)\]", gold_id)
        if match:
            old_doc_id = int(match.group(1))
            # now it must exist in old_to_new_id_map (we forced-add missing golds above)
            if old_doc_id in old_to_new_id_map:
                new_id = old_to_new_id_map[old_doc_id]
                new_gold_doc_ids.append(f"[DOC {new_id}]")
            else:
                # Defensive: should not happen because we appended missing golds.
                logger.error("Gold doc [DOC {}] could not be remapped — this should not happen.", old_doc_id)

    return new_training_context, new_gold_doc_ids


def filter_training_dataset(
    input_file: Path,
    output_file: Path,
    min_words: int,
    num_samples: int,
) -> None:
    """
    Filter training dataset and write filtered version.
    """
    logger.info("Configuration:")
    logger.info("  Input file:       {}", input_file)
    logger.info("  Output file:      {}", output_file)
    logger.info("  Min words:        {}", min_words)
    logger.info("  Num samples:      {}", num_samples)

    total_instances = 0
    total_docs_before = 0
    total_docs_after = 0
    total_gold_docs_before = 0
    total_gold_docs_after = 0

    total_words_before = 0
    total_words_after = 0

    total_gold_words_before = 0
    total_gold_words_after = 0

    instances_with_no_gold_left = 0

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(input_file, "r") as in_f, jsonlines.open(output_file, "w") as out_f:
        for i, line in enumerate(tqdm(in_f, desc="Processing")):

            if num_samples != -1 and i >= num_samples:
                break

            total_instances += 1

            training_context = line.get("training_context", "")
            gold_doc_ids = line.get("gold_doc_ids", [])

            # Count before
            original_docs = parse_training_context(training_context)
            total_docs_before += len(original_docs)
            total_gold_docs_before += len(gold_doc_ids)
            total_words_before += sum(count_words(content) for _, content in original_docs)

            # gold words before (only count those gold ids that exist in original parsed docs)
            original_doc_map = {doc_id: content for doc_id, content in original_docs}
            for gold_id in gold_doc_ids:
                m = re.match(r"\[DOC (\d+)\]", gold_id)
                if m:
                    old_id = int(m.group(1))
                    total_gold_words_before += count_words(original_doc_map.get(old_id, ""))

            # Filter & reformat (this guarantees gold docs are preserved)
            new_context, new_gold_ids = filter_and_reformat_context(
                training_context, gold_doc_ids, min_words
            )

            # Count after
            filtered_docs = parse_training_context(new_context)
            total_docs_after += len(filtered_docs)
            total_gold_docs_after += len(new_gold_ids)
            total_words_after += sum(count_words(content) for _, content in filtered_docs)

            # gold words after
            filtered_doc_map = {doc_id: content for doc_id, content in filtered_docs}
            for gold_id in new_gold_ids:
                m = re.match(r"\[DOC (\d+)\]", gold_id)
                if m:
                    new_id = int(m.group(1))
                    total_gold_words_after += count_words(filtered_doc_map.get(new_id, ""))

            # Defensive check: instances that had golds originally but lost them after processing
            if len(gold_doc_ids) > 0 and len(new_gold_ids) == 0:
                instances_with_no_gold_left += 1
                logger.warning("Instance {} originally had gold docs but none remain after filtering.", i)

            # Update the line and write it out
            filtered_line = line.copy()
            filtered_line["training_context"] = new_context
            filtered_line["gold_doc_ids"] = new_gold_ids

            out_f.write(filtered_line)

    # Print stats
    logger.info("\nFiltering Statistics:")
    logger.info("  Total instances:                      {}", total_instances)
    logger.info("  Total documents before filtering:     {}", total_docs_before)
    logger.info("  Total documents after filtering:      {}", total_docs_after)
    logger.info("  Total documents removed:              {}", total_docs_before - total_docs_after)
    logger.info("  Overall document retention rate:      {:.2%}", total_docs_after / max(total_docs_before, 1))
    logger.info("")
    logger.info("  Avg document length before:           {:.2f} words", total_words_before / max(total_docs_before, 1))
    logger.info("  Avg document length after:            {:.2f} words", total_words_after / max(total_docs_after, 1))
    logger.info("")
    logger.info("  Gold documents (always preserved):")
    logger.info("    Gold documents before:              {}", total_gold_docs_before)
    logger.info("    Gold documents after:               {}", total_gold_docs_after)
    logger.info("    Gold document retention rate:       {:.2%}", total_gold_docs_after / max(total_gold_docs_before, 1))
    logger.info("    Avg gold doc length before:         {:.2f} words", total_gold_words_before / max(total_gold_docs_before, 1))
    logger.info("    Avg gold doc length after:          {:.2f} words", total_gold_words_after / max(total_gold_docs_after, 1))
    logger.info("")
    logger.info("  Instances that lost all gold docs:    {}", instances_with_no_gold_left)
    logger.info("")

    logger.success("Filtering completed! Output written to '{}'", output_file)


def main():
    args = _parse_cli_args()
    filter_training_dataset(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        min_words=args.min_words,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
