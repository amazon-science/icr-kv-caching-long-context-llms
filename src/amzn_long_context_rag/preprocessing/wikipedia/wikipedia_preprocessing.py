# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import re
import jsonlines
import argparse
from tqdm import tqdm
from pathlib import Path

from datasets import load_dataset
from loguru import logger

# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------
def _parse_cli_args() -> argparse.Namespace:  # - imperative mood preferred
    """Parse command-line arguments and return an :class:`argparse.Namespace`."""
    parser = argparse.ArgumentParser(
        description=("Build one FAISS index per instance for a HuggingFace dataset's " "context field."),
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="wikimedia/wikipedia",
        help="HuggingFace dataset repository path.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="20231101.en",
        help="Dataset name to use.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data",
        help="Directory where per-instance index folders will be created.",
    )
    parser.add_argument(
        "--passage_len_words",
        type=int,
        default=100,
        help="Maximum tokens per chunk when splitting a context (via tiktoken).",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Name of the field containing the long context.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalise_whitespace(text: str) -> str:
    "Collapse all internal whitespace; strip leading/trailing."
    return re.sub(r"\s+", " ", text).strip()

def passage_iter(dataset_iter, passage_len=100):
    """
    Yields (title, passage_text) pairs of `passage_len` words
    from the streaming HF dataset iterator.
    """
    for item in dataset_iter:
        title = normalise_whitespace(item["title"])
        # DPR uses the raw WikiExtractor output. HF's `text` field is already
        # plain text with section markers removed – close enough for retrieval.
        words = normalise_whitespace(item["text"]).split(" ")
        for start in range(0, len(words), passage_len):
            chunk = words[start : start + passage_len]
            if chunk:                      # ignore empty trailing chunks
                yield title, " ".join(chunk)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def wikipedia_preprocessing(
    *,
    dataset_path: str,
    name: str,
    split: str,
    passage_len_words: int,
    save_dir: Path,
) -> None:  # - imperative mood preferred
    """Driver function that orchestrates per-instance indexing."""

    save_dir = Path(save_dir, dataset_path.split("/")[-1], name)

    logger.info("Configuration:")
    logger.info("  Dataset:          {}", dataset_path)
    logger.info("  Name:             {}", name)
    logger.info("  Split:            {}", split)
    logger.info("  Passage Len Words {}", passage_len_words)
    logger.info("  Save directory:   {}", save_dir)

    logger.info("Loading dataset '{}' (subset='{}') …", dataset_path, name)

    dataset = load_dataset(
        path=dataset_path, 
        name=name, 
        split=split, 
        trust_remote_code=True, 
        streaming=True
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f'passages_words={passage_len_words}.jsonl'

    with jsonlines.open(out_path, mode='w') as writer:
        pid = 0
        for title, passage in tqdm(passage_iter(dataset, passage_len_words),
                                    desc="Preprocessing Wikipedia", unit="passage"):
            writer.write({
                "pid": pid,
                "title": title,
                "passage": passage
            })

            pid += 1

    logger.success("All instances preprocessed successfully at '{}'.", out_path)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli_args = _parse_cli_args()

    wikipedia_preprocessing(
        dataset_path=cli_args.dataset_path,
        name=cli_args.name,
        split=cli_args.split,
        passage_len_words=cli_args.passage_len_words,
        save_dir=Path(cli_args.save_dir),
    )