# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""CLI utility that builds **one FAISS index per dataset instance**.

Given a HuggingFace dataset, the script:

1. Loads the specified *split*.
2. For every row, extracts a long context field (default: ``context``).
3. Uses :class:`~amzn_long_context_rag.retriever.retriever.Retriever`
   to token-split, embed, and FAISS-index the chunks.
4. Writes each per-instance index to ``<save_dir>/<instance_id>/`` containing:

   * ``faiss.index``  - the FAISS index.
   * ``mapping.pkl``  - the list of chunk strings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset, Features, Value, Sequence
from loguru import logger

from src.amzn_long_context_rag.retriever.retriever import (
    Retriever,
)

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
        default="THUDM/LongBench-v2",
        help="HuggingFace dataset repository path.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/indexes",
        help="Directory where per-instance index folders will be created.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens per chunk when splitting a context (via tiktoken).",
    )
    parser.add_argument(
        "--context_field",
        type=str,
        default="context",
        help="Name of the field containing the long context.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size to use when embedding chunks.",
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="Qwen/Qwen3-Embedding-4B",
        help="SentenceTransformer model to use for embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Retriever device.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def long_context_indexing(
    *,
    dataset_path: str,
    split_name: str,
    save_dir: Path,
    max_tokens: int,
    context_field: str,
    batch_size: int,
    embedding_model_name: str,
    device: str,
) -> None:  # - imperative mood preferred
    """Driver function that orchestrates per-instance indexing."""

    save_dir = Path(save_dir, dataset_path.split("/")[-1], split_name)

    logger.info("Configuration:")
    logger.info("  Dataset:          {}", dataset_path)
    logger.info("  Split:            {}", split_name)
    logger.info("  Save directory:   {}", save_dir)
    logger.info("  Max tokens/chunk: {}", max_tokens)
    logger.info("  Context field:    {}", context_field)
    logger.info("  Batch size:       {}", batch_size)
    logger.info("  Embedding model:  {}", embedding_model_name)
    logger.info("  Device         :  {}", device)

    # 1. Load the dataset split
    logger.info("Loading dataset '{}' (split='{}') …", dataset_path, split_name)

    dataset = None
    if "InfiniteBench" in dataset_path:
        ft = Features({
            "id": Value("int64"),
            "context": Value("string"),
            "input": Value("string"),
            "answer": Sequence(Value("string")),
            "options": Sequence(Value("string"))
        })
        dataset = load_dataset(path=dataset_path, split=split_name, features=ft)
    else:
        dataset = load_dataset(path=dataset_path, split=split_name)

    num_instances: int = len(dataset)
    logger.info("Found {} instances.", num_instances)

    # 2. Ensure output directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    # 3. Instantiate a Retriever with the requested configuration
    retriever = Retriever(
        embedding_model_name=embedding_model_name,
        max_tokens=max_tokens,
        device=device
    )

    # 4. Build an index for each dataset row
    for count, example in enumerate(dataset, start=1):
        # Prefer ``id``; fall back to ``_id`` if present.
        instance_id: str | int = example.get("id", example.get("_id"))
        if instance_id is None:
            err_msg = "Example is missing both 'id' and '_id' fields."
            raise KeyError(err_msg)

        if context_field not in example or not isinstance(example[context_field], str):
            err_msg = f"Context field '{context_field}' not found or not a string."
            raise KeyError(err_msg)

        context_text: str = example[context_field]
        inst_dir: Path = save_dir / str(instance_id)
        inst_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[{}/{}] Building index for instance_id='{}'", count, num_instances, instance_id)

        retriever.build_index_from_long_document(
            document=context_text,
            save_dir=inst_dir,
            batch_size=batch_size,
        )

    logger.success("All instance indexes built successfully at '{}'.", save_dir)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli_args = _parse_cli_args()

    long_context_indexing(
        dataset_path=cli_args.dataset_path,
        split_name=cli_args.split_name,
        save_dir=Path(cli_args.save_dir),
        max_tokens=cli_args.max_tokens,
        context_field=cli_args.context_field,
        batch_size=cli_args.batch_size,
        embedding_model_name=cli_args.embedding_model_name,
        device=cli_args.device
    )
