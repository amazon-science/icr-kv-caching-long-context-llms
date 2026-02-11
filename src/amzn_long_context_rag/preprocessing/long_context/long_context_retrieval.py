# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import argparse
import jsonlines
from pathlib import Path
from typing import Any

from datasets import load_dataset, Features, Value, Sequence
from loguru import logger

from src.amzn_long_context_rag.retriever.retriever import (
Retriever,
)

# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------
def _parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
    description="Retrieve top-k chunks from pre-built per-instance FAISS indexes.",
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
        "--index_root",
        type=str,
        default="data/indexes",
        help=(
            "Root directory that contains the per-instance index folders created "
            "by the indexing script."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/retriever/outputs",
        help="Path to the JSONL file that will be written.",
    )
    parser.add_argument(
        "--query_field",
        type=str,
        default=None,
        help="Field in the dataset row that contains the retrieval query.",
    )
    parser.add_argument(
        "--user_query",
        type=str,
        default=None,
        help="If the dataset does not provide a specific input, you can provide a custom one.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Number of chunks to retrieve (-1 means 'all').",
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="Qwen/Qwen3-Embedding-4B",
        help="SentenceTransformer model to initialise the Retriever.",
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
def retrieve_chunks(
    *,
    dataset_path: str,
    split_name: str,
    index_root: Path,
    output_dir: Path,
    query_field: str,
    user_query: str,
    top_k: int,
    embedding_model_name: str,
    device: str
) -> None:
    """Driver function that loads indexes and writes retrieved chunks to JSONL."""
    dataset_name: str = dataset_path.split("/")[-1]
    index_root = index_root / dataset_name / split_name
    output_dir = output_dir / dataset_name / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{embedding_model_name.split("/")[-1]}.jsonl'

    logger.info("Configuration:")
    logger.info("  Dataset:          {}", dataset_path)
    logger.info("  Split:            {}", split_name)
    logger.info("  Index root dir:   {}", index_root)
    logger.info("  Output dir:       {}", output_dir)
    logger.info("  Query field:      {}", query_field)
    logger.info("  User Query:       {}", user_query)
    logger.info("  Top-k:            {}", top_k)
    logger.info("  Embedding model:  {}", embedding_model_name)
    logger.info("  Device:           {}", device)

    # ------------------------------------------------------------------
    # Load the dataset split (handle InfiniteBench special-case)
    # ------------------------------------------------------------------
    logger.info("Loading dataset '{}' (split='{}') …", dataset_path, split_name)

    if "InfiniteBench" in dataset_path:
        ft = Features(
            {
                "id": Value("int64"),
                "context": Value("string"),
                "input": Value("string"),
                "answer": Sequence(Value("string")),
                "options": Sequence(Value("string")),
            }
        )
        dataset = load_dataset(path=dataset_path, split=split_name, features=ft)
    else:
        dataset = load_dataset(path=dataset_path, split=split_name)

    num_instances: int = len(dataset)
    logger.info("Found {} instances.", num_instances)

    # ------------------------------------------------------------------
    # Prepare a single Retriever (we'll reuse it, loading a new index each time)
    # ------------------------------------------------------------------
    retriever = Retriever(
        embedding_model_name=embedding_model_name,
        device=device
    )

    # ------------------------------------------------------------------
    # Iterate over rows, retrieve, and write JSONL
    # ------------------------------------------------------------------
    with jsonlines.open(output_path, "w",) as out_f:
        for count, example in enumerate(dataset, start=1):
            instance_id: str | int | None = example.get("id", example.get("_id"))
            if instance_id is None:
                raise KeyError("Example is missing both 'id' and '_id' fields.")

            if query_field is not None:
                query: str | None = example.get(query_field)
                if query is None or not isinstance(query, str):
                    raise KeyError(
                        f"Query field '{query_field}' not found or not a string."
                    )
            elif user_query is not None:
                query: str | None = str(user_query)
            
            else:
                raise ValueError(
                    f"Either 'query_field' or 'user_query' should be provided."
                )

            inst_dir: Path = index_root / str(instance_id)
            index_path: Path = inst_dir / "faiss.index"
            mapping_path: Path = inst_dir / "mapping.pkl"

            if not index_path.exists() or not mapping_path.exists():
                raise FileNotFoundError(
                    f"Index files missing for instance '{instance_id}' "
                    f"in directory '{inst_dir}'."
                )

            logger.info(
                "[{}/{}] Retrieving chunks for instance_id='{}'",
                count,
                num_instances,
                instance_id,
            )

            # Load the index + mapping and retrieve
            retriever.load_index(index_path, mapping_path)
            results = retriever.retrieve(query=query, top_k=top_k)
            chunks: list[str] = [chunk for chunk, _score in results]

            # Write JSONL line
            json_line: dict[str, Any] = {"id": instance_id, "chunks": chunks}
            out_f.write(json_line)

    logger.success("Finished! Retrieved chunks written to '{}'.", output_path)

# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli_args = _parse_cli_args()

    retrieve_chunks(
        dataset_path=cli_args.dataset_path,
        split_name=cli_args.split_name,
        index_root=Path(cli_args.index_root),
        output_dir=Path(cli_args.output_dir),
        query_field=cli_args.query_field,
        user_query=cli_args.user_query,
        top_k=cli_args.top_k,
        embedding_model_name=cli_args.embedding_model_name,
        device=cli_args.device
    )