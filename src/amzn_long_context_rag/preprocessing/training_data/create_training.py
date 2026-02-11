# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import random
import argparse
import jsonlines
from tqdm import tqdm
from pathlib import Path
from typing import Any

from datasets import load_dataset
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
        default="hotpotqa/hotpot_qa",
        help="HuggingFace dataset repository path.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name to use.",
    )
    parser.add_argument(
        "--index_root",
        type=str,
        default="data/indexes/wikipedia/20231101.en",
        help="Root directory that contains the Wikipedia index.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/train",
        help="Path to the JSONL file that will be written.",
    )
    parser.add_argument(
        "--query_field",
        type=str,
        default="question",
        help="Field in the dataset row that contains the retrieval query.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for retrieval.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=480, ## 64.000 / 133,3 ~= 480; 32.000 / 133,3 ~= 250; 133,3 tokens = 100 words * 4/3
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
        default="cuda:7",
        help="Device for the Retriever.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _create_training_instance(example: dict[str, Any], chunks: list[str]) -> dict[str, Any]:
    supporting_titles = example["supporting_facts"]["title"]
    supporting_sent_id = example["supporting_facts"]["sent_id"]
    supporting_facts = []
    for title, id in zip(supporting_titles, supporting_sent_id):
        for context_title, context_sentences in zip(example["context"]["title"], example["context"]["sentences"]):
            if context_title == title:
                if isinstance(id, list):
                    for ii in id:
                        try:
                            supporting_facts.append(context_sentences[ii])
                        except:
                            logger.warning(f'Missing gold passage for instance {example["id"]}')
                if isinstance(id, int):
                    try:
                        supporting_facts.append(context_sentences[id])
                    except:
                        logger.warning(f'Missing gold passage for instance {example["id"]}')

    all_documents = supporting_facts + chunks
    random.shuffle(all_documents)
    training_context = ""
    gold_doc_ids = []
    for i, doc in enumerate(all_documents):
        training_context += f'[DOC {i}]\n{doc}\n\n'
        if doc in supporting_facts:
            gold_doc_ids.append(f'[DOC {i}]')

    example["training_context"] = training_context
    example["gold_doc_ids"] = gold_doc_ids
    return example
    

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def create_training(
    *,
    dataset_path: str,
    dataset_split: str,
    dataset_name: str,
    index_root: Path,
    output_dir: Path,
    query_field: str,
    batch_size: int,
    top_k: int,
    embedding_model_name: str,
    device: str
) -> None:
    """Driver function that loads indexes and writes retrieved chunks to JSONL."""
    dataset_alias: str = dataset_path.split("/")[-1]
    output_dir = output_dir / dataset_alias
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'train.jsonl'

    logger.info("Configuration:")
    logger.info("  Dataset path:     {}", dataset_path)
    logger.info("  Dataset Split:    {}", dataset_split)
    logger.info("  Dataset Name:     {}", dataset_name)
    logger.info("  Index root dir:   {}", index_root)
    logger.info("  Output dir:       {}", output_dir)
    logger.info("  Query field:      {}", query_field)
    logger.info("  Top-k:            {}", top_k)
    logger.info("  Embedding model:  {}", embedding_model_name)
    logger.info("  Device:           {}", device)

    logger.info("Loading dataset '{}' (split='{}') …", dataset_path, dataset_split)

    dataset = load_dataset(
        path=dataset_path,
        split=dataset_split, 
        name=dataset_name, 
        trust_remote_code=True
    )

    num_instances: int = len(dataset)
    logger.info("Found {} instances.", num_instances)

    retriever = Retriever(embedding_model_name=embedding_model_name, device=device)

    index_path: Path = index_root / "faiss.index"
    mapping_path: Path = index_root / "mapping.pkl"

    if not index_path.exists() or not mapping_path.exists():
        raise FileNotFoundError(
            f"Index files missing in directory '{index_root}'."
        )

    # Load the index
    retriever.load_index(index_path, mapping_path)

    # ------------------------------------------------------------------
    # Iterate over rows, retrieve, and write JSONL
    # ------------------------------------------------------------------
    with jsonlines.open(output_path, "w") as out_f:
        for start in tqdm(range(0, num_instances, batch_size)):
            end = min(start + batch_size, num_instances)
            batch = dataset.select(list(range(start, end)))

            queries: list[str] = batch[query_field]
            batch_results = retriever.retrieve_batch(
                queries,
                batch_size=batch_size,
                top_k=top_k
            )

            for ex, chunks in zip(batch, batch_results, strict=True):
                training_line = _create_training_instance(
                    ex, [c for c, _ in chunks]
                )
                out_f.write(training_line)

            logger.info("Processed {}/{} examples.", end, num_instances)

    logger.success("Finished! Training set written to '{}'.", output_path)

# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli_args = _parse_cli_args()

    create_training(
        dataset_path=cli_args.dataset_path,
        dataset_split=cli_args.dataset_split,
        dataset_name=cli_args.dataset_name,
        index_root=Path(cli_args.index_root),
        output_dir=Path(cli_args.output_dir),
        query_field=cli_args.query_field,
        batch_size=cli_args.batch_size,
        top_k=cli_args.top_k,
        embedding_model_name=cli_args.embedding_model_name,
        device=cli_args.device
    )