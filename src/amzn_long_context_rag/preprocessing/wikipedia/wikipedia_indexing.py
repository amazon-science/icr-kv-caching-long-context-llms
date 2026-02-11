# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""CLI utility that builds a FAISS index for Wikipedia with optimized memory management.

Given the Wikipedia HuggingFace dataset, the script:

1. Loads the specified *split*.
2. For every row, extracts the textual field (default: ``passage``).
3. Uses :class:`~amzn_long_context_rag.retriever.retriever.Retriever`
   to embed, and FAISS-index the chunks
4. Writes an index containing:
   * ``faiss.index``  - the FAISS index.
   * ``mapping.pkl``  - the list of chunk strings.
"""

from __future__ import annotations

import argparse
import jsonlines
from tqdm import tqdm
from pathlib import Path
import gc
import torch

from loguru import logger

from src.amzn_long_context_rag.retriever.multi_gpu_retriever import (
    MultiGPURetriever,
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
        "--wikipedia_path",
        type=str,
        default="data/wikipedia/20231101.en/filtered_passages_words=100.jsonl",
        help="Path to the preprocessed Wikipedia.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/indexes",
        help="Directory where per-instance index folders will be created.",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="passage",
        help="Name of the field containing the long context.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size to use when embedding chunks.",
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="Qwen/Qwen3-Embedding-4B",
        help="SentenceTransformer model to use for embeddings.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500_000,
        help="Number of passages to process in each chunk.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def wikipedia_indexing(
    *,
    wikipedia_path: str,
    save_dir: Path,
    batch_size: int,
    embedding_model_name: str,
    chunk_size: int,
) -> None:

    save_dir = Path(save_dir, wikipedia_path.split("/")[-3], wikipedia_path.split("/")[-2])
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Configuration:")
    logger.info("  Wikipedia path:   {}", wikipedia_path)
    logger.info("  Save directory:   {}", save_dir)
    logger.info("  Batch size:       {}", batch_size)
    logger.info("  Chunk size:       {}", chunk_size)
    logger.info("  Embedding model:  {}", embedding_model_name)

    logger.info("Loading MultiGPURetriever {}", embedding_model_name)

    retriever = MultiGPURetriever(
        embedding_model_name=embedding_model_name,
        device="cuda",
        use_multi_gpu=True
    )

    logger.info("Loading preprocessed Wikipedia '{}'", wikipedia_path)

    passages     = []               # current chunk
    chunk_start  = 0                # first pid in the current chunk

    # Log initial GPU memory state
    if torch.cuda.is_available():
        memory_info = retriever.get_gpu_memory_usage()
        logger.info("Initial GPU memory usage: {}", memory_info)

    with jsonlines.open(wikipedia_path, "r") as wikipedia_dataset:
        for i, line in enumerate(tqdm(wikipedia_dataset, desc="Processing Wikipedia")):
            passages.append(line["passage"])

            if len(passages) == chunk_size:
                logger.info("Building index for Wikipedia: from pid {} to {}", chunk_start, i)
                
                # Log memory before processing
                if torch.cuda.is_available():
                    memory_info = retriever.get_gpu_memory_usage()
                    logger.info("GPU memory before chunk {}: {}", chunk_start // chunk_size, memory_info)

                retriever.build_index_from_documents(
                    documents=passages,
                    save_dir=save_dir,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    append=chunk_start != 0
                )

                # Aggressive cleanup after each chunk
                passages.clear()
                gc.collect()
                
                # Clear GPU cache
                retriever.clear_gpu_cache()
                
                # Log memory after processing
                if torch.cuda.is_available():
                    memory_info = retriever.get_gpu_memory_usage()
                    logger.info("GPU memory after chunk {}: {}", chunk_start // chunk_size, memory_info)

                chunk_start = i + 1

    # ---- loop finished: index any remainder ----
    if passages:
        logger.info("Building index for Wikipedia: from pid {} to {}", chunk_start, i)
        
        # Log memory before final processing
        if torch.cuda.is_available():
            memory_info = retriever.get_gpu_memory_usage()
            logger.info("GPU memory before final chunk: {}", memory_info)
        
        retriever.build_index_from_documents(
            documents=passages,
            save_dir=save_dir,
            batch_size=batch_size,
            show_progress_bar=True,
            append=chunk_start != 0
        )
        
        # Final cleanup
        passages.clear()
        gc.collect()
        retriever.clear_gpu_cache()
        
        # Log final memory state
        if torch.cuda.is_available():
            memory_info = retriever.get_gpu_memory_usage()
            logger.info("GPU memory after final chunk: {}", memory_info)

    logger.success("All instance indexes built successfully at '{}'.", save_dir)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli_args = _parse_cli_args()

    wikipedia_indexing(
        wikipedia_path=cli_args.wikipedia_path,
        save_dir=Path(cli_args.save_dir),
        batch_size=cli_args.batch_size,
        embedding_model_name=cli_args.embedding_model_name,
        chunk_size=cli_args.chunk_size,
    )