# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Neural text-chunk retriever backed by FAISS with multi-GPU support.

This utility class supports two indexing workflows:

1. **build_index_from_documents** - pass a pre-chunked list of strings.
2. **build_index_from_long_document** - supply one long document; it will be
   token-split into ≤ *max_tokens*-sized chunks.

Each chunk is embedded with a HuggingFace *SentenceTransformer* model, L2-
normalised, and indexed with FAISS (inner-product ≈ cosine similarity). The
class can persist the index and its accompanying chunk→text mapping to disk,
and later reload them.

Multi-GPU support is provided through manual batch distribution across GPUs
for efficient parallel processing with proper memory management.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Final
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import gc

import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from loguru import logger

__all__: list[str] = ["MultiGPURetriever"]

# ---------------------------------------------------------------------------
# Helper aliases
# ---------------------------------------------------------------------------
_TChunks = list[str]
_TResults = list[tuple[str, float]]

# ---------------------------------------------------------------------------
# MultiGPURetriever implementation
# ---------------------------------------------------------------------------

class MultiGPURetriever:  # pylint: disable=too-many-public-methods
    """A lightweight neural retriever built on *SentenceTransformers* + FAISS with multi-GPU support.

    Pipeline
    --------
    1. **Token-split** long documents with the model's own tokenizer into fixed-size chunks.
    2. **Embed** each chunk using a *SentenceTransformer* model across multiple GPUs.
    3. **Normalise** embeddings (L2) so that inner-product equals cosine sim.
    4. **Index** in FAISS (*IndexIDMap* over *IndexFlatIP*).
    5. **Persist / reload** index alongside its chunk mapping.
    """

    #: Default HuggingFace model for embeddings.
    _DEFAULT_MODEL: Final[str] = "Qwen/Qwen3-Embedding-4B"

    def __init__(
        self,
        embedding_model_name: str = _DEFAULT_MODEL,
        *,
        embedding_dim: int | None = None,
        max_tokens: int = 2048,
        device: str = "cuda",
        use_multi_gpu: bool = True,
    ) -> None:
        """Create a fresh *MultiGPURetriever* instance.

        Parameters
        ----------
        embedding_model_name : str, optional
            HuggingFace model name for both embeddings and tokenisation.
        embedding_dim : int | None, optional
            Explicit embedding dimension. If *None*, it is inferred.
        max_tokens : int, optional
            Chunk size for long-document splitting.
        device : str, optional
            Device specification. Use "cuda" for multi-GPU or specific device like "cuda:0".
        use_multi_gpu : bool, optional
            Whether to use multiple GPUs if available.
        """
        self.embedding_model_name = embedding_model_name
        self.use_multi_gpu = use_multi_gpu and device == "cuda" and torch.cuda.device_count() > 1
        self.num_gpus = torch.cuda.device_count() if self.use_multi_gpu else 1
        
        logger.info(f"Initializing MultiGPURetriever with {self.num_gpus} GPU(s)")
        
        if self.use_multi_gpu:
            # Create separate model instances for each GPU
            self.models = {}
            self.model_lock = threading.Lock()
            
            for gpu_id in range(self.num_gpus):
                device_name = f"cuda:{gpu_id}"
                logger.info(f"Loading model on {device_name}...")
                
                model = SentenceTransformer(
                    embedding_model_name,
                    device=device_name,
                    model_kwargs={
                        "attn_implementation": "flash_attention_2",
                        "torch_dtype": torch.bfloat16,
                        "trust_remote_code": True,
                    },
                    tokenizer_kwargs={"padding_side": "left"}
                )
                self.models[gpu_id] = model
                
            # Use the first model as the primary one for dimension inference and single queries
            self.primary_model = self.models[0]
            logger.info(f"Models loaded on GPUs: {list(self.models.keys())}")
        else:
            # Single GPU or CPU
            device_name = device if device != "cuda" else "cuda:0"
            self.primary_model = SentenceTransformer(
                embedding_model_name,
                device=device_name,
                model_kwargs={
                    "attn_implementation": "flash_attention_2",
                    "torch_dtype": torch.bfloat16,
                    "trust_remote_code": True,
                },
                tokenizer_kwargs={"padding_side": "left"}
            )
            self.models = {0: self.primary_model}

        self.embedding_dim: int = embedding_dim or self.primary_model.get_sentence_embedding_dimension()

        # Tokeniser matched to the same model.
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # FAISS index placeholder - populated after *build_* or *load_*.
        self.index: faiss.IndexIDMap | None = None

        # Mapping from FAISS row-id → text chunk.
        self.texts: _TChunks = []

        # Default chunk length.
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> _TChunks:
        """Token-level split of *text* into ≤ *self.max_tokens* chunks."""
        token_ids: list[int] = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        return [
            self.tokenizer.decode(token_ids[i : i + self.max_tokens], skip_special_tokens=False)
            for i in range(0, len(token_ids), self.max_tokens)
        ]

    def _embed_texts_single_gpu(self, texts: Sequence[str], *, batch_size: int = 32, show_progress_bar: bool = True) -> np.ndarray:
        """Vectorise *texts* using single GPU (fallback method)."""
        embeddings: np.ndarray = self.primary_model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        faiss.normalize_L2(embeddings)  # in-place
        return embeddings.astype("float16", copy=False)

    def _embed_batch_on_gpu(self, gpu_id: int, texts_batch: list[str], batch_size: int) -> np.ndarray:
        """Embed a batch of texts on a specific GPU with proper memory management."""
        model = self.models[gpu_id]
        device = f"cuda:{gpu_id}"
        
        try:
            torch.cuda.empty_cache()
            # Process the batch
            with torch.cuda.device(device):
                embeddings = model.encode(
                    texts_batch,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=True,  # Disable per-GPU progress bars
                    normalize_embeddings=False,  # We'll normalize manually
                )
            
            # Normalize embeddings
            faiss.normalize_L2(embeddings)
            torch.cuda.empty_cache()
            return embeddings.astype("float16", copy=False)
            
        except Exception as e:
            logger.error(f"Error processing batch on GPU {gpu_id}: {e}")
            raise




        # model = self.models[gpu_id]
        # device = f"cuda:{gpu_id}"
        
        # try:
        #     # Clear any existing cache before processing
        #     torch.cuda.empty_cache()
            
        #     # Process the batch in smaller sub-batches to avoid memory spikes
        #     all_embeddings = []
        #     total_batches = len(texts_batch)
            
        #     # Process in smaller chunks to control memory usage
        #     chunk_size = min(batch_size * 4, 128)  # Process at most 128 texts at once per GPU
            
        #     for i in range(0, len(texts_batch), chunk_size):
        #         chunk = texts_batch[i:i + chunk_size]
                
        #         # Encode the chunk
        #         with torch.cuda.device(device):
        #             embeddings = model.encode(
        #                 chunk,
        #                 batch_size=batch_size,
        #                 convert_to_numpy=True,
        #                 show_progress_bar=False,  # Disable per-chunk progress bars
        #                 normalize_embeddings=False,  # We'll normalize manually
        #             )
                
        #         # Convert to numpy immediately and normalize
        #         embeddings = np.array(embeddings, dtype=np.float32)
        #         faiss.normalize_L2(embeddings)
        #         embeddings = embeddings.astype("float16", copy=False)
                
        #         all_embeddings.append(embeddings)
                
        #         # Clear GPU cache after each chunk
        #         torch.cuda.empty_cache()
            
        #     # Concatenate all embeddings
        #     final_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([]).reshape(0, self.embedding_dim).astype('float16')
            
        #     # Final cleanup
        #     del all_embeddings
        #     torch.cuda.empty_cache()
            
        #     return final_embeddings
            
        # except Exception as e:
        #     logger.error(f"Error processing batch on GPU {gpu_id}: {e}")
        #     # Clean up on error
        #     torch.cuda.empty_cache()
        #     raise

    def _embed_texts_multi_gpu(self, texts: Sequence[str], *, batch_size: int = 32, show_progress_bar: bool = True) -> np.ndarray:
        """Vectorise *texts* using multiple GPUs with thread-based parallelization and memory management."""
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim).astype('float16')

        # Clear all GPU caches before starting
        self.clear_gpu_cache()

        texts_list = list(texts)
        total_texts = len(texts_list)
        
        # Calculate optimal batch distribution with memory considerations
        # Reduce per-GPU load for better memory management
        texts_per_gpu = math.ceil(total_texts / self.num_gpus)
        
        logger.info(f"Distributing {total_texts} texts across {self.num_gpus} GPUs")
        logger.info(f"Approximately {texts_per_gpu} texts per GPU")

        # Split texts across GPUs
        gpu_batches = []
        for gpu_id in range(self.num_gpus):
            start_idx = gpu_id * texts_per_gpu
            end_idx = min((gpu_id + 1) * texts_per_gpu, total_texts)
            
            if start_idx < total_texts:
                gpu_batch = texts_list[start_idx:end_idx]
                gpu_batches.append((gpu_id, gpu_batch))

        logger.info(f"Created {len(gpu_batches)} GPU batches")

        # Process batches in parallel using ThreadPoolExecutor
        all_embeddings = [None] * len(gpu_batches)
        
        def process_gpu_batch(args):
            gpu_id, texts_batch = args
            try:
                result = self._embed_batch_on_gpu(gpu_id, texts_batch, batch_size)
                # Force garbage collection after each GPU batch
                gc.collect()
                return gpu_id, result
            except Exception as e:
                logger.error(f"Error in GPU {gpu_id}: {e}")
                torch.cuda.empty_cache()
                raise

        # Use fewer workers to reduce memory pressure
        max_workers = min(self.num_gpus, 8)  # Limit concurrent workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_gpu = {
                executor.submit(process_gpu_batch, (gpu_id, texts_batch)): i
                for i, (gpu_id, texts_batch) in enumerate(gpu_batches)
            }

            completed = 0
            for future in as_completed(future_to_gpu):
                batch_idx = future_to_gpu[future]
                try:
                    gpu_id, embeddings = future.result()
                    all_embeddings[batch_idx] = embeddings
                    completed += 1
                    
                    if show_progress_bar:
                        progress = completed / len(gpu_batches)
                        logger.info(f"GPU processing progress: {progress:.1%} ({completed}/{len(gpu_batches)} batches)")
                        
                    # Clear cache after each completed batch
                    torch.cuda.empty_cache()
                    gc.collect()
                        
                except Exception as e:
                    logger.error(f"GPU batch {batch_idx} generated an exception: {e}")
                    # Clean up all GPUs on error
                    self.clear_gpu_cache()
                    raise

        # Concatenate results in order
        valid_embeddings = [emb for emb in all_embeddings if emb is not None]
        
        if valid_embeddings:
            final_embeddings = np.vstack(valid_embeddings)
            logger.info(f"Successfully created embeddings with shape: {final_embeddings.shape}")
            
            # Clean up intermediate results
            del valid_embeddings, all_embeddings
            self.clear_gpu_cache()
            gc.collect()
            
            return final_embeddings
        else:
            return np.array([]).reshape(0, self.embedding_dim).astype('float16')

    def _embed_texts(self, texts: Sequence[str], *, batch_size: int = 32, show_progress_bar: bool = True) -> np.ndarray:
        """Vectorise *texts* and L2-normalise them (shape: ``(N, d)``)."""
        if self.use_multi_gpu and len(self.models) > 1:
            return self._embed_texts_multi_gpu(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
        else:
            return self._embed_texts_single_gpu(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
    
    # ------------------------------------------------------------------
    # Public – incremental add
    # ------------------------------------------------------------------
    def add_documents(
        self,
        documents: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> None:
        """
        Incrementally embed **and add** pre-chunked `documents` to an
        existing index.

        • If no index exists yet, this falls back to
          `build_index_from_documents`.
        """
        if not documents:
            raise ValueError("`documents` must be a non-empty list of strings.")

        # Cold‑start → reuse the existing builder
        if self.index is None:
            self.build_index_from_documents(
                documents,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
            )
            return

        start_id = len(self.texts)
        self.texts.extend(documents)

        # Clear memory before embedding
        self.clear_gpu_cache()
        gc.collect()

        embeddings = self._embed_texts(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        ids = np.arange(start_id, start_id + len(documents), dtype="int64")
        self.index.add_with_ids(embeddings, ids)

        # Clean up after adding
        del embeddings, ids
        self.clear_gpu_cache()
        gc.collect()

        logger.info(
            f"Added {len(documents)} new documents "
            f"(index now has {len(self.texts)} vectors)"
        )

    # ------------------------------------------------------------------
    # Public - index construction
    # ------------------------------------------------------------------
    def build_index_from_documents(
        self,
        documents: Sequence[str],
        *,
        save_dir: str | Path | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        append: bool = False
    ) -> None:
        """Build a FAISS index from a pre-chunked list of *documents*."""
        if not documents:
            err_msg = "`documents` must be a non-empty list of strings."
            raise ValueError(err_msg)

        if append and self.index is not None:
            self.add_documents(
                documents,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
            )
            if save_dir is not None:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                self.save_index(save_path / "faiss.index",
                               save_path / "mapping.pkl")
            return

        # Clear memory before building new index
        self.clear_gpu_cache()
        gc.collect()

        self.texts = list(documents)
        num_chunks: int = len(self.texts)
        
        logger.info(f"Building index from {num_chunks} documents using {'multi-GPU' if self.use_multi_gpu else 'single-GPU'} processing...")

        # For multi-GPU, we can use larger batch sizes per GPU
        if self.use_multi_gpu:
            # Each GPU will process its portion with the specified batch size
            logger.info(f"Using batch size: {batch_size} per GPU")
        else:
            logger.info(f"Using batch size: {batch_size}")

        # 1. Embed
        embeddings = self._embed_texts(
            self.texts, 
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

        # 2. Index: cosine-sim via inner product on normalised vectors.
        index_flat_ip = faiss.IndexFlatIP(self.embedding_dim)
        index_id_map = faiss.IndexIDMap(index_flat_ip)

        # 3. Add with explicit int64 IDs.
        ids = np.arange(num_chunks, dtype="int64")
        index_id_map.add_with_ids(embeddings, ids)
        self.index = index_id_map
        
        # Clean up embeddings from memory
        del embeddings, ids
        self.clear_gpu_cache()
        gc.collect()
        
        logger.info(f"Successfully built index with {num_chunks} embeddings")

        # 4. Optionally persist.
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            self.save_index(save_path / "faiss.index", save_path / "mapping.pkl")

    def build_index_from_long_document(
        self,
        document: str,
        *,
        save_dir: str | Path | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        append: bool = False
    ) -> None:
        """Split *document* into chunks then delegate to `build_index_from_documents`."""
        if not document:
            err_msg = "`document` must be a non-empty string."
            raise ValueError(err_msg)

        chunks = self._split_text(document)
        logger.info(f"Split document into {len(chunks)} chunks")
        self.build_index_from_documents(
            chunks, 
            save_dir=save_dir, 
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            append=append,
        )

    # ------------------------------------------------------------------
    # Public - persistence helpers
    # ------------------------------------------------------------------

    def save_index(self, index_path: str | Path, mapping_path: str | Path) -> None:
        """Persist the FAISS index and its chunk mapping to disk."""
        if self.index is None:
            err_msg = "No FAISS index to save. Build or load one first."
            raise RuntimeError(err_msg)

        faiss.write_index(self.index, str(index_path))
        with open(mapping_path, "wb") as out_f:
            pickle.dump(self.texts, out_f)

    def load_index(self, index_path: str | Path, mapping_path: str | Path) -> None:
        """Load a previously saved FAISS index + mapping."""
        if not Path(index_path).exists():
            err_msg = f"FAISS index not found: {index_path}"
            raise FileNotFoundError(err_msg)
        if not Path(mapping_path).exists():
            err_msg = f"Mapping file not found: {mapping_path}"
            raise FileNotFoundError(err_msg)

        self.index = faiss.read_index(str(index_path))

        with open(mapping_path, "rb") as in_f:
            loaded_texts = pickle.load(in_f)  # noqa: S301 (trusted source)
        if not isinstance(loaded_texts, list):
            err_msg = "Expected mapping to be a List[str]."
            raise ValueError(err_msg)
        self.texts = loaded_texts.copy()

        # Recover embedding dimension if missing.
        if getattr(self, "embedding_dim", None) is None and hasattr(self.index, "d"):
            self.embedding_dim = (
                getattr(self.index, "d", None)
                or getattr(getattr(self.index, "index", None), "d", None)
                or self.primary_model.get_sentence_embedding_dimension()
            )

    # ------------------------------------------------------------------
    # Public - retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, *, top_k: int = 5) -> _TResults:
        """Return *(chunk, score)* pairs ranked by cosine similarity to *query*.

        Parameters
        ----------
        query : str
            Natural-language query.
        top_k : int, optional
            Number of results to return. If *top_k* is negative (e.g. ``-1``),
            *all* indexed chunks are returned in ranked order.
        """
        if self.index is None or not self.texts:
            err_msg = "Index not built or loaded. Call build_index_* or load_index first."
            raise RuntimeError(err_msg)

        if top_k < 0:  # "return everything" mode
            top_k = len(self.texts)

        # Use the primary model for single query encoding (more efficient)
        query_vec: np.ndarray = self.primary_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)

        distances, indices = self.index.search(query_vec, top_k)

        results: _TResults = []
        for idx, score in zip(indices[0], distances[0], strict=False):
            if 0 <= idx < len(self.texts):
                results.append((self.texts[idx], float(score)))
        return results

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    
    def get_gpu_memory_usage(self) -> dict:
        """Get current GPU memory usage for monitoring."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        memory_info = {}
        for i in range(torch.cuda.device_count()):
            memory_info[f"cuda:{i}"] = {
                "allocated": torch.cuda.memory_allocated(i) / 1024**3,  # GB
                "reserved": torch.cuda.memory_reserved(i) / 1024**3,    # GB
                "max_allocated": torch.cuda.max_memory_allocated(i) / 1024**3  # GB
            }
        return memory_info
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU memory cache for all devices."""
        if torch.cuda.is_available():
            # Clear cache on all GPUs
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            logger.debug("GPU cache cleared on all devices")
    
    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        info = {
            "num_gpus": self.num_gpus,
            "use_multi_gpu": self.use_multi_gpu,
            "embedding_dim": self.embedding_dim,
            "model_name": self.embedding_model_name,
        }
        
        if self.use_multi_gpu:
            info["gpu_models"] = {}
            for gpu_id, model in self.models.items():
                info["gpu_models"][f"cuda:{gpu_id}"] = {
                    "device": str(model.device),
                    "max_seq_length": getattr(model, "max_seq_length", "unknown")
                }
        
        return info