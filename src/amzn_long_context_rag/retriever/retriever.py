# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Neural text-chunk retriever backed by FAISS.

This utility class supports two indexing workflows:

1. **build_index_from_documents** - pass a pre-chunked list of strings.
2. **build_index_from_long_document** - supply one long document; it will be
   token-split into ≤ *max_tokens*-sized chunks.

Each chunk is embedded with a HuggingFace *SentenceTransformer* model, L2-
normalised, and indexed with FAISS (inner-product ≈ cosine similarity). The
class can persist the index and its accompanying chunk→text mapping to disk,
and later reload them.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

__all__: list[str] = ["Retriever"]

# ---------------------------------------------------------------------------
# Helper aliases
# ---------------------------------------------------------------------------
_TChunks = list[str]
_TResults = list[tuple[str, float]]

# ---------------------------------------------------------------------------
# Retriever implementation
# ---------------------------------------------------------------------------


class Retriever:  # pylint: disable=too-many-public-methods
    """A lightweight neural retriever built on *SentenceTransformers* + FAISS.

    Pipeline
    --------
    1. **Token-split** long documents with the model's own tokenizer into fixed-size chunks.
    2. **Embed** each chunk using a *SentenceTransformer* model.
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
        device: str = "cuda:7",
    ) -> None:
        """Create a fresh *Retriever* instance.

        Parameters
        ----------
        embedding_model_name : str, optional
            HuggingFace model name for both embeddings and tokenisation.
        embedding_dim : int | None, optional
            Explicit embedding dimension. If *None*, it is inferred.
        max_tokens : int, optional
            Chunk size for long-document splitting.
        """
        self.device = torch.device(device)

        self.embedding_model: SentenceTransformer = SentenceTransformer(
            embedding_model_name,
            device=device,
            model_kwargs={
                "attn_implementation": "flash_attention_2",
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
            },
            tokenizer_kwargs={"padding_side": "left"}
        )
        self.embedding_model._target_device = self.device
        # self.embedding_model: SentenceTransformer = SentenceTransformer(embedding_model_name)
        self.embedding_dim: int = embedding_dim or self.embedding_model.get_sentence_embedding_dimension()

        # 2. Tokeniser matched to the same model.
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, use_fast=True)

        # 3. FAISS index placeholder - populated after *build_* or *load_*.
        self.index: faiss.IndexIDMap | None = None

        # 4. Mapping from FAISS row-id → text chunk.
        self.texts: _TChunks = []

        # 5. Default chunk length.
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

    def _embed_texts(self, texts: Sequence[str], *, batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        """Vectorise *texts* and L2-normalise them (shape: ``(N, d)``)."""
        embeddings: np.ndarray = self.embedding_model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
            device=self.embedding_model.device
        )
        faiss.normalize_L2(embeddings)  # in-place
        return embeddings.astype("float32", copy=False)

    # ------------------------------------------------------------------
    # Public - index construction
    # ------------------------------------------------------------------

    def build_index_from_documents(
        self,
        documents: Sequence[str],
        *,
        save_dir: str | Path | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> None:
        """Build a FAISS index from a pre-chunked list of *documents*."""
        if not documents:
            err_msg = "`documents` must be a non-empty list of strings."
            raise ValueError(err_msg)

        self.texts = list(documents)
        num_chunks: int = len(self.texts)

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
    ) -> None:
        """Split *document* into chunks then delegate to `build_index_from_documents`."""
        if not document:
            err_msg = "`document` must be a non-empty string."
            raise ValueError(err_msg)

        chunks = self._split_text(document)
        self.build_index_from_documents(chunks, save_dir=save_dir, batch_size=batch_size)

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
                or self.embedding_model.get_sentence_embedding_dimension()
            )

    # ------------------------------------------------------------------
    # Public - batched retrieval
    # ------------------------------------------------------------------
    def retrieve_batch(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 5,
        batch_size: int = 32,
    ) -> list[_TResults]:
        """
        Retrieve *top_k* chunks for a list of *queries*.

        Returns
        -------
        list[list[tuple[str, float]]]
            ``results[i]`` contains the (chunk, score) pairs for ``queries[i]``.
        """
        if self.index is None or not self.texts:
            raise RuntimeError(
                "Index not built or loaded. Call build_index_* or load_index first."
            )

        if len(queries) == 0:
            raise ValueError("`queries` must be a non-empty sequence of strings.")

        if top_k < 0:
            top_k = len(self.texts)

        # 1. Embed queries in a single call (fast!)
        q_vecs: np.ndarray = self.embedding_model.encode(
            list(queries),
            batch_size=batch_size,
            convert_to_numpy=True
        )
        faiss.normalize_L2(q_vecs)

        # 2. FAISS batched search (N × d) → (N × top_k)
        distances, indices = self.index.search(q_vecs, top_k)

        # 3. Collect results for every query
        batched_results: list[_TResults] = []
        for row_idx in range(len(queries)):
            one_query: _TResults = []
            for col_idx in range(top_k):
                idx = indices[row_idx, col_idx]
                score = distances[row_idx, col_idx]
                if 0 <= idx < len(self.texts):
                    one_query.append((self.texts[idx], float(score)))
            batched_results.append(one_query)

        return batched_results

    # ------------------------------------------------------------------
    # Public - single‑query retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, *, top_k: int = 5) -> _TResults:
        """Wrapper around *retrieve_batch* for one query."""
        return self.retrieve_batch([query], top_k=top_k)[0]
