# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import sys
import math
import random
import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Tuple, List
from copy import deepcopy
from bisect import bisect_right
import hashlib

import jsonlines
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from transformers import (
    set_seed,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ---------------------------------------------------------------------------
# Dataset → loader mapping
# ---------------------------------------------------------------------------
from src.amzn_long_context_rag.data.dataloader import (
    LongBenchV2Loader,
    InfiniteBenchLoader,
    LoongLoader,
)
from src.amzn_long_context_rag.retriever.retriever import (
    Retriever,
)

DATASET_TO_LOADER: dict[str, type[LongBenchV2Loader]] = {
    "LongBench-v2": LongBenchV2Loader,
    "InfiniteBench": InfiniteBenchLoader,
    "Loong": LoongLoader,
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_cli() -> Tuple[str, list[str]]:
    """Return ``(config_path, dotlist_overrides)``."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "-c",
        "--config_path",
        required=True,
        help="Path to YAML with base hyper-parameters",
    )
    args, dotlist = p.parse_known_args()  # *dotlist* is everything else
    return args.config_path, dotlist


def load_config(config_path: str, dotlist: list[str]) -> DictConfig:
    base = OmegaConf.load(config_path)
    cli = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(base, cli)
    OmegaConf.resolve(cfg)  # resolve ${...} interpolations, if any
    return OmegaConf.to_container(cfg, resolve=True)

# ---------------------------------------------------------------------------
# Context-trimming helpers
# ---------------------------------------------------------------------------
def _trim_text_end(token_ids: list[int], max_len: int, buffer: int) -> list[int]:
    """Return *token_ids* truncated to *max_len* from the **end**."""
    return token_ids[: max_len - buffer]


def trim_context_end(
    row: Mapping[str, Any],
    tokenizer,
    context_max_tokens: int,
    data_loader,
    buffer: int = 0,
) -> bool:
    """Trim ``row["context"]`` in-place **from the end** so that it contains no
    more than ``context_max_tokens`` tokens.  The row's *user* message is then
    regenerated via ``data_loader._format_user_content`` so that the prompt
    stays consistent.

    Return ``True`` if trimming actually happened, else ``False``.
    """
    context: str | None = row.get("context")
    if not context:
        return False

    ctx_tokens = tokenizer(context, add_special_tokens=False).input_ids
    if len(ctx_tokens) <= context_max_tokens:
        return False

    trimmed_ctx = tokenizer.decode(
        _trim_text_end(ctx_tokens, context_max_tokens, buffer),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    row["context"] = trimmed_ctx
    row["messages"][1]["content"] = data_loader._format_user_content(**row)
    return True


def _prepare_prompt(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> tuple[list[int], list[tuple[int, int]], str]:
    """
    Build prompt string + token ids + offset map.

    If the tokenizer has no ``bos_token_id`` we simply *don't* prepend one.
    """
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    enc = tokenizer(
        prompt_str,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    # ---- BOS handling -------------------------------------------------
    bos_id = tokenizer.bos_token_id  # may be None
    if bos_id is None:
        input_ids = enc["input_ids"]
        offsets   = enc["offset_mapping"]
    else:
        input_ids = [bos_id] + enc["input_ids"]
        offsets   = [(-1, -1)] + enc["offset_mapping"]   # dummy span
    # -------------------------------------------------------------------

    return input_ids, offsets, prompt_str


# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------
def _sentence_bounds(
    text: str,
) -> list[tuple[int, int, str]]:
    """
    Return a list with one entry per *sentence* in *text*, each entry being
    ``(start_char, end_char, sentence_text)``.

    Sentence segmentation is deliberately simple and deterministic: we split
    on “.”, “?” or “!” **and** on bare new-lines.  This guarantees that the
    full-context and RAG variants are segmented in the *exact* same way,
    because the retriever glues passages together with “\\n”.
    """
    bounds: list[tuple[int, int, str]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in ".!?" or ch == "\n":
            end = i + (1 if ch in ".!?" else 0)
            sent = text[start:end].strip()
            if sent:
                bounds.append((start, end, sent))
            # advance *start* past this delimiter + any following whitespace
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            start = j
    if start < len(text):                       # last trailing span
        sent = text[start:].strip()
        if sent:
            bounds.append((start, len(text), sent))
    return bounds


def _topk_attended_sentences(
    *,
    model,
    tokenizer,
    messages,
    context_str: str,
    top_k: int,
) -> list[str]:
    """
    Aggregate the *received* attention score for every sentence in
    *context_str* and return the **top-k** most attended sentences.
    """
    # 1) prompt → ids / offsets / str
    input_ids, offsets, prompt_str = _prepare_prompt(tokenizer, messages)

    # 2) attention per model token
    tok_scores = _token_attention_scores(model=model, input_ids=input_ids)

    # 3) locate context span in the full prompt
    ctx_start = prompt_str.find(context_str)
    if ctx_start < 0:
        raise RuntimeError("Failed to locate context in the prompt string.")

    # 4) sentence boundaries *inside* the context
    sent_bounds = _sentence_bounds(context_str)
    sent_ends   = [end for _, end, _ in sent_bounds]

    # 5) accumulate attention per sentence
    from collections import defaultdict

    sent_scores = defaultdict(float)
    for score, (beg, _end) in zip(tok_scores.tolist(), offsets, strict=False):
        if beg < 0 or beg < ctx_start or beg >= ctx_start + len(context_str):
            continue
        rel_pos = beg - ctx_start
        idx = bisect_right(sent_ends, rel_pos)
        if idx < len(sent_bounds):
            _, _, sent_txt = sent_bounds[idx]
            sent_scores[sent_txt] += score

    ranked = sorted(sent_scores.items(), key=lambda kv: kv[1], reverse=True)
    if top_k < 0:
        return [t for t, _ in ranked]
    return [t for t, _ in ranked[: min(top_k, len(ranked))]]


def _token_attention_scores(
    *,
    model: AutoModelForCausalLM,
    input_ids: List[int],
    slice_size: int = 128,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return a 1-D tensor of length *S* containing the cumulative attention **score
    received** by each key token in the *last* transformer block.
    """
    with torch.no_grad():
        out = model(
            torch.tensor([input_ids], device=next(model.parameters()).device),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    last_hidden = out.hidden_states[-2]             # (1, S, d_model)
    seq_len     = last_hidden.size(1)
    last_block  = model.model.layers[-1]

    with torch.no_grad():
        q = last_block.self_attn.q_proj(last_hidden)
        k = last_block.self_attn.k_proj(last_hidden)

        num_q_heads  = model.config.num_attention_heads
        head_dim     = q.size(-1) // num_q_heads
        num_kv_heads = getattr(model.config, "num_key_value_heads", num_q_heads)
        scale        = 1.0 / math.sqrt(head_dim)

        q = q.view(1, seq_len, num_q_heads, head_dim).transpose(1, 2).to(dtype)
        k = k.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2).to(dtype)
        if num_kv_heads != num_q_heads:
            k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

        k_t = k.transpose(-2, -1)
        scores = torch.zeros(seq_len, device=q.device, dtype=dtype)

        for start in range(0, seq_len, slice_size):
            end   = min(start + slice_size, seq_len)
            qi    = q[:, :, start:end, :]
            probs = ((qi @ k_t) * scale).softmax(dim=-1)
            slice_scores = probs.sum(dim=2).mean(dim=1).squeeze(0)  # (S,)
            scores += slice_scores
    return scores  # (S,)


def _short_hash(text: str) -> str:
    """16-hex-char digest - keeps the JSONL output compact & comparable."""
    return hashlib.md5(text.encode("utf8")).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------
def sentence_attn_analysis(config_path: str, dotlist: list[str] | None) -> None:  # noqa: C901
    """Run sentence-attention analysis on a dataset, with and without RAG."""
    cfg_path = Path(config_path)
    cfg = load_config(cfg_path, dotlist or [])

    # ------------------------------------------------------------------
    # Parse config
    # ------------------------------------------------------------------
    model_name: str = cfg["model_name"]
    seed: int = cfg["seed"]
    dcfg: Mapping[str, Any] = cfg["dataset"]
    model_cfg: Mapping[str, Any] = cfg.get("model_config", {})
    retrieval_cfg: Mapping[str, Any] = cfg.get("retrieval", {})
    retrieval_mode: str = retrieval_cfg.get("mode", "full")
    use_offline_hits: bool = bool(retrieval_cfg.get("use_offline_hits", False))
    offline_hits_path: Path | None = retrieval_cfg.get("offline_hits_path")
    if retrieval_mode not in {"full", "topk", "none"}:
        raise ValueError("retrieval.mode must be one of {'full', 'topk', 'none'}")

    # This is to decide when to trim input instances.
    context_max_tokens: int | None = cfg.get("context_max_tokens")

    # This is to decide which instances to keep for the analysis 
    # based on their maximum input length.
    maximum_input_length = cfg.get("maximum_input_length", 64000)

    # The number of sentence for the attention intersection
    chunk_size: int = retrieval_cfg.get("max_tokens", 2048)
    sentence_top_k: int = cfg.get("sentence_top_k", 10)

    # ------------------------------------------------------------------
    # Output setup
    # ------------------------------------------------------------------
    out_dir = Path(cfg["output_dir"])
    config_name = cfg_path.stem
    out_subdir = (
        out_dir / dcfg["data_loader"] / dcfg["split"] / f"{model_name.split('/')[-1]}"
    )
    out_subdir.mkdir(parents=True, exist_ok=True)

    if context_max_tokens:
        out_path = out_subdir / f"sentence_{config_name}_maximum_input_length={maximum_input_length}_sentence_top_k={sentence_top_k}_context_max_tokens={context_max_tokens}.jsonl"
    else:
        out_path = out_subdir / f"sentence_{config_name}_maximum_input_length={maximum_input_length}_sentence_top_k={sentence_top_k}.jsonl"

    # ------------------------------------------------------------------
    # Log sinks
    # ------------------------------------------------------------------
    if context_max_tokens:
        log_path = out_subdir / f"sentence_{config_name}_maximum_input_length={maximum_input_length}_sentence_top_k={sentence_top_k}_context_max_tokens={context_max_tokens}.log"
    else:
        log_path = out_subdir / f"sentence_{config_name}_maximum_input_length={maximum_input_length}_sentence_top_k={sentence_top_k}.log"

    logger.remove()
    logger.add(sys.stderr, level="INFO", enqueue=True, backtrace=True, diagnose=True)
    logger.add(
        log_path,
        level="DEBUG",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
    )
    logger.success("Logging to {}", log_path)
    logger.info(OmegaConf.to_yaml(cfg).rstrip())
    logger.info("Writing outputs to {}", out_path)

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------
    _set_seeds(seed)

    # ------------------------------------------------------------------
    # Data loader
    # ------------------------------------------------------------------
    loader_cls = DATASET_TO_LOADER[dcfg["data_loader"]]
    data_loader = loader_cls(
        dataset_path=dcfg["path"],
        dataset_split=dcfg["split"],
        dataset_name=dcfg["name"],
        prompt_obj=dcfg["prompt_obj"],
    )

    # ------------------------------------------------------------------
    # Tokenizer & model
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(
        model_name,
        **model_cfg,
        output_attentions=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=cfg.get("device_map", "auto"),
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).eval()

    # ------------------------------------------------------------------
    # Context-length limit
    # ------------------------------------------------------------------
    if not context_max_tokens:
        context_max_tokens = tokenizer.model_max_length
    if context_max_tokens <= 0:
        raise ValueError("context_max_tokens must be > 0")
    logger.info("Setting context max tokens to {}", context_max_tokens)

    # ------------------------------------------------------------------
    # Retrieval setup
    # ------------------------------------------------------------------
    retriever: Retriever | None = None
    # When use_offline_hits==True we *skip* Retriever initialisation.
    if not use_offline_hits:
        logger.info("Initialising Retriever for top-k chunk selection …")
        retriever = Retriever(
            embedding_model_name=retrieval_cfg.get(
                "embedding_model_name", "Qwen/Qwen3-Embedding-4B"
            ),
            max_tokens=retrieval_cfg.get("max_tokens", 2048),
        )
        index_root: Path = (
            Path(retrieval_cfg.get("index_dir", "data/indexes"))
            / dcfg["data_loader"]
            / dcfg["split"]
        )
        top_k: int = int(retrieval_cfg.get("top_k", -1))

    else:
        # Load once: {instance_id -> [chunk_1, ..., chunk_k]}
        if offline_hits_path is None:
            raise ValueError(
                "use_offline_hits=True but 'offline_hits_path' not set in config."
            )
        offline_hits_path: Path = (
            Path(offline_hits_path)
            / dcfg["data_loader"] 
            / dcfg["split"] 
            / f'{retrieval_cfg.get("embedding_model_name", "Qwen/Qwen3-Embedding-4B").split("/")[-1]}.jsonl'
        )
        logger.info("Loading offline retrieved passages from {}", offline_hits_path)
        offline_hits: dict[str, list[str]] = {}
        with jsonlines.open(offline_hits_path, "r") as fh:
            for obj in fh:
                offline_hits[str(obj["id"])] = obj["chunks"]
        top_k: int = int(retrieval_cfg.get("top_k", -1))

    # ------------------------------------------------------------------
    # Analysis loop
    # ------------------------------------------------------------------
    limit_samples = cfg.get("limit_samples", None)
    total_instances = 0
    sum_intersections = 0.0
    logger.info("chunk_size = {}, sentence_top_k = {}", chunk_size, sentence_top_k)
    logger.info("maximum_input_length = {}", maximum_input_length)

    with jsonlines.open(out_path, mode="w") as writer:
        for i, row in tqdm(
            enumerate(data_loader.iterate(), start=1), desc="sentence-attention"
        ):
            if limit_samples and i > limit_samples:
                break

            instance_id = row.get("id", row.get("_id"))

            tokens = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=False
            )

            if len(tokens) > maximum_input_length:
                continue

            total_instances += 1

            # --------------------------------------------------
            # FULL context variant (no retrieval)
            # --------------------------------------------------
            row_full = deepcopy(row)

            # Trim if necessary
            trim_context_end(
                row=row_full,
                tokenizer=tokenizer,
                context_max_tokens=tokenizer.model_max_length, ## We avoid truncation in the FC setup.
                data_loader=data_loader,
            )
            tokens_full = tokenizer.apply_chat_template(
                row_full["messages"], tokenize=True, add_generation_prompt=False
            )

            # Skip if still too long for the model
            if len(tokens_full) > tokenizer.model_max_length:
                logger.warning(
                    "Instance id {} - prompt exceeds model limit ({}); skipping.",
                    instance_id,
                    len(tokens_full),
                )
                continue

            topk_full_sents = [
                _short_hash(t) for t in _topk_attended_sentences(
                    model=model,
                    tokenizer=tokenizer,
                    messages=row_full["messages"],
                    context_str=row_full["context"],
                    top_k=sentence_top_k,
                )
            ]

            # -----------------------------------------------------
            # RAG variant (retrieval_mode implicitly set to 'topk')
            # -----------------------------------------------------
            row_rag = deepcopy(row)

            # We retrieve the chunks offline.
            if use_offline_hits:
                hits = offline_hits.get(str(instance_id))
                if hits:
                    reduced_context = "\n".join(
                        hits[: top_k] if top_k > 0 else hits
                    )
                    row_rag["context"] = reduced_context
                    row_rag["messages"][1]["content"] = (
                        data_loader._format_user_content(**row_rag)
                    )
                else:
                    logger.warning(
                        "Offline hits missing for id {}; using full context.",
                        instance_id,
                    )
            else:
                idx_dir = index_root / str(instance_id)
                idx_path = idx_dir / "faiss.index"
                map_path = idx_dir / "mapping.pkl"
                if idx_path.exists() and map_path.exists():
                    try:
                        retriever.load_index(idx_path, map_path)
                        if "question" in row_rag:
                            hits = retriever.retrieve(
                                row_rag["question"], top_k=top_k
                            )
                        elif "input" in row_rag:
                            hits = retriever.retrieve(
                                row_rag["input"], top_k=top_k
                            )
                        else:
                            hits = []
                        reduced_context = "\n".join(chunk for chunk, _ in hits)
                        row_rag["context"] = reduced_context

                        # Regenerate user message with the *new* context.
                        row_rag["messages"][1]["content"] = (
                            data_loader._format_user_content(**row_rag)
                        )
                    except Exception as exc:  # pragma: no cover
                        logger.error(
                            "Retrieval failed for id {} ({}); using full context.",
                            instance_id,
                            exc,
                        )
                else:
                    logger.warning(
                        "Index missing for id '%s'; using full context.", instance_id
                    )
            # else: fall back to full

            # Trim if necessary
            trim_context_end(
                row=row_rag,
                tokenizer=tokenizer,
                context_max_tokens=context_max_tokens, ## We truncate up to context_max_tokens for the RAG setup.
                data_loader=data_loader,
            )

            tokens_rag = tokenizer.apply_chat_template(
                row_rag["messages"], tokenize=True, add_generation_prompt=False
            )

            if len(tokens_rag) > tokenizer.model_max_length:
                logger.warning(
                    "Instance id {} - RAG prompt exceeds model limit ({}); skipping.",
                    instance_id,
                    len(tokens_rag),
                )
                continue

            topk_rag_sents = [
                _short_hash(t) for t in _topk_attended_sentences(
                    model=model,
                    tokenizer=tokenizer,
                    messages=row_rag["messages"],
                    context_str=row_rag["context"],
                    top_k=sentence_top_k,
                )
            ]

            # --------------------------------------------------
            # Intersection & output
            # --------------------------------------------------
            num_sentences = min(len(topk_full_sents), len(topk_rag_sents))
            topk_full_sents = topk_full_sents[: num_sentences]
            topk_rag_sents = topk_rag_sents[: num_sentences]

            intersection = len(set(topk_full_sents) & set(topk_rag_sents)) / num_sentences
            sum_intersections += intersection

            writer.write(
                {
                    "id": instance_id,
                    "topk_full_sents": topk_full_sents,
                    "topk_rag_sents": topk_rag_sents,
                    "intersection": intersection,
                }
            )

    avg_inter = sum_intersections / total_instances if total_instances else 0.0
    logger.success(
        "Done - analysed {} instances | average sentence-overlap = {:.4f}",
        total_instances,
        avg_inter,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg_path, dotlist_overrides = parse_cli()
    sentence_attn_analysis(cfg_path, dotlist_overrides)
