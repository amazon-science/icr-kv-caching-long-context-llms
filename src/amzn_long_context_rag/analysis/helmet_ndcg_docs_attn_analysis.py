# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import re
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
from collections import defaultdict
from transformers import (
    set_seed,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from src.amzn_long_context_rag.data.dataloader import HELMETLoader


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
    args, dotlist = p.parse_known_args()
    return args.config_path, dotlist


def load_config(config_path: str, dotlist: list[str]) -> DictConfig:
    base = OmegaConf.load(config_path)
    cli = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(base, cli)
    OmegaConf.resolve(cfg)
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
    """Trim ``row["context"]`` in-place **from the end**."""
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
    """Build prompt string + token ids + offset map."""
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

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        input_ids = enc["input_ids"]
        offsets   = enc["offset_mapping"]
    else:
        input_ids = [bos_id] + enc["input_ids"]
        offsets   = [(-1, -1)] + enc["offset_mapping"]

    return input_ids, offsets, prompt_str


# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------
def _doc_bounds(text: str) -> list[tuple[int, int, str]]:
    """Return list of (start_char, end_char, doc_text) for each document."""
    bounds: list[tuple[int, int, str]] = []
    doc_pattern = r'\[DOC\s+(\d+|\w+)\]'
    matches = list(re.finditer(doc_pattern, text))
    
    if not matches:
        if text.strip():
            bounds.append((0, len(text), text.strip()))
        return bounds
    
    for i, match in enumerate(matches):
        start_pos = match.start()
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)
        
        doc_content = text[start_pos:end_pos].strip()
        if doc_content:
            bounds.append((start_pos, end_pos, doc_content))
    
    return bounds


def _token_attention_scores_single_layer(
    *,
    model: AutoModelForCausalLM,
    input_ids: List[int],
    layer_idx: int = -1,
    slice_size: int = 128,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return attention scores for a specific layer.
    
    Args:
        layer_idx: Which layer to analyze (-1 for last, 0 for first, etc.)
    """
    input_device = next(model.parameters()).device
    
    with torch.no_grad():
        out = model(
            torch.tensor([input_ids], device=input_device),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    if layer_idx == -1:
        target_hidden = out.hidden_states[-2]
        target_block = model.model.layers[-1]
    else:
        target_hidden = out.hidden_states[layer_idx]
        target_block = model.model.layers[layer_idx]

    seq_len = target_hidden.size(1)

    with torch.no_grad():
        q = target_block.self_attn.q_proj(target_hidden)
        k = target_block.self_attn.k_proj(target_hidden)

        num_q_heads = model.config.num_attention_heads
        head_dim = q.size(-1) // num_q_heads
        num_kv_heads = getattr(model.config, "num_key_value_heads", num_q_heads)
        scale = 1.0 / math.sqrt(head_dim)

        q = q.view(1, seq_len, num_q_heads, head_dim).transpose(1, 2).to(dtype)
        k = k.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2).to(dtype)
        if num_kv_heads != num_q_heads:
            k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

        k_t = k.transpose(-2, -1)
        scores = torch.zeros(seq_len, device=q.device, dtype=dtype)

        for start in range(0, seq_len, slice_size):
            end = min(start + slice_size, seq_len)
            qi = q[:, :, start:end, :]
            probs = ((qi @ k_t) * scale).softmax(dim=-1)
            slice_scores = probs.sum(dim=2).mean(dim=1).squeeze(0)
            scores += slice_scores
    
    return scores.cpu()


def _token_attention_scores_all_layers(
    *,
    model: AutoModelForCausalLM,
    input_ids: List[int],
    slice_size: int = 128,
    dtype: torch.dtype = torch.float32,
) -> dict[int, torch.Tensor]:
    """Return attention scores for ALL layers."""
    input_device = next(model.parameters()).device
    
    with torch.no_grad():
        out = model(
            torch.tensor([input_ids], device=input_device),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    num_layers = len(model.model.layers)
    all_scores = {}
    
    for layer_idx in range(num_layers):
        target_hidden = out.hidden_states[layer_idx]
        target_block = model.model.layers[layer_idx]
        seq_len = target_hidden.size(1)

        with torch.no_grad():
            q = target_block.self_attn.q_proj(target_hidden)
            k = target_block.self_attn.k_proj(target_hidden)

            num_q_heads = model.config.num_attention_heads
            head_dim = q.size(-1) // num_q_heads
            num_kv_heads = getattr(model.config, "num_key_value_heads", num_q_heads)
            scale = 1.0 / math.sqrt(head_dim)

            q = q.view(1, seq_len, num_q_heads, head_dim).transpose(1, 2).to(dtype)
            k = k.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2).to(dtype)
            if num_kv_heads != num_q_heads:
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

            k_t = k.transpose(-2, -1)
            scores = torch.zeros(seq_len, device=q.device, dtype=dtype)

            for start in range(0, seq_len, slice_size):
                end = min(start + slice_size, seq_len)
                qi = q[:, :, start:end, :]
                probs = ((qi @ k_t) * scale).softmax(dim=-1)
                slice_scores = probs.sum(dim=2).mean(dim=1).squeeze(0)
                scores += slice_scores
        
        all_scores[layer_idx] = scores.cpu()

    return all_scores


def _token_attention_scores_averaged(
    *,
    model: AutoModelForCausalLM,
    input_ids: List[int],
    layers: str = "all",
    last_n: int = 4,
    slice_size: int = 128,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return attention scores averaged across multiple layers."""
    all_layer_scores = _token_attention_scores_all_layers(
        model=model, input_ids=input_ids, slice_size=slice_size, dtype=dtype
    )
    
    if layers == "all":
        layer_indices = list(all_layer_scores.keys())
    elif layers == "last_n":
        total_layers = len(all_layer_scores)
        layer_indices = list(range(max(0, total_layers - last_n), total_layers))
    else:
        layer_indices = layers
    
    selected_scores = [all_layer_scores[i] for i in layer_indices]
    stacked_scores = torch.stack(selected_scores)
    averaged_scores = stacked_scores.mean(dim=0)
    
    return averaged_scores


def _rank_docs_by_attention(
    *,
    model,
    tokenizer,
    messages,
    context_str: str,
    attention_strategy: str = "all_layers",
    layer_idx: int = -1,
    last_n: int = 4,
) -> list[tuple[str, float]]:
    """
    Rank ALL documents by attention score.
    
    Returns:
        List of (doc_text, attention_score) tuples, sorted by score (highest first)
    """
    input_ids, offsets, prompt_str = _prepare_prompt(tokenizer, messages)

    # Get attention scores based on strategy
    if attention_strategy == "last_layer":
        tok_scores = _token_attention_scores_single_layer(
            model=model, input_ids=input_ids, layer_idx=-1
        )
    elif attention_strategy == "specific_layer":
        tok_scores = _token_attention_scores_single_layer(
            model=model, input_ids=input_ids, layer_idx=layer_idx
        )
    elif attention_strategy == "all_layers":
        tok_scores = _token_attention_scores_averaged(
            model=model, input_ids=input_ids, layers="all"
        )
    elif attention_strategy == "last_n":
        tok_scores = _token_attention_scores_averaged(
            model=model, input_ids=input_ids, layers="last_n", last_n=last_n
        )
    else:
        raise ValueError(f"Unknown attention_strategy: {attention_strategy}")

    # Locate context span in the full prompt
    ctx_start = prompt_str.find(context_str)
    if ctx_start < 0:
        raise RuntimeError("Failed to locate context in the prompt string.")

    # Document boundaries inside the context
    doc_bounds = _doc_bounds(context_str)
    doc_ends = [end for _, end, _ in doc_bounds]

    # Accumulate attention per document
    doc_scores = defaultdict(float)
    
    for score, (beg, _end) in zip(tok_scores.tolist(), offsets, strict=False):
        if beg < 0 or beg < ctx_start or beg >= ctx_start + len(context_str):
            continue
        
        rel_pos = beg - ctx_start
        idx = bisect_right(doc_ends, rel_pos)
        
        if idx < len(doc_bounds):
            _, _, doc_text = doc_bounds[idx]
            doc_scores[doc_text] += score

    # Sort by attention score (highest first)
    ranked = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked


def extract_doc_id(doc_text: str) -> str:
    """Extract document ID from document text."""
    match = re.match(r"\[DOC\s+[^\]]+?\]", doc_text.strip())
    return match.group(0) if match else ""


def build_gold_ranking(ctxs: list[dict]) -> list[tuple[str, float]]:
    """
    Build gold ranking from ctxs field.
    
    Documents with 'psg_id' (gold docs) are ranked highest.
    Other documents are ranked by their 'score' field.
    
    Returns:
        List of (doc_id, relevance_score) tuples in descending relevance order
    """
    gold_docs = []
    scored_docs = []
    
    for i, ctx in enumerate(ctxs):
        # Gold documents have 'psg_id' instead of 'id' and no 'score'
        if 'psg_id' in ctx and ctx["psg_id"] is not None:
            doc_id = f'[DOC {i}]'
            # Assign highest score to gold docs
            gold_docs.append((doc_id, 1.0))
        elif 'id' in ctx and 'score' in ctx and ctx["id"] is not None:
            doc_id = f'[DOC {i}]'
            score = float(ctx['score'])
            scored_docs.append((doc_id, score))
    
    # Sort scored docs by score (highest first)
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Gold docs come first, then scored docs
    return gold_docs + scored_docs


def calculate_ndcg(
    attention_ranking: list[tuple[str, float]],
    gold_ranking: list[tuple[str, float]],
    k: int = 10
) -> dict[str, float]:
    """
    Calculate NDCG@k and related ranking metrics.
    
    Args:
        attention_ranking: List of (doc_id, attn_score) from model attention
        gold_ranking: List of (doc_id, relevance) from ground truth
        k: Cutoff for NDCG@k calculation
    
    Returns:
        Dict with NDCG@k and other ranking metrics
    """
    # Create relevance map from gold ranking
    relevance_map = {doc_id: relevance for doc_id, relevance in gold_ranking}
    
    # Get doc IDs from attention ranking (in order)
    attention_doc_ids = [doc_id for doc_id, _ in attention_ranking]
    
    # Calculate DCG@k for attention ranking
    dcg = 0.0
    for i, doc_id in enumerate(attention_doc_ids[:k]):
        relevance = relevance_map.get(doc_id, 0.0)
        dcg += relevance / math.log2(i + 2)  # i+2 because rank starts at 1
    
    # Calculate IDCG@k (ideal DCG with perfect ranking)
    gold_doc_ids = [doc_id for doc_id, _ in gold_ranking]
    idcg = 0.0
    for i, doc_id in enumerate(gold_doc_ids[:k]):
        relevance = relevance_map.get(doc_id, 0.0)
        idcg += relevance / math.log2(i + 2)
    
    # NDCG@k
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    # Additional metrics
    # TODO: fix precision@k and recall@k. These two metrics are wrong. Use topk_docs_attn_analysis.py instead.
    # Precision@k: fraction of top-k that are gold docs
    gold_doc_ids_set = set(gold_doc_ids)
    top_k_attention = set(attention_doc_ids[:k])
    precision_at_k = len(top_k_attention & gold_doc_ids_set) / k if k > 0 else 0.0
    
    # Recall@k: fraction of gold docs in top-k
    recall_at_k = len(top_k_attention & gold_doc_ids_set) / len(gold_doc_ids_set) if gold_doc_ids_set else 0.0
    
    # Mean Reciprocal Rank (MRR): position of first gold doc
    mrr = 0.0
    for i, doc_id in enumerate(attention_doc_ids):
        if doc_id in gold_doc_ids_set:
            mrr = 1.0 / (i + 1)
            break
    
    return {
        "ndcg@k": ndcg,
        "precision@k": precision_at_k,
        "recall@k": recall_at_k,
        "mrr": mrr,
        "dcg": dcg,
        "idcg": idcg,
    }


# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------
def docs_attn_ranking_analysis(config_path: str, dotlist: list[str] | None) -> None:
    """Run doc-attention ranking analysis on a dataset."""
    cfg_path = Path(config_path)
    cfg = load_config(cfg_path, dotlist or [])

    # Parse config
    model_name: str = cfg["model_name"]
    seed: int = cfg["seed"]
    dcfg: Mapping[str, Any] = cfg["dataset"]
    model_cfg: Mapping[str, Any] = cfg.get("model_config", {})

    context_max_tokens: int | None = cfg.get("context_max_tokens")
    attention_strategy: str = cfg.get("attention_strategy", "all_layers")
    layer_idx: int = cfg.get("layer_idx", -1)
    last_n: int = cfg.get("last_n", 4)
    ndcg_k: int = cfg.get("ndcg_k", 10)

    # Output setup
    out_dir = Path(cfg["output_dir"])
    setting_name = Path(dcfg["prompt_obj"]).stem
    
    if dcfg["name"] is not None:
        out_subdir = out_dir / dcfg["data_loader"] / dcfg["split"] / dcfg["name"] / f"{model_name.split('/')[-1]}"
    else:
        out_subdir = out_dir / dcfg["data_loader"] / dcfg["split"] / f"{model_name.split('/')[-1]}"
    out_subdir.mkdir(parents=True, exist_ok=True)

    # Create file names based on attention strategy
    if attention_strategy == "last_n":
        strategy_suffix = f"strategy={attention_strategy}_n={last_n}"
    else:
        strategy_suffix = f"strategy={attention_strategy}"

    if context_max_tokens:
        out_path = out_subdir / f"{setting_name}_ranking_ndcg@{ndcg_k}_{strategy_suffix}_context_max_tokens={context_max_tokens}.jsonl"
        log_path = out_subdir / f"{setting_name}_ranking_ndcg@{ndcg_k}_{strategy_suffix}_context_max_tokens={context_max_tokens}.log"
    else:
        out_path = out_subdir / f"{setting_name}_ranking_ndcg@{ndcg_k}_{strategy_suffix}.jsonl"
        log_path = out_subdir / f"{setting_name}_ranking_ndcg@{ndcg_k}_{strategy_suffix}.log"

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

    # Determinism
    _set_seeds(seed)

    # Data loader
    data_loader = HELMETLoader(
        dataset_path=dcfg["path"],
        dataset_split=dcfg["split"],
        dataset_name=dcfg["name"],
        prompt_obj=dcfg["prompt_obj"],
    )

    # Tokenizer & model
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

    # Context-length limit
    if not context_max_tokens:
        context_max_tokens = tokenizer.model_max_length
    if context_max_tokens <= 0:
        raise ValueError("context_max_tokens must be > 0")
    logger.info("Setting context max tokens to {}", context_max_tokens)

    # Analysis loop
    limit_samples = cfg.get("limit_samples", None)
    total_instances = 0
    sum_metrics = {
        "ndcg@k": 0.0,
        "precision@k": 0.0,
        "recall@k": 0.0,
        "mrr": 0.0,
    }
    logger.info("Computing NDCG@{}", ndcg_k)

    with jsonlines.open(out_path, mode="w") as writer:
        for i, row in tqdm(
            enumerate(data_loader.iterate(), start=1), desc="doc-ranking"
        ):
            if limit_samples and i > limit_samples:
                break

            instance_id = row.get("id", row.get("_id"))
            
            # Extract gold ranking from ctxs
            ctxs = row.get("ctxs", [])
            if not ctxs:
                logger.warning("Instance {} has no ctxs field; skipping.", instance_id)
                continue
            
            gold_ranking = build_gold_ranking(ctxs)

            total_instances += 1
            
            # Prepare row
            row_full = deepcopy(row)

            # Trim if necessary
            trim_context_end(
                row=row_full,
                tokenizer=tokenizer,
                context_max_tokens=tokenizer.model_max_length,
                data_loader=data_loader,
            )
            tokens_full = tokenizer.apply_chat_template(
                row_full["messages"], tokenize=True, add_generation_prompt=False
            )

            # Skip if still too long
            if len(tokens_full) > tokenizer.model_max_length:
                logger.warning(
                    "Instance id {} - prompt exceeds model limit ({}); skipping.",
                    instance_id,
                    len(tokens_full),
                )
                continue

            # Rank documents by attention
            attention_ranking_raw = _rank_docs_by_attention(
                model=model, 
                tokenizer=tokenizer, 
                messages=row_full["messages"],
                context_str=row_full["context"],
                attention_strategy=attention_strategy,
                layer_idx=layer_idx,
                last_n=last_n
            )
            
            # Convert to (doc_id, score) format
            attention_ranking = [
                (extract_doc_id(doc_text), score) 
                for doc_text, score in attention_ranking_raw
            ]

            # Calculate ranking metrics
            metrics = calculate_ndcg(attention_ranking, gold_ranking, k=ndcg_k)
            
            # Accumulate metrics
            for key in sum_metrics:
                sum_metrics[key] += metrics[key]

            logger.debug(
                "Instance {} - NDCG@{}: {:.4f}, Precision@{}: {:.4f}, Recall@{}: {:.4f}, MRR: {:.4f}",
                instance_id,
                ndcg_k,
                metrics["ndcg@k"],
                ndcg_k,
                metrics["precision@k"],
                ndcg_k,
                metrics["recall@k"],
                metrics["mrr"],
            )

            # Write output
            writer.write(
                {
                    "id": instance_id,
                    "attention_ranking": attention_ranking,
                    "gold_ranking": gold_ranking,
                    "metrics": metrics,
                }
            )

    # Compute averages
    avg_metrics = {key: value / total_instances if total_instances else 0.0 
                   for key, value in sum_metrics.items()}
    
    logger.success(
        "Done - analyzed {} instances | Avg NDCG@{}: {:.4f}, Avg Precision@{}: {:.4f}, Avg Recall@{}: {:.4f}, Avg MRR: {:.4f}",
        total_instances,
        ndcg_k,
        avg_metrics["ndcg@k"],
        ndcg_k,
        avg_metrics["precision@k"],
        ndcg_k,
        avg_metrics["recall@k"],
        avg_metrics["mrr"],
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg_path, dotlist_overrides = parse_cli()
    docs_attn_ranking_analysis(cfg_path, dotlist_overrides)