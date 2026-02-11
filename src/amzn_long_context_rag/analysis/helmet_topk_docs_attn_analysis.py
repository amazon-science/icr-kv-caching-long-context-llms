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
def _doc_bounds(
    text: str,
) -> list[tuple[int, int, str]]:
    """
    Return a list with one entry per *document* in *text*, each entry being
    ``(start_char, end_char, doc_text)``.

    Document segmentation is based on the "[DOC X]" pattern where X is the document ID.
    Each document includes the "[DOC X]" header and the content that follows until
    the next "[DOC X]" or end of text.
    """
    bounds: list[tuple[int, int, str]] = []
    
    # Pattern to match document headers like "[DOC 1]", "[DOC 2]", etc.
    doc_pattern = r'\[DOC\s+(\d+|\w+)\]'
    
    # Find all document header matches
    matches = list(re.finditer(doc_pattern, text))
    
    if not matches:
        # If no document headers found, return the entire text as one document
        if text.strip():
            bounds.append((0, len(text), text.strip()))
        return bounds
    
    for i, match in enumerate(matches):
        start_pos = match.start()
        
        # Determine end position (start of next document or end of text)
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)
        
        # Extract document content (including the [DOC X] header)
        doc_content = text[start_pos:end_pos].strip()
        
        if doc_content:
            bounds.append((start_pos, end_pos, doc_content))
    
    return bounds


def _token_attention_scores_single_layer(
    *,
    model: AutoModelForCausalLM,
    input_ids: List[int],
    layer_idx: int = -1,  # -1 for last layer, 0 for first, etc.
    slice_size: int = 128,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return a 1-D tensor of length *S* containing the cumulative attention score
    received by each key token in a specific transformer layer.
    
    Args:
        layer_idx: Which layer to analyze (-1 for last, 0 for first, etc.)
    """
    # Get the device of the input  
    input_device = next(model.parameters()).device
    
    with torch.no_grad():
        out = model(
            torch.tensor([input_ids], device=input_device),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    # Get hidden states from the layer BEFORE the target layer
    # (since we need the input to that layer's attention)
    if layer_idx == -1:
        target_hidden = out.hidden_states[-2]  # Input to last layer
        target_block = model.model.layers[-1]  # Last layer
    else:
        target_hidden = out.hidden_states[layer_idx]  # Input to specified layer
        target_block = model.model.layers[layer_idx]  # Specified layer

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
            slice_scores = probs.sum(dim=2).mean(dim=1).squeeze(0)  # (S,)
            scores += slice_scores
    
    # Return scores on CPU for consistency
    return scores.cpu()  # (S,)


def _token_attention_scores_all_layers(
    *,
    model: AutoModelForCausalLM,
    input_ids: List[int],
    slice_size: int = 128,
    dtype: torch.dtype = torch.float32,
) -> dict[int, torch.Tensor]:
    """
    Return attention scores for ALL layers in the model.
    
    Returns:
        Dict mapping layer_idx -> attention_scores_tensor (all on CPU for consistency)
    """
    # Get the device of the input
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
        # Get input to this layer
        target_hidden = out.hidden_states[layer_idx]  # (1, S, d_model)
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
                slice_scores = probs.sum(dim=2).mean(dim=1).squeeze(0)  # (S,)
                scores += slice_scores
        
        # Move scores to CPU to avoid device conflicts when stacking later
        all_scores[layer_idx] = scores.cpu()

    return all_scores


def _token_attention_scores_averaged(
    *,
    model: AutoModelForCausalLM,
    input_ids: List[int],
    layers: str = "all",  # "all", "last_n", or list of layer indices
    last_n: int = 4,  # if layers="last_n", how many layers to average
    slice_size: int = 128,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return attention scores averaged across multiple layers.
    
    Args:
        layers: "all" for all layers, "last_n" for last N layers, or list of specific indices
        last_n: Number of last layers to average if layers="last_n"
    """
    all_layer_scores = _token_attention_scores_all_layers(
        model=model, input_ids=input_ids, slice_size=slice_size, dtype=dtype
    )
    
    if layers == "all":
        layer_indices = list(all_layer_scores.keys())
    elif layers == "last_n":
        total_layers = len(all_layer_scores)
        layer_indices = list(range(max(0, total_layers - last_n), total_layers))
    else:
        layer_indices = layers  # Assume it's a list of indices
    
    # Since all scores are now on CPU, we can safely stack them
    selected_scores = [all_layer_scores[i] for i in layer_indices]
    stacked_scores = torch.stack(selected_scores)
    averaged_scores = stacked_scores.mean(dim=0)
    
    return averaged_scores

# Updated topk function that can use different layer strategies
def _topk_attended_docs_multilayer(
    *,
    model,
    tokenizer,
    messages,
    context_str: str,
    top_k: int,
    attention_strategy: str = "all_layers",  # "last_layer", "all_layers", "last_n", "specific_layer"
    layer_idx: int = -1,  # For specific_layer strategy
    last_n: int = 4,  # For last_n strategy
) -> list[str]:
    """
    Get top-k attended documents using different attention aggregation strategies.
    
    Args:
        attention_strategy: How to compute attention scores
            - "all_layers": Average across all layers (default)
            - "last_layer": Only use last layer 
            - "last_n": Average across last N layers
            - "specific_layer": Use specific layer index
        layer_idx: Which layer to use for "specific_layer" strategy
        last_n: How many last layers to average for "last_n" strategy
    """
    # 1) prompt → ids / offsets / str
    input_ids, offsets, prompt_str = _prepare_prompt(tokenizer, messages)

    # 2) attention per model token based on strategy
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

    # 3) locate context span in the full prompt
    ctx_start = prompt_str.find(context_str)
    if ctx_start < 0:
        raise RuntimeError("Failed to locate context in the prompt string.")

    # 4) document boundaries *inside* the context
    doc_bounds = _doc_bounds(context_str)
    doc_ends = [end for _, end, _ in doc_bounds]

    # 5) accumulate attention per document
    doc_scores = defaultdict(float)
    
    for score, (beg, _end) in zip(tok_scores.tolist(), offsets, strict=False):
        if beg < 0 or beg < ctx_start or beg >= ctx_start + len(context_str):
            continue
        
        rel_pos = beg - ctx_start
        idx = bisect_right(doc_ends, rel_pos)
        
        if idx < len(doc_bounds):
            _, _, doc_text = doc_bounds[idx]
            doc_scores[doc_text] += score

    # Sort documents by attention score (highest first)
    ranked = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)
    
    if top_k < 0:
        return [doc_text for doc_text, _ in ranked]
    
    return [doc_text for doc_text, _ in ranked[:min(top_k, len(ranked))]]


def calculate_intersection(topk_attn_docs: list[str], docs_attn_target: list[str]) -> dict:
    """
    Calculate intersection metrics between top-k attended documents and gold documents.
    
    Args:
        topk_attn_docs: List of top-k attended documents in format "[DOC X]\n{content}\n\n"
        docs_attn_target: List of gold documents in format "[DOC X]\n{content}\n\n"
    
    Returns:
        Dict with intersection metrics
    """
    def extract_doc_id(doc_text: str) -> str:
        """Extract document ID from document text."""
        match = re.match(r'\[DOC\s+(\d+|\w+)\]', doc_text.strip())
        return match.group(1) if match else ""
    
    # Extract document IDs from both lists
    attn_doc_ids = set()
    for doc in topk_attn_docs:
        doc_id = extract_doc_id(doc)
        if doc_id:
            attn_doc_ids.add(doc_id)
    
    target_doc_ids = set()
    for doc in docs_attn_target:
        doc_id = extract_doc_id(doc)
        if doc_id:
            target_doc_ids.add(doc_id)
    
    # Calculate intersection
    intersection_ids = attn_doc_ids.intersection(target_doc_ids)
    
    # Calculate metrics
    precision = len(intersection_ids) / len(attn_doc_ids) if attn_doc_ids else 0.0
    recall = len(intersection_ids) / len(target_doc_ids) if target_doc_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "intersection_count": len(intersection_ids),
        "intersection_ids": sorted(list(intersection_ids)),
        "attn_doc_ids": sorted(list(attn_doc_ids)),
        "target_doc_ids": sorted(list(target_doc_ids)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------
def docs_attn_analysis(config_path: str, dotlist: list[str] | None) -> None:  # noqa: C901
    """Run doc-attention analysis on a dataset, with and without RAG."""
    cfg_path = Path(config_path)
    cfg = load_config(cfg_path, dotlist or [])

    # ------------------------------------------------------------------
    # Parse config
    # ------------------------------------------------------------------
    model_name: str = cfg["model_name"]
    seed: int = cfg["seed"]
    dcfg: Mapping[str, Any] = cfg["dataset"]
    model_cfg: Mapping[str, Any] = cfg.get("model_config", {})

    # This is to decide when to trim input instances.
    context_max_tokens: int | None = cfg.get("context_max_tokens")
    attention_strategy: str = cfg.get("attention_strategy", "all_layers")
    layer_idx: int = cfg.get("layer_idx", -1)
    last_n: int = cfg.get("last_n", 4)
    docs_top_k: int = cfg.get("docs_top_k", 10)

    # ------------------------------------------------------------------
    # Output setup
    # ------------------------------------------------------------------
    out_dir = Path(cfg["output_dir"])
    setting_name = Path(dcfg["prompt_obj"]).stem
    
    if dcfg["name"] is not None:
        out_subdir = out_dir / dcfg["data_loader"] / dcfg["split"] / dcfg["name"] / f"{model_name.split('/')[-1]}"
    else:
        out_subdir = out_dir / dcfg["data_loader"] / dcfg["split"] / f"{model_name.split('/')[-1]}"
    out_subdir.mkdir(parents=True, exist_ok=True)

    # Create file names based on attention strategy
    if attention_strategy == "specific_layer":
        strategy_suffix = f"strategy={attention_strategy}_layer={layer_idx}"
    elif attention_strategy == "last_n":
        strategy_suffix = f"strategy={attention_strategy}_n={last_n}"
    else:
        strategy_suffix = f"strategy={attention_strategy}"

    if context_max_tokens:
        out_path = out_subdir / f"{setting_name}_docs_top_k={docs_top_k}_{strategy_suffix}_context_max_tokens={context_max_tokens}.jsonl"
        log_path = out_subdir / f"{setting_name}_docs_top_k={docs_top_k}_{strategy_suffix}_context_max_tokens={context_max_tokens}.log"
    else:
        out_path = out_subdir / f"{setting_name}_docs_top_k={docs_top_k}_{strategy_suffix}.jsonl"
        log_path = out_subdir / f"{setting_name}_docs_top_k={docs_top_k}_{strategy_suffix}.log"

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
    data_loader = HELMETLoader(
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
    # Analysis loop
    # ------------------------------------------------------------------
    limit_samples = cfg.get("limit_samples", None)
    total_instances = 0
    sum_intersections = {"precision": 0, "recall": 0, "f1": 0}
    logger.info("Document top k: {}", docs_top_k)

    with jsonlines.open(out_path, mode="w") as writer:
        for i, row in tqdm(
            enumerate(data_loader.iterate(), start=1), desc="doc-attention"
        ):
            if limit_samples and i > limit_samples:
                break

            instance_id = row.get("id", row.get("_id"))

            gold_doc_ids = row["gold_doc_ids"]
            gold_docs = row["gold_docs"]
            docs_attn_target = [f'{id}\n{content}\n\n' for id, content in zip(gold_doc_ids, gold_docs)]

            total_instances += 1
            # --------------------------------------------------
            # FULL context 
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

            # Average across all layers
            topk_attn_docs = _topk_attended_docs_multilayer(
                model=model, 
                tokenizer=tokenizer, 
                messages=row_full["messages"],
                context_str=row_full["context"],
                top_k=docs_top_k,
                attention_strategy=attention_strategy,
                layer_idx=layer_idx,
                last_n=last_n
            )

            # --------------------------------------------------
            # Intersection & output: TODO
            # --------------------------------------------------
            intersection = calculate_intersection(topk_attn_docs, docs_attn_target)
            sum_intersections["precision"] += intersection["precision"]
            sum_intersections["recall"] += intersection["recall"]
            sum_intersections["f1"] += intersection["f1"]

            logger.debug(
                "Instance {} - Found {} intersecting docs out of {} attended, {} target. Precision: {:.4f}. Recall: {:.4f}. F1: {:.4f}",
                instance_id,
                intersection["intersection_count"],
                len(topk_attn_docs),
                len(docs_attn_target),
                intersection["precision"],
                intersection["recall"],
                intersection["f1"]
            )

            # --------------------------------------------------
            # Write output  
            # --------------------------------------------------
            writer.write(
                {
                    "id": instance_id,
                    "topk_attn_docs": topk_attn_docs,
                    "topk_attn_doc_ids": intersection["attn_doc_ids"],
                    "docs_attn_target": docs_attn_target,
                    "target_doc_ids": intersection["target_doc_ids"],
                    "intersection": intersection,
                }
            )

    avg_precision = sum_intersections["precision"] / total_instances if total_instances else 0.0
    avg_recall = sum_intersections["recall"] / total_instances if total_instances else 0.0
    avg_f1 = sum_intersections["f1"] / total_instances if total_instances else 0.0
    logger.success(
        "Done - analysed {} instances | avg_precision: {:.4f}. avg_recall: {:.4f}. avg_f1: {:.4f}.",
        total_instances,
        avg_precision,
        avg_recall,
        avg_f1
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg_path, dotlist_overrides = parse_cli()
    docs_attn_analysis(cfg_path, dotlist_overrides)
