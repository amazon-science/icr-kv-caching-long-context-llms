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
from typing import Any, Tuple, List, Dict
from copy import deepcopy

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
    more than ``context_max_tokens`` tokens."""
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

    # BOS handling
    bos_id = tokenizer.bos_token_id  # may be None
    if bos_id is None:
        input_ids = enc["input_ids"]
        offsets   = enc["offset_mapping"]
    else:
        input_ids = [bos_id] + enc["input_ids"]
        offsets   = [(-1, -1)] + enc["offset_mapping"]   # dummy span

    return input_ids, offsets, prompt_str


# ---------------------------------------------------------------------------
# Attention computation functions
# ---------------------------------------------------------------------------
def _token_attention_scores_single_layer(
    *,
    model: AutoModelForCausalLM,
    input_ids: List[int],
    layer_idx: int = -1,  # -1 for last layer, 0 for first, etc.
    slice_size: int = 128,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return attention scores for a specific layer."""
    input_device = next(model.parameters()).device
    
    with torch.no_grad():
        out = model(
            torch.tensor([input_ids], device=input_device),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

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
    
    return scores.cpu()


def _token_attention_scores_all_layers(
    *,
    model: AutoModelForCausalLM,
    input_ids: List[int],
    slice_size: int = 128,
    dtype: torch.dtype = torch.float32,
) -> dict[int, torch.Tensor]:
    """Return attention scores for ALL layers in the model."""
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
        layer_indices = layers  # Assume it's a list of indices
    
    # Since all scores are now on CPU, we can safely stack them
    selected_scores = [all_layer_scores[i] for i in layer_indices]
    stacked_scores = torch.stack(selected_scores)
    averaged_scores = stacked_scores.mean(dim=0)
    
    return averaged_scores


# ---------------------------------------------------------------------------
# Attention sink detection and removal functions
# ---------------------------------------------------------------------------
def _identify_attention_sinks(
    *,
    tok_scores: torch.Tensor,
    offsets: List[Tuple[int, int]], 
    prompt_str: str,
    context_str: str,
    ctx_start: int,
    sink_threshold_pct: float = 5.0,  # Tokens above this % are considered sinks
    max_sink_position: int = 1000,  # Only consider sinks in first N characters
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Identify attention sink tokens that receive disproportionate attention.
    
    Args:
        tok_scores: Attention scores for all tokens
        offsets: Character offset mapping for tokens
        prompt_str: Full prompt string
        context_str: Context portion of prompt
        ctx_start: Start position of context in prompt
        sink_threshold_pct: Minimum attention % to be considered a sink
        max_sink_position: Only consider tokens before this character position as potential sinks
        
    Returns:
        Tuple of (sink_token_indices, sink_info_dict)
    """
    total_attention = tok_scores.sum().item()
    sink_indices = []
    sink_info = {
        "detected_sinks": [],
        "total_sink_attention": 0.0,
        "total_sink_attention_pct": 0.0,
        "num_sinks": 0,
    }
    
    for i, (score, (token_start, token_end)) in enumerate(zip(tok_scores.tolist(), offsets)):
        if token_start < 0:  # Skip special tokens like BOS
            continue
            
        # Only consider tokens in the early part of the prompt as potential sinks
        if token_start > max_sink_position:
            continue
            
        # Only consider tokens before the context as potential sinks
        if token_start >= ctx_start:
            continue
            
        attention_pct = (score / total_attention * 100) if total_attention > 0 else 0.0
        
        if attention_pct >= sink_threshold_pct:
            # Get the token text
            if token_end > token_start:
                token_text = prompt_str[token_start:token_end]
            else:
                token_text = "[unknown]"
                
            sink_indices.append(i)
            sink_info["detected_sinks"].append({
                "token_idx": i,
                "token_text": repr(token_text),  # Use repr to show special characters
                "attention_score": score,
                "attention_pct": attention_pct,
                "char_position": token_start,
            })
            sink_info["total_sink_attention"] += score
    
    sink_info["total_sink_attention_pct"] = (sink_info["total_sink_attention"] / total_attention * 100) if total_attention > 0 else 0.0
    sink_info["num_sinks"] = len(sink_indices)
    
    return sink_indices, sink_info


def _remove_attention_sinks(
    *,
    tok_scores: torch.Tensor,
    sink_indices: List[int],
) -> torch.Tensor:
    """
    Create a new attention tensor with sink tokens removed and remaining scores rescaled.
    
    Args:
        tok_scores: Original attention scores
        sink_indices: Indices of tokens to remove (attention sinks)
        
    Returns:
        Rescaled attention scores with sinks set to 0 and remaining scores normalized
    """
    if not sink_indices:
        return tok_scores  # No sinks to remove
        
    # Create a copy of the scores
    adjusted_scores = tok_scores.clone()
    
    # Set sink token scores to 0
    for idx in sink_indices:
        if idx < len(adjusted_scores):
            adjusted_scores[idx] = 0.0
    
    # Rescale remaining scores to sum to the original total
    original_total = tok_scores.sum().item()
    remaining_total = adjusted_scores.sum().item()
    
    if remaining_total > 0:
        # Rescale so the non-sink tokens sum to the original total
        scale_factor = original_total / remaining_total
        adjusted_scores = adjusted_scores * scale_factor
    
    return adjusted_scores


# ---------------------------------------------------------------------------
# Gold document attention analysis functions
# ---------------------------------------------------------------------------
def _find_gold_doc_spans(
    context_str: str,
    gold_doc_ids: List[str],
    gold_docs: List[str]
) -> List[Tuple[str, int, int, str]]:
    """Find the character spans of gold documents in the context string."""
    gold_spans = []
    
    for doc_id, doc_content in zip(gold_doc_ids, gold_docs):
        # Create the expected format: "[DOC {doc_id}]\n{doc_content}\n\n"
        expected_format = f"{doc_id}\n{doc_content}\n\n"
        
        # Try to find this exact format first
        start_pos = context_str.find(expected_format)
        if start_pos >= 0:
            end_pos = start_pos + len(expected_format)
            gold_spans.append((doc_id, start_pos, end_pos, expected_format.strip()))
            continue
        
        # If exact format not found, try to find the document header
        doc_header_pattern = f"{doc_id}"
        header_start = context_str.find(doc_header_pattern)
        if header_start >= 0:
            # Find the end of this document (start of next [DOC] or end of context)
            next_doc_pattern = r'\[DOC\s+\w+\]'
            next_match = None
            for match in re.finditer(next_doc_pattern, context_str[header_start + len(doc_header_pattern):]):
                next_match = match
                break
            
            if next_match:
                end_pos = header_start + len(doc_header_pattern) + next_match.start()
            else:
                end_pos = len(context_str)
            
            doc_text = context_str[header_start:end_pos].strip()
            gold_spans.append((doc_id, header_start, end_pos, doc_text))
        else:
            logger.warning(f"Could not find gold document with ID {doc_id} in context")
    
    return gold_spans


def _find_all_doc_spans(context_str: str) -> List[Tuple[str, int, int, str]]:
    """
    Find ALL document spans in the context string (not just gold ones).
    
    Returns:
        List of tuples: (doc_id, start_char, end_char, doc_content)
    """
    doc_spans = []
    
    # Pattern to match document headers like "[DOC 1]", "[DOC 2]", etc.
    doc_pattern = r'\[DOC\s+(\d+|\w+)\]'
    
    # Find all document header matches
    matches = list(re.finditer(doc_pattern, context_str))
    
    if not matches:
        # If no document headers found, return the entire context as one span
        if context_str.strip():
            doc_spans.append(("unknown", 0, len(context_str), context_str.strip()))
        return doc_spans
    
    for i, match in enumerate(matches):
        start_pos = match.start()
        doc_id = match.group(1)
        
        # Determine end position (start of next document or end of context)
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(context_str)
        
        # Extract document content (including the [DOC X] header)
        doc_content = context_str[start_pos:end_pos].strip()
        
        if doc_content:
            doc_spans.append((doc_id, start_pos, end_pos, doc_content))
    
    return doc_spans


def _get_top_attended_content(
    *,
    tok_scores: torch.Tensor,
    offsets: List[Tuple[int, int]], 
    prompt_str: str,
    context_str: str,
    ctx_start: int,
    top_k: int = 10,
    min_span_length: int = 20,
) -> Dict[str, Any]:
    """
    Identify the top-k most attended content spans in the input.
    
    Returns both document-level and token-level high attention content.
    """
    # 1. Document-level attention (if documents are present)
    all_doc_spans = _find_all_doc_spans(context_str)
    doc_attention_scores = {}
    
    if all_doc_spans:
        for doc_id, start_char, end_char, doc_text in all_doc_spans:
            doc_attention = 0.0
            token_count = 0
            
            for score, (token_start, token_end) in zip(tok_scores.tolist(), offsets):
                if token_start < 0:  # Skip special tokens
                    continue
                
                # Check if token is within this document span
                if token_start >= ctx_start and token_start < ctx_start + len(context_str):
                    rel_token_start = token_start - ctx_start
                    if start_char <= rel_token_start < end_char:
                        doc_attention += score
                        token_count += 1
            
            if token_count > 0:  # Only include docs that have tokens
                doc_attention_scores[doc_id] = {
                    "total_attention": doc_attention,
                    "avg_attention_per_token": doc_attention / token_count,
                    "token_count": token_count,
                    "doc_text_preview": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text,
                }
    
    # Sort documents by total attention
    top_docs = sorted(
        doc_attention_scores.items(),
        key=lambda x: x[1]["total_attention"],
        reverse=True
    )[:top_k]
    
    # 2. Token-level attention (find individual high-attention tokens)
    token_attention_details = []
    total_attention = tok_scores.sum().item()
    
    for i, (score, (token_start, token_end)) in enumerate(zip(tok_scores.tolist(), offsets)):
        if token_start < 0:  # Skip special tokens
            continue
        
        # Get the token text
        if token_end > token_start:
            token_text = prompt_str[token_start:token_end]
        else:
            token_text = "[unknown]"
        
        # Determine what section this token belongs to
        section = "other"
        if token_start >= ctx_start and token_start < ctx_start + len(context_str):
            section = "context"
        elif token_start < ctx_start:
            section = "prompt"
        
        token_attention_details.append({
            "token_idx": i,
            "token_text": token_text,
            "attention_score": score,
            "relative_attention_pct": (score / total_attention * 100) if total_attention > 0 else 0,
            "section": section,
            "char_start": token_start,
            "char_end": token_end,
        })
    
    # Sort tokens by attention score and get top-k
    top_tokens = sorted(
        token_attention_details,
        key=lambda x: x["attention_score"],
        reverse=True
    )[:top_k]
    
    # 3. Find high-attention contiguous spans
    high_attention_spans = []
    current_span = []
    span_threshold = np.percentile([t["attention_score"] for t in token_attention_details], 90)
    
    for token_info in token_attention_details:
        if token_info["attention_score"] > span_threshold:
            current_span.append(token_info)
        else:
            if len(current_span) >= 3:  # At least 3 consecutive high-attention tokens
                span_text = "".join([t["token_text"] for t in current_span])
                if len(span_text.strip()) >= min_span_length:
                    total_span_attention = sum(t["attention_score"] for t in current_span)
                    high_attention_spans.append({
                        "span_text": span_text.strip(),
                        "total_attention": total_span_attention,
                        "token_count": len(current_span),
                        "avg_attention": total_span_attention / len(current_span),
                        "section": current_span[0]["section"],
                        "char_start": current_span[0]["char_start"],
                        "char_end": current_span[-1]["char_end"],
                    })
            current_span = []
    
    # Handle final span
    if len(current_span) >= 3:
        span_text = "".join([t["token_text"] for t in current_span])
        if len(span_text.strip()) >= min_span_length:
            total_span_attention = sum(t["attention_score"] for t in current_span)
            high_attention_spans.append({
                "span_text": span_text.strip(),
                "total_attention": total_span_attention,
                "token_count": len(current_span),
                "avg_attention": total_span_attention / len(current_span),
                "section": current_span[0]["section"],
                "char_start": current_span[0]["char_start"],
                "char_end": current_span[-1]["char_end"],
            })
    
    # Sort spans by total attention
    top_spans = sorted(high_attention_spans, key=lambda x: x["total_attention"], reverse=True)[:top_k]
    
    # 4. Attention distribution by section
    section_attention = {"prompt": 0.0, "context": 0.0, "other": 0.0}
    section_token_counts = {"prompt": 0, "context": 0, "other": 0}
    
    for token_info in token_attention_details:
        section = token_info["section"]
        section_attention[section] += token_info["attention_score"]
        section_token_counts[section] += 1
    
    section_stats = {}
    for section in section_attention:
        total_attn = section_attention[section]
        token_count = section_token_counts[section]
        section_stats[section] = {
            "total_attention": total_attn,
            "relative_attention_pct": (total_attn / total_attention * 100) if total_attention > 0 else 0,
            "token_count": token_count,
            "avg_attention_per_token": total_attn / token_count if token_count > 0 else 0,
        }
    
    return {
        "top_attended_documents": dict(top_docs),
        "top_attended_tokens": top_tokens,
        "top_attended_spans": top_spans,
        "attention_by_section": section_stats,
        "total_documents_found": len(all_doc_spans),
        "total_input_attention": total_attention,
    }


# ---------------------------------------------------------------------------
# Main attention analysis functions
# ---------------------------------------------------------------------------
def _calculate_attention_for_step(
    *,
    model,
    tokenizer,
    input_ids: List[int],
    context_str: str,
    gold_spans: List[Tuple[str, int, int, str]],
    attention_strategy: str = "all_layers",
    layer_idx: int = -1,
    last_n: int = 4,
    prompt_str: str,
    ctx_start: int,
    analyze_top_attention: bool = True,
    top_k: int = 5,
    remove_attention_sinks: bool = False,
    sink_threshold_pct: float = 5.0,
    max_sink_position: int = 1000,
) -> Dict[str, Any]:
    """Calculate gold document attention and top-attended content for a single generation step."""
    
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
        raise ValueError(f"Unsupported attention_strategy: {attention_strategy}")
    
    # Create offset mapping for current sequence
    current_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    enc = tokenizer(
        current_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    
    # Handle BOS token if present
    if tokenizer.bos_token_id is not None and input_ids[0] == tokenizer.bos_token_id:
        offsets = [(-1, -1)] + enc["offset_mapping"]
    else:
        offsets = enc["offset_mapping"]
    
    # Store original attention scores for analysis
    original_tok_scores = tok_scores.clone()
    sink_info = None
    
    # Handle attention sink removal if requested
    if remove_attention_sinks:
        sink_indices, sink_info = _identify_attention_sinks(
            tok_scores=original_tok_scores,
            offsets=offsets,
            prompt_str=current_text,
            context_str=context_str,
            ctx_start=ctx_start,
            sink_threshold_pct=sink_threshold_pct,
            max_sink_position=max_sink_position,
        )
        
        # Use rescaled attention scores for calculations
        tok_scores = _remove_attention_sinks(
            tok_scores=original_tok_scores,
            sink_indices=sink_indices,
        )
    
    # Calculate attention scores for each gold document using adjusted scores
    total_attention = tok_scores.sum().item()
    gold_doc_attention = {}
    
    for doc_id, start_char, end_char, doc_text in gold_spans:
        # Find tokens that correspond to this document span
        doc_attention = 0.0
        token_count = 0
        
        for i, (score, (token_start, token_end)) in enumerate(zip(tok_scores.tolist(), offsets)):
            if token_start < 0:  # Skip special tokens like BOS
                continue
            
            # Check if this token is within the gold document span
            # Convert absolute positions to context-relative positions
            if token_start >= ctx_start and token_start < ctx_start + len(context_str):
                rel_token_start = token_start - ctx_start
                
                if start_char <= rel_token_start < end_char:
                    doc_attention += score
                    token_count += 1
        
        # Calculate metrics for this document
        relative_attention = (doc_attention / total_attention) * 100 if total_attention > 0 else 0.0
        
        gold_doc_attention[doc_id] = {
            "absolute_attention": doc_attention,
            "relative_attention_pct": relative_attention,
            "token_count": token_count,
        }
    
    # Calculate summary statistics for gold documents
    total_gold_attention = sum(doc_info["absolute_attention"] for doc_info in gold_doc_attention.values())
    total_gold_relative = sum(doc_info["relative_attention_pct"] for doc_info in gold_doc_attention.values())
    
    step_result = {
        "gold_docs_attention": gold_doc_attention,
        "step_summary": {
            "total_gold_docs": len(gold_doc_attention),
            "found_gold_docs": len([d for d in gold_doc_attention.values() if d["token_count"] > 0]),
            "total_gold_attention_absolute": total_gold_attention,
            "total_gold_attention_relative_pct": total_gold_relative,
            "total_input_attention": total_attention,
            "gold_attention_coverage": (total_gold_attention / total_attention * 100) if total_attention > 0 else 0.0,
        }
    }
    
    # Include attention sink information if sink removal was applied
    if remove_attention_sinks and sink_info is not None:
        step_result["attention_sink_info"] = sink_info
    
    # Analyze top-attended content if requested
    if analyze_top_attention:
        try:
            top_attention_analysis = _get_top_attended_content(
                tok_scores=tok_scores,
                offsets=offsets,
                prompt_str=current_text,
                context_str=context_str,
                ctx_start=ctx_start,
                top_k=top_k,
            )
            step_result["top_attention_analysis"] = top_attention_analysis
        except Exception as e:
            logger.debug(f"Failed to analyze top attention: {e}")
            step_result["top_attention_analysis"] = None
    
    return step_result


def calculate_gold_docs_attention_during_generation(
    *,
    model,
    tokenizer,
    messages,
    context_str: str,
    gold_doc_ids: List[str],
    gold_docs: List[str],
    attention_strategy: str = "all_layer",
    layer_idx: int = -1,
    last_n: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    do_sample: bool = False,
    analyze_top_attention: bool = True,
    top_k: int = 5,
    remove_attention_sinks: bool = False,
    sink_threshold_pct: float = 5.0,
    max_sink_position: int = 1000,
) -> dict:
    """
    Calculate cumulative attention scores for gold documents during generation
    and analyze what content gets the highest attention, with optional attention sink removal.
    
    Args:
        remove_attention_sinks: If True, identify and remove attention sinks before analysis
        sink_threshold_pct: Minimum attention % for a token to be considered a sink
        max_sink_position: Only consider tokens before this char position as potential sinks
    
    Returns:
        Dict with attention metrics averaged across all generation steps
    """
    # 1) Build initial prompt
    input_ids, initial_offsets, prompt_str = _prepare_prompt(
        tokenizer, messages, add_generation_prompt=True
    )
    
    # 2) Locate context span in the full prompt
    ctx_start = prompt_str.find(context_str)
    if ctx_start < 0:
        raise RuntimeError("Failed to locate context in the prompt string.")
    
    # 3) Find gold document spans in the context
    gold_spans = _find_gold_doc_spans(context_str, gold_doc_ids, gold_docs)
    
    # 4) Generate tokens step by step and collect attention at each step
    device = next(model.parameters()).device
    current_input_ids = torch.tensor([input_ids], device=device)
    
    step_results = []
    generated_tokens = []
    
    for step in range(max_new_tokens):
        # Calculate attention for current state
        step_attention = _calculate_attention_for_step(
            model=model,
            tokenizer=tokenizer,
            input_ids=current_input_ids[0].tolist(),
            context_str=context_str,
            gold_spans=gold_spans,
            attention_strategy=attention_strategy,
            layer_idx=layer_idx,
            last_n=last_n,
            prompt_str=prompt_str,
            ctx_start=ctx_start,
            analyze_top_attention=analyze_top_attention,
            top_k=top_k,
            remove_attention_sinks=remove_attention_sinks,
            sink_threshold_pct=sink_threshold_pct,
            max_sink_position=max_sink_position,
        )
        
        step_results.append(step_attention)
        
        # Generate next token
        with torch.no_grad():
            outputs = model(
                current_input_ids,
                use_cache=True,
                return_dict=True,
            )
            
            logits = outputs.logits[:, -1, :]  # Get logits for last position
            
            if do_sample:
                # Apply temperature and sample
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1)
            else:
                # Greedy decoding
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Check for EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token_id.item())
            
            # Append new token to sequence
            current_input_ids = torch.cat([current_input_ids, next_token_id], dim=1)
            
            # Check if we're exceeding model's max length
            if current_input_ids.shape[1] >= tokenizer.model_max_length:
                break
    
    # 5) Aggregate attention metrics across all steps
    if not step_results:
        return {"error": "No generation steps completed"}
    
    # Average attention metrics across all generation steps
    num_steps = len(step_results)
    
    # Initialize aggregated metrics for gold documents
    aggregated_gold_attention = {}
    for doc_id in gold_doc_ids:
        aggregated_gold_attention[doc_id] = {
            "doc_id": doc_id,
            "avg_absolute_attention": 0.0,
            "avg_relative_attention_pct": 0.0,
            "avg_token_count": 0.0,
            "steps_present": 0,
        }
    
    # Aggregate step-level summary statistics
    total_coverage = 0.0
    total_found_docs = 0.0
    total_absolute_attention = 0.0
    total_relative_attention = 0.0
    
    # Aggregate top attention analysis if available
    aggregated_top_attention = {
        "most_attended_documents": {},
        "most_attended_sections": {"prompt": 0.0, "context": 0.0, "other": 0.0},
        "top_attended_spans_summary": [],
        "avg_documents_found": 0.0,
    }
    
    section_attention_totals = {"prompt": 0.0, "context": 0.0, "other": 0.0}
    document_attention_aggregator = {}
    
    # Aggregate attention sink information if sink removal is enabled
    sink_aggregation = None
    if remove_attention_sinks:
        sink_aggregation = {
            "total_sinks_detected": 0,
            "avg_sink_attention_pct": 0.0,
            "avg_sinks_per_step": 0.0,
            "most_common_sink_tokens": {},
            "steps_with_sinks": 0,
        }
        sink_token_counter = {}
    
    for step_result in step_results:
        step_summary = step_result["step_summary"]
        total_coverage += step_summary["gold_attention_coverage"]
        total_found_docs += step_summary["found_gold_docs"]
        total_absolute_attention += step_summary["total_gold_attention_absolute"]
        total_relative_attention += step_summary["total_gold_attention_relative_pct"]
        
        # Aggregate per-document metrics for gold documents
        for doc_id, doc_metrics in step_result["gold_docs_attention"].items():
            if doc_id in aggregated_gold_attention:
                agg_doc = aggregated_gold_attention[doc_id]
                agg_doc["avg_absolute_attention"] += doc_metrics["absolute_attention"]
                agg_doc["avg_relative_attention_pct"] += doc_metrics["relative_attention_pct"]
                agg_doc["avg_token_count"] += doc_metrics["token_count"]
                if doc_metrics["token_count"] > 0:
                    agg_doc["steps_present"] += 1
        
        # Aggregate attention sink information
        if remove_attention_sinks and "attention_sink_info" in step_result:
            sink_info = step_result["attention_sink_info"]
            if sink_info["num_sinks"] > 0:
                sink_aggregation["steps_with_sinks"] += 1
                sink_aggregation["total_sinks_detected"] += sink_info["num_sinks"]
                sink_aggregation["avg_sink_attention_pct"] += sink_info["total_sink_attention_pct"]
                
                # Count sink tokens
                for sink_token_info in sink_info["detected_sinks"]:
                    token_text = sink_token_info["token_text"]
                    if token_text not in sink_token_counter:
                        sink_token_counter[token_text] = 0
                    sink_token_counter[token_text] += 1
        
        # Aggregate top attention analysis if available
        if analyze_top_attention and step_result.get("top_attention_analysis"):
            top_analysis = step_result["top_attention_analysis"]
            
            # Aggregate section attention
            for section, stats in top_analysis["attention_by_section"].items():
                section_attention_totals[section] += stats["relative_attention_pct"]
            
            # Aggregate document attention (all documents, not just gold)
            for doc_id, doc_stats in top_analysis["top_attended_documents"].items():
                if doc_id not in document_attention_aggregator:
                    document_attention_aggregator[doc_id] = {
                        "total_attention": 0.0,
                        "appearances": 0,
                        "sample_preview": doc_stats["doc_text_preview"],
                    }
                document_attention_aggregator[doc_id]["total_attention"] += doc_stats["total_attention"]
                document_attention_aggregator[doc_id]["appearances"] += 1
            
            aggregated_top_attention["avg_documents_found"] += top_analysis["total_documents_found"]
    
    # Finalize averages for gold documents
    for doc_id in aggregated_gold_attention:
        agg_doc = aggregated_gold_attention[doc_id]
        agg_doc["avg_absolute_attention"] /= num_steps
        agg_doc["avg_relative_attention_pct"] /= num_steps
        agg_doc["avg_token_count"] /= num_steps
        agg_doc["presence_ratio"] = agg_doc["steps_present"] / num_steps
    
    # Finalize attention sink aggregation
    if remove_attention_sinks and sink_aggregation is not None:
        sink_aggregation["avg_sinks_per_step"] = sink_aggregation["total_sinks_detected"] / num_steps
        sink_aggregation["avg_sink_attention_pct"] = sink_aggregation["avg_sink_attention_pct"] / num_steps
        sink_aggregation["sink_presence_ratio"] = sink_aggregation["steps_with_sinks"] / num_steps
        
        # Get most common sink tokens
        most_common_sinks = sorted(sink_token_counter.items(), key=lambda x: x[1], reverse=True)[:10]
        sink_aggregation["most_common_sink_tokens"] = dict(most_common_sinks)
    
    # Finalize top attention aggregation
    if analyze_top_attention:
        # Average section attention percentages
        for section in section_attention_totals:
            aggregated_top_attention["most_attended_sections"][section] = section_attention_totals[section] / num_steps
        
        # Sort and get most attended documents (across all steps)
        sorted_docs = sorted(
            document_attention_aggregator.items(),
            key=lambda x: x[1]["total_attention"],
            reverse=True
        )[:top_k]
        
        aggregated_top_attention["most_attended_documents"] = {
            doc_id: {
                "avg_total_attention": stats["total_attention"] / num_steps,
                "appearance_ratio": stats["appearances"] / num_steps,
                "sample_preview": stats["sample_preview"],
            }
            for doc_id, stats in sorted_docs
        }
        
        aggregated_top_attention["avg_documents_found"] /= num_steps
    
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    result = {
        "gold_docs_attention": aggregated_gold_attention,
        "generation_summary": {
            "total_generation_steps": num_steps,
            "avg_gold_attention_coverage": total_coverage / num_steps,
            "avg_found_gold_docs": total_found_docs / num_steps,
            "avg_total_gold_attention_absolute": total_absolute_attention / num_steps,
            "avg_total_gold_attention_relative_pct": total_relative_attention / num_steps,
            "generated_tokens_count": len(generated_tokens),
            "generated_text": generated_text,
        },
        "step_by_step_results": step_results,  # Include detailed step results for debugging
    }
    
    if analyze_top_attention:
        result["top_attention_summary"] = aggregated_top_attention
    
    if remove_attention_sinks and sink_aggregation is not None:
        result["attention_sink_summary"] = sink_aggregation
    
    return result


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------
def gold_docs_generation_attn_analysis(config_path: str, dotlist: list[str] | None) -> None:
    """Run gold document attention analysis during generation with top attention tracking and optional attention sink removal."""
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
    
    # Generation parameters
    max_new_tokens: int = cfg.get("max_new_tokens", 512)
    temperature: float = cfg.get("temperature", 0.0)
    do_sample: bool = cfg.get("do_sample", False)
    
    # Top attention analysis parameters
    analyze_top_attention: bool = cfg.get("analyze_top_attention", True)
    top_k: int = cfg.get("top_k", 5)
    
    # Attention sink removal parameters
    remove_attention_sinks: bool = cfg.get("remove_attention_sinks", False)
    sink_threshold_pct: float = cfg.get("sink_threshold_pct", 5.0)
    max_sink_position: int = cfg.get("max_sink_position", 1000)

    # Output setup
    out_dir = Path(cfg["output_dir"])
    setting_name = Path(dcfg["prompt_obj"]).stem
    
    if dcfg["name"] is not None:
        out_subdir = out_dir / dcfg["data_loader"] / dcfg["split"] / dcfg["name"] / f"{model_name.split('/')[-1]}"
    else:
        out_subdir = out_dir / dcfg["data_loader"] / dcfg["split"] / f"{model_name.split('/')[-1]}"
    out_subdir.mkdir(parents=True, exist_ok=True)

    # Create file names based on attention strategy and sink removal
    if attention_strategy == "specific_layer":
        strategy_suffix = f"strategy={attention_strategy}_layer={layer_idx}"
    elif attention_strategy == "last_n":
        strategy_suffix = f"strategy={attention_strategy}_n={last_n}"
    else:
        strategy_suffix = f"strategy={attention_strategy}"

    sink_suffix = f"remove_sinks={remove_attention_sinks}"
    
    if context_max_tokens:
        out_path = out_subdir / f"{setting_name}_cumulative_step_docs_attn_{strategy_suffix}_{sink_suffix}_context_max_tokens={context_max_tokens}.jsonl"
        log_path = out_subdir / f"{setting_name}_cumulative_step_docs_attn_{strategy_suffix}_{sink_suffix}_context_max_tokens={context_max_tokens}.log"
    else:
        out_path = out_subdir / f"{setting_name}_cumulative_step_docs_attn_{strategy_suffix}_{sink_suffix}.jsonl"
        log_path = out_subdir / f"{setting_name}_cumulative_step_docs_attn_{strategy_suffix}_{sink_suffix}.log"

    # Logging setup
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    logger.info("Using attention strategy: {}", attention_strategy)
    logger.info("Generation parameters: max_new_tokens={}, temperature={}, do_sample={}", max_new_tokens, temperature, do_sample)
    logger.info("Top attention analysis: enabled={}, top_k={}", analyze_top_attention, top_k)
    logger.info("Attention sink removal: enabled={}, threshold={}%, max_position={}", remove_attention_sinks, sink_threshold_pct, max_sink_position)
    
    if attention_strategy == "specific_layer":
        logger.info("Using layer: {}", layer_idx)
    elif attention_strategy == "last_n":
        logger.info("Averaging over last {} layers", last_n)

    # Analysis loop
    limit_samples = cfg.get("limit_samples", None)
    total_instances = 0
    total_avg_gold_coverage = 0.0
    total_avg_found_gold_docs = 0.0
    total_gold_docs_count = 0
    
    # Track top attention statistics across instances
    global_top_attended_docs = {}
    global_section_attention = {"prompt": 0.0, "context": 0.0, "other": 0.0}
    instances_with_top_analysis = 0
    
    # Track attention sink statistics
    global_sink_stats = {
        "total_instances_with_sinks": 0,
        "avg_sinks_per_instance": 0.0,
        "avg_sink_attention_pct": 0.0,
        "global_sink_tokens": {},
    }

    with jsonlines.open(out_path, mode="w") as writer:
        for i, row in tqdm(
            enumerate(data_loader.iterate(), start=1), desc="cumulative-docs-step-attention"
        ):
            if limit_samples and i > limit_samples:
                break

            instance_id = row.get("id", row.get("_id"))
            gold_doc_ids = row["gold_doc_ids"]
            gold_docs = row["gold_docs"]
            total_instances += 1
            total_gold_docs_count += len(gold_docs)

            # Prepare row (with potential trimming)
            row_analysis = deepcopy(row)
            trim_context_end(
                row=row_analysis,
                tokenizer=tokenizer,
                context_max_tokens=context_max_tokens - max_new_tokens,  # Leave room for generation
                data_loader=data_loader,
            )

            # Check initial token length
            initial_tokens = tokenizer.apply_chat_template(
                row_analysis["messages"], tokenize=True, add_generation_prompt=True
            )
            if len(initial_tokens) + max_new_tokens > tokenizer.model_max_length:
                logger.warning(
                    "Instance id {} - prompt + generation would exceed model limit; skipping.",
                    instance_id,
                )
                continue

            # Calculate gold document attention during generation
            try:
                attention_results = calculate_gold_docs_attention_during_generation(
                    model=model,
                    tokenizer=tokenizer,
                    messages=row_analysis["messages"],
                    context_str=row_analysis["context"],
                    gold_doc_ids=gold_doc_ids,
                    gold_docs=gold_docs,
                    attention_strategy=attention_strategy,
                    layer_idx=layer_idx,
                    last_n=last_n,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    analyze_top_attention=analyze_top_attention,
                    top_k=top_k,
                    remove_attention_sinks=remove_attention_sinks,
                    sink_threshold_pct=sink_threshold_pct,
                    max_sink_position=max_sink_position,
                )

                if "error" in attention_results:
                    logger.error("Error processing instance {}: {}", instance_id, attention_results["error"])
                    continue

                # Update running statistics
                gen_summary = attention_results["generation_summary"]
                total_avg_gold_coverage += gen_summary["avg_gold_attention_coverage"]
                total_avg_found_gold_docs += gen_summary["avg_found_gold_docs"]
                
                # Track attention sink statistics
                if remove_attention_sinks and "attention_sink_summary" in attention_results:
                    sink_summary = attention_results["attention_sink_summary"]
                    if sink_summary["steps_with_sinks"] > 0:
                        global_sink_stats["total_instances_with_sinks"] += 1
                    global_sink_stats["avg_sinks_per_instance"] += sink_summary["avg_sinks_per_step"]
                    global_sink_stats["avg_sink_attention_pct"] += sink_summary["avg_sink_attention_pct"]
                    
                    # Aggregate sink tokens
                    for token, count in sink_summary["most_common_sink_tokens"].items():
                        if token not in global_sink_stats["global_sink_tokens"]:
                            global_sink_stats["global_sink_tokens"][token] = 0
                        global_sink_stats["global_sink_tokens"][token] += count

                # Aggregate global top attention statistics
                if analyze_top_attention and "top_attention_summary" in attention_results:
                    instances_with_top_analysis += 1
                    top_summary = attention_results["top_attention_summary"]
                    
                    # Aggregate section attention
                    for section, pct in top_summary["most_attended_sections"].items():
                        global_section_attention[section] += pct
                    
                    # Aggregate most attended documents
                    for doc_id, doc_stats in top_summary["most_attended_documents"].items():
                        if doc_id not in global_top_attended_docs:
                            global_top_attended_docs[doc_id] = {
                                "total_avg_attention": 0.0,
                                "instances_seen": 0,
                                "sample_preview": doc_stats["sample_preview"],
                            }
                        global_top_attended_docs[doc_id]["total_avg_attention"] += doc_stats["avg_total_attention"]
                        global_top_attended_docs[doc_id]["instances_seen"] += 1

                # Log instance results with detailed info
                log_msg = "Instance {} - Generated {} tokens, Steps: {}, Avg gold coverage: {:.2f}%"
                log_args = [instance_id, gen_summary["generated_tokens_count"], gen_summary["total_generation_steps"], gen_summary["avg_gold_attention_coverage"]]
                
                if analyze_top_attention and "top_attention_summary" in attention_results:
                    top_summary = attention_results["top_attention_summary"]
                    section_attn = top_summary["most_attended_sections"]
                    log_msg += f", Attn dist: prompt={section_attn['prompt']:.1f}%, context={section_attn['context']:.1f}%, other={section_attn['other']:.1f}%"
                
                if remove_attention_sinks and "attention_sink_summary" in attention_results:
                    sink_summary = attention_results["attention_sink_summary"]
                    log_msg += f", Sinks: {sink_summary['avg_sinks_per_step']:.1f}/step ({sink_summary['avg_sink_attention_pct']:.1f}% attn)"
                
                logger.debug(log_msg, *log_args)

                # Write output
                output_data = {
                    "id": instance_id,
                    "gold_doc_ids": gold_doc_ids,
                    "attention_results": attention_results,
                    "attention_strategy": attention_strategy,
                    "generation_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "do_sample": do_sample,
                    },
                    "analysis_params": {
                        "analyze_top_attention": analyze_top_attention,
                        "top_k": top_k,
                        "remove_attention_sinks": remove_attention_sinks,
                        "sink_threshold_pct": sink_threshold_pct,
                        "max_sink_position": max_sink_position,
                    },
                    "layer_info": {
                        "layer_idx": layer_idx if attention_strategy == "specific_layer" else None,
                        "last_n": last_n if attention_strategy == "last_n" else None,
                    },
                    "initial_tokens_count": len(initial_tokens),
                    "context_length_chars": len(row_analysis["context"]),
                }
                
                writer.write(output_data)

            except Exception as e:
                logger.error("Error processing instance {}: {}", instance_id, str(e))
                import traceback
                logger.error("Traceback: {}", traceback.format_exc())
                continue

    # Final statistics
    avg_coverage = total_avg_gold_coverage / total_instances if total_instances > 0 else 0.0
    avg_found_ratio = total_avg_found_gold_docs / total_gold_docs_count if total_gold_docs_count > 0 else 0.0

    # Global attention sink summary
    if remove_attention_sinks and total_instances > 0:
        global_sink_stats["avg_sinks_per_instance"] /= total_instances
        global_sink_stats["avg_sink_attention_pct"] /= total_instances
        global_sink_stats["sink_presence_ratio"] = global_sink_stats["total_instances_with_sinks"] / total_instances
        
        # Sort global sink tokens
        sorted_sink_tokens = sorted(
            global_sink_stats["global_sink_tokens"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        global_sink_stats["top_global_sink_tokens"] = dict(sorted_sink_tokens)

    # Global top attention summary
    if analyze_top_attention and instances_with_top_analysis > 0:
        # Average section attention across all instances
        avg_section_attention = {
            section: total / instances_with_top_analysis 
            for section, total in global_section_attention.items()
        }
        
        # Get globally most attended documents
        global_top_docs_sorted = sorted(
            global_top_attended_docs.items(),
            key=lambda x: x[1]["total_avg_attention"] / x[1]["instances_seen"],
            reverse=True
        )[:10]
        
        logger.success(
            "Done - Analyzed {} instances | Avg generation-time gold coverage: {:.4f}% | Avg found gold docs: {:.4f}",
            total_instances,
            avg_coverage,
            avg_found_ratio,
        )
        
        logger.info("=== GLOBAL ATTENTION DISTRIBUTION ===")
        logger.info("Average attention by section across all instances:")
        for section, pct in avg_section_attention.items():
            logger.info(f"  {section}: {pct:.2f}%")
        
        logger.info("Most attended documents globally (top 5):")
        for doc_id, stats in global_top_docs_sorted[:5]:
            avg_attn = stats["total_avg_attention"] / stats["instances_seen"]
            instances = stats["instances_seen"]
            preview = stats["sample_preview"][:100] + "..." if len(stats["sample_preview"]) > 100 else stats["sample_preview"]
            logger.info(f"  DOC {doc_id}: {avg_attn:.4f} avg attention (seen in {instances} instances)")
            logger.info(f"    Preview: {preview}")
        
        # Attention sink summary
        if remove_attention_sinks:
            logger.info("=== ATTENTION SINK SUMMARY ===")
            logger.info(f"Instances with attention sinks: {global_sink_stats['total_instances_with_sinks']}/{total_instances} ({global_sink_stats['sink_presence_ratio']*100:.1f}%)")
            logger.info(f"Average sinks per instance: {global_sink_stats['avg_sinks_per_instance']:.2f}")
            logger.info(f"Average sink attention per instance: {global_sink_stats['avg_sink_attention_pct']:.2f}%")
            logger.info("Most common sink tokens:")
            for token, count in sorted_sink_tokens[:5]:
                logger.info(f"  {token}: {count} occurrences")
    else:
        logger.success(
            "Done - Analyzed {} instances | Avg generation-time gold coverage: {:.4f}% | Avg found gold docs: {:.4f}",
            total_instances,
            avg_coverage,
            avg_found_ratio,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg_path, dotlist_overrides = parse_cli()
    gold_docs_generation_attn_analysis(cfg_path, dotlist_overrides)