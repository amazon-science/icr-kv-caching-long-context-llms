# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Multi-GPU inference script with optional retrieval-augmented generation (RAG).

This CLI utility supports three *context modes*:

* **full**  - pass the entire context field to the model (default).
* **topk**  - use :class:`Retriever` to fetch the *k* most similar chunks from a
  per-instance FAISS index and pass only those chunks.
* **none**  - perform zero-shot inference without any context at all.
"""

from __future__ import annotations

import os
import re
import sys
import random
import argparse
import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Tuple

import jsonlines
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from transformers import set_seed, AutoConfig
from vllm import EngineArgs, AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

from src.amzn_long_context_rag.data.dataloader import (
    LongBenchV2Loader, InfiniteBenchLoader, LoongLoader, HELMETLoader
)
from src.amzn_long_context_rag.retriever.retriever import (
    Retriever,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

DATASET_TO_LOADER: dict[str, type[LongBenchV2Loader]] = {
    "LongBench-v2": LongBenchV2Loader,
    "InfiniteBench": InfiniteBenchLoader,
    "Loong": LoongLoader,
    "HELMET": HELMETLoader,
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def _set_seeds(seed: int) -> None:
    """Seed every RNG we can reasonably find."""
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
# Context‑trimming helpers
# ---------------------------------------------------------------------------
def _trim_text_end(token_ids: list[int], max_len: int, buffer: int) -> list[int]:
    """Return *token_ids* truncated to *max_len* from the **end**."""
    return token_ids[:max_len - buffer]


def trim_context_end(
    row: Mapping[str, Any],
    tokenizer,  # HF tokenizer
    context_max_tokens: int,
    data_loader,
    buffer: int
) -> bool:
    """Trim ``row["context"]`` in-place **from the end** so that it contains no
    more than ``max_ctx_tokens`` tokens.  The row's *user* message is then
    regenerated via ``data_loader._format_user_content`` so that the prompt
    stays consistent.

    Return ``True`` if trimming actually happened, else ``False``.
    """
    context: str | None = row.get("context")
    if not context:
        return False

    ctx_tokens = tokenizer(context, add_special_tokens=False).input_ids
    if len(ctx_tokens) <= context_max_tokens:
        return False  # no need to trim

    trimmed_ctx = tokenizer.decode(
        _trim_text_end(ctx_tokens, context_max_tokens, buffer),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    row["context"] = trimmed_ctx
    # Regenerate the user message to reflect the new (shorter) context.
    row["messages"][1]["content"] = data_loader._format_user_content(**row)
    return True


# ---------------------------------------------------------------------------
# Helpers to split context into multiple DOCs for fine-tuned models.
# ---------------------------------------------------------------------------
def normalise_whitespace(text: str) -> str:
    "Collapse all internal whitespace; strip leading/trailing."
    return re.sub(r"\s+", " ", text).strip()

def split_context_into_documents(
    row: Mapping[str, Any],
    data_loader,
    passage_len: int=100 
) -> bool:
    """Splits ``row["context"]`` in-place into documents of ``passage_len`` words. 
    The row's *user* message is then regenerated via ``data_loader._format_user_content`` 
    so that the prompt stays consistent.

    Return ``True`` if splitting actually happened, else ``False``.
    """
    context: str | None = row.get("context")
    if not context:
        return False
    
    all_documents = []
    splitted_context = normalise_whitespace(context).split(" ")
    for start in range(0, len(splitted_context), passage_len):
        chunk = splitted_context[start : start + passage_len]
        if chunk:
            all_documents.append(" ".join(chunk))

    documents_ctx = ""
    for i, doc in enumerate(all_documents):
        documents_ctx += f'[DOC {i}]\n{doc}\n\n'
    
    row["context"] = documents_ctx
    row["messages"][1]["content"] = data_loader._format_user_content(**row)
    return True


# ---------------------------------------------------------------------------
# Core inference routine
# ---------------------------------------------------------------------------
async def inference(config_path: str, dotlist: list[str] | None) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run generation on a dataset, optionally using RAG."""

    cfg_path = Path(config_path)
    cfg = load_config(cfg_path, dotlist)

    # ------------------------------------------------------------------
    # Parse config
    # ------------------------------------------------------------------
    model_name: str = cfg["model_name"]
    seed: int = cfg["seed"]
    dcfg: Mapping[str, Any] = cfg["dataset"]
    vllm_params: Mapping[str, Any] = cfg["vllm_params"]
    samp_params_cfg: Mapping[str, Any] = cfg["sampling_params"]
    out_dir = Path(cfg["output_dir"])

    retrieval_cfg: Mapping[str, Any] = cfg.get("retrieval", {})
    retrieval_mode: str = retrieval_cfg.get("mode", "full")
    use_offline_hits: bool = bool(retrieval_cfg.get("use_offline_hits", False))
    offline_hits_path: Path | None = retrieval_cfg.get("offline_hits_path")
    if retrieval_mode not in {"full", "topk", "none"}:
        err_msg = "retrieval.mode must be one of {'full', 'topk', 'none'}"
        raise ValueError(err_msg)

    split_docs: bool = cfg.get("split_docs", False)
    context_max_tokens: int = cfg.get("context_max_tokens")
    continue_final_message: bool = dcfg.get("continue_final_message", False)

    # ------------------------------------------------------------------
    # Output setup
    # ------------------------------------------------------------------
    setting_name = Path(dcfg["prompt_obj"]).stem
    if dcfg["name"] is not None:
        out_subdir = out_dir / dcfg["data_loader"] / dcfg["split"] / dcfg["name"] / f"{model_name.split('/')[-1]}"
    else:
        out_subdir = out_dir / dcfg["data_loader"] / dcfg["split"] / f"{model_name.split('/')[-1]}"
    out_subdir.mkdir(parents=True, exist_ok=True)
    if context_max_tokens:
        out_path = out_subdir / f"{setting_name}_context_max_tokens={context_max_tokens}.jsonl"
    else:
        out_path = out_subdir / f"{setting_name}.jsonl"

    # ------------------------------------------------------------------
    # Log-file sink
    # ------------------------------------------------------------------
    if context_max_tokens:
        log_path = out_subdir / f"{setting_name}_context_max_tokens={context_max_tokens}.log"
    else:
        log_path = out_subdir / f"{setting_name}.log"

    logger.remove()

    # Console sink
    logger.add(
        sink=sys.stderr,
        level="INFO",
        enqueue=True,
        backtrace=True,
        diagnose=True
    )

    # File sink
    logger.add(
        sink=log_path,
        level="DEBUG",  # write *everything* to disk
        enqueue=True,  # safe for multiprocessing / vLLM workers
        backtrace=True,
        diagnose=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
    )

    logger.success("Logging to {}", log_path)
    logger.info("Loaded config from: {}", cfg_path)
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
        continue_final_message=continue_final_message
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True
    )

    total_gpus = torch.cuda.device_count()

    # Choose the highest even tensor_parallel_size that divides num_attention_heads
    tensor_parallel_size = vllm_params.get("tensor_parallel_size", next(
        (n for n in range(total_gpus, 0, -1)
        if model_config.num_attention_heads % n == 0 and n % 2 == 0),
        1  # fallback
    ))

    # Compute matching pipeline parallel size to fully utilize all GPUs
    # pipeline_parallel_size = total_gpus // tensor_parallel_size
    pipeline_parallel_size = vllm_params.get("pipeline_parallel_size", 1)

    logger.info("Loading model (AsyncLLMEngine): {}", model_name)

    rope_scaling_cfg = vllm_params.get("rope_scaling", None)

    engine_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        enable_lora=vllm_params.get("enable_lora", False), ## Added for models trained with LoRA
        gpu_memory_utilization=vllm_params.get("gpu_memory_utilization", 0.9),
        max_model_len=vllm_params.get("max_model_len"),
        max_num_batched_tokens=vllm_params.get("max_num_batched_tokens"),
        enforce_eager=vllm_params.get("enforce_eager", False),
        device=cfg.get("device", "cuda"),
        dtype=torch.bfloat16,
        enable_chunked_prefill=True,  # False for V100‑class GPUs
        trust_remote_code=True
    )
    if rope_scaling_cfg is not None:
        engine_kwargs["rope_scaling"] = rope_scaling_cfg

    engine_args = EngineArgs(**engine_kwargs)
    setattr(engine_args, "disable_log_requests", False)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Load LoRA adapter if enabled
    lora_request = None
    if vllm_params.get("enable_lora", False):
        lora_adapter_path = vllm_params.get("lora_adapter_path", f'{model_name}/lora_adapter')  # Path to LoRA weights
        adapter_name = vllm_params.get("adapter_name", "lora_adapter")
        adapter_id = vllm_params.get("adapter_id", 1)
        
        lora_request = LoRARequest(
            lora_name=adapter_name,
            lora_int_id=adapter_id,
            lora_local_path=lora_adapter_path
        )
        await engine.add_lora(lora_request)

    tokenizer = await engine.get_tokenizer()

    sampling_params = SamplingParams(
        **samp_params_cfg,
        # stop_token_ids=[tokenizer.eos_token_id],
    )

    # ------------------------------------------------------------------
    # Configurable context‑length limit
    # ------------------------------------------------------------------
    if not context_max_tokens:
        context_max_tokens = min(engine.model_config.max_model_len, tokenizer.model_max_length)
    # context_max_tokens: int = int(
    #     cfg.get("context_max_tokens", tokenizer.model_max_length)
    # )
    if context_max_tokens <= 0:
        err_msg = "context_max_tokens must be > 0"
        raise ValueError(err_msg)

    logger.info("Setting context max tokens to {}", context_max_tokens)

    # ------------------------------------------------------------------
    # Retrieval setup (if required)
    # ------------------------------------------------------------------
    retriever: Retriever | None = None
    # When use_offline_hits==True we *skip* Retriever initialisation.
    if retrieval_mode == "topk" and not use_offline_hits:
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

    elif retrieval_mode == "topk" and use_offline_hits:
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

    else:
        index_root = Path(".")  # unused placeholder
        top_k = 0  # unused

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    limit_samples = cfg.get("limit_samples", None)
    total_instances = 0
    truncated_instances = 0
    skept_instances = 0

    with jsonlines.open(out_path, mode="w") as writer:
        for i, row in tqdm(enumerate(data_loader.iterate(), start=1), desc="inference"):
            if limit_samples and i > limit_samples:
                break

            # Derive instance ID (supports either 'id' or '_id').
            instance_id = row.get("id", row.get("_id"))
            total_instances += 1
            
            # ------------------------------------------------------
            # Context handling & retrieval
            # ------------------------------------------------------
            if retrieval_mode == "none":
                row.pop("context", None)
            
            if "context" in row:
                # Write original context length before truncation/retrieval
                original_ctx_length: int = len(tokenizer(row["context"], add_special_tokens=False).input_ids)
                row["original_ctx_length"] = original_ctx_length

            # We retrieve the chunks offline.
            elif retrieval_mode == "topk" and use_offline_hits:
                hits = offline_hits.get(str(instance_id))
                if hits:
                    reduced_context = "\n".join(
                        hits[: top_k] if top_k > 0 else hits
                    )
                    row["context"] = reduced_context
                    row["messages"][1]["content"] = data_loader._format_user_content(**row)
                else:
                    logger.warning(
                        "Offline hits missing for id {}; using full context.",
                        instance_id,
                    )
            # We retrieve the chunks online.
            elif retrieval_mode == "topk" and not use_offline_hits:
                if instance_id is None:
                    logger.warning("Row without 'id'/'_id'; falling back to full context.")
                else:
                    idx_dir = index_root / str(instance_id)
                    idx_path = idx_dir / "faiss.index"
                    map_path = idx_dir / "mapping.pkl"
                    if idx_path.exists() and map_path.exists():
                        try:
                            retriever.load_index(idx_path, map_path)
                            if "question" in row:
                                hits = retriever.retrieve(row["question"], top_k=top_k)
                            elif "input" in row:
                                hits = retriever.retrieve(row["input"], top_k=top_k)
                            else:
                                hits = []
                            reduced_context = "\n".join(chunk for chunk, _ in hits)
                            row["context"] = reduced_context

                            # Regenerate user message with the *new* context.
                            row["messages"][1]["content"] = data_loader._format_user_content(**row)  # type: ignore[attr-defined]
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
            # Else: 'full' – leave context untouched (may still trim below).

            # ------------------------------------------------------
            # Context end‑trimming (both 'full' and 'topk')
            # ------------------------------------------------------
            if retrieval_mode in {"full", "topk"} and row.get("context"):
                if trim_context_end(
                    row=row,
                    tokenizer=tokenizer,
                    context_max_tokens=context_max_tokens,
                    data_loader=data_loader,
                    buffer=samp_params_cfg.get("max_tokens", 512)
                ):
                    logger.warning(
                        "Instance id {} exceeds {} tokens; truncating.", 
                        instance_id, 
                        context_max_tokens
                    )
                    truncated_instances += 1

            # ------------------------------------------------------
            # Global token length guard (skip if still too long)
            # ------------------------------------------------------
            tokens = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=False
            )
            if len(tokens) > tokenizer.model_max_length:
                logger.warning(
                    "Instance id {} - {} tokens still exceed model limit ({}); skipping.",
                    instance_id,
                    len(tokens),
                    tokenizer.model_max_length,
                )
                skept_instances += 1
                continue

            # ---------------------------------------------------------------
            # Context splitting into DOCs to match fine-tuned model behavior.
            # ---------------------------------------------------------------
            if split_docs:
                if split_context_into_documents(
                    row=row,
                    data_loader=data_loader,
                    passage_len=100
                ):
                    logger.info(
                        "Instance id {} successfully splitted into multiple documents.", 
                        instance_id
                    )

            if "context" in row:
                # Write final context length after truncation/retrieval
                final_ctx_length: int = len(tokenizer(row["context"], add_special_tokens=False).input_ids)
                row["final_ctx_length"] = final_ctx_length

            # ------------------------------------------------------
            # Generation
            # ------------------------------------------------------
            try:
                # 1) Build one prompt string from the chat messages
                prompt = tokenizer.apply_chat_template(
                    row["messages"],
                    tokenize=False,  # we want a string
                    add_generation_prompt=not continue_final_message, # add_generation_prompt is not compatible with continue_final_message
                    continue_final_message=continue_final_message # Used to test generation with gold documents.
                )

                # 2) Kick off generation (returns an *async generator*)
                request_id = f"req-{instance_id}"
                stream = engine.generate(
                    prompt=prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    lora_request=lora_request ## Added lora request for models trained with LoRA
                )

                # 3) Collect the streamed tokens
                full_answer = ""
                async for chunk in stream:
                    full_answer = chunk.outputs[0].text

                # 4) Write the result
                payload = row.copy()
                payload.pop("messages", None)
                if retrieval_mode == "none":
                    payload.pop("context", None)
                payload["model_output"] = full_answer
                writer.write(payload)

            except Exception as exc:  # pragma: no cover - log but keep going
                logger.error("Generation failed for id {}: {}", instance_id, exc)

    logger.success(
        "Done. {} total instances, {} contexts truncated, {} skipped.",
        total_instances,
        truncated_instances,
        skept_instances,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # grab config path + overrides
    config_path, dotlist_overrides = parse_cli()
    asyncio.run(inference(config_path, dotlist_overrides))
