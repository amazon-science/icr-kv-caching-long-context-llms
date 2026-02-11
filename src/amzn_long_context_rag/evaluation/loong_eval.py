# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from pathlib import Path
import re
import json
import jsonlines
import random
import argparse
from typing import Tuple, Any, Mapping

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from vllm import LLM, SamplingParams

PromptObject = Mapping[str, str]


def _load_prompt_obj(path: Path) -> PromptObject:
    with path.expanduser().open(encoding="utf-8") as fh:
        return json.load(fh)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str, dotlist: list[str]) -> DictConfig:
    base = OmegaConf.load(config_path)
    cli = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(base, cli)
    OmegaConf.resolve(cfg)                    # resolve ${…} interpolations
    return OmegaConf.to_container(cfg, resolve=True)  # plain Python dict


def parse_cli() -> Tuple[str, list[str], Path]:
    """
    Return ``(config_path, dotlist_overrides, result_file)``.

    We parse *config_path* with a first parser (everything else becomes
    the dot-list), then we parse ``--file`` with a second minimal parser
    so that the two do not clash.
    """
    # first stage – grab the config + dot-list
    p_cfg = argparse.ArgumentParser(add_help=False)
    p_cfg.add_argument(
        "-c",
        "--config_path",
        required=True,
        type=Path,
        help="Path to YAML with base hyper-parameters",
    )
    cfg_args, dotlist = p_cfg.parse_known_args()

    # second stage – grab the mandatory JSONL result file
    p_file = argparse.ArgumentParser(add_help=False)
    p_file.add_argument(
        "--file",
        required=True,
        type=Path,
        help="JSONL result file with model predictions",
    )
    p_file.add_argument(
        "--maximum_input_length",
        type=int,
        nargs='+',
        help="JSONL result file with model predictions",
    )
    file_args, _ = p_file.parse_known_args()

    return cfg_args.config_path, dotlist, file_args.file, file_args.maximum_input_length


# =========================================================
# Core routine -------------------------------------------------------------
# =========================================================
def llm_as_a_judge(cfg: Mapping[str, Any], input_path: Path, maximum_input_length: int | list) -> None:
    """
    Read *input_path*, build judgement prompts, run them through the model
    specified in *cfg*, and write a JSONL with an extra ``judge_output`` field
    to ``cfg['output_dir']`` (mirroring the sub-directory structure of
    *input_path*).
    """

    set_seed(cfg.get("seed", 42))

    if maximum_input_length and len(maximum_input_length) == 1:
        maximum_input_length = maximum_input_length[0]

    out_subdir = (
        Path(cfg["output_dir"]).expanduser()
        / Path("/".join(input_path.parts[2:-1]))
    )
    out_subdir.mkdir(parents=True, exist_ok=True)
    if maximum_input_length:
        if isinstance(maximum_input_length, int):
            out_path = out_subdir / f"{input_path.stem}_maximum_input_length={maximum_input_length}.jsonl"
            score_path = out_subdir / f"{input_path.stem}_maximum_input_length={maximum_input_length}.json"
        if isinstance(maximum_input_length, list):
            out_path = out_subdir / f"{input_path.stem}_maximum_input_length=[{maximum_input_length[0]},{maximum_input_length[1]}].jsonl"
            score_path = out_subdir / f"{input_path.stem}_maximum_input_length=[{maximum_input_length[0]},{maximum_input_length[1]}].json"      
    else:
        out_path = out_subdir / f"{input_path.stem}.jsonl"
        score_path = out_subdir / f"{input_path.stem}.json"

    prompt_obj = _load_prompt_obj(Path(cfg["prompt_obj"]))
    judge_prompts: list[dict[str, Any]] = []

    logger.info("Loading file from {}", input_path)
    logger.info("Building judge prompts...")

    with jsonlines.open(input_path, "r") as fin:
        for line in tqdm(fin):
            # Skip instances with a context longer than maximum_input_length tokens.
            original_ctx_length = line["original_ctx_length"]

            if isinstance(maximum_input_length, int):
                if original_ctx_length > maximum_input_length:
                    continue
            
            if isinstance(maximum_input_length, list):
                if original_ctx_length < maximum_input_length[0] or original_ctx_length > maximum_input_length[1]:
                    continue

            judge_template = prompt_obj["judge_template"]

            # prune keys that are irrelevant for the judge
            line.pop("docs", None)
            line.pop("context", None)

            doc_type = line["type"]
            question = line["question"]
            instruction = line["instruction"]
            prompt_template = line["prompt_template"]

            if doc_type != "paper":
                prompt_template = prompt_template.replace("{docs}", "")

            question = (
                prompt_template.replace("{question}", question)
                .replace("{instruction}", instruction)
            )

            answer = line["answer"]
            predict = line["model_output"]

            line["prompt"] = judge_template.format(question, answer, predict)
            judge_prompts.append(line)

    logger.info("Loading model: {}", cfg["model_name"])

    model_name = cfg["model_name"]

    # model_config = AutoConfig.from_pretrained(
    #     pretrained_model_name_or_path=model_name,
    #     trust_remote_code=True
    # )

    # total_gpus = torch.cuda.device_count()

    # # Choose the highest even tensor_parallel_size that divides num_attention_heads
    # tensor_parallel_size = next(
    #     (n for n in range(total_gpus, 0, -1)
    #     if model_config.num_attention_heads % n == 0 and n % 2 == 0),
    #     1  # fallback
    # )

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        # tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_size=4,
        dtype=torch.bfloat16,
        **cfg.get("vllm_params", {}),
    )

    sampling_params = SamplingParams(**cfg.get("sampling_params", {}))

    logger.info("Evaluating...")

    scores = 0
    with jsonlines.open(out_path, "w") as fout:
        for item in tqdm(judge_prompts, total=len(judge_prompts)):
            prompt = item["prompt"]
            out = llm.generate([prompt], sampling_params)[0]
            judge_text = out.outputs[0].text.strip()

            # save the raw judge output
            item["judge_output"] = judge_text

            # Extract “Rating: [<num>]” into item["score"]
            m = re.search(r"Rating:\s*\[\[?\s*(\d{1,3})\s*\]\]?", judge_text, re.IGNORECASE)
            item["score"] = int(m.group(1)) if m else None   # None if pattern not found
            scores += float(item["score"])
            fout.write(item)
    
    with open(score_path, "w") as score_fout:
        json.dump(
            {
                "score": scores / len(judge_prompts),
                "instances": len(judge_prompts),
                "maximum_input_length": maximum_input_length
            }, 
            score_fout, indent=2
        )
        logger.info("Score: {}", scores / len(judge_prompts))
        logger.info("Number of instances: {}", len(judge_prompts))
        logger.info("Maximum input length: {}", maximum_input_length)
        logger.info("Score written to: {}", score_path)

    logger.success(f"LLM-as-a-judge results written to: {out_path}")


if __name__ == "__main__":
    config_path, dotlist, result_file, maximum_input_length = parse_cli()
    cfg = load_config(config_path, dotlist)
    llm_as_a_judge(cfg, result_file, maximum_input_length)