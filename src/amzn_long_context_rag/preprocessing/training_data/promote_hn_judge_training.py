# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Remove distractor documents that are near-duplicates of any gold document,
re-index documents / gold_doc_ids accordingly, and optionally promote
similar documents to gold status if they contain useful information.
"""
from __future__ import annotations

import re
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

from loguru import logger
import jsonlines

from vllm import LLM, SamplingParams

_DOC_REGEX = re.compile(r"\[DOC (\d+)]", re.DOTALL)
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean training JSONL from near-duplicate docs.")
    parser.add_argument(
        "--input_path", 
        type=str, 
        # required=True,
        default="data/train/hotpot_qa/promoted_clean_train_tokens=[16000-32000].jsonl",
        help="Path to the original JSONL file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        # required=True,
        default="data/train/hotpot_qa/promoted_judge_clean_train_tokens=[16000-32000].jsonl",
        help="Where the cleaned JSONL will be written.",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        help="The name of the judge model.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_training_context(training_context: str) -> list[tuple[int, str]]:
    """
    Parse training_context to extract document ID and content pairs robustly.

    Returns:
        List of (doc_id, content) tuples in order of appearance.
    """
    # Find all [DOC N] markers and capture content until the next marker or end
    marker_re = re.compile(r"\[DOC (\d+)\]\s*")
    matches = list(marker_re.finditer(training_context))

    documents: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        doc_id = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(training_context)
        content = training_context[start:end].strip()
        # include all docs (even if content is empty) — gold docs must not be lost
        documents.append((doc_id, content))

    return documents

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def clean_file(
    input_path: Path,
    output_path: Path,
    judge: str
) -> None:

    llm = LLM(
        model=judge,
        trust_remote_code=True,
        tensor_parallel_size=4,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
        max_model_len=8192
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        n=1,
        max_tokens=8192
    )

    judge_instruction = (
        "Documents: {promoted_documents}\n\nGold Documents: {gold_documents}\n\nQuestion: {question}\n\nAnswer: {answer}\n\n"
        "You are given as input a set of documents, a set of gold relevant documents, a question, and the correct answer. "
        "You task is to generate only the IDs of the documents that can be deemed relevant with respect to the gold documents, the question and the correct answer. "
        "Provide your answer in the following format:\n"
        "Relevant Documents IDs: [DOC i], [DOC j], [DOC k]...\n"
    )

    n_in, n_out = 0, 0
    original_promoted_counts, final_promoted_counts = 0, 0
    with jsonlines.open(input_path, "r") as reader, jsonlines.open(output_path, "w") as writer:
        for example in tqdm(reader):

            n_in += 1
            original_promoted_counts += len(example["promoted_gold_doc_ids"])

            all_documents = parse_training_context(example["training_context"])
            gold_documents = []
            promoted_documents = []
            for i, d in all_documents:
                if f'[DOC {i}]' in example["promoted_gold_doc_ids"] and f'[DOC {i}]' not in example["gold_doc_ids"]:
                    promoted_documents.append((i, d))
                if f'[DOC {i}]' in example["gold_doc_ids"]:
                    gold_documents.append((i, d))

            promoted_documents = "".join([f'[DOC {i}]\n{d}\n\n' for i, d in promoted_documents])
            gold_documents = "".join([f'[DOC {i}]\n{d}\n\n' for i, d in gold_documents])

            user_content = judge_instruction.format(
                promoted_documents=promoted_documents,
                gold_documents=gold_documents,
                question=example["question"],
                answer=example["answer"]
            )

            prompt = [
                {"role": "system", "content": "You are an helpful assistant"},
                {"role": "user", "content": user_content}
            ]

            tokenizer = llm.get_tokenizer()
            prompt = tokenizer.apply_chat_template(
                prompt, 
                tokenize=False, 
                add_generation_prompt=True
            )
                
            out = llm.generate(user_content, sampling_params)[0]
            judge_text = out.outputs[0].text.strip()
            judge_text = judge_text.split("</think>")[1]

            relevant_doc_ids = [m.group(0).strip() for m in _DOC_REGEX.finditer(judge_text)]

            for g_doc_id in example["gold_doc_ids"]:
                if g_doc_id not in relevant_doc_ids:
                    relevant_doc_ids.append(g_doc_id)
            
            example["relevant_gold_doc_ids"] = sorted(relevant_doc_ids)

            final_promoted_counts += len(example["relevant_gold_doc_ids"])

            n_out += 1

            writer.write(example)

    logger.success("Finished! Wrote {} examples ({} processed) to '{}'.", n_out, n_in, output_path)
    logger.info("Average number of gold documents per example (before): {}", original_promoted_counts / n_out)
    logger.info("Average number of gold documents per example (after): {}", final_promoted_counts / n_out)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    judge = args.judge

    logger.info("Input JSONL:  {}", input_path)
    logger.info("Output JSONL: {}", output_path)
    logger.info("Judge: {}", judge)

    clean_file(
        input_path=input_path,
        output_path=output_path,
        judge=judge
    )

if __name__ == "__main__":
    main()