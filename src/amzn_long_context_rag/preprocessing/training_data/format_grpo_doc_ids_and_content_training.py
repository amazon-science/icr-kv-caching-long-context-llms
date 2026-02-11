# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import re
import argparse
from typing import List, Set, Iterable
from datasets import load_dataset

_DOC_REGEX = re.compile(r"\[DOC (\d+)]\s*(.*?)(?=\s*\[DOC \d+]|$)", re.DOTALL)

def _split_docs(training_context: str) -> List[str]:
    """Return list of document strings in the order they appear."""
    return [m.group(2).strip() for m in _DOC_REGEX.finditer(training_context)]


def _extract_gold_idxs(gold_doc_ids: Iterable[str]) -> Set[int]:
    """Extract integer indices from '[DOC k]' strings."""
    idxs: Set[int] = set()
    for tag in gold_doc_ids:
        m = re.match(r"\[DOC (\d+)]", tag)
        if m:
            idxs.add(int(m.group(1)))
    return idxs

def sort_doc_ids_by_index(doc_ids: List[str]) -> List[str]:
    """Sort document IDs by their numeric index."""
    def extract_index(doc_id: str) -> int:
        match = re.match(r"\[DOC (\d+)\]", doc_id)
        return int(match.group(1)) if match else 0
    
    return sorted(doc_ids, key=extract_index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--data_source", required=True)

    args = parser.parse_args()

    data_source = args.data_source
    
    dataset = load_dataset("json", data_files=args.input_file)["train"]

    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    user_template = 'Documents: {context}\n\nInstruction: {instruction}\n\nQuestion: {question}'

    instruction = (
        "You are given a set of documents and a question. Your task is to:\n\n"
        "1. Identify which documents are most relevant to answering the question\n"
        "2. Extract and reproduce the IDs and the full content of those relevant documents\n"
        "3. Provide the final answer based on the relevant documents\n\n"
        "Follow this exact format in your response:\n\n"
        "Relevant documents:\n"
        "[DOC X]\n"
        "<full content of first relevant document>\n\n"
        "[DOC Y]\n"
        "<full content of second relevant document>\n\n"
        "(continue for all relevant documents)\n\n"
        "The answer is: <your final answer>\n\n"
        "Important guidelines:\n"
        "- Only include documents that directly help answer the question\n"
        "- Reproduce the ID and the complete content of each relevant document exactly as provided\n"
        "- The final answer should be concise and directly address the question\n"
        "- Base your answer on information found in the relevant documents you identified\n"
        "- Output your answer after the relevant documents you identified\n"
        "- Do not include documents that are less related or irrelevant"
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            answer_raw = example.pop("answer")
            training_context = example["training_context"]
            gold_doc_ids = sort_doc_ids_by_index(example["gold_doc_ids"])
            promoted_gold_doc_ids = sort_doc_ids_by_index(example["promoted_gold_doc_ids"])
            relevant_gold_doc_ids = sort_doc_ids_by_index(example["relevant_gold_doc_ids"])

            docs = _split_docs(example["training_context"])
            relevant_gold_idxs = _extract_gold_idxs(relevant_gold_doc_ids)
            relevant_gold_docs = [docs[i] for i in sorted(relevant_gold_idxs)]

            user_content = user_template.format(
                context=training_context,
                instruction=instruction,
                question=question_raw,
            )

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": user_content,
                    }
                ],
                "ability": "question answering",
                "reward_model": {"style": "rule", "ground_truth": answer_raw},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "gold_doc_ids": gold_doc_ids,
                    "promoted_gold_doc_ids": promoted_gold_doc_ids,
                    "relevant_gold_doc_ids": relevant_gold_doc_ids,
                    "relevant_gold_docs": relevant_gold_docs,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.output_dir, "train_doc_ids_and_content.parquet"))
    test_dataset.to_parquet(os.path.join(args.output_dir, "test_doc_ids_and_content.parquet"))

    train_dataset.to_json(os.path.join(args.output_dir, "train_doc_ids_and_content.jsonl"))
    test_dataset.to_json(os.path.join(args.output_dir, "test_doc_ids_and_content.jsonl"))