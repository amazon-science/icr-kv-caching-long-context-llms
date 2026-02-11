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
        "Given the set of documents above, answer the following question by following these steps:\n\n"
        "1. Extract relevant quotes: Find and extract short quotes or passages (≤30 tokens each) from the documents that help answer the question. "
        "Present each quote in the following format:\n"
        "   Quote 1: \"<exact text from document>\"\n"
        "   Quote 2: \"<exact text from document>\"\n"
        "   (Continue as needed)\n\n"
        "2. List the source documents using this exact format:\n"
        "   Relevant Document IDs: [DOC i], [DOC j], [DOC k]\n"
        "   (Where i, j, k are the indices of documents that contain your selected quotes)\n"
        "   If no documents are relevant, use: Relevant Document IDs: [DOC -1]\n\n"
        "3. Provide your final answer in the following format:\n"
        "   The answer is: <your answer here>\n\n"
        "Important: Keep quotes short (≤30 tokens), select only the most relevant passages, and ensure your document IDs correspond to the documents containing your quotes."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            answer_raw = example.pop("answer")
            training_context = example["training_context"]
            gold_doc_ids = example["gold_doc_ids"]
            promoted_gold_doc_ids = example["promoted_gold_doc_ids"]
            relevant_gold_doc_ids = example["relevant_gold_doc_ids"]

            docs = _split_docs(example["training_context"])
            relevant_gold_idxs = _extract_gold_idxs(example["relevant_gold_doc_ids"])
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

    train_dataset.to_parquet(os.path.join(args.output_dir, "train_quote.parquet"))
    test_dataset.to_parquet(os.path.join(args.output_dir, "test_quote.parquet"))

    train_dataset.to_json(os.path.join(args.output_dir, "train_quote.jsonl"))
    test_dataset.to_json(os.path.join(args.output_dir, "test_quote.jsonl"))