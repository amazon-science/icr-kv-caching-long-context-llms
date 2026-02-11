# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import argparse
import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--data_source", required=True)

    args = parser.parse_args()

    data_source = args.data_source
    
    dataset = datasets.load_dataset('json', data_files=args.input_file)["train"]
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    user_template = 'Documents: {context}\n\nInstruction: {instruction}\n\nQuestion: {question}.'
    response_template = 'Relevant documents: {gold_doc_ids}\n\nThe answer is: {answer}.'
    instruction = (
        "Given the set of documents above, answer the following question by following these steps:\n\n"
        "1. Identify the relevant documents. Each document is identified by a tag in the form [DOC i], where i is the index of the document in the context.\n"
        "   - If you find relevant documents, output their IDs exactly as shown, separated by commas. Example: [DOC i], [DOC j], [DOC k].\n"
        "   - If no documents are relevant, output only: [DOC -1].\n\n"
        "2. On a new line, provide your answer in the following format:\n"
        "   The answer is: <your answer here>."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            answer_raw = example.pop("answer")
            training_context = example["training_context"]
            gold_doc_ids = example["gold_doc_ids"]
            parsed_gold_doc_ids = ", ".join(gold_doc_ids)

            user_content = user_template.format(
                context=training_context,
                instruction=instruction,
                question=question_raw,
            )

            response_content = response_template.format(
                gold_doc_ids=parsed_gold_doc_ids,
                answer=answer_raw,
            )

            data = {
                "prompt": user_content,
                "response": response_content,
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.output_dir, "test.parquet"))
