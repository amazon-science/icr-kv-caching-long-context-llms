# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import jsonlines
import os
from tqdm import tqdm
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Filter rows from a JSONL file by token length.")
    parser.add_argument("--input_path", help="Path to the JSONL file.")
    parser.add_argument("--model_name", help="Path to the HuggingFace model for tokenization.")
    parser.add_argument("--min_num_tokens", type=int, default=0, help="Minimum number of tokens in each sample.")
    parser.add_argument("--max_num_tokens", type=int, default=32000, help="Maximum number of tokens in each sample.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for tokenization.")
    args = parser.parse_args()

    base, ext = os.path.splitext(args.input_path)
    output_path = f"{base}_tokens_{args.min_num_tokens}_{args.max_num_tokens}{ext}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Read everything once (streaming could also be done in chunks)
    with jsonlines.open(args.input_path) as fin:
        data = list(fin)

    total = len(data)
    kept = 0

    with jsonlines.open(output_path, mode="w") as fout:
        # Process in batches
        for start in tqdm(range(0, total, args.batch_size)):
            batch = data[start:start + args.batch_size]
            texts = [line["training_context"] for line in batch]

            # Tokenize without tensors — just get lengths
            encodings = tokenizer(texts, add_special_tokens=True, truncation=False)
            lengths = [len(ids) for ids in encodings["input_ids"]]

            for line, curr_len in zip(batch, lengths):
                if args.min_num_tokens <= curr_len <= args.max_num_tokens:
                    fout.write(line)
                    kept += 1

    print(f"Original Num. of Instances: {total}")
    print(f"Final Num. of Instances: {kept}")
    print(f"Filtered data written to: {output_path}")

if __name__ == "__main__":
    main()
