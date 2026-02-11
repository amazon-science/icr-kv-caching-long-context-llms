# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import random
import jsonlines
import os

def main():
    parser = argparse.ArgumentParser(description="Randomly sample rows from a JSONL file.")
    parser.add_argument("--input_path", help="Path to the JSONL file.")
    parser.add_argument("--num_samples", type=int, help="Number of rows to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)

    with jsonlines.open(args.input_path) as reader:
        data = list(reader)

    if args.num_samples > len(data):
        raise ValueError(f"Requested {args.num_samples} samples, but file only has {len(data)} rows.")

    sampled_data = random.sample(data, args.num_samples)

    base, ext = os.path.splitext(args.input_path)
    output_path = f"{base}_samples={args.num_samples}{ext}"

    with jsonlines.open(output_path, mode="w") as writer:
        writer.write_all(sampled_data)

    print(f"Sampled data written to: {output_path}")

if __name__ == "__main__":
    main()
