#!bin/bash

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    $@