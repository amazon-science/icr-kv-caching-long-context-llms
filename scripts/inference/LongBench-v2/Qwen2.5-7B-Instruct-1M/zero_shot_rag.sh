#!bin/bash

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/LongBench-v2/Qwen2.5-7B-Instruct-1M/zero_shot_rag.yaml \
    context_max_tokens=16000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/LongBench-v2/Qwen2.5-7B-Instruct-1M/zero_shot_rag.yaml \
    context_max_tokens=8000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/LongBench-v2/Qwen2.5-7B-Instruct-1M/zero_shot_rag.yaml \
    context_max_tokens=4000