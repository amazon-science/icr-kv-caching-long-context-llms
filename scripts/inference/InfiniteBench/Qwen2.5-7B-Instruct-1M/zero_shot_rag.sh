# !bin/bash

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct-1M/zero_shot_rag.yaml \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct-1M/zero_shot_rag.yaml \
    context_max_tokens=128000 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct-1M/zero_shot_rag.yaml \
    context_max_tokens=64000 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct-1M/zero_shot_rag.yaml \
    context_max_tokens=32000 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct-1M/zero_shot_rag.yaml \
    context_max_tokens=16000 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct-1M/zero_shot_rag.yaml \
    context_max_tokens=8000 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct-1M/zero_shot_rag.yaml \
    context_max_tokens=4000 \
    $@