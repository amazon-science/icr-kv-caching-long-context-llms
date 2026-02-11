#!bin/bash

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/LongBench-v2/internlm2_5-7b-chat-1m/zero_shot_rag.yaml

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/LongBench-v2/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    context_max_tokens=128000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/LongBench-v2/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    context_max_tokens=64000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/LongBench-v2/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    context_max_tokens=32000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/LongBench-v2/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    context_max_tokens=16000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/LongBench-v2/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    context_max_tokens=8000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/LongBench-v2/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    context_max_tokens=4000