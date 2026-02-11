#!bin/bash

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct/zero_shot_rag.yaml \
    dataset.split=longbook_choice_eng \
    context_max_tokens=16000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct/zero_shot_rag.yaml \
    dataset.split=longbook_choice_eng \
    context_max_tokens=8000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct/zero_shot_rag.yaml \
    dataset.split=longbook_choice_eng \
    context_max_tokens=4000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct/zero_shot_rag.yaml \
    dataset.split=longbook_qa_eng \
    context_max_tokens=16000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct/zero_shot_rag.yaml \
    dataset.split=longbook_qa_eng \
    context_max_tokens=8000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct/zero_shot_rag.yaml \
    dataset.split=longbook_qa_eng \
    context_max_tokens=4000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct/zero_shot_rag.yaml \
    dataset.split=longbook_sum_eng \
    context_max_tokens=16000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct/zero_shot_rag.yaml \
    dataset.split=longbook_sum_eng \
    context_max_tokens=8000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/Qwen2.5-7B-Instruct/zero_shot_rag.yaml \
    dataset.split=longbook_sum_eng \
    context_max_tokens=4000