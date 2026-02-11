#!bin/bash

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_choice_eng

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_choice_eng \
    context_max_tokens=128000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_choice_eng \
    context_max_tokens=64000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_choice_eng \
    context_max_tokens=32000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_choice_eng \
    context_max_tokens=16000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_choice_eng \
    context_max_tokens=8000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_choice_eng \
    context_max_tokens=4000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_qa_eng

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_qa_eng \
    context_max_tokens=128000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_qa_eng \
    context_max_tokens=64000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_qa_eng \
    context_max_tokens=32000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_qa_eng \
    context_max_tokens=16000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_qa_eng \
    context_max_tokens=8000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_qa_eng \
    context_max_tokens=4000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_sum_eng

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_sum_eng \
    context_max_tokens=128000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_sum_eng \
    context_max_tokens=64000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_sum_eng \
    context_max_tokens=32000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_sum_eng \
    context_max_tokens=16000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_sum_eng \
    context_max_tokens=8000

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/internlm2_5-7b-chat-1m/zero_shot_rag.yaml \
    dataset.split=longbook_sum_eng \
    context_max_tokens=4000