#!bin/bash

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/glm-4-9b-chat/zero_shot.yaml \
    dataset.split=longbook_choice_eng

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/glm-4-9b-chat/zero_shot.yaml \
    dataset.split=longbook_qa_eng

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/InfiniteBench/glm-4-9b-chat/zero_shot.yaml \
    dataset.split=longbook_sum_eng