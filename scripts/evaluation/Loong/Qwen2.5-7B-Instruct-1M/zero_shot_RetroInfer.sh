#!bin/bash

MODEL_NAME=$1

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/$MODEL_NAME/zero_shot_RetroInfer.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \