#!bin/bash

MODEL_NAME=$1

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/$MODEL_NAME/zero_shot_RetroInfer.jsonl" \