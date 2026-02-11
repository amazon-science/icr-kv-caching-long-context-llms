#!bin/bash

MODEL_NAME=$1

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/$MODEL_NAME/zero_shot_RetroInfer.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/$MODEL_NAME/zero_shot_RetroInfer.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/$MODEL_NAME/zero_shot_RetroInfer.jsonl" \