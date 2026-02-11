#!bin/bash

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot.jsonl" \
    --maximum_input_length 9918 34644

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot.jsonl" \
    --maximum_input_length 35217 99128

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot.jsonl" \
    --maximum_input_length 99135 196429

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot.jsonl" \
    --maximum_input_length 198708 4144848