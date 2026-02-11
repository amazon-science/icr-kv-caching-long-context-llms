#!bin/bash

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot.jsonl" \
    --maximum_input_length 9975 37147

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot.jsonl" \
    --maximum_input_length 37626 99412

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot.jsonl" \
    --maximum_input_length 99412 210686

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot.jsonl" \
    --maximum_input_length 211922 4163702