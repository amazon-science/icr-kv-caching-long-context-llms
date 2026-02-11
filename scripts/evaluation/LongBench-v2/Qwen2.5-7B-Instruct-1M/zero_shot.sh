#!bin/bash

MODEL_NAME=$1
PROMPT_SETTING=$2

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --maximum_input_length 9975 37147

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --maximum_input_length 37626 99412

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --maximum_input_length 99412 210686

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --maximum_input_length 211922 4163702