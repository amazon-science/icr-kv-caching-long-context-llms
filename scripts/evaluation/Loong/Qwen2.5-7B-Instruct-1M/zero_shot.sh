#!bin/bash

MODEL_NAME=$1
PROMPT_SETTING=$2

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 50370 93863

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 93863 136823

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 136887 204392

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 204392 365522