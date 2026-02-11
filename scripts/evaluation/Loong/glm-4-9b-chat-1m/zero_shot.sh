#!bin/bash

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 45996 87360

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 87360 126193

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 127661 185941

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 185941 339901