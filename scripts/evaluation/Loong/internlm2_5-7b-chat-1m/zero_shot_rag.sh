#!bin/bash

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml"

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 47448 90766

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 90766 129492

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 131321 196186

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 196186 355715

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml"

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 47448 90766

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 90766 129492

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 131321 196186

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 196186 355715

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml"

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 47448 90766

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 90766 129492

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 131321 196186

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 196186 355715

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml"

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 47448 90766

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 90766 129492

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 131321 196186

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 196186 355715

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml"

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 47448 90766

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 90766 129492

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 131321 196186

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 196186 355715

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml"

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 47448 90766

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 90766 129492

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 131321 196186

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 196186 355715

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml"

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 47448 90766

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 90766 129492

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 131321 196186

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 196186 355715