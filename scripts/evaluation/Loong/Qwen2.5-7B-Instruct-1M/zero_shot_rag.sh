#!bin/bash

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml"

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 50370 93863

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 93863 136823

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 136887 204392

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 204392 365522

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml"

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 50370 93863

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 93863 136823

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 136887 204392

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 204392 365522

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml"

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 50370 93863

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 93863 136823

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 136887 204392

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 204392 365522

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml"

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 50370 93863

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 93863 136823

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 136887 204392

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 204392 365522

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml"

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 50370 93863

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 93863 136823

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 136887 204392

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 204392 365522

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml"

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 50370 93863

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 93863 136823

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 136887 204392

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 204392 365522

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml"

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 50370 93863

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 93863 136823

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 136887 204392

python src/amzn_long_context_rag/evaluation/loong_eval.py \
    --file "data/outputs/Loong/financial/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --config_path "config/evaluation/Loong/DeepSeek-R1-Distill-Qwen-32B/llm_as_a_judge.yaml" \
    --maximum_input_length 204392 365522