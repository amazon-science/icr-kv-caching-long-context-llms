#!bin/bash

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag.jsonl" \
    --maximum_input_length 9975 37147

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag.jsonl" \
    --maximum_input_length 37626 99412

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag.jsonl" \
    --maximum_input_length 99412 210686

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag.jsonl" \
    --maximum_input_length 211922 4163702

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=128000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --maximum_input_length 9975 37147

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --maximum_input_length 37626 99412

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --maximum_input_length 99412 210686

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --maximum_input_length 211922 4163702

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=64000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --maximum_input_length 9975 37147

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --maximum_input_length 37626 99412

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --maximum_input_length 99412 210686

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --maximum_input_length 211922 4163702

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=32000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --maximum_input_length 9975 37147

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --maximum_input_length 37626 99412

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --maximum_input_length 99412 210686

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --maximum_input_length 211922 4163702

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=16000.jsonl" \
    
python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --maximum_input_length 9975 37147

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --maximum_input_length 37626 99412

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --maximum_input_length 99412 210686

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --maximum_input_length 211922 4163702

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=8000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --maximum_input_length 9975 37147

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --maximum_input_length 37626 99412

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --maximum_input_length 99412 210686

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --maximum_input_length 211922 4163702

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=4000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --maximum_input_length 9975 37147

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --maximum_input_length 37626 99412

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --maximum_input_length 99412 210686

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --maximum_input_length 211922 4163702