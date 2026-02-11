#!bin/bash

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag.jsonl" \
    --maximum_input_length 10515 36232

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag.jsonl" \
    --maximum_input_length 36360 101870

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag.jsonl" \
    --maximum_input_length 102730 201755

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag.jsonl" \
    --maximum_input_length 202707 4411426

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=128000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --maximum_input_length 10515 36232

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --maximum_input_length 36360 101870

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --maximum_input_length 102730 201755

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --maximum_input_length 202707 4411426

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=64000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --maximum_input_length 10515 36232

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --maximum_input_length 36360 101870

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --maximum_input_length 102730 201755

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --maximum_input_length 202707 4411426

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=32000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --maximum_input_length 10515 36232

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --maximum_input_length 36360 101870

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --maximum_input_length 102730 201755

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --maximum_input_length 202707 4411426

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=16000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --maximum_input_length 10515 36232

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --maximum_input_length 36360 101870

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --maximum_input_length 102730 201755

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --maximum_input_length 202707 4411426

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=8000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --maximum_input_length 10515 36232

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --maximum_input_length 36360 101870

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --maximum_input_length 102730 201755

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --maximum_input_length 202707 4411426

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=4000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --maximum_input_length 10515 36232

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --maximum_input_length 36360 101870

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --maximum_input_length 102730 201755

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/internlm2_5-7b-chat-1m/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --maximum_input_length 202707 4411426