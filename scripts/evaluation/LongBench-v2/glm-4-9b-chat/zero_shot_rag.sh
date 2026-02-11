#!bin/bash

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag.jsonl" \
    --maximum_input_length 9918 34644

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag.jsonl" \
    --maximum_input_length 35217 99128

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag.jsonl" \
    --maximum_input_length 99135 196429

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag.jsonl" \
    --maximum_input_length 198708 4144848

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=128000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --maximum_input_length 9918 34644

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --maximum_input_length 35217 99128

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --maximum_input_length 99135 196429

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=128000.jsonl" \
    --maximum_input_length 198708 4144848

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=64000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --maximum_input_length 9918 34644

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --maximum_input_length 35217 99128

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --maximum_input_length 99135 196429

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=64000.jsonl" \
    --maximum_input_length 198708 4144848

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=32000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --maximum_input_length 9918 34644

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --maximum_input_length 35217 99128

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --maximum_input_length 99135 196429

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=32000.jsonl" \
    --maximum_input_length 198708 4144848

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=16000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --maximum_input_length 9918 34644

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --maximum_input_length 35217 99128

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --maximum_input_length 99135 196429

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=16000.jsonl" \
    --maximum_input_length 198708 4144848

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=8000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --maximum_input_length 9918 34644

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --maximum_input_length 35217 99128

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --maximum_input_length 99135 196429

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=8000.jsonl" \
    --maximum_input_length 198708 4144848

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=4000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --maximum_input_length 9918 34644

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --maximum_input_length 35217 99128

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --maximum_input_length 99135 196429

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/LongBench-v2/train/glm-4-9b-chat/zero_shot_rag_context_max_tokens=4000.jsonl" \
    --maximum_input_length 198708 4144848