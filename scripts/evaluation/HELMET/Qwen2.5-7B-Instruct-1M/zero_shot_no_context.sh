#!bin/bash

## Qwen2.5-7B-Instruct-1M
## HotpotQA
python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/hotpotqa-dev-multikilt_1000_k20_dep3/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/hotpotqa-dev-multikilt_1000_k50_dep3/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/hotpotqa-dev-multikilt_1000_k105_dep3/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/hotpotqa-dev-multikilt_1000_k220_dep3/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/hotpotqa-dev-multikilt_1000_k440_dep3/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/hotpotqa-dev-multikilt_1000_k1000_dep3/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

## NQ
python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/nq-dev-multikilt_1000_k20_dep6/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/nq-dev-multikilt_1000_k50_dep6/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/nq-dev-multikilt_1000_k105_dep6/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/nq-dev-multikilt_1000_k220_dep6/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/nq-dev-multikilt_1000_k440_dep6/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/nq-dev-multikilt_1000_k1000_dep6/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

## TriviaQA
python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/triviaqa-dev-multikilt_1000_k20_dep6/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/triviaqa-dev-multikilt_1000_k50_dep6/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/triviaqa-dev-multikilt_1000_k105_dep6/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/triviaqa-dev-multikilt_1000_k220_dep6/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/triviaqa-dev-multikilt_1000_k440_dep6/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/triviaqa-dev-multikilt_1000_k1000_dep6/Qwen2.5-7B-Instruct-1M/zero_shot_no_context.jsonl" \