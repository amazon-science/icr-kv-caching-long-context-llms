#!bin/bash

## HotpotQA
python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=hotpotqa-dev-multikilt_1000_k20_dep3 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=hotpotqa-dev-multikilt_1000_k50_dep3 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=hotpotqa-dev-multikilt_1000_k105_dep3 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=hotpotqa-dev-multikilt_1000_k220_dep3 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=hotpotqa-dev-multikilt_1000_k440_dep3 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=hotpotqa-dev-multikilt_1000_k1000_dep3 \
    $@

## NQ
python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=nq-dev-multikilt_1000_k20_dep6 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=nq-dev-multikilt_1000_k50_dep6 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=nq-dev-multikilt_1000_k105_dep6 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=nq-dev-multikilt_1000_k220_dep6 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=nq-dev-multikilt_1000_k440_dep6 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=nq-dev-multikilt_1000_k1000_dep6 \
    $@

## TriviaQA
python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=triviaqa-dev-multikilt_1000_k20_dep6 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=triviaqa-dev-multikilt_1000_k50_dep6 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=triviaqa-dev-multikilt_1000_k105_dep6 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=triviaqa-dev-multikilt_1000_k220_dep6 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=triviaqa-dev-multikilt_1000_k440_dep6 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=triviaqa-dev-multikilt_1000_k1000_dep6 \
    $@

## PopQA
python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=popqa_test_1000_k20_dep6 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=popqa_test_1000_k50_dep6 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=popqa_test_1000_k105_dep6 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=popqa_test_1000_k220_dep6 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=popqa_test_1000_k440_dep6 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=kilt \
    dataset.name=popqa_test_1000_k1000_dep6 \
    $@

## ASQA
python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=alce_asqa \
    dataset.name=asqa_eval_gtr_top15 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=alce_asqa \
    dataset.name=asqa_eval_gtr_top30 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=alce_asqa \
    dataset.name=asqa_eval_gtr_top75 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=alce_asqa \
    dataset.name=asqa_eval_gtr_top165 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=alce_asqa \
    dataset.name=asqa_eval_gtr_top345 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=alce_asqa \
    dataset.name=asqa_eval_gtr_top700 \
    $@

## Qampari
python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=alce_qampari \
    dataset.name=qampari_eval_gtr_top15 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=alce_qampari \
    dataset.name=qampari_eval_gtr_top30 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=alce_qampari \
    dataset.name=qampari_eval_gtr_top75 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=alce_qampari \
    dataset.name=qampari_eval_gtr_top165 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=alce_qampari \
    dataset.name=qampari_eval_gtr_top345 \
    $@

python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    dataset.split=alce_qampari \
    dataset.name=qampari_eval_gtr_top700 \
    $@