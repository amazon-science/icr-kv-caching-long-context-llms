#!bin/bash

MODEL_NAME=$1
PROMPT_SETTING=$2

## HotpotQA
python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/hotpotqa-dev-multikilt_1000_k20_dep3/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/hotpotqa-dev-multikilt_1000_k50_dep3/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/hotpotqa-dev-multikilt_1000_k105_dep3/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/hotpotqa-dev-multikilt_1000_k220_dep3/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/hotpotqa-dev-multikilt_1000_k440_dep3/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/hotpotqa-dev-multikilt_1000_k1000_dep3/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

## NQ
python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/nq-dev-multikilt_1000_k20_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/nq-dev-multikilt_1000_k50_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/nq-dev-multikilt_1000_k105_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/nq-dev-multikilt_1000_k220_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/nq-dev-multikilt_1000_k440_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/nq-dev-multikilt_1000_k1000_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

## TriviaQA
python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/triviaqa-dev-multikilt_1000_k20_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/triviaqa-dev-multikilt_1000_k50_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/triviaqa-dev-multikilt_1000_k105_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/triviaqa-dev-multikilt_1000_k220_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/triviaqa-dev-multikilt_1000_k440_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/triviaqa-dev-multikilt_1000_k1000_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

## PopQA
python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/popqa_test_1000_k20_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/popqa_test_1000_k50_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/popqa_test_1000_k105_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/popqa_test_1000_k220_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/popqa_test_1000_k440_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --task kilt \
    --file "data/outputs/HELMET/kilt/popqa_test_1000_k1000_dep6/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

## ASQA
python src/amzn_long_context_rag/evaluation/helmet_alce_eval.py \
    --file "data/outputs/HELMET/alce_asqa/asqa_eval_gtr_top15/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --qa \
    --citations \

python src/amzn_long_context_rag/evaluation/helmet_alce_eval.py \
    --file "data/outputs/HELMET/alce_asqa/asqa_eval_gtr_top30/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --qa \
    --citations \

python src/amzn_long_context_rag/evaluation/helmet_alce_eval.py \
    --file "data/outputs/HELMET/alce_asqa/asqa_eval_gtr_top75/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --qa \
    --citations \

python src/amzn_long_context_rag/evaluation/helmet_alce_eval.py \
    --file "data/outputs/HELMET/alce_asqa/asqa_eval_gtr_top165/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --qa \
    --citations \

python src/amzn_long_context_rag/evaluation/helmet_alce_eval.py \
    --file "data/outputs/HELMET/alce_asqa/asqa_eval_gtr_top345/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --qa \
    --citations \

python src/amzn_long_context_rag/evaluation/helmet_alce_eval.py \
    --file "data/outputs/HELMET/alce_asqa/asqa_eval_gtr_top700/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --qa \
    --citations \

## Qampari
python src/amzn_long_context_rag/evaluation/helmet_alce_eval.py \
    --file "data/outputs/HELMET/alce_qampari/qampari_eval_gtr_top15/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --qa \
    --citations \

python src/amzn_long_context_rag/evaluation/helmet_alce_eval.py \
    --file "data/outputs/HELMET/alce_qampari/qampari_eval_gtr_top30/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --qa \
    --citations \

python src/amzn_long_context_rag/evaluation/helmet_alce_eval.py \
    --file "data/outputs/HELMET/alce_qampari/qampari_eval_gtr_top75/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --qa \
    --citations \

python src/amzn_long_context_rag/evaluation/helmet_alce_eval.py \
    --file "data/outputs/HELMET/alce_qampari/qampari_eval_gtr_top165/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --qa \
    --citations \

python src/amzn_long_context_rag/evaluation/helmet_alce_eval.py \
    --file "data/outputs/HELMET/alce_qampari/qampari_eval_gtr_top345/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --qa \
    --citations \

python src/amzn_long_context_rag/evaluation/helmet_alce_eval.py \
    --file "data/outputs/HELMET/alce_qampari/qampari_eval_gtr_top700/$MODEL_NAME/$PROMPT_SETTING.jsonl" \
    --qa \
    --citations \