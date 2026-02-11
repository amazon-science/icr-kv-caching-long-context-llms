#!bin/bash

MODEL_NAME=$1
PROMPT_SETTING=$2

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "passkey" \
    --file "data/outputs/InfiniteBench/passkey/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "number_string" \
    --file "data/outputs/InfiniteBench/number_string/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "kv_retrieval" \
    --file "data/outputs/InfiniteBench/kv_retrieval/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "math_find" \
    --file "data/outputs/InfiniteBench/math_find/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "math_calc" \
    --file "data/outputs/InfiniteBench/math_calc/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "code_run" \
    --file "data/outputs/InfiniteBench/code_run/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/code_debug/$MODEL_NAME/$PROMPT_SETTING.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longdialogue_qa_eng" \
    --file "data/outputs/InfiniteBench/longdialogue_qa_eng/$MODEL_NAME/$PROMPT_SETTING.jsonl" \