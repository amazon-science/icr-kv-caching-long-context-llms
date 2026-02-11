#!bin/bash

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=128000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=64000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=32000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=16000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=8000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=4000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=128000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=64000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=32000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=16000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=8000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=4000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=128000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=64000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=32000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=16000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=8000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=4000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "passkey" \
    --file "data/outputs/InfiniteBench/passkey/Qwen2.5-7B-Instruct-1M/zero_shot_rag.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "passkey" \
    --file "data/outputs/InfiniteBench/passkey/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=128000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "passkey" \
    --file "data/outputs/InfiniteBench/passkey/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=64000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "passkey" \
    --file "data/outputs/InfiniteBench/passkey/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=32000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "passkey" \
    --file "data/outputs/InfiniteBench/passkey/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=16000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "passkey" \
    --file "data/outputs/InfiniteBench/passkey/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=8000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "passkey" \
    --file "data/outputs/InfiniteBench/passkey/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=4000.jsonl" \


python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "number_string" \
    --file "data/outputs/InfiniteBench/number_string/Qwen2.5-7B-Instruct-1M/zero_shot_rag.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "number_string" \
    --file "data/outputs/InfiniteBench/number_string/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=128000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "number_string" \
    --file "data/outputs/InfiniteBench/number_string/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=64000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "number_string" \
    --file "data/outputs/InfiniteBench/number_string/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=32000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "number_string" \
    --file "data/outputs/InfiniteBench/number_string/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=16000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "number_string" \
    --file "data/outputs/InfiniteBench/number_string/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=8000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "number_string" \
    --file "data/outputs/InfiniteBench/number_string/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=4000.jsonl" \


python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "kv_retrieval" \
    --file "data/outputs/InfiniteBench/kv_retrieval/Qwen2.5-7B-Instruct-1M/zero_shot_rag.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "kv_retrieval" \
    --file "data/outputs/InfiniteBench/kv_retrieval/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=128000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "kv_retrieval" \
    --file "data/outputs/InfiniteBench/kv_retrieval/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=64000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "kv_retrieval" \
    --file "data/outputs/InfiniteBench/kv_retrieval/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=32000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "kv_retrieval" \
    --file "data/outputs/InfiniteBench/kv_retrieval/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=16000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "kv_retrieval" \
    --file "data/outputs/InfiniteBench/kv_retrieval/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=8000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "kv_retrieval" \
    --file "data/outputs/InfiniteBench/kv_retrieval/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=4000.jsonl" \


python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "math_find" \
    --file "data/outputs/InfiniteBench/math_find/Qwen2.5-7B-Instruct-1M/zero_shot_rag.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "math_find" \
    --file "data/outputs/InfiniteBench/math_find/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=128000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "math_find" \
    --file "data/outputs/InfiniteBench/math_find/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=64000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "math_find" \
    --file "data/outputs/InfiniteBench/math_find/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=32000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "math_find" \
    --file "data/outputs/InfiniteBench/math_find/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=16000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "math_find" \
    --file "data/outputs/InfiniteBench/math_find/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=8000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "math_find" \
    --file "data/outputs/InfiniteBench/math_find/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=4000.jsonl" \


python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "math_calc" \
    --file "data/outputs/InfiniteBench/math_calc/Qwen2.5-7B-Instruct-1M/zero_shot_rag.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "math_calc" \
    --file "data/outputs/InfiniteBench/math_calc/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=128000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "math_calc" \
    --file "data/outputs/InfiniteBench/math_calc/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=64000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "math_calc" \
    --file "data/outputs/InfiniteBench/math_calc/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=32000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "math_calc" \
    --file "data/outputs/InfiniteBench/math_calc/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=16000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "math_calc" \
    --file "data/outputs/InfiniteBench/math_calc/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=8000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "math_calc" \
    --file "data/outputs/InfiniteBench/math_calc/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=4000.jsonl" \


python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "code_run" \
    --file "data/outputs/InfiniteBench/code_run/Qwen2.5-7B-Instruct-1M/zero_shot_rag.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "code_run" \
    --file "data/outputs/InfiniteBench/code_run/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=128000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "code_run" \
    --file "data/outputs/InfiniteBench/code_run/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=64000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "code_run" \
    --file "data/outputs/InfiniteBench/code_run/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=32000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "code_run" \
    --file "data/outputs/InfiniteBench/code_run/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=16000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "code_run" \
    --file "data/outputs/InfiniteBench/code_run/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=8000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "code_run" \
    --file "data/outputs/InfiniteBench/code_run/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=4000.jsonl" \


python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/code_debug/Qwen2.5-7B-Instruct-1M/zero_shot_rag.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/code_debug/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=128000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/code_debug/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=64000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/code_debug/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=32000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/code_debug/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=16000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/code_debug/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=8000.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/code_debug/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=4000.jsonl" \


python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longdialogue_qa_eng" \
    --file "data/outputs/InfiniteBench/longdialogue_qa_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longdialogue_qa_eng" \
    --file "data/outputs/InfiniteBench/longdialogue_qa_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=128000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longdialogue_qa_eng" \
    --file "data/outputs/InfiniteBench/longdialogue_qa_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=64000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longdialogue_qa_eng" \
    --file "data/outputs/InfiniteBench/longdialogue_qa_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=32000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longdialogue_qa_eng" \
    --file "data/outputs/InfiniteBench/longdialogue_qa_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=16000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longdialogue_qa_eng" \
    --file "data/outputs/InfiniteBench/longdialogue_qa_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=8000.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longdialogue_qa_eng" \
    --file "data/outputs/InfiniteBench/longdialogue_qa_eng/Qwen2.5-7B-Instruct-1M/zero_shot_rag_context_max_tokens=4000.jsonl" \
