#!bin/bash

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/Qwen2.5-7B-Instruct/zero_shot.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/Qwen2.5-7B-Instruct/zero_shot.jsonl" \
    --maximum_input_length 69657 104304

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/Qwen2.5-7B-Instruct/zero_shot.jsonl" \
    --maximum_input_length 104304 149474

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/Qwen2.5-7B-Instruct/zero_shot.jsonl" \
    --maximum_input_length 149474 244909

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/Qwen2.5-7B-Instruct/zero_shot.jsonl" \
    --maximum_input_length 244909 745586

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/Qwen2.5-7B-Instruct/zero_shot.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/Qwen2.5-7B-Instruct/zero_shot.jsonl" \
    --maximum_input_length 69657 104304

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/Qwen2.5-7B-Instruct/zero_shot.jsonl" \
    --maximum_input_length 104304 163712

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/Qwen2.5-7B-Instruct/zero_shot.jsonl" \
    --maximum_input_length 163712 237831

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/Qwen2.5-7B-Instruct/zero_shot.jsonl" \
    --maximum_input_length 237831 745586

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/Qwen2.5-7B-Instruct/zero_shot.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/Qwen2.5-7B-Instruct/zero_shot.jsonl" \
    --maximum_input_length 66455 92046

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/Qwen2.5-7B-Instruct/zero_shot.jsonl" \
    --maximum_input_length 94337 134553

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/Qwen2.5-7B-Instruct/zero_shot.jsonl" \
    --maximum_input_length 142829 218613

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/Qwen2.5-7B-Instruct/zero_shot.jsonl" \
    --maximum_input_length 223252 745586