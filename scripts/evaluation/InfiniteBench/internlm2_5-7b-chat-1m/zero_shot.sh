#!bin/bash

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/internlm2_5-7b-chat-1m/zero_shot.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/internlm2_5-7b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 71286 107162

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/internlm2_5-7b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 107162 158607

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/internlm2_5-7b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 158607 253491

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/internlm2_5-7b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 253491 789998

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/internlm2_5-7b-chat-1m/zero_shot.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/internlm2_5-7b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 71286 107162

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/internlm2_5-7b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 107162 169793

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/internlm2_5-7b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 169793 253491

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/internlm2_5-7b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 254138 789998

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/internlm2_5-7b-chat-1m/zero_shot.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/internlm2_5-7b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 68516 94272

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/internlm2_5-7b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 96320 139150

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/internlm2_5-7b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 145775 224999

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/internlm2_5-7b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 229140 789998