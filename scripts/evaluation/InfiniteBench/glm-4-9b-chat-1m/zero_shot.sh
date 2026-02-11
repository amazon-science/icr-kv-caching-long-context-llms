#!bin/bash

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/glm-4-9b-chat-1m/zero_shot.jsonl" \

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 69634 104032

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 104032 149360

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 149360 244794

python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files "data/outputs/InfiniteBench/longbook_choice_eng/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 244794 743814

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/glm-4-9b-chat-1m/zero_shot.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 69634 104032

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 104032 163517

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 163517 237378

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_qa_eng" \
    --file "data/outputs/InfiniteBench/longbook_qa_eng/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 237378 743814

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/glm-4-9b-chat-1m/zero_shot.jsonl" \

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 66419 91967

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 94218 134427

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 142787 218419

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task "longbook_sum_eng" \
    --file "data/outputs/InfiniteBench/longbook_sum_eng/glm-4-9b-chat-1m/zero_shot.jsonl" \
    --maximum_input_length 222777 743814