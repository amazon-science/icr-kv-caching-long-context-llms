#!bin/bash

## HotpotQA
python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen/Qwen2.5-7B-Instruct-1M \
    dataset.name=hotpotqa-dev-multikilt_1000_k220_dep3 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen2.5-7B-Instruct-1M-GRPO-v2 \
    dataset.name=hotpotqa-dev-multikilt_1000_k220_dep3 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_doc_ids_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen2.5-7B-Instruct-1M-GRPO-AnsOnly-v2 \
    dataset.name=hotpotqa-dev-multikilt_1000_k220_dep3 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen2.5-7B-Instruct-1M-GRPO-Quote \
    dataset.name=hotpotqa-dev-multikilt_1000_k220_dep3 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_quote_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen2.5-7B-Instruct-1M-GRPO-IDC \
    dataset.name=hotpotqa-dev-multikilt_1000_k220_dep3 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_doc_ids_and_content_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen2.5-7B-Instruct-1M-GRPO-Judge \
    dataset.name=hotpotqa-dev-multikilt_1000_k220_dep3 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_judge_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

## NQ
python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen/Qwen2.5-7B-Instruct-1M \
    dataset.name=nq-dev-multikilt_1000_k220_dep6 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen2.5-7B-Instruct-1M-GRPO-v2 \
    dataset.name=nq-dev-multikilt_1000_k220_dep6 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_doc_ids_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen2.5-7B-Instruct-1M-GRPO-AnsOnly-v2 \
    dataset.name=nq-dev-multikilt_1000_k220_dep6 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen2.5-7B-Instruct-1M-GRPO-Quote \
    dataset.name=nq-dev-multikilt_1000_k220_dep6 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_quote_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen2.5-7B-Instruct-1M-GRPO-IDC \
    dataset.name=nq-dev-multikilt_1000_k220_dep6 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_doc_ids_and_content_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen2.5-7B-Instruct-1M-GRPO-Judge \
    dataset.name=nq-dev-multikilt_1000_k220_dep6 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_judge_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

## TriviaQA
python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen/Qwen2.5-7B-Instruct-1M \
    dataset.name=triviaqa-dev-multikilt_1000_k220_dep6 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen2.5-7B-Instruct-1M-GRPO-v2 \
    dataset.name=triviaqa-dev-multikilt_1000_k220_dep6 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_doc_ids_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen2.5-7B-Instruct-1M-GRPO-AnsOnly-v2 \
    dataset.name=triviaqa-dev-multikilt_1000_k220_dep6 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen2.5-7B-Instruct-1M-GRPO-Quote \
    dataset.name=triviaqa-dev-multikilt_1000_k220_dep6 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_quote_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen2.5-7B-Instruct-1M-GRPO-IDC \
    dataset.name=triviaqa-dev-multikilt_1000_k220_dep6 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_doc_ids_and_content_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@

python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/docs_attn_analysis.yaml \
    model_name=Qwen2.5-7B-Instruct-1M-GRPO-Judge \
    dataset.name=triviaqa-dev-multikilt_1000_k220_dep6 \
    dataset.prompt_obj=prompts/analysis/HELMET/zero_shot_judge_attn.json \
    attention_strategy=all_layers \
    ndcg_k=10 \
    limit_samples=100
    $@