# Exploring Fine-Tuning for In-Context Retrieval and Efficient KV-Caching in Long-Context Language Models
Official repository for the paper "Exploring Fine-Tuning for In-Context Retrieval and Efficient KV-Caching in Long-Context Language Models".

## Table of Contents

- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Training](#training)
- [Preprocessing](#preprocessing)
- [Analysis](#analysis)
- [Advanced Topics](#advanced-topics)

---

## Repository Structure

```
icr-kv-caching-long-context-llms/
├── config/              # YAML configuration files
│   ├── inference/       # Inference configs per benchmark/model
│   ├── evaluation/      # Evaluation configs
│   ├── training/        # Training configs (Kubernetes YAML)
│   ├── analysis/        # Attention analysis configs
│   └── server_inference/ # vLLM server configs
├── data/                # Data storage (outputs, indexes, benchmarks)
│   ├── outputs/         # Model inference outputs
│   ├── evaluation/      # Evaluation results
│   ├── indexes/         # FAISS indexes for RAG
│   ├── train/           # Training data
│   ├── benchmarks/      # Downloaded benchmarks
│   └── wikipedia/       # Wikipedia passages
├── plots/               # Plotting scripts and images
├── prompts/             # Prompt templates (JSON format)
│   ├── inference/       # Inference prompts
│   ├── evaluation/      # Evaluation prompts
│   └── analysis/        # Analysis prompts
├── scripts/             # Bash scripts for running experiments
│   ├── inference/       # Inference scripts per benchmark/model
│   ├── evaluation/      # Evaluation scripts per benchmark/model
│   └── training/        # Training launch scripts
├── src/amzn_long_context_rag/
│   ├── inference/       # Inference logic (local & vLLM)
│   ├── evaluation/      # Evaluation metrics
│   ├── training/        # Reward functions for RL training
│   ├── data/            # Data loaders
│   ├── retriever/       # RAG retriever implementation
│   ├── preprocessing/   # Data preprocessing utilities
│   └── analysis/        # Attention analysis tools
└── notebooks/           # Jupyter notebooks for exploration
```

---

## Installation

### Prerequisites

- Python 3.10
- CUDA-compatible GPU (for inference and training)
- Conda or virtualenv

### Setup

```bash
# Create conda environment
conda create -n longcontext python==3.10
conda activate longcontext

# Install the package
pip install -e .

# Install flash-attention (required for efficient inference)
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

### Verify Installation

```bash
python -c "import src.amzn_long_context_rag; print('Installation successful!')"
```

---

## Quick Start

### Run Inference on LongBench-v2

```bash
# Navigate to the repository root
cd icr-kv-caching-long-context-llms

# Run zero-shot inference
bash scripts/inference/LongBench-v2/Qwen2.5-7B-Instruct-1M/zero_shot.sh

# Run with RAG
bash scripts/inference/LongBench-v2/Qwen2.5-7B-Instruct-1M/zero_shot_rag.sh
```

### Evaluate Results

```bash
# Evaluate the inference outputs
bash scripts/evaluation/LongBench-v2/Qwen2.5-7B-Instruct-1M/zero_shot.sh \
    Qwen2.5-7B-Instruct-1M zero_shot
```

---

## Inference

### Overview

The main inference script is `src/amzn_long_context_rag/inference/async_inference.py`. It supports:
- **Full context mode**: Pass entire context to the model
- **Top-k RAG mode**: Retrieve top-k relevant chunks
- **No context mode**: Zero-shot inference without context

### Running Inference

#### Method 1: Using Bash Scripts (Recommended)

```bash
# General pattern
bash scripts/inference/<BENCHMARK>/<MODEL>/<SETTING>.sh [OVERRIDES]

# Examples
bash scripts/inference/InfiniteBench/Qwen2.5-7B-Instruct-1M/zero_shot.sh
bash scripts/inference/Loong/glm-4-9b-chat-1m/zero_shot_rag.sh
```

#### Method 2: Direct Python Execution

```bash
python src/amzn_long_context_rag/inference/async_inference.py \
    --config_path config/inference/LongBench-v2/Qwen2.5-7B-Instruct-1M/zero_shot.yaml \
    [KEY=VALUE ...]
```

### Inference Parameters

#### Command-Line Arguments

- `--config_path` or `-c`: Path to YAML configuration file (required)
- Additional arguments can override config values using dotlist notation: `key.subkey=value`

#### Configuration File Structure

```yaml
seed: 42                          # Random seed
output_dir: data/outputs          # Output directory
model_name: Qwen/Qwen2.5-7B-Instruct-1M  # HuggingFace model name
device: cuda                      # Device (cuda/cpu)

vllm_params:
  gpu_memory_utilization: 0.8     # GPU memory fraction
  max_model_len: 1010000          # Maximum sequence length
  max_num_batched_tokens: 131072  # Batch size in tokens
  enforce_eager: true             # Disable CUDA graphs
  tensor_parallel_size: 2         # Tensor parallelism
  pipeline_parallel_size: 1       # Pipeline parallelism
  enable_lora: false              # Enable LoRA adapters
  lora_adapter_path: null         # Path to LoRA weights

sampling_params:
  temperature: 0.0                # Sampling temperature
  n: 1                            # Number of completions
  max_tokens: 2048                # Max generation length

dataset:
  data_loader: LongBench-v2       # Dataset loader class
  path: THUDM/LongBench-v2        # HuggingFace dataset path
  split: train                    # Dataset split
  name: null                      # Dataset subset name
  prompt_obj: prompts/inference/LongBench-v2/zero_shot.json
  continue_final_message: false   # Continue from last message

retrieval:
  mode: full                      # full/topk/none
  top_k: 5                        # Number of chunks (for topk)
  embedding_model_name: Qwen/Qwen3-Embedding-4B
  index_dir: data/indexes         # FAISS index directory
  use_offline_hits: false         # Use pre-computed retrieval
  offline_hits_path: null         # Path to offline hits

context_max_tokens: null          # Max context tokens (null=auto)
split_docs: false                 # Split context into [DOC i] format
limit_samples: null               # Limit number of samples (for testing)
```

#### Runtime Overrides

Override any config parameter at runtime:

```bash
# Change max generation length
bash scripts/inference/LongBench-v2/Qwen2.5-7B-Instruct-1M/zero_shot.sh \
    sampling_params.max_tokens=512

# Change model
bash scripts/inference/LongBench-v2/Qwen2.5-7B-Instruct-1M/zero_shot.sh \
    model_name=Qwen/Qwen2.5-14B-Instruct-1M

# Enable RAG with top-k=10
bash scripts/inference/LongBench-v2/Qwen2.5-7B-Instruct-1M/zero_shot.sh \
    retrieval.mode=topk retrieval.top_k=10

# Limit to 100 samples for testing
bash scripts/inference/LongBench-v2/Qwen2.5-7B-Instruct-1M/zero_shot.sh \
    limit_samples=100
```

### Output Format

Inference outputs are saved as JSONL files in `data/outputs/<BENCHMARK>/<SPLIT>/<MODEL>/<SETTING>.jsonl`:

```json
{
  "id": "instance_id",
  "question": "What is...",
  "context": "Long context...",
  "model_output": "The answer is...",
  "original_ctx_length": 50000,
  "final_ctx_length": 50000
}
```

### Supported Benchmarks

- **LongBench-v2**: Multi-choice QA benchmark
- **InfiniteBench**: 12 diverse long-context tasks
- **Loong**: Financial domain benchmark
- **HELMET**: Multi-hop QA with citations

---

## Evaluation

### Overview

Evaluation scripts compute task-specific metrics on inference outputs:
- **LongBench-v2 and InfiniteBench MC**: Accuracy using XFinder
- **InfiniteBench**: Task-specific metrics (F1, SubEM, Rouge, etc.)
- **Loong**: LLM-as-judge evaluation
- **HELMET**: Citation accuracy and answer quality

### Running Evaluation

#### Method 1: Using Bash Scripts

```bash
# General pattern
bash scripts/evaluation/<BENCHMARK>/<MODEL>/<SETTING>.sh <MODEL_NAME> <SETTING_NAME>

# Example: Evaluate LongBench-v2 zero-shot results
bash scripts/evaluation/LongBench-v2/Qwen2.5-7B-Instruct-1M/zero_shot.sh \
    Qwen2.5-7B-Instruct-1M zero_shot
```

#### Method 2: Direct Python Execution

**InfiniteBench:**

```bash
python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task longbook_qa_eng \
    --file data/outputs/InfiniteBench/test/Qwen2.5-7B-Instruct-1M/zero_shot.jsonl \
    --output_dir data/evaluation
```

**LongBench-v2:**

```bash
python src/amzn_long_context_rag/evaluation/xfinder_eval_mcqa.py \
    --files data/outputs/LongBench-v2/train/Qwen2.5-7B-Instruct-1M/zero_shot.jsonl
```

**HELMET:**

```bash
python src/amzn_long_context_rag/evaluation/helmet_eval.py \
    --file data/outputs/HELMET/kilt/hotpotqa/Qwen2.5-7B-Instruct-1M/zero_shot.jsonl \
    --output_dir data/evaluation
```

### Evaluation Parameters

#### InfiniteBench Evaluation

- `--task`: Task name (e.g., `longbook_qa_eng`, `code_run`, `passkey`)
- `--file`: Path to inference output JSONL
- `--output_dir`: Directory for evaluation results
- `--maximum_input_length`: Filter by context length (single int or range)

**Example with context length filtering:**

```bash
# Evaluate only instances with context < 50000 tokens
python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task longbook_qa_eng \
    --file data/outputs/InfiniteBench/test/model/zero_shot.jsonl \
    --maximum_input_length 50000

# Evaluate instances with context between 10000-50000 tokens
python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task longbook_qa_eng \
    --file data/outputs/InfiniteBench/test/model/zero_shot.jsonl \
    --maximum_input_length 10000 50000
```

#### LongBench-v2 Evaluation

- `--files`: Path(s) to inference output JSONL (can specify multiple)
- `--maximum_input_length`: Filter by context length (optional)

#### HELMET Evaluation

- `--file`: Path to inference output JSONL
- `--output_dir`: Directory for evaluation results
- `--task`: Task name (e.g., `hotpotqa`, `nq`, `triviaqa`)

### Output Format

Evaluation results are saved as JSON files in `data/evaluation/<BENCHMARK>/<SPLIT>/<MODEL>/<SETTING>.json`:

```json
{
  "task": "longbook_qa_eng",
  "score": 0.85,
  "f1": 0.82,
  "sub_em": 0.88,
  "instances": 200,
  "maximum_input_length": null
}
```

### Supported Tasks

**InfiniteBench:**
- `passkey`, `number_string`, `kv_retrieval`
- `longbook_qa_eng`, `longbook_choice_eng`, `longbook_sum_eng`
- `longdialogue_qa_eng`
- `math_find`, `math_calc`
- `code_run`, `code_debug`

**LongBench-v2:**
- Multiple-choice QA tasks (evaluated with XFinder)

**HELMET:**
- `hotpotqa`, `nq`, `triviaqa`, `popqa`
- `alce_asqa`, `alce_qampari`

---

## Training

### Overview

Training uses the VERL library for reinforcement learning with custom reward functions. The project supports:
- **GRPO (Group Relative Policy Optimization)**: Main RL algorithm
- **SFT (Supervised Fine-Tuning)**: Baseline training
- **LoRA**: Parameter-efficient fine-tuning

### Training Data

Training data is stored in `data/train/` with the following structure:

```
data/train/
├── hotpot_qa/
│   ├── train.jsonl              # Raw training data
│   ├── clean_train.jsonl        # Cleaned data
│   ├── promoted_clean_train_tokens_16000_32000.jsonl  # Filtered by length
│   └── grpo/
│       ├── train.parquet        # GRPO format
│       └── test.parquet
└── 2WikiMultihopQA/
    └── ...
```

### Reward Functions

Reward functions are defined in `src/amzn_long_context_rag/training/reward_functions.py` and `src/amzn_long_context_rag/training/judge_reward_functions.py`:

1. **my_reward_fn_answer_only**: Rewards correct answers only
2. **my_reward_fn_f1_score**: Rewards answer + document ID F1
3. **my_reward_fn_quote**: Rewards quote extraction + coverage
4. **my_reward_fn_doc_ids_and_content**: Rewards document selection + content accuracy
5. **compute_score**: Rewards the policy with a judge

### Running Training

#### Local Training (Single Node)

Training is typically run on Kubernetes clusters, but for local testing:

```bash
# Activate VERL environment
source /path/to/verl/bin/activate

# Run training script
bash scripts/training/run_qwen2_5_7b_instruct_1m_grpo.sh
```

#### Kubernetes Training (Multi-Node)

```bash
# Submit training job to Kubernetes
kubectl apply -f config/training/multi_node_training_grpo.yaml

# Monitor job status
kubectl get pods -w | grep <your-alias>

# View logs
kubectl logs -f <pod-name>
```

Note: this yaml config will call the above training script to be executed on a set of replicas.

### Training Configuration

Training scripts use VERL's configuration system. Key parameters in bash scripts:

```bash
# Model and data paths
BASE_MODEL="/path/to/Qwen2.5-7B-Instruct-1M"
hotpot_qa_train_path="data/train/hotpot_qa/grpo/train.parquet"
hotpot_qa_test_path="data/train/hotpot_qa/grpo/test.parquet"

# Reward function
reward_func_name="my_reward_fn_f1_score"
reward_func_path="src/amzn_long_context_rag/training/reward_functions.py"

# Training hyperparameters
data.train_batch_size=256
data.max_prompt_length=32256
data.max_response_length=512
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.rollout.n=5  # Number of samples per prompt
trainer.total_epochs=2
```

### Available Training Scripts

```bash
# GRPO variants
scripts/training/run_qwen2_5_7b_instruct_1m_grpo.sh              # Standard GRPO
scripts/training/run_qwen2_5_7b_instruct_1m_grpo_answer_only.sh # Answer-only reward
scripts/training/run_qwen2_5_7b_instruct_1m_grpo_quote.sh       # Quote extraction
scripts/training/run_qwen2_5_7b_instruct_1m_grpo_judge.sh       # Judge-based reward
scripts/training/run_qwen2_5_7b_instruct_1m_grpo_doc_ids_and_content.sh

# LoRA training
scripts/training/run_qwen2_5_7b_instruct_1m_grpo_lora.sh
scripts/training/run_qwen2_5_7b_instruct_1m_sft_lora.sh

# SFT
scripts/training/run_qwen2_5_7b_instruct_1m_sft.sh
```

### Modifying Training Parameters

Edit the bash script directly or override via environment variables:

```bash
# Change learning rate
export LEARNING_RATE=5e-7
bash scripts/training/run_qwen2_5_7b_instruct_1m_grpo.sh

# Change number of epochs
# Edit the script and modify: trainer.total_epochs=5
```

### Training Outputs

- **Checkpoints**: Saved to `OUTPUT_DIR` (specified in script)
- **Logs**: Saved to `LOG_DIR`
- **Wandb**: Training metrics logged to Weights & Biases

---

## Preprocessing

### Overview

Preprocessing scripts prepare data for training and create indexes for RAG.

### Training Data Preprocessing

Located in `src/amzn_long_context_rag/preprocessing/training_data/`:

#### 1. Create Training Data

```bash
python src/amzn_long_context_rag/preprocessing/training_data/create_training.py \
    --dataset hotpot_qa \
    --output data/train/hotpot_qa/train.jsonl
```

#### 2. Clean Training Data

```bash
python src/amzn_long_context_rag/preprocessing/training_data/clean_training.py \
    --input data/train/hotpot_qa/train.jsonl \
    --output data/train/hotpot_qa/clean_train.jsonl
```

#### 3. Filter by Token Length

```bash
python src/amzn_long_context_rag/preprocessing/training_data/filter_training.py \
    --input data/train/hotpot_qa/clean_train.jsonl \
    --output data/train/hotpot_qa/clean_train_tokens_16000_32000.jsonl \
    --min_tokens 16000 \
    --max_tokens 32000
```

#### 4. Format for GRPO

```bash
python src/amzn_long_context_rag/preprocessing/training_data/format_grpo_training.py \
    --input data/train/hotpot_qa/clean_train_tokens_16000_32000.jsonl \
    --output data/train/hotpot_qa/grpo/
```

### RAG Index Creation

#### Index Long-Context Benchmarks

```bash
python src/amzn_long_context_rag/preprocessing/long_context/long_context_indexing.py \
    --benchmark InfiniteBench \
    --split longbook_qa_eng \
    --embedding_model Qwen/Qwen3-Embedding-4B \
    --output_dir data/indexes
```

#### Index Wikipedia

```bash
# Preprocess Wikipedia dump
python src/amzn_long_context_rag/preprocessing/wikipedia/wikipedia_preprocessing.py \
    --input wikipedia_dump.jsonl \
    --output data/wikipedia/20231101.en/passages_words=100.jsonl \
    --passage_length 100

# Create FAISS index
python src/amzn_long_context_rag/preprocessing/wikipedia/wikipedia_indexing.py \
    --input data/wikipedia/20231101.en/passages_words=100.jsonl \
    --output data/indexes/wikipedia/20231101.en/ \
    --embedding_model Qwen/Qwen3-Embedding-4B
```

#### Offline Retrieval

Pre-compute retrieval results for faster inference:

```bash
python src/amzn_long_context_rag/preprocessing/long_context/long_context_retrieval.py \
    --benchmark InfiniteBench \
    --split longbook_qa_eng \
    --index_dir data/indexes \
    --output data/retriever/outputs/InfiniteBench/ \
    --top_k 10
```

---

## Analysis

### Attention Analysis

Analyze model attention patterns to understand document relevance.

### Available Analysis Scripts

Located in `src/amzn_long_context_rag/analysis/`:

1. **helmet_ndcg_docs_attn_analysis.py**: NDCG@K for document ranking
2. **helmet_topk_docs_attn_analysis.py**: Top-K document attention
3. **helmet_cumulative_step_attn_analysis.py**: Cumulative attention over generation
4. **token_attn_analysis.py**: Token-level attention
5. **sentence_attn_analysis.py**: Sentence-level attention
6. **chunk_attn_analysis.py**: Chunk-level attention

### Running Attention Analysis

```bash
python src/amzn_long_context_rag/analysis/helmet_ndcg_docs_attn_analysis.py \
    --config_path config/analysis/HELMET/Qwen2.5-7B-Instruct-1M/zero_shot_attn.yaml \
    attention_strategy=all_layers \
    ndcg_k=10
```

### Analysis Parameters

- `attention_strategy`: `last_layer`, `all_layers`, `last_n`, `specific_layer`
- `layer_idx`: Layer index for `specific_layer` strategy
- `last_n`: Number of last layers for `last_n` strategy
- `ndcg_k`: K value for NDCG@K metric

### Analysis Outputs

Results saved to `data/analysis/<BENCHMARK>/<SPLIT>/<MODEL>/<SETTING>_analysis.jsonl`:

```json
{
  "id": "instance_id",
  "attention_ranking": [["[DOC 0]", 0.85], ["[DOC 1]", 0.72]],
  "gold_ranking": [["[DOC 0]", 1.0], ["[DOC 1]", 0.8]],
  "metrics": {
    "ndcg@k": 0.92,
    "precision@k": 0.8,
    "recall@k": 0.9,
    "mrr": 1.0
  }
}
```

---

## Advanced Topics

### vLLM Server Inference

For high-throughput inference, use vLLM server mode:

#### 1. Start vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-1M \
    --max-model-len 1010000 \
    --tensor-parallel-size 2
```

#### 2. Run Inference Against Server

```bash
python src/amzn_long_context_rag/inference/vllm_server_inference.py \
    --config_path config/server_inference/server.yaml
```

### Custom Prompts

Prompts are defined in JSON files under `prompts/`. Example:

```json
{
  "system_instruction": "You are an expert assistant.",
  "user_content_template": "Context: {DOC}\n\nQuestion: {Q}\n\nAnswer:"
}
```

### Multi-GPU Inference

The inference script automatically detects available GPUs and configures tensor parallelism:

```yaml
vllm_params:
  tensor_parallel_size: 2      # Use 2 GPUs
  pipeline_parallel_size: 1    # Pipeline parallelism
```

### LoRA Inference

To run inference with LoRA adapters:

```yaml
vllm_params:
  enable_lora: true
  lora_adapter_path: /path/to/lora_adapter
  adapter_name: my_lora
  adapter_id: 1
```

### Context Length Experiments

Run experiments with different context lengths:

```bash
# Inference with context limit
bash scripts/inference/LongBench-v2/Qwen2.5-7B-Instruct-1M/zero_shot.sh \
    context_max_tokens=50000

# Evaluate by context length bins
python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task longbook_qa_eng \
    --file data/outputs/InfiniteBench/test/model/zero_shot.jsonl \
    --maximum_input_length 0 50000

python src/amzn_long_context_rag/evaluation/infinitebench_eval.py \
    --task longbook_qa_eng \
    --file data/outputs/InfiniteBench/test/model/zero_shot.jsonl \
    --maximum_input_length 50000 100000
```

---

## Troubleshooting

### Common Issues

**Out of Memory:**
- Reduce `vllm_params.gpu_memory_utilization`
- Increase `vllm_params.tensor_parallel_size`
- Enable `vllm_params.enforce_eager=true`

**Slow Inference:**
- Increase `vllm_params.max_num_batched_tokens`
- Disable `vllm_params.enforce_eager`
- Use vLLM server mode

**CUDA Errors:**
- Ensure flash-attention is properly installed
- Check CUDA version compatibility
- Try `vllm_params.enforce_eager=true`

---

## Citation

If you use the resources presented in this repository, please cite:

```
@inproceedings{Molfese:2026:EACL,
  title        = "Exploring Fine-Tuning for In-Context Retrieval and Efficient {KV}-Caching in Long-Context Language Models",
  author       = "Molfese, Francesco Maria and Hardalov, Momchil and Blloshmi, Rexhina and Byrne, Bill and de Gispert, Adri{\`a}",
  year         = 2026,
  booktitle    = "Proceedings of the 19th Conference of the {E}uropean Chapter of the Association for Computational Linguistics",
  publisher    = "Association for Computational Linguistics",
  series       = "EACL ’26",
  url          = "https://arxiv.org/abs/2601.18527"
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the [CC-BY-NC-4.0](LICENSE) License.
