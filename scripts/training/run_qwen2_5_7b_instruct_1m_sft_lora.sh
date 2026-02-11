set -x
#!/bin/bash

# Activate the verl enviroment
source code/venvs/verl/bin/activate

export EXPERIMENT_NAME="qwen2_5_7b_instruct_1m_sft_lora"
export DATA_DIR="code/verl/data"
export OUTPUT_DIR="data/outputs/${EXPERIMENT_NAME}"
export LOG_DIR="data/logs/${EXPERIMENT_NAME}"
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct-1M"
export NNODES="${REPLICA}"
export NPROC_PER_NODE="${PET_NPROC_PER_NODE}"
export WANDB_MODE="online"
export WANDB_DIR="data/wandb/${EXPERIMENT_NAME}"
export WANDB_API_KEY=<your key here>

mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR
mkdir -p $WANDB_DIR

wandb login

hotpot_qa_train_path="$DATA_DIR/hotpot_qa/sft/train.parquet"
hotpot_qa_test_path="$DATA_DIR/hotpot_qa/sft/test.parquet"
WikiMultihopQA_train_path="$DATA_DIR/2WikiMultihopQA/sft/train.parquet"
WikiMultihopQA_test_path="$DATA_DIR/2WikiMultihopQA/sft/test.parquet"

train_files="['$hotpot_qa_train_path','$WikiMultihopQA_train_path']"
test_files="['$hotpot_qa_test_path','$WikiMultihopQA_test_path']"

NODE_RANK=${REPLICA_IDX:-0}

# Generate timestamp once
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Setup logging
LOG_FILE="$LOG_DIR/node_${NODE_RANK}_training_${TIMESTAMP}.log"
echo "🚀 Node $NODE_RANK - Logging to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# Updates
apt update
apt-get install -y software-properties-common python3-dev cuda-minimal-build-12-5=12.5.1-1

# trainer.nnodes=$NNODES \
# trainer.n_gpus_per_node=$NPROC_PER_NODE \

torchrun --standalone --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=32768 \
    data.truncation=left \
    optim.lr=1e-6 \
    optim.warmup_steps_ratio=0.1 \
    optim.weight_decay=0.01 \
    optim.clip_grad=1.0 \
    model.partial_pretrain=$BASE_MODEL \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    model.use_liger=true \
    model.lora_rank=64 \
    model.lora_alpha=32 \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.project_name=icr_kv_sft_debugging \
    trainer.experiment_name=$EXPERIMENT_NAME-${TIMESTAMP} \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=1 \
    trainer.save_freq=70 \
    trainer.test_freq=30 \
    trainer.seed=42 \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true \

echo ""
echo "Training completed at $(date)!"