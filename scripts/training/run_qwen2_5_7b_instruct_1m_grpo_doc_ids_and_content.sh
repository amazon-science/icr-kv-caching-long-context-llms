set -x
#!/bin/bash

# Activate the verl enviroment
source code/venvs/verl/bin/activate

export EXPERIMENT_NAME="qwen2_5_7b_instruct_1m_grpo_doc_ids_and_content"
export DATA_DIR="data"
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

hotpot_qa_train_path="$DATA_DIR/train/hotpot_qa/grpo/train_doc_ids_and_content.parquet"
hotpot_qa_test_path="$DATA_DIR/train/hotpot_qa/grpo/test_doc_ids_and_content.parquet"
WikiMultihopQA_train_path="$DATA_DIR/train/2WikiMultihopQA/grpo/train_doc_ids_and_content.parquet"
WikiMultihopQA_test_path="$DATA_DIR/train/2WikiMultihopQA/grpo/test_doc_ids_and_content.parquet"

train_files="['$hotpot_qa_train_path', '$WikiMultihopQA_train_path']"
test_files="['$hotpot_qa_test_path', '$WikiMultihopQA_test_path']"

reward_func_name="my_reward_fn_doc_ids_and_content"
reward_func_path="src/amzn_long_context_rag/training/reward_functions.py"

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

# Synchronize workers
if [ "${HOSTNAME##*-}" -eq 0 ]; then
    ray start --head --port=6379
    until [ "$(ray status | grep node_ | wc -l | awk '{print $1}')" -eq $NNODES ]; do
        echo "waiting for all workers up..."
        sleep 10
    done
else
    HEAD_ADDR="${HOSTNAME%-*}-0"
    HEAD_PORT=6379

    echo "Waiting for head node (${HEAD_ADDR}:${HEAD_PORT}) to become reachable..."
    until (echo > /dev/tcp/${HEAD_ADDR}/${HEAD_PORT}) >/dev/null 2>&1; do
        sleep 5
    done

    echo "Head node is reachable, starting ray worker..."
    ray start --address="${HEAD_ADDR}:${HEAD_PORT}" --block
fi
echo "Ray all worker nodes started"

# Start training from worker 0
if [ "${HOSTNAME##*-}" -eq 0 ]; then
    # Command 1
    echo "Executing command 1 because DIST_NODE_RANK is 0"
    python -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.train_batch_size=256 \
        data.val_batch_size=128 \
        data.max_prompt_length=32256 \
        data.max_response_length=512 \
        data.truncation='middle' \
        data.shuffle=False \
        custom_reward_function.path=$reward_func_path \
        custom_reward_function.name=$reward_func_name \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.actor.strategy=fsdp \
        actor_rollout_ref.model.use_shm=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.enable_activation_offload=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.rollout.load_format=safetensors \
        actor_rollout_ref.rollout.layered_summon=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.default_local_dir="$OUTPUT_DIR" \
        trainer.critic_warmup=0 \
        trainer.logger='["console", "wandb"]' \
        trainer.project_name='icr_kv_grpo_debugging' \
        trainer.experiment_name=$EXPERIMENT_NAME-${TIMESTAMP} \
        trainer.n_gpus_per_node=$NPROC_PER_NODE \
        trainer.nnodes=$NNODES \
        trainer.save_freq=60 \
        trainer.test_freq=20 \
        trainer.val_before_train=False \
        trainer.total_epochs=2
else
    sleep infinity
fi

echo ""
echo "Training completed at $(date)!"