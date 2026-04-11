export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

math_train_path=./data/math/train.parquet
math_test_path=./data/math/test.parquet
aime2025_test_path=./data/aime2025/test.parquet
amc23_test_path=./data/amc23/test.parquet

train_files="['$math_train_path']"
test_files="['$math_test_path', '$aime2025_test_path', '$amc23_test_path']"

advantage="positive"
kl_coef=0.0
lr=1e-6
model_name=Qwen/Qwen2.5-Math-1.5B
model_dtype=bf16

# vllm currently fails in this environment due to a torch/vllm ABI mismatch.
# Use HF rollout by default; set ROLLOUT_BACKEND=vllm once vllm is rebuilt
# against the current torch installation.
rollout_backend=${ROLLOUT_BACKEND:-hf}
rollout_top_k=0
if [ "$rollout_backend" = "vllm" ]; then
    rollout_top_k=-1
fi

# Auto-detect visible GPUs (can still be overridden with N_GPUS_PER_NODE).
if command -v nvidia-smi >/dev/null 2>&1; then
    detected_gpus=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    detected_gpus=0
fi
n_gpus_per_node=${N_GPUS_PER_NODE:-$detected_gpus}

if [ "${n_gpus_per_node}" -lt 1 ]; then
    echo "No GPU detected by this shell (nvidia-smi)."
    echo "Ray/VERL requires at least one visible GPU for this FSDP+FlashAttention setup."
    echo "Set CUDA visibility correctly, then rerun. Example:"
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_qwen2.5-math-7b_psr_nsr.sh"
    exit 1
fi

# Default to console logging to avoid mandatory wandb auth.
# Set USE_WANDB=1 to enable wandb logging.
trainer_logger="['console']"
if [ "${USE_WANDB:-0}" = "1" ]; then
    trainer_logger="['wandb']"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=psr_nsr \
    algorithm.advantage=$advantage \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20000 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=$model_dtype \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$rollout_backend \
    actor_rollout_ref.rollout.top_k=$rollout_top_k \
    actor_rollout_ref.rollout.dtype=$model_dtype \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.experiment_name="MATH-Qwen2.5-Math-7B-$advantage" \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger="$trainer_logger" \
    trainer.project_name='verl' \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    +trainer.val_before_train=True \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=10