#!/bin/bash
# Entropy-Aware On-Policy Distillation (EOPD)
# Implements: arXiv:2603.07079 "Entropy-Aware On-Policy Distillation of Language Models"
#
# Core idea: augment the standard OPD reverse-KL objective with a forward-KL
# term on tokens where the teacher distribution has high entropy:
#   L_EOPD_t = L_OPD_t + alpha * I[H_te_t > tau] * L_FKL_t
# where L_FKL is approximated over the teacher's top-k tokens (renormalized).
# Default hyperparameters from the paper: tau=0.8, alpha=1.0, k=16.

set -x
ray stop --force
# Configure logging when running outside SBATCH.
if [ -z "$SLURM_JOB_ID" ]; then
    LOG_DIR=${LOG_DIR:-logs}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo "Start time: $(date)"
    echo "=========================================="
fi

export RAY_memory_usage_threshold=0.99
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,6,7}
export PYTHONUNBUFFERED=1
# Force HuggingFace / transformers to use local files only (no network access).
# All models and datasets in this script are local paths, so offline mode is safe
# and avoids LocalEntryNotFoundError when the Hub is unreachable.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=true
export PROJECT_NAME='OPDqwen2.5distill-eopd'
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT_SECONDS=7200
export NCCL_TIMEOUT=7200
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export ADV_ESTIMATOR=token_reward_direct

# ---- EOPD hyperparameters (paper defaults) ----
export EOPD_ENABLE=${EOPD_ENABLE:-True}
export EOPD_ENTROPY_THRESHOLD=${EOPD_ENTROPY_THRESHOLD:-0.8}   # tau
export EOPD_FKL_COEF=${EOPD_FKL_COEF:-1.0}                      # alpha
# k for forward KL == teacher top-k used by the OPD rollout (LOG_PROB_TOP_K)

# DeepMath-103K / DAPO-Math-17k
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-7168}
export MAX_MODEL_LEN=$((MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ))
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-66}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
export N_RESPONSES=${N_RESPONSES:-4}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}  # paper: k=16
export ROLLOUT_CALCULATE_LOG_PROBS=${ROLLOUT_CALCULATE_LOG_PROBS:-False}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-"only_stu"}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-"student_p"}
export OPD_CONSISTENCY_ENABLE=${OPD_CONSISTENCY_ENABLE:-False}
export OPD_CONSISTENCY_TOP_PERCENT_RESPONSES=${OPD_CONSISTENCY_TOP_PERCENT_RESPONSES:-100}
export OPD_CONSISTENCY_MASK_TOP_PERCENT_SEGMENTS=${OPD_CONSISTENCY_MASK_TOP_PERCENT_SEGMENTS:-30}
export OPD_CONSISTENCY_MIN_SEGMENTS=${OPD_CONSISTENCY_MIN_SEGMENTS:-3}
export OPD_SEGMENT_MIN_SENTENCES=${OPD_SEGMENT_MIN_SENTENCES:-3}
export OPD_CONSISTENCY_DROP_LOW=${OPD_CONSISTENCY_DROP_LOW:-True}
export OPD_PROCESS_REWARD_ENABLE=${OPD_PROCESS_REWARD_ENABLE:-$OPD_CONSISTENCY_ENABLE}
export OPD_PROCESS_REWARD_K_EVAL=${OPD_PROCESS_REWARD_K_EVAL:-8}
export OPD_PROCESS_REWARD_TEMPERATURE=${OPD_PROCESS_REWARD_TEMPERATURE:-0.7}
export OPD_PROCESS_REWARD_TOP_K=${OPD_PROCESS_REWARD_TOP_K:-50}
export OPD_PROCESS_REWARD_TOP_P=${OPD_PROCESS_REWARD_TOP_P:-1.0}
export OPD_PROCESS_REWARD_MAX_TOKENS=${OPD_PROCESS_REWARD_MAX_TOKENS:-300}
export OPD_PROCESS_REWARD_MAX_PROMPT_LENGTH=${OPD_PROCESS_REWARD_MAX_PROMPT_LENGTH:-8192}
export OPD_PROCESS_REWARD_MAX_TOTAL_LENGTH=${OPD_PROCESS_REWARD_MAX_TOTAL_LENGTH:-8192}
export OPD_PROCESS_REWARD_BATCH_SIZE=${OPD_PROCESS_REWARD_BATCH_SIZE:-64}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.3}
export ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export USE_KL=${USE_KL:-False}
export ENABLE_FORMAT_REWARD=${ENABLE_FORMAT_REWARD:-False}
export REWARD_MODEL_PARAM_OFFLOAD=${REWARD_MODEL_PARAM_OFFLOAD:-False}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
export MODEL_DTYPE=${MODEL_DTYPE:-bf16}
export IS_PLOT=${IS_PLOT:-False}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-"token-mean"}

export PROJECT_ROOT=/data/chenyang/OPD
export TRAIN_DATASET=$PROJECT_ROOT/datasets/dapo-math-17k-processed.parquet
export TEST_DATA_DIR=$PROJECT_ROOT/datasets/test_data
export PROJECT_PATH=$PROJECT_ROOT/checkpoint
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}

TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AIME25/test.parquet", "$TEST_DATA_DIR/AMC23/test.parquet", "$TEST_DATA_DIR/AIME24/test.parquet"]}

export ACTOR_MODEL_PATH=/data/chenyang/models/DeepSeek-R1-Distill-Qwen-1.5B
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
export REWARD_MODEL_PATH=/data/chenyang/models/JustRL-1.5B
export REWARD_MODEL_NAME=$(basename "$REWARD_MODEL_PATH")

export PROJECT_PATH=/data/chenyang/OPD/checkpoint
export PARALLEL_SIZE=1
_RUN_NAME_SUFFIX=eopd_${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}-tau_${EOPD_ENTROPY_THRESHOLD}-alpha_${EOPD_FKL_COEF}
export RESUME_MODE=${RESUME_MODE:-disable}

if [ -n "$RESUME_CKPT_DIR" ]; then
    export CKPT_PATH="$RESUME_CKPT_DIR"
    export EXPERIMENT_NAME=$(basename "$RESUME_CKPT_DIR")
    export RESUME_MODE=${RESUME_MODE:-resume_path}
    export RESUME_FROM_PATH="$RESUME_CKPT_DIR"
    echo "Resuming training from: $CKPT_PATH"
else
    _TS=$(date +%Y-%m-%d_%H-%M-%S)
    export CKPT_PATH=${PROJECT_PATH}/${_RUN_NAME_SUFFIX}-${_TS}
    export EXPERIMENT_NAME=${_RUN_NAME_SUFFIX}-${_TS}
    echo "Starting new EOPD training run: $CKPT_PATH"
fi

RESUME_ARGS="trainer.resume_mode=$RESUME_MODE"
if [ "$RESUME_MODE" = "resume_path" ]; then
    RESUME_ARGS="$RESUME_ARGS trainer.resume_from_path=$RESUME_FROM_PATH"
fi
export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)
export NCCL_DEBUG=WARN
export SWANLAB_LOG_DIR=${PROJECT_PATH}/swanlab_log
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="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"

KL_ARGS=""
if [ "$USE_KL" = "True" ]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi

LR_ARGS=""
if [ "$LR_SCHEDULER" = "cosine" ]; then
    LR_ARGS="actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03"
fi

export PPO_MAX_TOKEN_LEN_PER_GPU=8192
echo "PPO_MAX_TOKEN_LEN_PER_GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"

ray start
sleep 5

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.grpo_outcome_weight=$GRPO_OUTCOME_WEIGHT \
    data.shuffle=False \
    data.train_files="$TRAIN_DATASET" \
    data.train_max_samples=$TRAIN_MAX_SAMPLES \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size=$((${MINI_BATCH_SIZE}*${PARALLEL_SIZE})) \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.nccl_timeout=7200 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    $LR_ARGS \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    $KL_ARGS \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.actor.eopd_enable=$EOPD_ENABLE \
    actor_rollout_ref.actor.eopd_entropy_threshold=$EOPD_ENTROPY_THRESHOLD \
    actor_rollout_ref.actor.eopd_fkl_coef=$EOPD_FKL_COEF \
    actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    +actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY \
    +actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE \
    +actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_VAL_RESP_LENGTH \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.repetition_penalty=$REPETITION_PENALTY \
    actor_rollout_ref.rollout.calculate_log_probs=$ROLLOUT_CALCULATE_LOG_PROBS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    +reward_model.nccl_timeout=7200 \
    reward_model.enable=True \
    +reward_model.reward_kwargs.enable_format_reward=$ENABLE_FORMAT_REWARD \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=$REWARD_MODEL_PARAM_OFFLOAD \
    +reward_model.model.dtype=$MODEL_DTYPE \
    reward_model.micro_batch_size_per_gpu=1 \
    custom_reward_function.path="/data/chenyang/OPD/verl/verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.log_val_generations=6 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    $RESUME_ARGS \
    trainer.validation_data_dir=validation_log/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot=$IS_PLOT

if [ -z "$SLURM_JOB_ID" ]; then
    echo "=========================================="
    echo "End time: $(date)"
    echo "=========================================="
fi
