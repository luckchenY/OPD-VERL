#!/bin/bash
set -e

cd /data/chenyang/OPD
export PYTHONPATH=/data/chenyang/OPD/verl
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true

# Isolated Ray session for eval — do not attach to other jobs' global cluster.
RAY_SESSION_DIR="/tmp/ray_eval_${CUDA_VISIBLE_DEVICES//,/_}"
export RAY_TMPDIR="$RAY_SESSION_DIR"
mkdir -p "$RAY_SESSION_DIR"
unset RAY_ADDRESS

# Only stop Ray/vLLM processes on our visible GPU(s), never the whole-machine cluster.
stop_ray_on_visible_gpus() {
    if [[ -f "$RAY_SESSION_DIR/ray_current_cluster" ]]; then
        RAY_TMPDIR="$RAY_SESSION_DIR" ray stop --force 2>/dev/null || true
    fi

    local gpu pid args
    IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
    for gpu in "${GPU_LIST[@]}"; do
        gpu="${gpu// /}"
        [[ -z "$gpu" ]] && continue
        while IFS= read -r pid; do
            pid="${pid// /}"
            [[ "$pid" =~ ^[0-9]+$ ]] || continue
            args=$(ps -p "$pid" -o args= 2>/dev/null || true)
            if [[ "$args" == *ray* || "$args" == *vllm* || "$args" == *VLLM* ]]; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        done < <(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    done
}
stop_ray_on_visible_gpus

MODEL="${1:-/data/chenyang/OPD/checkpoint/token_reward_direct_OpenThoughts3_DeepSeek-R1-Distill-Qwen-1.5B_Skywork-OR1-7B_7168-T_1.0-Tch_1.0-n_4-mbs_4-topk_16-topk_strategy_only_stu-rw_student_p-2026-06-27_02-14-07/global_step_550/actor/merged_hf_550}"
BENCH="${2:-AIME24}"   # AIME24 / AIME25 / AMC23
STEP="${3:-1250}"

OUT="eval_outputs/${BENCH}_${STEP}.parquet"
mkdir -p eval_outputs

echo "=========================================="
echo "Model : $MODEL"
echo "Bench : $BENCH"
echo "GPU   : $CUDA_VISIBLE_DEVICES"
echo "Out   : $OUT"
echo "=========================================="
echo "Step 1/2: generation (Ray + vLLM init may take 2-5 min with no new logs)"

/home/chenyang2/miniconda3/envs/verl/bin/python -m verl.trainer.main_generation \
    "+ray_kwargs.ray_init._temp_dir=$RAY_SESSION_DIR" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=datasets/test_data/${BENCH}/test.parquet \
    data.prompt_key=prompt \
    data.output_path=$OUT \
    data.batch_size=4 \
    data.n_samples=1 \
    model.path="$MODEL" \
    rollout.name=vllm \
    rollout.temperature=0.6 \
    rollout.top_p=0.95 \
    rollout.prompt_length=1024 \
    rollout.response_length=32768 \
    rollout.max_model_len=32768 \
    rollout.tensor_model_parallel_size=1 \
    +rollout.pipeline_model_parallel_size=1 \
    rollout.dtype=bfloat16 \
    rollout.gpu_memory_utilization=0.8 \
    rollout.max_num_batched_tokens=32768 \
    rollout.enforce_eager=True \
    rollout.load_format=safetensors

echo "Step 2/2: scoring"
/home/chenyang2/miniconda3/envs/verl/bin/python -m verl.trainer.main_eval \
    data.path=$OUT \
    data.response_key=responses \
    data.data_source_key=data_source \
    data.reward_model_key=reward_model \
    custom_reward_function.path=verl/verl/utils/reward_score/ttrl_math/__init__.py \
    custom_reward_function.name=reward_func_eval

echo "Done: $OUT"

#prompt_length是对prompt做截断的
# responselength是VLLM最多生成多少token
# max_model_len是VLLM的上下文窗口