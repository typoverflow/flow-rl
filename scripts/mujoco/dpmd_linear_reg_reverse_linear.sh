#!/bin/bash
# Specify which GPUs to use
GPUS=(0 1 2 3)  # Modify this array to specify which GPUs to use
SEEDS=(0 1 2 3)
WEIGHTS_OFFSETS=(-0.2)
NUM_EACH_GPU=1

PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

TASKS=(
    # "Ant-v5"
    # "Walker2d-v5"
    "HalfCheetah-v5"
    # "Swimmer-v5"
    # "Humanoid-v5"
    # "Hopper-v5"
)

SHARED_ARGS=(
    "algo=dpmd_linear_reg_reverse_linear"
    "log.tag=weights_offset_v3_reverse_linear_kl1.5"
    "log.entity=haitongma-harvard-university"
    "log.project=simpo-neurips"
    "algo.target_kl=1.5"
)


run_task() {
    task=$1
    seed=$2
    slot=$3
    weights_offset=$4
    num_gpus=${#GPUS[@]}
    device_idx=$((slot % num_gpus))
    device=${GPUS[$device_idx]}
    echo "Running $task seed=$seed weights_offset=$weights_offset on GPU $device"
    export CUDA_VISIBLE_DEVICES=$device
    export XLA_PYTHON_CLIENT_PREALLOCATE="false"
    command="python3 examples/online/main_mujoco_offpolicy.py task=$task seed=$seed algo.weights_offset=$weights_offset ${SHARED_ARGS[@]}"
    if [ -n "$DRY_RUN" ]; then
        echo $command
    else
        echo $command
        $command
    fi
}

. env_parallel.bash
if [ -n "$DRY_RUN" ]; then
    env_parallel -P${PARALLEL} run_task {1} {2} {%} {3} ::: ${TASKS[@]} ::: ${SEEDS[@]} ::: ${WEIGHTS_OFFSETS[@]}
else
    env_parallel --bar --results log/parallel/$name -P${PARALLEL} run_task {1} {2} {%} {3} ::: ${TASKS[@]} ::: ${SEEDS[@]} ::: ${WEIGHTS_OFFSETS[@]}
fi
