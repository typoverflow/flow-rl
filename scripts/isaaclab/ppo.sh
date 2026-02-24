# Specify which GPUs to use
GPUS=(0 1 2 3 4 5 6 7)  # Modify this array to specify which GPUs to use
SEEDS=(0 1)
NUM_EACH_GPU=1

PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

TASKS=(
    "Isaac-Velocity-Flat-Anymal-D-v0"
    "Isaac-Velocity-Flat-G1-v0"
)

SHARED_ARGS=(
    "algo=ppo"
    "log.tag=default"
)

run_task() {
    task=$1
    seed=$2
    slot=$3
    num_gpus=${#GPUS[@]}
    device_idx=$((slot % num_gpus))
    device=${GPUS[$device_idx]}
    echo "Running $task $seed on GPU $device"
    unset CUDA_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=$device
    export XLA_PYTHON_CLIENT_PREALLOCATE="false"
    command="python3 examples/online/main_isaaclab_onpolicy.py task=$task seed=$seed ${SHARED_ARGS[@]}"
    if [ -n "$DRY_RUN" ]; then
        echo $command
    else
        echo $command
        $command
    fi
}

. env_parallel.bash
if [ -n "$DRY_RUN" ]; then
    env_parallel -P${PARALLEL} run_task {1} {2} {%} ::: ${TASKS[@]} ::: ${SEEDS[@]}
else
    env_parallel --bar --results log/parallel/$name -P${PARALLEL} run_task {1} {2} {%} ::: ${TASKS[@]} ::: ${SEEDS[@]}
fi
