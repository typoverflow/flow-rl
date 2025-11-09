export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
# Specify which GPUs to use
GPUS=(0)  # Modify this array to specify which GPUs to use
SEEDS=(0)
NUM_EACH_GPU=1

PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

TASKS=(
    "8gaussians"
    "2spirals"
    "checkerboard"
    "moons"
    "rings"
    "swissroll"
)


SHARED_ARGS=(
    "algo=dac"
    "log.tag=toy2d"
    "log.project=flow-rl"
    "log.entity=lamda-rl"
)


run_task() {
    task=$1
    seed=$2
    slot=$3
    # Calculate device index based on available GPUs
    num_gpus=${#GPUS[@]}
    device_idx=$((slot % num_gpus))
    device=${GPUS[$device_idx]}
    echo "Running $task $seed on GPU $device"
    command="python3 examples/offline/main_toy2d.py task=$task device=$device seed=$seed ${SHARED_ARGS[@]}"
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
    env_parallel --bar --results logs/parallel/dac-toy2d -P${PARALLEL} run_task {1} {2} {%} ::: ${TASKS[@]} ::: ${SEEDS[@]}
fi
