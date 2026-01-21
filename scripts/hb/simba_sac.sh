# Specify which GPUs to use
GPUS=(0 1 2 3 4 5 6 7)  # Modify this array to specify which GPUs to use
SEEDS=(0 1)
NUM_EACH_GPU=3

PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

TASKS=(
    "h1-walk-v0"
    "h1-stand-v0"
    "h1-run-v0"
    "h1-crawl-v0"
    # "h1-stair-v0"
    # "h1-pole-v0"
    # "h1-sit-v0"
    # "h1-hurry-v0"
    # "h1-maze-v0"
    # "h1hand-walk-v0"
    # "h1hand-stand-v0"
    # "h1hand-run-v0"
    # "h1hand-stair-v0"
    # "h1hand-crawl-v0"
    # "h1hand-pole-v0"
    # "h1hand-slide-v0"
    # "h1hand-hurdle-v0"
    # "h1hand-maze-v0"
    # "h1hand-sit-simple-v0"
    # "h1hand-sit-hard-v0"
    # "h1hand-balance-simple-v0"
    # "h1hand-balance-hard-v0"
    # "h1hand-reach-v0"
    # "h1hand-spoon-v0"
    # "h1hand-window-v0"
    # "h1hand-insert-small-v0"
    # "h1hand-insert-normal-v0"
    # "h1hand-bookshelf-simple-v0"
    # "h1hand-bookshelf-hard-v0"
)

SHARED_ARGS=(
    "algo=simba_sac"
    "log.tag=default"
    "log.project=flow-rl"
    "log.entity=lamda-rl"
)

run_task() {
    task=$1
    seed=$2
    slot=$3
    num_gpus=${#GPUS[@]}
    device_idx=$((slot % num_gpus))
    device=${GPUS[$device_idx]}
    echo "Running $env $seed on GPU $device"
    export MUJOCO_EGL_DEVICE_ID=$device
    export CUDA_VISIBLE_DEVICES=$device
    export XLA_PYTHON_CLIENT_PREALLOCATE="false"
    command="python3 examples/online/main_hb_offpolicy.py task=$task seed=$seed ${SHARED_ARGS[@]}"
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
