# Specify which GPUs to use
GPUS=(0 1 2 3 4 5 6 7)  # Modify this array to specify which GPUs to use
SEEDS=(0 1 2 3 4)
NUM_EACH_GPU=3 # think we should increase this?

PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

TASKS=(
    "h1hand-walk-v0"
    "h1hand-reach-v0"
    "h1hand-hurdle-v0"
    "h1hand-crawl-v0"
    "h1hand-maze-v0"
    "h1hand-push-v0"
    "h1hand-cabinet-v0"
    "h1strong-highbar_hard-v0"
    "h1hand-door-v0"
    "h1hand-truck-v0"
    "h1hand-cube-v0"
    "h1hand-bookshelf_simple-v0"
    "h1hand-bookshelf_hard-v0"
    "h1hand-basketball-v0"
    "h1hand-window-v0"
    "h1hand-spoon-v0"
    "h1hand-kitchen-v0"
    "h1hand-package-v0"
    "h1hand-powerlift-v0"
    "h1hand-room-v0"
    "h1hand-stand-v0"
    "h1hand-run-v0"
    "h1hand-sit_simple-v0"
    "h1hand-sit_hard-v0"
    "h1hand-balance_simple-v0"
    "h1hand-balance_hard-v0"
    "h1hand-stair-v0"
    "h1hand-slide-v0"
    "h1hand-pole-v0"
    "h1hand-insert_normal-v0"
    "h1hand-insert_small-v0"
)

SHARED_ARGS=(
    "algo=td3"
    "log.tag=default"
    "log.project=flow-rl"
    "log.entity=lambda-rl"
)

run_task() {
    task=$1
    seed=$2
    slot=$3
    num_gpus=${#GPUS[@]}
    device_idx=$((slot % num_gpus))
    device=${GPUS[$device_idx]}
    echo "Running $env $seed on GPU $device"
    command="python3 examples/online/main_humanoidbench_offpolicy.py task=$task device=$device seed=$seed ${SHARED_ARGS[@]}"
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
