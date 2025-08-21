export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'

TASKS=(
    "hopper-medium-v2"
    "hopper-medium-replay-v2"
    "hopper-medium-expert-v2"
    "walker2d-medium-v2"
    "walker2d-medium-replay-v2"
    "walker2d-medium-expert-v2"
    "halfcheetah-medium-v2"
    "halfcheetah-medium-replay-v2"
    "halfcheetah-medium-expert-v2"
)
SEEDS=(0 1 2 3 4)
PARALLEL=${PARALLEL:-4}

SHARED_ARGS=(
    "algo=iql"
    "log.project=flow-rl"
    "log.entity=lamda-rl"
    "log.tag=default"
)

declare -A TASK_ARGS
TASK_ARGS=(
    ["halfcheetah-medium-v2"]=""
    ["hopper-medium-v2"]=""
    ["walker2d-medium-v2"]=""
    ["halfcheetah-medium-replay-v2"]=""
    ["hopper-medium-replay-v2"]=""
    ["walker2d-medium-replay-v2"]=""
    ["halfcheetah-medium-expert-v2"]=""
    ["hopper-medium-expert-v2"]="algo.expectile=0.5 algo.beta=6.0"
    ["walker2d-medium-expert-v2"]=""
)

run_task() {
    task=$1
    seed=$2
    slot=$3
    device=$((slot % 1))
    echo "Running $env $level $seed on GPU $device"
    command="python examples/offline/main_d4rl.py task=$task device=$device seed=$seed ${SHARED_ARGS[@]} ${TASK_ARGS[$task]}"
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
    env_parallel --bar --results logs/parallel/$name -P${PARALLEL} run_task {1} {2} {%} ::: ${TASKS[@]} ::: ${SEEDS[@]}
fi
