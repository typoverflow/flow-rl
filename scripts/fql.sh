export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'

TASKS=( # NOTE: FQL's paper did not report results on mujoco datasets, so we skip them
    # antmaze
	"antmaze-umaze-v2"  # NOTE: fql uses -v2 antmaze datasets
    "antmaze-umaze-diverse-v2"
	"antmaze-medium-play-v2"
	"antmaze-medium-diverse-v2"
	"antmaze-large-play-v2"
	"antmaze-large-diverse-v2"
    # TODO: pen, door, hammer, relocate
)
SEEDS=(0 1 2 3 4)
PARALLEL=${PARALLEL:-4}

SHARED_ARGS=(
    "algo=fql"
    "log.project=flow-rl"
    "log.entity=lamda-rl"
    "log.tag=default"
)

MUJOCO_ARGS=(
    "algo.actor_dropout=0"
    "algo.value_dropout=0"
    "algo.alpha=1.0"
)

declare -A TASK_ARGS
TASK_ARGS=(
    # antmaze
	["antmaze-umaze-v2"]="algo.alpha=10"
    ["antmaze-umaze-diverse-v2"]="algo.alpha=10"
	["antmaze-medium-play-v2"]="algo.alpha=10"
	["antmaze-medium-diverse-v2"]="algo.alpha=10"
	["antmaze-large-play-v2"]="algo.alpha=3"
	["antmaze-large-diverse-v2"]="algo.alpha=3"
)

run_task() {
    task=$1
    seed=$2
    slot=$3
    device=$((slot % 1))
    echo "Running $env $level $seed on GPU $device"
    command="python examples/main_d4rl.py task=$task device=$device seed=$seed ${SHARED_ARGS[@]} ${TASK_ARGS[$task]}"
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
