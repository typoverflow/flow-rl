export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'

TASKS=(
    # mujoco
    "hopper-medium-v2"
    "hopper-medium-replay-v2"
    "hopper-medium-expert-v2"
    "walker2d-medium-v2"
    "walker2d-medium-replay-v2"
    "walker2d-medium-expert-v2"
    "halfcheetah-medium-v2"
    "halfcheetah-medium-replay-v2"
    "halfcheetah-medium-expert-v2"
    # antmaze
	"antmaze-umaze-v2"  # NOTE: it seems ivr uses -v2 antmaze datasets
    "antmaze-umaze-diverse-v2"
	"antmaze-medium-play-v2"
	"antmaze-medium-diverse-v2"
	"antmaze-large-play-v2"
	"antmaze-large-diverse-v2"
    # kitchen
	"kitchen-complete-v0"
	"kitchen-partial-v0"
	"kitchen-mixed-v0"
)
SEEDS=(0 1 2 3 4)
PARALLEL=${PARALLEL:-4}

SHARED_ARGS=(
    "algo=ivr"
    "algo.method=eql"
    "log.project=flow-rl"
    "log.entity=lamda-rl"
    "log.tag=default"
)

MUJOCO_ARGS=(
    "algo.actor_dropout=0"
    "algo.value_dropout=0"
    "algo.alpha=2.0"
)

ANTMAZE_ARGS=(
    "algo.actor_lr=2e-4"
    "algo.value_lr=2e-4"
    "algo.critic_lr=2e-4"
    "algo.actor_dropout=0"
    "algo.value_dropout=0.5"
    "data.norm_reward=iql_antmaze"
    "eval.num_episodes=100"
    "algo.alpha=0.5"
)

KITCHEN_ARGS=(
    "algo.actor_dropout=0.1"
    "algo.value_dropout=0"
    "algo.alpha=2.0"
)

declare -A TASK_ARGS
TASK_ARGS=(
    # mujoco
    ["halfcheetah-medium-v2"]="${MUJOCO_ARGS[@]}"
    ["hopper-medium-v2"]="${MUJOCO_ARGS[@]}"
    ["walker2d-medium-v2"]="${MUJOCO_ARGS[@]}"
    ["halfcheetah-medium-replay-v2"]="${MUJOCO_ARGS[@]}"
    ["hopper-medium-replay-v2"]="${MUJOCO_ARGS[@]}"
    ["walker2d-medium-replay-v2"]="${MUJOCO_ARGS[@]}"
    ["halfcheetah-medium-expert-v2"]="${MUJOCO_ARGS[@]}"
    ["hopper-medium-expert-v2"]="${MUJOCO_ARGS[@]}"
    ["walker2d-medium-expert-v2"]="${MUJOCO_ARGS[@]}"
    # antmaze
	["antmaze-umaze-v2"]="${ANTMAZE_ARGS[@]}"
    ["antmaze-umaze-diverse-v2"]="${ANTMAZE_ARGS[@]} algo.alpha=5.0"
	["antmaze-medium-play-v2"]="${ANTMAZE_ARGS[@]}"
	["antmaze-medium-diverse-v2"]="${ANTMAZE_ARGS[@]}"
	["antmaze-large-play-v2"]="${ANTMAZE_ARGS[@]}"
	["antmaze-large-diverse-v2"]="${ANTMAZE_ARGS[@]}"
    # kitchen
	["kitchen-complete-v0"]="${KITCHEN_ARGS[@]}"
	["kitchen-partial-v0"]="${KITCHEN_ARGS[@]}"
	["kitchen-mixed-v0"]="${KITCHEN_ARGS[@]}"
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
