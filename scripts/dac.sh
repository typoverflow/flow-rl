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
	"antmaze-umaze-v0"
    "antmaze-umaze-diverse-v0"
	"antmaze-medium-play-v0"
	"antmaze-medium-diverse-v0"
	"antmaze-large-play-v0"
	"antmaze-large-diverse-v0"
)
SEEDS=(0 1 2 3 4)
PARALLEL=${PARALLEL:-4}

SHARED_ARGS=(
    "algo=dac"
    "log.project=flow-rl"
    "log.entity=lamda-rl"
    "log.tag=default"
)

MUJOCO_ARGS=(
    "algo.eta=1.0"
    "algo.eta_lr=0.001"
)

ANTMAZE_ARGS=(
    "algo.eta=0.1"
    "algo.eta_lr=0.0"
    "eval.num_episodes=100"
)


declare -A TASK_ARGS
TASK_ARGS=(
    # mujoco
    ["halfcheetah-medium-v2"]="${MUJOCO_ARGS[@]} algo.eta_threshold=1.0 algo.critic.rho=0.0"
    ["halfcheetah-medium-replay-v2"]="${MUJOCO_ARGS[@]} algo.eta_threshold=1.0 algo.critic.rho=0.0"
    ["halfcheetah-medium-expert-v2"]="${MUJOCO_ARGS[@]} algo.eta_threshold=0.1 algo.critic.rho=0.0"
    ["hopper-medium-v2"]="${MUJOCO_ARGS[@]} algo.eta_threshold=1.0 algo.critic.rho=1.5"
    ["hopper-medium-replay-v2"]="${MUJOCO_ARGS[@]} algo.eta_threshold=1.0 algo.critic.rho=1.5"
    ["hopper-medium-expert-v2"]="${MUJOCO_ARGS[@]} algo.eta_threshold=0.05 algo.critic.rho=1.5"
    ["walker2d-medium-v2"]="${MUJOCO_ARGS[@]} algo.eta_threshold=1.0 algo.critic.rho=1.0"
    ["walker2d-medium-replay-v2"]="${MUJOCO_ARGS[@]} algo.eta_threshold=1.0 algo.critic.rho=1.0"
    ["walker2d-medium-expert-v2"]="${MUJOCO_ARGS[@]} algo.eta_threshold=1.0 algo.critic.rho=1.0"
    # antmaze
	["antmaze-umaze-v0"]="${ANTMAZE_ARGS[@]} algo.critic.rho=1.0"
    ["antmaze-umaze-diverse-v0"]="${ANTMAZE_ARGS[@]} algo.critic.rho=1.0"
	["antmaze-medium-play-v0"]="${ANTMAZE_ARGS[@]} algo.critic.rho=1.0"
	["antmaze-medium-diverse-v0"]="${ANTMAZE_ARGS[@]} algo.critic.rho=1.0"
	["antmaze-large-play-v0"]="${ANTMAZE_ARGS[@]} algo.critic.rho=1.1"
	["antmaze-large-diverse-v0"]="${ANTMAZE_ARGS[@]} algo.critic.rho=1.0"
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
