export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
# Specify which GPUs to use
GPUS=(0 1 2 3 4 5 6 7)  # Modify this array to specify which GPUs to use
SEEDS=(0 1 2 3 4)
NUM_EACH_GPU=3

PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))
TASKS=(
    # mujoco
    # "hopper-medium-v2"
    # "hopper-medium-replay-v2"
    # "hopper-medium-expert-v2"
    # "walker2d-medium-v2"
    # "walker2d-medium-replay-v2"
    # "walker2d-medium-expert-v2"
    # "halfcheetah-medium-v2"
    # "halfcheetah-medium-replay-v2"
    # "halfcheetah-medium-expert-v2"
    # # antmaze
	# "antmaze-umaze-v0"
    # "antmaze-umaze-diverse-v0"
	# "antmaze-medium-play-v0"
	# "antmaze-medium-diverse-v0"
	# "antmaze-large-play-v0"
	"antmaze-large-diverse-v0"
)


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
    "algo.critic.maxQ=true"
    "algo.eta=0.1"
    "algo.eta_lr=0.0"
    "algo.critic.rho=1.0"
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
	["antmaze-umaze-v0"]="${ANTMAZE_ARGS[@]}"
    ["antmaze-umaze-diverse-v0"]="${ANTMAZE_ARGS[@]}"
	["antmaze-medium-play-v0"]="${ANTMAZE_ARGS[@]}"
	["antmaze-medium-diverse-v0"]="${ANTMAZE_ARGS[@]}"
	["antmaze-large-play-v0"]="${ANTMAZE_ARGS[@]} algo.critic.rho=1.1 algo.critic.lr=0.001"
	["antmaze-large-diverse-v0"]="${ANTMAZE_ARGS[@]} algo.critic.rho=1.1 algo.critic.lr=0.001"
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
    command="python3 examples/offline/main_d4rl.py task=$task device=$device seed=$seed ${SHARED_ARGS[@]} ${TASK_ARGS[$task]}"
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
