export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
# Specify which GPUs to use
GPUS=(0 1 2 3 4 5 6 7)  # Modify this array to specify which GPUs to use
SEEDS=(0 1 2 3 4)
NUM_EACH_GPU=2

PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

TASKS=(
    # locomotion tasks
    "hopper-medium-v2"
    "hopper-medium-replay-v2"
    "hopper-medium-expert-v2"
    "walker2d-medium-v2"
    "walker2d-medium-replay-v2"
    "walker2d-medium-expert-v2"
    "halfcheetah-medium-v2"
    "halfcheetah-medium-replay-v2"
    "halfcheetah-medium-expert-v2"
    # antmaze tasks
    "antmaze-umaze-v0"
    "antmaze-umaze-diverse-v0"
    "antmaze-medium-play-v0"
    "antmaze-medium-diverse-v0"
    "antmaze-large-play-v0"
    "antmaze-large-diverse-v0"
)


SHARED_ARGS=(
    "algo=bdpo"
    "log.tag=default"
    "log.project=flow-rl"
    "log.entity=lamda-rl"
)

ANTMAZE_ARGS=(
    "algo.critic.layer_norm=true"
    "algo.critic.maxQ=true"
    "algo.critic.discount=0.995"
    "data.norm_reward=antmaze100"
    "eval.num_episodes=100"
    "algo.diffusion.mlp_hidden_dims=[512,512,512,512]"
    "algo.diffusion.behavior.lr=1e-4"
)

declare -A TASK_ARGS
TASK_ARGS=(
    ["halfcheetah-medium-v2"]="algo.critic.eta=0.05 algo.critic.rho=0.5"
    ["halfcheetah-medium-replay-v2"]="algo.critic.eta=0.05 algo.critic.rho=0.5"
    ["halfcheetah-medium-expert-v2"]="algo.critic.eta=0.05 algo.critic.rho=0.5"
    ["hopper-medium-v2"]="algo.critic.eta=0.2 algo.critic.rho=2.0 algo.critic.ensemble_size=20"
    ["hopper-medium-replay-v2"]="algo.critic.eta=0.2 algo.critic.rho=2.0"
    ["hopper-medium-expert-v2"]="algo.critic.eta=0.2 algo.critic.rho=2.0"
    ["walker2d-medium-v2"]="algo.critic.eta=0.15 algo.critic.rho=1.0"
    ["walker2d-medium-replay-v2"]="algo.critic.eta=0.15 algo.critic.rho=1.0"
    ["walker2d-medium-expert-v2"]="algo.critic.eta=0.15 algo.critic.rho=1.0"
    # antmaze
    ["antmaze-umaze-v0"]="algo.critic.eta=0.5 algo.critic.rho=0.8 ${ANTMAZE_ARGS[@]}"
    ["antmaze-umaze-diverse-v0"]="algo.temperature=null algo.critic.eta=0.5 algo.critic.rho=0.8 ${ANTMAZE_ARGS[@]}"
    ["antmaze-medium-play-v0"]="algo.critic.eta=0.2 algo.critic.rho=0.8 ${ANTMAZE_ARGS[@]}"
    ["antmaze-medium-diverse-v0"]="algo.critic.eta=0.2 algo.critic.rho=0.8 ${ANTMAZE_ARGS[@]}"
    ["antmaze-large-play-v0"]="algo.critic.eta=1.0 algo.critic.rho=0.8 ${ANTMAZE_ARGS[@]}"
    ["antmaze-large-diverse-v0"]="algo.critic.eta=1.0 algo.critic.rho=0.8 ${ANTMAZE_ARGS[@]}"
)


# first arugment is the name of the experiment
# any other arguments will be added to the command
# if no arguments are given, exit


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
