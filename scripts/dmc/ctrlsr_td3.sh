# Specify which GPUs to use
GPUS=(0 1 2 3 4 5 6 7)  # Modify this array to specify which GPUs to use
SEEDS=(0 1 2 3 4)
NUM_EACH_GPU=3

PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

TASKS=(
    # "acrobot-swingup"
    # "ball_in_cup-catch"
    # "cartpole-balance"
    # "cartpole-balance_sparse"
    # "cartpole-swingup"
    # "cartpole-swingup_sparse"
    "cheetah-run"
    "dog-run"
    "dog-stand"
    "dog-trot"
    "dog-walk"
    # "finger-spin"
    # "finger-turn_easy"
    # "finger-turn_hard"
    # "fish-swim"
    # "hopper-hop"
    # "hopper-stand"
    "humanoid-run"
    "humanoid-stand"
    "humanoid-walk"
    # "pendulum-swingup"
    # "quadruped-run"
    # "quadruped-walk"
    # "reacher-easy"
    # "reacher-hard"
    # "walker-run"
    # "walker-stand"
    # "walker-walk"
)

SHARED_ARGS=(
    "algo=ctrlsr_td3"
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
    command="python3 examples/online/main_dmc_offpolicy.py task=$task device=$device seed=$seed ${SHARED_ARGS[@]}"
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
