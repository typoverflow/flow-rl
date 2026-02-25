# Specify which GPUs to use
GPUS=(0 1 2 3 4 5 6 7)  # Modify this array to specify which GPUs to use
SEEDS=(0 1 2 3)
NUM_EACH_GPU=1

PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

TASKS=(
    "acrobot-swingup"
    "ball_in_cup-catch"
    "cartpole-balance"
    "cartpole-balance_sparse"
    "cartpole-swingup"
    "cartpole-swingup_sparse"
    "cheetah-run"
    "dog-run"
    "dog-stand"
    "dog-trot"
    "dog-walk"
    "finger-spin"
    "finger-turn_easy"
    "finger-turn_hard"
    "fish-swim"
    "hopper-hop"
    "hopper-stand"
    "humanoid-run"
    "humanoid-stand"
    "humanoid-walk"
    "pendulum-swingup"
    "quadruped-run"
    "quadruped-walk"
    "reacher-easy"
    "reacher-hard"
    "walker-run"
    "walker-stand"
    "walker-walk"
)

SHARED_ARGS=(
    "algo=sac"
    "log.tag=default"
)


run_task() {
    task=$1
    seed=$2
    slot=$3
    num_gpus=${#GPUS[@]}
    device_idx=$((slot % num_gpus))
    device=${GPUS[$device_idx]}
    echo "Running $env $seed on GPU $device"
    export CUDA_VISIBLE_DEVICES=$device
    export EGL_VISIBLE_DEVICES=$device
    export XLA_PYTHON_CLIENT_PREALLOCATE="false"
    command="python3 examples/online/main_dmc_offpolicy.py task=$task seed=$seed ${SHARED_ARGS[@]}"
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
