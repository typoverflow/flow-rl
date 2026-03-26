# Specify which GPUs to use
GPUS=(0 1 2 3 4 5 6 7)  # Modify this array to specify which GPUs to use
SEEDS=(0 1 2 3 4)
NUM_EACH_GPU=3

PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

TASKS=(
    "dog-run"
    # "dog-stand"
    # "dog-trot"
    "dog-walk"
    "humanoid-run"
    # "humanoid-stand"
    "humanoid-walk"
)

SHARED_ARGS=(
    "algo=dpmd_exp"
    "log.tag=default"
    "algo.backbone_cls=simba"
    "algo.critic_hidden_dims=[512,512]"
    "algo.critic_activation=relu"
    "algo.diffusion.hidden_dims=[256]"
    "algo.diffusion.activation=relu"
    "norm_obs=true"
    "log.tag=simba"
    "log.project=flow-rl-dmc"
    "log.entity=gaochenxiao"
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
