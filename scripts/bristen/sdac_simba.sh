#!/bin/bash
#SBATCH --job-name=sdac
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=log/sdac_%j.out
#SBATCH --error=log/sdac_%j.err
#SBATCH --partition=normal
# Uncomment and set if your cluster requires:
#SBATCH --account=a0134

# 4 envs x 5 seeds = 20 experiments; 4 GPUs x 2 runs per GPU = 8 concurrent
NUM_GPUS=4
RUNS_PER_GPU=1
PARALLEL=$((NUM_GPUS * RUNS_PER_GPU))

TASKS=(
    "dog-run"
    # "dog-stand"
    # "dog-trot"
    # "dog-walk"
    # "humanoid-run"
    # "humanoid-stand"
    # "humanoid-walk"
    # "quadruped-run"
)

SEEDS=(0 1 2 3)

# Always run from repo root (where flowrl and examples/ live) so behavior matches command line
cd /iopsstor/scratch/cscs/cgao304/flow-rl-clariden
mkdir -p log


run_task() {
    local task=$1
    local seed=$2
    local slot=$3
    local device=$((slot % NUM_GPUS))
    # Define inside function so it's available when run via parallel (arrays are not exported)
    local shared_args=(
        "algo=sdac"
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
    export XLA_PYTHON_CLIENT_PREALLOCATE="false"
    export CUDA_VISIBLE_DEVICES=$device
    source .venv_online_bristen/bin/activate
    export PYTHONPATH=".":$PYTHONPATH
    echo Running on GPU $device
    echo python3 examples/online/main_dmc_offpolicy.py "${shared_args[@]}" task="$task" seed="$seed"
    python3 examples/online/main_dmc_offpolicy.py "${shared_args[@]}" task="$task" seed="$seed"
}

export -f run_task
export NUM_GPUS

# Generate all (task, seed) pairs and run with 8 in parallel; {%} is job slot 1..8 from GNU parallel
# If GNU parallel is not available, we fall back to a bash loop with background jobs
if command -v parallel &>/dev/null; then
    parallel -j "$PARALLEL" --line-buffer run_task {1} {2} {3} {%} ::: "${TASKS[@]}" ::: "${SEEDS[@]}"
else
    # Fallback: run 8 at a time; slot cycles 0..7 so GPU = slot % 4 gives 2 per GPU
    slot=0
    for task in "${TASKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_task "$task" "$seed" "$slot" &
                slot=$(( (slot + 1) % PARALLEL ))
                if [[ $slot -eq 0 ]]; then
                    wait
                fi
        done
    done
    wait
fi

echo "All 20 sdac runs finished."
