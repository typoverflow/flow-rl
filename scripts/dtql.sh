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
    # antmaze (v0)
	"antmaze-umaze-v0"
    "antmaze-umaze-diverse-v0"
	"antmaze-medium-play-v0"
	"antmaze-medium-diverse-v0"
	"antmaze-large-play-v0"
	"antmaze-large-diverse-v0"
    # antmaze (v2)
	"antmaze-umaze-v2"
    "antmaze-umaze-diverse-v2"
	"antmaze-medium-play-v2"
	"antmaze-medium-diverse-v2"
	"antmaze-large-play-v2"
	"antmaze-large-diverse-v2"
    # pen
    "pen-human-v1"
    "pen-cloned-v1"
    # kitchen
	"kitchen-complete-v0"
	"kitchen-partial-v0"
	"kitchen-mixed-v0"
)
SEEDS=(0 1 2 3 4)
PARALLEL=${PARALLEL:-4}

SHARED_ARGS=(
    "algo=dtql"
    "log.project=flow-rl"
    "log.entity=lamda-rl"
    "log.tag=default"
)

# these are same args as DTQL's [official implementation](https://github.com/TianyuCodings/Diffusion_Trusted_Q_Learning)
declare -A TASK_ARGS
TASK_ARGS=(
    ["halfcheetah-medium-v2"]="algo.lr=0.0003 algo.alpha=1.0 algo.gamma=0.0 algo.lr_decay=false train_steps=1000000 data.batch_size=256 algo.expectile=0.7 data.norm_reward=none"
    ["halfcheetah-medium-replay-v2"]="algo.lr=0.0003 algo.alpha=5.0 algo.gamma=0.0 algo.lr_decay=false train_steps=1000000 data.batch_size=256 algo.expectile=0.7 data.norm_reward=none"
    ["halfcheetah-medium-expert-v2"]="algo.lr=0.0003 algo.alpha=50.0 algo.gamma=0.0 algo.lr_decay=false train_steps=1000000 data.batch_size=256 algo.expectile=0.7 data.norm_reward=none"
    ["hopper-medium-v2"]="algo.lr=0.0001 algo.alpha=5.0 algo.gamma=0.0 algo.lr_decay=true train_steps=1000000 data.batch_size=256 algo.expectile=0.7 data.norm_reward=none"
    ["hopper-medium-replay-v2"]="algo.lr=0.0003 algo.alpha=5.0 algo.gamma=0.0 algo.lr_decay=false train_steps=1000000 data.batch_size=256 algo.expectile=0.7 data.norm_reward=none"
    ["hopper-medium-expert-v2"]="algo.lr=0.0003 algo.alpha=20.0 algo.gamma=0.0 algo.lr_decay=false train_steps=1000000 data.batch_size=256 algo.expectile=0.7 data.norm_reward=none"
    ["walker2d-medium-v2"]="algo.lr=0.0003 algo.alpha=5.0 algo.gamma=0.0 algo.lr_decay=true train_steps=1000000 data.batch_size=256 algo.expectile=0.7 data.norm_reward=none"
    ["walker2d-medium-replay-v2"]="algo.lr=0.0003 algo.alpha=5.0 algo.gamma=0.0 algo.lr_decay=true train_steps=1000000 data.batch_size=256 algo.expectile=0.7 data.norm_reward=none"
    ["walker2d-medium-expert-v2"]="algo.lr=0.0003 algo.alpha=5.0 algo.gamma=0.0 algo.lr_decay=true train_steps=1000000 data.batch_size=256 algo.expectile=0.7 data.norm_reward=none"
    ["antmaze-umaze-v0"]="algo.lr=0.0003 algo.alpha=1.0 algo.gamma=1.0 algo.lr_decay=false train_steps=500000 data.batch_size=2048 algo.expectile=0.9 data.norm_reward=iql_antmaze"
    ["antmaze-umaze-diverse-v0"]="algo.lr=3e-05 algo.alpha=1.0 algo.gamma=1.0 algo.lr_decay=true train_steps=500000 data.batch_size=2048 algo.expectile=0.9 data.norm_reward=iql_antmaze"
    ["antmaze-medium-play-v0"]="algo.lr=0.0003 algo.alpha=1.0 algo.gamma=1.0 algo.lr_decay=false train_steps=400000 data.batch_size=2048 algo.expectile=0.9 data.norm_reward=iql_antmaze"
    ["antmaze-medium-diverse-v0"]="algo.lr=0.0003 algo.alpha=1.0 algo.gamma=1.0 algo.lr_decay=false train_steps=400000 data.batch_size=2048 algo.expectile=0.9 data.norm_reward=iql_antmaze"
    ["antmaze-large-play-v0"]="algo.lr=0.0003 algo.alpha=1.0 algo.gamma=1.0 algo.lr_decay=false train_steps=350000 data.batch_size=2048 algo.expectile=0.9 data.norm_reward=iql_antmaze"
    ["antmaze-large-diverse-v0"]="algo.lr=0.0003 algo.alpha=0.5 algo.gamma=1.0 algo.lr_decay=false train_steps=300000 data.batch_size=2048 algo.expectile=0.9 data.norm_reward=iql_antmaze"
    ["antmaze-umaze-v2"]="algo.lr=0.0003 algo.alpha=1.0 algo.gamma=1.0 algo.lr_decay=false train_steps=500000 data.batch_size=2048 algo.expectile=0.9 data.norm_reward=iql_antmaze"
    ["antmaze-umaze-diverse-v2"]="algo.lr=3e-05 algo.alpha=1.0 algo.gamma=1.0 algo.lr_decay=true train_steps=500000 data.batch_size=2048 algo.expectile=0.9 data.norm_reward=iql_antmaze"
    ["antmaze-medium-play-v2"]="algo.lr=0.0003 algo.alpha=1.0 algo.gamma=1.0 algo.lr_decay=false train_steps=400000 data.batch_size=2048 algo.expectile=0.9 data.norm_reward=iql_antmaze"
    ["antmaze-medium-diverse-v2"]="algo.lr=0.0003 algo.alpha=1.0 algo.gamma=1.0 algo.lr_decay=false train_steps=400000 data.batch_size=2048 algo.expectile=0.9 data.norm_reward=iql_antmaze"
    ["antmaze-large-play-v2"]="algo.lr=0.0003 algo.alpha=1.0 algo.gamma=1.0 algo.lr_decay=false train_steps=350000 data.batch_size=2048 algo.expectile=0.9 data.norm_reward=iql_antmaze"
    ["antmaze-large-diverse-v2"]="algo.lr=0.0003 algo.alpha=0.5 algo.gamma=1.0 algo.lr_decay=false train_steps=300000 data.batch_size=2048 algo.expectile=0.9 data.norm_reward=iql_antmaze"
    ["pen-human-v1"]="algo.lr=3e-05 algo.alpha=1500.0 algo.gamma=0.0 algo.lr_decay=true train_steps=300000 data.batch_size=256 algo.expectile=0.9 data.norm_reward=none"
    ["pen-cloned-v1"]="algo.lr=1e-05 algo.alpha=1500.0 algo.gamma=0.0 algo.lr_decay=false train_steps=200000 data.batch_size=256 algo.expectile=0.7 data.norm_reward=none"
    ["kitchen-complete-v0"]="algo.lr=0.0001 algo.alpha=200.0 algo.gamma=0.0 algo.lr_decay=true train_steps=500000 data.batch_size=256 algo.expectile=0.7 data.norm_reward=none"
    ["kitchen-partial-v0"]="algo.lr=0.0001 algo.alpha=100.0 algo.gamma=0.0 algo.lr_decay=true train_steps=1000000 data.batch_size=256 algo.expectile=0.7 data.norm_reward=none"
    ["kitchen-mixed-v0"]="algo.lr=0.0003 algo.alpha=200.0 algo.gamma=0.0 algo.lr_decay=true train_steps=500000 data.batch_size=256 algo.expectile=0.7 data.norm_reward=none"
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
