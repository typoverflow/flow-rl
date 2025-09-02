#!/bin/bash

# Default configuration
CONFIG="ppo_config"
TASK="HalfCheetah-v5"
SEED=0
ENTITY=""
PROJECT="flowrl-ppo"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --entity)
      ENTITY="$2"
      shift 2
      ;;
    --project)
      PROJECT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Set entity flag if provided
ENTITY_FLAG=""
if [[ -n "$ENTITY" ]]; then
    ENTITY_FLAG="log.entity=$ENTITY"
fi

# Run training
python examples/online/main_ppo.py \
    --config-path ./examples/online/config \
    --config-name $CONFIG \
    task=$TASK \
    seed=$SEED \
    log.project=$PROJECT \
    $ENTITY_FLAG
