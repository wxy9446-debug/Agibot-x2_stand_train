#!/bin/bash
# ============================================================
# X2 PPO Walk Training Launch Script
# Usage:
#   ./run_train_ppo.sh              # full training
#   ./run_train_ppo.sh debug        # 64 envs, 100 iters
#   ./run_train_ppo.sh resume PATH  # resume from checkpoint
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_ENVS=4096
MAX_ITER=5000
LOG_DIR="logs/x2_walk_ppo"
RESUME=""

case "$1" in
  debug)
    NUM_ENVS=64
    MAX_ITER=100
    echo "[X2-Train] Debug mode: num_envs=$NUM_ENVS, max_iter=$MAX_ITER"
    ;;
  resume)
    RESUME="$2"
    echo "[X2-Train] Resume from: $RESUME"
    ;;
  *)
    echo "[X2-Train] Full training: num_envs=$NUM_ENVS, max_iter=$MAX_ITER"
    ;;
esac

CMD="python train_x2_walk_ppo.py \
  --num_envs $NUM_ENVS \
  --max_iterations $MAX_ITER \
  --log_dir $LOG_DIR \
  --headless"

if [ -n "$RESUME" ]; then
  CMD="$CMD --resume $RESUME"
fi

echo "[X2-Train] Command: $CMD"
echo "[X2-Train] Starting..."
eval $CMD
