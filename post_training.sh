#!/bin/bash
# Post-training automation: waits for training PID, runs tournament, starts next run.
# Usage: bash post_training.sh [TRAINING_PID]

set -e
cd "$(dirname "$0")"
TRAINING_PID=${1:-487725}
LOG="post_training_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Post-training monitor started: $(date) ==="
echo "Waiting for training PID $TRAINING_PID to finish..."

# Poll until the training process exits (works for any PID, not just children)
while kill -0 "$TRAINING_PID" 2>/dev/null; do
    sleep 30
    REMAIN=$(python3 -c "import json; d=json.load(open('training_state.json')); print(f'{d[\"remaining_hours\"]:.2f}h remaining, eps={d[\"epsilon\"]:.4f}')" 2>/dev/null || echo "?")
    echo "  [$(date +%H:%M)] Training still running: $REMAIN"
done

echo ""
echo "=== Training finished: $(date) ==="

# 1. Verify the final checkpoint was saved
echo "Checking final checkpoint..."
EPS=$(python3 -c "
import torch
ck = torch.load('checkpoints_normalized/final.pt', map_location='cpu', weights_only=False)
eps = ck.get('epsilon', '?')
print(f'{eps:.4f}')
" 2>/dev/null)
echo "  Final checkpoint epsilon: $EPS"

# 2. Run mini-tournament to benchmark current model
echo ""
echo "=== Mini-tournament: biased-final vs normalized-final (5 games each direction) ==="
python3 tournament.py \
    --checkpoint  checkpoints_spark/final.pt \
    --checkpoint2 checkpoints_normalized/final.pt \
    --games 5 --time 1.0 --no-slow

echo ""
echo "Tournament done: $(date)"

# 3. Start next 10h training run with calibrated decay rate.
# Target: eps=0.678 -> eps=0.05 in 10 hours (at ~9300 steps/hour = 93000 steps).
# Decay rate: (0.05/0.678)^(1/93000) = 0.9999720
echo ""
echo "=== Starting next training run: 10h, epsilon-decay=0.9999720 ==="
echo "Goal: eps $EPS -> 0.05 (fully trained normalized model)"

nohup python3 train_spark.py \
    --hours 10.0 \
    --epsilon-decay 0.9999720 \
    --checkpoint-dir checkpoints_normalized \
    --log-dir logs_normalized \
    --resume checkpoints_normalized/final.pt \
    --n-cpu-workers 8 \
    --games-per-worker 32 \
    --save-every 300 \
    --log-every 60 \
    --skip-analysis > train_run3.log 2>&1 &

RUN3_PID=$!
echo "Started training run 3 (PID $RUN3_PID), log: train_run3.log"
echo "ETA: $(date -d '+10 hours')"
echo ""
echo "After run 3 completes, run the full tournament:"
echo "  python3 tournament.py --checkpoint checkpoints_spark/final.pt --checkpoint2 checkpoints_normalized/final.pt --games 10 --time 2.0"
