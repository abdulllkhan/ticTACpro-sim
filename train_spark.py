#!/usr/bin/env python3
"""
train_spark.py — DGX Spark GB10 training entry point for Tic Tac Pro.

Runs a time-based training loop (default 2 hours) that:
  - Uses the full 121 GB GPU / CPU unified memory.
  - Saturates all 48 SMs with batch_size=16384.
  - Employs BF16 AMP (native on Blackwell) for ~2× throughput.
  - Optionally distributes game collection across all 20 ARM cores.
  - Saves checkpoints every 10 minutes + a final checkpoint.
  - Writes training_state.json so external monitors can check health.
  - Prints a strategy summary (first-player vs second-player advantage, top
    opening moves, Q-value heatmap) once training completes.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# Maximise CPU-side parallelism for PyTorch ops (ARM cores on Grace)
_N_CPU = os.cpu_count() or 20
torch.set_num_threads(_N_CPU)
os.environ.setdefault("OMP_NUM_THREADS", str(_N_CPU))

# Blackwell tuning flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True



from rl.agent import GPUDQNAgent
from rl.gpu_trainer import GPUTrainer, collect_episodes_vectorized
from game.tictacpro import TicTacPro, Player, PieceSize


# ---------------------------------------------------------------------------
# Strategy analysis
# ---------------------------------------------------------------------------

def analyze_strategies(agent: GPUDQNAgent, n_games: int = 2000) -> None:
    """
    Post-training analysis: first-player advantage, best opening moves,
    Q-value heatmap printed to stdout.
    """
    print("\n" + "=" * 72)
    print("  POST-TRAINING STRATEGY ANALYSIS")
    print("=" * 72)

    saved_eps = agent.epsilon
    agent.epsilon = 0.0  # Fully greedy

    # ---- 1. First-player vs second-player advantage ----------------------
    red_wins = blue_wins = draws = 0
    batch = 64
    collected = 0
    while collected < n_games:
        b = min(batch, n_games - collected)
        _, stats = collect_episodes_vectorized(agent, b)
        red_wins += stats["red_wins"]
        blue_wins += stats["blue_wins"]
        draws += stats["draws"]
        collected += b

    total = red_wins + blue_wins + draws
    print(f"\n[Greedy self-play, {total} games]")
    print(f"  First player  (RED)  wins: {red_wins:>5}  ({red_wins/total:.1%})")
    print(f"  Second player (BLUE) wins: {blue_wins:>5}  ({blue_wins/total:.1%})")
    print(f"  Draws                    : {draws:>5}  ({draws/total:.1%})")

    if red_wins / total > 0.55:
        print("\n  Insight: RED (first player) has a significant ADVANTAGE — "
              "going first is crucial in this configuration.")
    elif blue_wins / total > 0.55:
        print("\n  Insight: BLUE (second player) has a significant ADVANTAGE — "
              "the second player benefits from reactive play.")
    else:
        print("\n  Insight: The game appears roughly balanced between first and second player.")

    # ---- 2. Opening move Q-values ----------------------------------------
    print("\n[Best opening moves from empty board — Q-values by position and size]")
    game = TicTacPro()
    qv = agent.get_q_values(game)

    size_names = {1: "S", 2: "M", 3: "L"}
    rows_out = []
    for row in range(3):
        for col in range(3):
            for size in [1, 2, 3]:
                idx = row * 9 + col * 3 + (size - 1)
                rows_out.append((qv[idx], row, col, size))

    rows_out.sort(reverse=True)

    print(f"\n  {'Rank':<5} {'Position':<10} {'Size':<6} {'Q-value':<10}")
    print(f"  {'-'*35}")
    for rank, (q, r, c, sz) in enumerate(rows_out[:10], 1):
        print(f"  {rank:<5} ({r},{c})      {size_names[sz]:<6} {q:>+.4f}")

    # ---- 3. Q-value heatmap (large pieces) -------------------------------
    print("\n[Q-value heatmap for LARGE pieces from empty board]")
    print("  (higher = agent prefers this cell for its first large piece)")
    print()
    for row in range(3):
        row_str = "  "
        for col in range(3):
            idx = row * 9 + col * 3 + (3 - 1)  # size=LARGE
            row_str += f"{qv[idx]:+.3f}  "
        print(row_str)

    # ---- 4. Best strategies summarised -----------------------------------
    top_move = rows_out[0]
    _, r, c, sz = top_move
    size_name_full = {1: "SMALL", 2: "MEDIUM", 3: "LARGE"}[sz]
    pos_name = {(0, 0): "top-left corner", (0, 1): "top-center edge",
                (0, 2): "top-right corner", (1, 0): "left edge",
                (1, 1): "CENTER", (1, 2): "right edge",
                (2, 0): "bottom-left corner", (2, 1): "bottom-center edge",
                (2, 2): "bottom-right corner"}.get((r, c), f"({r},{c})")

    print(f"\n[Summary: Best strategies]")
    print(f"  STARTING FIRST  — Best opening: play a {size_name_full} piece at {pos_name}")
    print(f"  STARTING SECOND — React to opponent's opening; the agent's top counter-moves")
    print(f"                    depend on the specific first move.")
    print()

    # Show top 3 opening moves with readable description
    print("  Top 3 opening moves (by Q-value):")
    for rank, (q, r, c, sz) in enumerate(rows_out[:3], 1):
        pname = {(0, 0): "top-left", (0, 1): "top-mid", (0, 2): "top-right",
                 (1, 0): "mid-left", (1, 1): "CENTER", (1, 2): "mid-right",
                 (2, 0): "bot-left", (2, 1): "bot-mid", (2, 2): "bot-right"}.get(
                     (r, c), f"({r},{c})")
        sname = size_name_full = {1: "Small", 2: "Medium", 3: "Large"}[sz]
        print(f"    {rank}. {pname} — {sname} piece  (Q={q:+.4f})")

    agent.epsilon = saved_eps
    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Tic Tac Pro DQN on NVIDIA DGX Spark GB10"
    )
    p.add_argument("--hours", type=float, default=2.0,
                   help="Training duration in hours (default: 2.0)")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate (default: 3e-4)")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor (default: 0.99)")
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.999987,
                   help="Per-gradient-step epsilon decay (default: 0.999987 — "
                        "decays 1.0→0.05 over ~242k steps at 32 steps/collect)")
    p.add_argument("--batch-size", type=int, default=65536,
                   help="Gradient update batch size (default: 65536)")
    p.add_argument("--buffer-size", type=int, default=10_000_000,
                   help="Replay buffer capacity (default: 10M)")
    p.add_argument("--n-envs", type=int, default=512,
                   help="Parallel environments per collection round when n_cpu_workers=0 (default: 512)")
    p.add_argument("--train-steps", type=int, default=32,
                   help="Gradient steps per collection round (default: 32)")
    p.add_argument("--target-update", type=int, default=5000,
                   help="Target network sync frequency in gradient steps (default: 5000)")
    p.add_argument("--n-cpu-workers", type=int, default=16,
                   help="Parallel CPU collector processes (default: 16; uses all ARM cores "
                        "while GPU trains — enables pipelined collection+training)")
    p.add_argument("--games-per-worker", type=int, default=32,
                   help="Games per CPU worker per round (default: 32; 16×32=512 games/round)")
    p.add_argument("--hidden", nargs=3, type=int, default=[1024, 512, 256],
                   metavar=("H1", "H2", "H3"),
                   help="Hidden layer sizes (default: 1024 512 256)")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable BF16 Automatic Mixed Precision")
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile (Inductor)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_spark")
    p.add_argument("--log-dir", type=str, default="logs_spark")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--save-every", type=int, default=600,
                   help="Checkpoint save interval in seconds (default: 600)")
    p.add_argument("--log-every", type=int, default=60,
                   help="Progress log interval in seconds (default: 60)")
    p.add_argument("--skip-analysis", action="store_true",
                   help="Skip post-training strategy analysis")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Sanity checks ---------------------------------------------------
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. This script requires the DGX Spark GPU.")
        sys.exit(1)

    gpu_props = torch.cuda.get_device_properties(0)
    print("\n" + "=" * 72)
    print("  Tic Tac Pro — DGX Spark Training")
    print("=" * 72)
    print(f"  GPU       : {gpu_props.name}")
    print(f"  GPU Mem   : {gpu_props.total_memory / 1024**3:.1f} GB")
    print(f"  SMs       : {gpu_props.multi_processor_count}")
    print(f"  CPU cores : {_N_CPU}")
    print(f"  Duration  : {args.hours:.1f} h")
    print(f"  Batch size: {args.batch_size:,}")
    print(f"  Buffer    : {args.buffer_size:,}")
    print(f"  n_envs    : {args.n_envs}")
    print(f"  Train steps/collect: {args.train_steps}")
    print(f"  Hidden    : {args.hidden}")
    print(f"  AMP       : {'disabled' if args.no_amp else 'BF16'}")
    print(f"  Compile   : {'disabled' if args.no_compile else 'inductor'}")
    total_games_per_round = (
        args.n_cpu_workers * args.games_per_worker if args.n_cpu_workers > 0
        else args.n_envs
    )
    mode_str = (
        f"pipelined ({args.n_cpu_workers} workers × {args.games_per_worker} games = "
        f"{total_games_per_round}/round)" if args.n_cpu_workers > 0
        else f"vectorized ({args.n_envs} envs)"
    )
    print(f"  Mode      : {mode_str}")
    print("=" * 72 + "\n")

    # ---- Agent -----------------------------------------------------------
    device = torch.device("cuda:0")
    agent = GPUDQNAgent(
        state_size=61,
        action_size=27,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,  # calibrated for 32 steps/collect, 2.1h
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update,
        use_amp=not args.no_amp,
        compile_model=not args.no_compile,
        device=device,
        hidden_sizes=tuple(args.hidden),
    )

    # ---- Trainer ---------------------------------------------------------
    trainer = GPUTrainer(
        agent=agent,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        n_envs=args.n_envs,
        train_steps_per_collect=args.train_steps,
        log_interval=args.log_every,
        checkpoint_interval=args.save_every,
        state_file="training_state.json",
        n_cpu_workers=args.n_cpu_workers,
        games_per_worker=args.games_per_worker,
    )

    # ---- Train -----------------------------------------------------------
    trainer.train(duration_hours=args.hours, resume_from=args.resume)

    # ---- Strategy analysis -----------------------------------------------
    if not args.skip_analysis:
        try:
            analyze_strategies(agent, n_games=2000)
        except Exception as exc:
            print(f"Strategy analysis failed: {exc}")

    print(f"\nDone. Final checkpoint: {args.checkpoint_dir}/final.pt")
    print(f"TensorBoard logs     : {args.log_dir}/")
    print("To view logs: tensorboard --logdir", args.log_dir)
    print("To play vs agent: python play.py --checkpoint", args.checkpoint_dir + "/final.pt")


if __name__ == "__main__":
    main()
