#!/usr/bin/env python3
"""
Analyze a trained Tic Tac Pro agent and print a strategy summary.

Requires a GPUDQNAgent checkpoint (produced by train_spark.py).
For legacy DQNAgent checkpoints from train_local.py, use play.py instead.

Reports:
  - First-player vs second-player win rates (greedy self-play)
  - Top opening moves by Q-value
  - Q-value heatmap (Large pieces)

Usage:
    python3 analyze.py --checkpoint checkpoints_spark/final.pt
    python3 analyze.py --checkpoint checkpoints_spark/final.pt --games 500
"""

import argparse
import os
import sys

import torch

from rl.agent import GPUDQNAgent
from train_spark import analyze_strategies


def main():
    parser = argparse.ArgumentParser(description="Analyze a trained Tic Tac Pro agent")
    parser.add_argument("--checkpoint", default="checkpoints_spark/final.pt",
                        help="Path to GPUDQNAgent checkpoint (default: checkpoints_spark/final.pt)")
    parser.add_argument("--games", type=int, default=2000,
                        help="Games for self-play win-rate analysis (default: 2000)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        print("\nTrain an agent first:")
        print("  python3 train_spark.py --hours 2")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "policy_net" not in ckpt:
        print("ERROR: this script requires a GPUDQNAgent checkpoint (train_spark.py output).")
        print("For legacy DQNAgent checkpoints, use: python3 play.py --checkpoint", args.checkpoint)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint}  [{device}]")

    agent = GPUDQNAgent(compile_model=False, device=device, buffer_size=1)
    if not agent.load(args.checkpoint):
        print("ERROR: failed to load checkpoint")
        sys.exit(1)

    agent.epsilon = 0.0
    analyze_strategies(agent, n_games=args.games)


if __name__ == "__main__":
    main()
