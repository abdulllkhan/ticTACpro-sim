#!/usr/bin/env python3
"""
Local training script for Tic Tac Pro DQN Agent

This script trains the agent on your local machine (Mac).
Suitable for quick experiments and testing (10K-100K games).
"""

import argparse
import torch
from rl.agent import DQNAgent
from rl.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train Tic Tac Pro DQN Agent locally')

    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of episodes to train (default: 10000)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                       help='Initial exploration rate (default: 1.0)')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                       help='Final exploration rate (default: 0.01)')
    parser.add_argument('--epsilon-decay', type=float, default=0.9995,
                       help='Exploration decay rate (default: 0.9995)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size (default: 64)')
    parser.add_argument('--buffer-size', type=int, default=100000,
                       help='Replay buffer size (default: 100000)')
    parser.add_argument('--target-update', type=int, default=1000,
                       help='Target network update frequency (default: 1000)')
    parser.add_argument('--save-freq', type=int, default=1000,
                       help='Checkpoint save frequency (default: 1000)')
    parser.add_argument('--eval-freq', type=int, default=100,
                       help='Evaluation frequency (default: 100)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--use-conv', action='store_true',
                       help='Use convolutional network architecture')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory for checkpoints (default: checkpoints)')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for logs (default: logs)')

    args = parser.parse_args()

    print("=" * 60)
    print("Tic Tac Pro - Local Training")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Learning rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"Epsilon: {args.epsilon_start} -> {args.epsilon_end} (decay: {args.epsilon_decay})")
    print(f"Batch size: {args.batch_size}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Architecture: {'Convolutional' if args.use_conv else 'Fully Connected'}")
    print("=" * 60)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if device.type == "cpu":
        print("Warning: Training on CPU. This may be slow.")
        print("For faster training, consider using the DGX Spark cluster.")

    # Create agent
    agent = DQNAgent(
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update,
        use_conv=args.use_conv,
        device=device
    )

    # Create trainer
    trainer = Trainer(
        agent=agent,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq
    )

    # Train
    try:
        trainer.train(
            num_episodes=args.episodes,
            resume_from=args.resume
        )

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        print(f"\nCheckpoints saved in: {args.checkpoint_dir}")
        print(f"Logs saved in: {args.log_dir}")
        print("\nTo view training progress, run:")
        print(f"  tensorboard --logdir {args.log_dir}")
        print("\nTo play against the trained agent, run:")
        print(f"  python play.py --checkpoint {args.checkpoint_dir}/final.pt")
        print("\nTo analyze strategies, run:")
        print(f"  python analyze.py --checkpoint {args.checkpoint_dir}/final.pt")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Latest checkpoint saved in: {args.checkpoint_dir}/latest.pt")
        print("You can resume training with:")
        print(f"  python train_local.py --resume {args.checkpoint_dir}/latest.pt")


if __name__ == "__main__":
    main()
