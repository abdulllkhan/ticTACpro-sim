#!/usr/bin/env python3
"""
DGX Spark training script for Tic Tac Pro DQN Agent

This script is optimized for training on NVIDIA DGX Spark cluster.
Supports multi-GPU distributed training for intensive training runs (millions of games).
"""

import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rl.agent import DQNAgent
from rl.trainer import Trainer
import os


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


def train_worker(rank, world_size, args):
    """
    Training worker for distributed training

    Args:
        rank: Process rank
        world_size: Total number of processes
        args: Training arguments
    """
    print(f"[Rank {rank}] Starting worker...")

    # Setup distributed
    if world_size > 1:
        setup_distributed(rank, world_size)

    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

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

    # Wrap model for distributed training
    if world_size > 1:
        agent.policy_net = torch.nn.parallel.DistributedDataParallel(
            agent.policy_net,
            device_ids=[rank]
        )
        agent.target_net = torch.nn.parallel.DistributedDataParallel(
            agent.target_net,
            device_ids=[rank]
        )

    # Create trainer (only rank 0 saves checkpoints)
    trainer = Trainer(
        agent=agent,
        checkpoint_dir=args.checkpoint_dir if rank == 0 else None,
        log_dir=args.log_dir if rank == 0 else None,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq
    )

    # Train
    try:
        trainer.train(
            num_episodes=args.episodes // world_size,  # Distribute episodes
            resume_from=args.resume
        )

        if rank == 0:
            print("\n" + "=" * 60)
            print("DGX Training completed successfully!")
            print("=" * 60)

    except KeyboardInterrupt:
        if rank == 0:
            print("\n\nTraining interrupted by user.")

    finally:
        if world_size > 1:
            cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='Train Tic Tac Pro DQN Agent on DGX Spark')

    parser.add_argument('--episodes', type=int, default=1000000,
                       help='Number of episodes to train (default: 1000000)')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='Learning rate (default: 0.0005)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                       help='Initial exploration rate (default: 1.0)')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                       help='Final exploration rate (default: 0.01)')
    parser.add_argument('--epsilon-decay', type=float, default=0.99995,
                       help='Exploration decay rate (default: 0.99995)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Training batch size (default: 256)')
    parser.add_argument('--buffer-size', type=int, default=500000,
                       help='Replay buffer size (default: 500000)')
    parser.add_argument('--target-update', type=int, default=2000,
                       help='Target network update frequency (default: 2000)')
    parser.add_argument('--save-freq', type=int, default=5000,
                       help='Checkpoint save frequency (default: 5000)')
    parser.add_argument('--eval-freq', type=int, default=1000,
                       help='Evaluation frequency (default: 1000)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--use-conv', action='store_true',
                       help='Use convolutional network architecture')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_dgx',
                       help='Directory for checkpoints (default: checkpoints_dgx)')
    parser.add_argument('--log-dir', type=str, default='logs_dgx',
                       help='Directory for logs (default: logs_dgx)')
    parser.add_argument('--num-gpus', type=int, default=-1,
                       help='Number of GPUs to use (-1 for all available, default: -1)')

    args = parser.parse_args()

    # Determine number of GPUs
    if args.num_gpus == -1:
        world_size = torch.cuda.device_count()
    else:
        world_size = min(args.num_gpus, torch.cuda.device_count())

    if world_size == 0:
        print("ERROR: No GPUs available. This script requires GPUs.")
        print("For CPU training, use train_local.py instead.")
        return

    print("=" * 60)
    print("Tic Tac Pro - DGX Spark Training")
    print("=" * 60)
    print(f"Total Episodes: {args.episodes}")
    print(f"Episodes per GPU: {args.episodes // world_size}")
    print(f"Number of GPUs: {world_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"Epsilon: {args.epsilon_start} -> {args.epsilon_end} (decay: {args.epsilon_decay})")
    print(f"Batch size: {args.batch_size}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Architecture: {'Convolutional' if args.use_conv else 'Fully Connected'}")
    print("=" * 60)

    # List available GPUs
    print("\nAvailable GPUs:")
    for i in range(world_size):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    print("\nStarting distributed training...\n")

    # Launch distributed training
    if world_size == 1:
        # Single GPU training
        train_worker(0, 1, args)
    else:
        # Multi-GPU distributed training
        mp.spawn(
            train_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )

    print("\n" + "=" * 60)
    print("All training workers completed!")
    print("=" * 60)
    print(f"\nCheckpoints saved in: {args.checkpoint_dir}")
    print(f"Logs saved in: {args.log_dir}")
    print("\nTo view training progress, run:")
    print(f"  tensorboard --logdir {args.log_dir}")
    print("\nTo play against the trained agent, run:")
    print(f"  python play.py --checkpoint {args.checkpoint_dir}/final.pt")
    print("\nTo analyze strategies, run:")
    print(f"  python analyze.py --checkpoint {args.checkpoint_dir}/final.pt")


if __name__ == "__main__":
    main()
