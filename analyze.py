#!/usr/bin/env python3
"""
Analyze trained Tic Tac Pro agent and generate strategy guide

This script analyzes the trained agent to extract optimal strategies,
patterns, and insights.
"""

import argparse
import torch
import os

from rl.agent import DQNAgent
from analysis.strategy_analyzer import StrategyAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Analyze Tic Tac Pro AI strategies')

    parser.add_argument('--checkpoint', type=str, default='checkpoints/final.pt',
                       help='Path to trained agent checkpoint (default: checkpoints/final.pt)')
    parser.add_argument('--output-dir', type=str, default='analysis',
                       help='Output directory for analysis (default: analysis)')
    parser.add_argument('--use-conv', action='store_true',
                       help='Use convolutional network architecture')
    parser.add_argument('--skip-guide', action='store_true',
                       help='Skip strategy guide generation')
    parser.add_argument('--opening-games', type=int, default=1000,
                       help='Number of games for opening analysis (default: 1000)')
    parser.add_argument('--pattern-games', type=int, default=500,
                       help='Number of games for pattern analysis (default: 500)')

    args = parser.parse_args()

    print("=" * 60)
    print("Tic Tac Pro - Strategy Analysis")
    print("=" * 60)

    # Load agent
    if not os.path.exists(args.checkpoint):
        print(f"\nERROR: Checkpoint not found: {args.checkpoint}")
        print("\nPlease train an agent first using:")
        print("  python train_local.py")
        return

    print(f"\nLoading agent from: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create agent
    agent = DQNAgent(
        use_conv=args.use_conv,
        device=device
    )

    # Load checkpoint
    if agent.load(args.checkpoint):
        print("Agent loaded successfully!")
    else:
        print("ERROR: Failed to load agent.")
        return

    # Set to evaluation mode
    agent.epsilon = 0.0

    print("\n" + "=" * 60)
    print("Running Analysis")
    print("=" * 60)

    # Create analyzer
    analyzer = StrategyAnalyzer(agent)

    # Run opening analysis
    print("\n1. Opening Move Analysis")
    print("-" * 60)
    opening_analysis = analyzer.analyze_opening_moves(num_games=args.opening_games)

    # Run position analysis
    print("\n2. Position Value Analysis")
    print("-" * 60)
    position_values = analyzer.analyze_position_values(num_positions=1000)

    # Generate heatmap
    print("\n3. Generating Position Heatmap")
    print("-" * 60)
    analyzer.plot_position_heatmap(output_dir=args.output_dir)

    # Run win pattern analysis
    print("\n4. Win Pattern Analysis")
    print("-" * 60)
    win_patterns = analyzer.analyze_win_patterns(num_games=args.pattern_games)

    # Run piece usage analysis
    print("\n5. Piece Usage Analysis")
    print("-" * 60)
    piece_usage = analyzer.analyze_piece_usage(num_games=args.pattern_games)

    # Export statistics
    print("\n6. Exporting Statistics")
    print("-" * 60)
    analyzer.export_statistics(output_path=os.path.join(args.output_dir, 'statistics.json'))

    # Generate strategy guide
    if not args.skip_guide:
        print("\n7. Generating Strategy Guide")
        print("-" * 60)
        analyzer.generate_strategy_guide(
            output_path=os.path.join(args.output_dir, 'strategy_guide.md')
        )

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nGenerated files:")
    print(f"  - {os.path.join(args.output_dir, 'strategy_guide.md')}")
    print(f"  - {os.path.join(args.output_dir, 'position_heatmap.png')}")
    print(f"  - {os.path.join(args.output_dir, 'statistics.json')}")

    print("\nView the strategy guide to learn optimal strategies!")


if __name__ == "__main__":
    main()
