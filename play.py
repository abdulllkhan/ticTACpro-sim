#!/usr/bin/env python3
"""
Play Tic Tac Pro against the trained AI agent

This script launches the visual interface where you can play against
the trained agent and learn from its moves.
"""

import argparse
import torch
import os

from rl.agent import DQNAgent
from gui.game_gui import TicTacProGUI
from game.tictacpro import Player


def main():
    parser = argparse.ArgumentParser(description='Play Tic Tac Pro against AI')

    parser.add_argument('--checkpoint', type=str, default='checkpoints/final.pt',
                       help='Path to trained agent checkpoint (default: checkpoints/final.pt)')
    parser.add_argument('--ai-player', type=str, default='blue', choices=['red', 'blue'],
                       help='Which player the AI controls (default: blue)')
    parser.add_argument('--no-suggestions', action='store_true',
                       help='Disable move suggestions (default: show suggestions)')
    parser.add_argument('--use-conv', action='store_true',
                       help='Use convolutional network architecture')

    args = parser.parse_args()

    print("=" * 60)
    print("Tic Tac Pro - Play Against AI")
    print("=" * 60)

    # Load agent
    if not os.path.exists(args.checkpoint):
        print(f"\nERROR: Checkpoint not found: {args.checkpoint}")
        print("\nPlease train an agent first using:")
        print("  python train_local.py")
        print("\nOr download a pre-trained checkpoint.")
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

    # Set agent to evaluation mode (no exploration)
    agent.epsilon = 0.0

    # Determine AI player
    ai_player = Player.BLUE if args.ai_player == 'blue' else Player.RED
    human_player = Player.RED if ai_player == Player.BLUE else Player.BLUE

    print(f"\nAI plays as: {ai_player.name}")
    print(f"You play as: {human_player.name}")

    print("\n" + "=" * 60)
    print("Controls:")
    print("  - Click to select piece size and place pieces")
    print("  - Keys 1/2/3: Select Small/Medium/Large piece")
    print("  - S: Toggle move suggestions")
    print("  - R: Reset game")
    print("  - ESC: Quit")
    print("=" * 60)

    print("\nStarting game...\n")

    # Create and run GUI
    gui = TicTacProGUI(agent=agent, ai_player=ai_player)
    gui.show_suggestions = not args.no_suggestions
    gui.run()

    print("\nThanks for playing!")


if __name__ == "__main__":
    main()
