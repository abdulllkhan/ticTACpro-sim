"""
Training loop for Tic Tac Pro DQN Agent

This module implements the self-play training loop with logging and checkpointing.
"""

import torch
import numpy as np
import time
from tqdm import tqdm
from typing import Optional, Dict, List
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from game.tictacpro import TicTacPro, Player, PieceSize
from .agent import DQNAgent


class Trainer:
    """
    Trainer for DQN Agent using self-play

    The agent plays against itself and learns from the experiences.
    """

    def __init__(
        self,
        agent: DQNAgent,
        checkpoint_dir='checkpoints',
        log_dir='logs',
        save_freq=1000,
        eval_freq=100,
        eval_games=50
    ):
        """
        Initialize trainer

        Args:
            agent: DQN agent to train
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
            save_freq: How often to save checkpoints (in episodes)
            eval_freq: How often to evaluate (in episodes)
            eval_games: Number of games for evaluation
        """
        self.agent = agent
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.eval_games = eval_games

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Tensorboard writer
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(os.path.join(log_dir, run_name))

        # Training statistics
        self.episode = 0
        self.total_steps = 0
        self.win_rates = {'red': [], 'blue': [], 'draw': []}
        self.avg_game_lengths = []
        self.training_start_time = None

    def play_episode(self, training=True) -> Dict:
        """
        Play one episode of self-play

        Args:
            training: Whether to train the agent during play

        Returns:
            Episode statistics dictionary
        """
        game = TicTacPro()
        episode_data = []
        move_count = 0

        while not game.game_over:
            current_player = game.current_player
            state = game.get_state_tensor()

            # Get action from agent
            action = self.agent.get_action(game, current_player)

            if action is None:
                # No legal moves (shouldn't happen in normal play)
                break

            row, col, size = action
            action_idx = self.agent.policy_net.get_action_index(row, col, size)

            # Make move
            game.make_move(row, col, size)
            move_count += 1

            # Get next state
            next_state = game.get_state_tensor()

            # Calculate reward
            if game.game_over:
                if game.winner == current_player:
                    reward = 1.0  # Win
                elif game.winner == Player.NONE:
                    reward = 0.0  # Draw
                else:
                    reward = -1.0  # Loss (shouldn't happen in self-play immediately)
            else:
                reward = 0.0  # Intermediate move

            # Store experience
            episode_data.append({
                'state': state,
                'action': action_idx,
                'reward': reward,
                'next_state': next_state,
                'done': game.game_over,
                'player': current_player
            })

        # Process episode data - assign rewards retroactively
        winner = game.winner
        for i, data in enumerate(episode_data):
            player = data['player']

            if winner == Player.NONE:
                # Draw
                final_reward = 0.0
            elif winner == player:
                # This player won
                final_reward = 1.0
            else:
                # This player lost
                final_reward = -1.0

            # Store in replay buffer
            self.agent.store_experience(
                data['state'],
                data['action'],
                final_reward,
                data['next_state'],
                data['done']
            )

            # Train the agent
            if training:
                loss = self.agent.learn()
                if loss is not None and i == len(episode_data) - 1:
                    # Log final loss of episode
                    self.writer.add_scalar('Loss/train', loss, self.episode)

        # Episode statistics
        stats = {
            'winner': winner,
            'moves': move_count,
            'red_pieces_used': 9 - sum(game.pieces_remaining[Player.RED].values()),
            'blue_pieces_used': 9 - sum(game.pieces_remaining[Player.BLUE].values())
        }

        return stats

    def train(self, num_episodes: int, resume_from: Optional[str] = None):
        """
        Train the agent for a number of episodes

        Args:
            num_episodes: Number of episodes to train
            resume_from: Path to checkpoint to resume from
        """
        if resume_from and os.path.exists(resume_from):
            self.agent.load(resume_from)
            print(f"Resumed training from {resume_from}")

        self.training_start_time = time.time()
        print(f"\nStarting training for {num_episodes} episodes...")
        print(f"Device: {self.agent.device}")
        print(f"Epsilon: {self.agent.epsilon:.4f}")

        episode_bar = tqdm(range(num_episodes), desc="Training")

        recent_winners = []
        recent_moves = []

        for ep in episode_bar:
            self.episode += 1

            # Play episode
            stats = self.play_episode(training=True)

            # Track statistics
            recent_winners.append(stats['winner'])
            recent_moves.append(stats['moves'])

            # Keep only recent history
            if len(recent_winners) > 100:
                recent_winners.pop(0)
                recent_moves.pop(0)

            # Log to tensorboard
            self.writer.add_scalar('Episode/moves', stats['moves'], self.episode)
            self.writer.add_scalar('Training/epsilon', self.agent.epsilon, self.episode)

            # Periodic evaluation
            if self.episode % self.eval_freq == 0:
                eval_stats = self.evaluate()

                # Log evaluation
                self.writer.add_scalar('Eval/red_win_rate', eval_stats['red_win_rate'], self.episode)
                self.writer.add_scalar('Eval/blue_win_rate', eval_stats['blue_win_rate'], self.episode)
                self.writer.add_scalar('Eval/draw_rate', eval_stats['draw_rate'], self.episode)
                self.writer.add_scalar('Eval/avg_moves', eval_stats['avg_moves'], self.episode)

                # Update progress bar
                episode_bar.set_postfix({
                    'epsilon': f"{self.agent.epsilon:.3f}",
                    'red_wr': f"{eval_stats['red_win_rate']:.2f}",
                    'blue_wr': f"{eval_stats['blue_win_rate']:.2f}",
                    'draw': f"{eval_stats['draw_rate']:.2f}",
                    'moves': f"{eval_stats['avg_moves']:.1f}"
                })

            # Save checkpoint
            if self.episode % self.save_freq == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_ep{self.episode}.pt"
                )
                self.agent.save(checkpoint_path)

                # Also save as latest
                latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
                self.agent.save(latest_path)

        # Final save
        final_path = os.path.join(self.checkpoint_dir, "final.pt")
        self.agent.save(final_path)

        training_time = time.time() - self.training_start_time
        print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

        # Final evaluation
        print("\nFinal Evaluation:")
        final_stats = self.evaluate(num_games=200)
        self._print_eval_stats(final_stats)

        self.writer.close()

    def evaluate(self, num_games: Optional[int] = None) -> Dict:
        """
        Evaluate the agent's performance

        Args:
            num_games: Number of games to evaluate (uses self.eval_games if None)

        Returns:
            Evaluation statistics dictionary
        """
        if num_games is None:
            num_games = self.eval_games

        # Save current epsilon
        old_epsilon = self.agent.epsilon

        # Evaluate with greedy policy (epsilon=0)
        self.agent.epsilon = 0.0

        winners = []
        move_counts = []

        for _ in range(num_games):
            stats = self.play_episode(training=False)
            winners.append(stats['winner'])
            move_counts.append(stats['moves'])

        # Restore epsilon
        self.agent.epsilon = old_epsilon

        # Calculate statistics
        red_wins = winners.count(Player.RED)
        blue_wins = winners.count(Player.BLUE)
        draws = winners.count(Player.NONE)

        eval_stats = {
            'red_wins': red_wins,
            'blue_wins': blue_wins,
            'draws': draws,
            'red_win_rate': red_wins / num_games,
            'blue_win_rate': blue_wins / num_games,
            'draw_rate': draws / num_games,
            'avg_moves': np.mean(move_counts),
            'std_moves': np.std(move_counts)
        }

        return eval_stats

    def _print_eval_stats(self, stats: Dict):
        """Print evaluation statistics"""
        print(f"Red wins: {stats['red_wins']} ({stats['red_win_rate']*100:.1f}%)")
        print(f"Blue wins: {stats['blue_wins']} ({stats['blue_win_rate']*100:.1f}%)")
        print(f"Draws: {stats['draws']} ({stats['draw_rate']*100:.1f}%)")
        print(f"Avg moves: {stats['avg_moves']:.2f} ± {stats['std_moves']:.2f}")

    def play_against_random(self, num_games=100, agent_player=Player.RED) -> Dict:
        """
        Evaluate agent against random player

        Args:
            num_games: Number of games to play
            agent_player: Which player the agent controls

        Returns:
            Statistics dictionary
        """
        old_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0  # Greedy play

        wins = 0
        losses = 0
        draws = 0

        for _ in tqdm(range(num_games), desc=f"Playing as {agent_player.name}"):
            game = TicTacPro()

            while not game.game_over:
                current_player = game.current_player

                if current_player == agent_player:
                    # Agent's turn
                    action = self.agent.get_action(game, current_player)
                else:
                    # Random player's turn
                    legal_moves = game.get_legal_moves(current_player)
                    if legal_moves:
                        action = np.random.choice(len(legal_moves))
                        action = legal_moves[action]
                    else:
                        action = None

                if action is None:
                    break

                row, col, size = action
                game.make_move(row, col, size)

            # Record result
            if game.winner == agent_player:
                wins += 1
            elif game.winner == Player.NONE:
                draws += 1
            else:
                losses += 1

        self.agent.epsilon = old_epsilon

        stats = {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / num_games,
            'loss_rate': losses / num_games,
            'draw_rate': draws / num_games
        }

        print(f"\nAgent as {agent_player.name} vs Random:")
        print(f"Wins: {wins} ({stats['win_rate']*100:.1f}%)")
        print(f"Losses: {losses} ({stats['loss_rate']*100:.1f}%)")
        print(f"Draws: {draws} ({stats['draw_rate']*100:.1f}%)")

        return stats


if __name__ == "__main__":
    # Test the trainer
    print("Testing Trainer...")

    agent = DQNAgent(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995)
    trainer = Trainer(agent, save_freq=100, eval_freq=50)

    # Short training run
    print("\nRunning short training test (100 episodes)...")
    trainer.train(num_episodes=100)

    print("\nTest completed!")
