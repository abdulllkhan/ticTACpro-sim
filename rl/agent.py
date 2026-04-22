"""
DQN Agent with Experience Replay for Tic Tac Pro

This module implements a Deep Q-Learning agent that learns to play Tic Tac Pro
through self-play and experience replay.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional
import os

from .network import DQNNetwork, ConvDQNNetwork
from game.tictacpro import TicTacPro, Player, PieceSize


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add an experience to the buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)

        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor(np.array([e.action for e in experiences]))
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences]))
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor(np.array([e.done for e in experiences]))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Learning Agent for Tic Tac Pro

    Features:
    - Experience replay
    - Target network for stable learning
    - Epsilon-greedy exploration
    - Support for GPU acceleration
    """

    def __init__(
        self,
        state_size=61,
        action_size=27,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000,
        use_conv=False,
        device=None
    ):
        """
        Initialize DQN Agent

        Args:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            buffer_size: Size of replay buffer
            batch_size: Size of training batches
            target_update_freq: How often to update target network
            use_conv: Whether to use convolutional network
            device: torch device (cuda/cpu)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"DQN Agent initialized on device: {self.device}")

        # Networks
        if use_conv:
            self.policy_net = ConvDQNNetwork(action_size).to(self.device)
            self.target_net = ConvDQNNetwork(action_size).to(self.device)
        else:
            self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
            self.target_net = DQNNetwork(state_size, action_size).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Training stats
        self.losses = []
        self.episode_rewards = []

    def get_action(self, game: TicTacPro, player: Player, epsilon: float = None) -> Optional[Tuple[int, int, PieceSize]]:
        """
        Select an action using epsilon-greedy policy

        Args:
            game: Current game state
            player: Player to select action for
            epsilon: Exploration rate (uses self.epsilon if None)

        Returns:
            (row, col, size) tuple or None if no legal moves
        """
        if epsilon is None:
            epsilon = self.epsilon

        legal_moves = game.get_legal_moves(player)

        if not legal_moves:
            return None

        # Epsilon-greedy exploration
        if random.random() < epsilon:
            return random.choice(legal_moves)

        # Greedy action selection
        state = torch.FloatTensor(game.get_state_tensor()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state).cpu().numpy()[0]

        # Mask illegal actions with very negative values
        action_mask = np.full(self.action_size, -1e9)

        for row, col, size in legal_moves:
            action_idx = self.policy_net.get_action_index(row, col, size)
            action_mask[action_idx] = q_values[action_idx]

        # Select best legal action
        best_action_idx = np.argmax(action_mask)
        return self.policy_net.get_action_from_index(best_action_idx)

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """Train the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.losses.append(loss.item())
        return loss.item()

    def get_q_values(self, game: TicTacPro) -> np.ndarray:
        """
        Get Q-values for all actions in current state

        Args:
            game: Current game state

        Returns:
            Q-values array of shape (action_size,)
        """
        state = torch.FloatTensor(game.get_state_tensor()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state).cpu().numpy()[0]

        return q_values

    def get_move_suggestions(self, game: TicTacPro, player: Player, top_k=3) -> List[Tuple[Tuple[int, int, int], float]]:
        """
        Get top-k move suggestions with their Q-values

        Args:
            game: Current game state
            player: Player to get suggestions for
            top_k: Number of top moves to return

        Returns:
            List of ((row, col, size), q_value) tuples
        """
        legal_moves = game.get_legal_moves(player)

        if not legal_moves:
            return []

        q_values = self.get_q_values(game)

        # Get Q-values for legal moves
        move_q_values = []
        for row, col, size in legal_moves:
            action_idx = self.policy_net.get_action_index(row, col, size)
            move_q_values.append(((row, col, size), q_values[action_idx]))

        # Sort by Q-value and return top-k
        move_q_values.sort(key=lambda x: x[1], reverse=True)
        return move_q_values[:top_k]

    def save(self, filepath):
        """Save the agent's network weights and training stats"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step_counter': self.learn_step_counter,
            'losses': self.losses,
            'episode_rewards': self.episode_rewards
        }

        torch.save(checkpoint, filepath)
        print(f"Agent saved to {filepath}")

    def load(self, filepath):
        """Load the agent's network weights and training stats"""
        if not os.path.exists(filepath):
            print(f"Checkpoint not found: {filepath}")
            return False

        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.learn_step_counter = checkpoint['learn_step_counter']
        self.losses = checkpoint.get('losses', [])
        self.episode_rewards = checkpoint.get('episode_rewards', [])

        print(f"Agent loaded from {filepath}")
        print(f"Epsilon: {self.epsilon:.4f}, Steps: {self.learn_step_counter}")
        return True


if __name__ == "__main__":
    # Test the agent
    print("Testing DQN Agent...")
    agent = DQNAgent()

    # Test with a game
    game = TicTacPro()
    print("\nGame state:")
    print(game)

    # Get action
    action = agent.get_action(game, Player.RED, epsilon=0.5)
    print(f"\nSelected action: {action}")

    # Get Q-values
    q_values = agent.get_q_values(game)
    print(f"\nQ-values shape: {q_values.shape}")

    # Get move suggestions
    suggestions = agent.get_move_suggestions(game, Player.RED, top_k=5)
    print("\nTop 5 move suggestions:")
    for i, ((row, col, size), q_val) in enumerate(suggestions, 1):
        print(f"{i}. Position ({row}, {col}), Size {size}: Q-value = {q_val:.4f}")
