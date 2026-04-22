"""
Deep Q-Network (DQN) architecture for Tic Tac Pro

The network takes the game state as input and outputs Q-values for all possible actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Tic Tac Pro

    Input: State tensor (61 features)
      - 54 board features (3x3x3x2 for red and blue boards)
      - 6 piece count features (remaining pieces for each size and player)
      - 1 current player feature

    Output: Q-values for all possible actions
      - 81 possible actions (9 positions * 9 piece types, though many will be invalid)
      - We use 3x3x3 = 27 actions (row, col, size)
    """

    def __init__(self, state_size=61, action_size=27, hidden_sizes=[256, 256, 128]):
        """
        Initialize the DQN network

        Args:
            state_size: Size of the input state vector
            action_size: Number of possible actions (3x3 positions * 3 sizes = 27)
            hidden_sizes: List of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        # Build fully connected layers
        layers = []
        prev_size = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, action_size))

        self.network = nn.Sequential(*layers)

        # Alternative architecture with separate value and advantage streams (Dueling DQN)
        self.use_dueling = True
        if self.use_dueling:
            self.feature_layers = nn.Sequential(
                nn.Linear(state_size, hidden_sizes[0]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
                nn.Dropout(0.2)
            )

            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[2], 1)
            )

            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[2], action_size)
            )

    def forward(self, state):
        """
        Forward pass through the network

        Args:
            state: Tensor of shape (batch_size, state_size) or (state_size,)

        Returns:
            Q-values tensor of shape (batch_size, action_size) or (action_size,)
        """
        if self.use_dueling:
            # Dueling DQN architecture
            features = self.feature_layers(state)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)

            # Combine value and advantage: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
            return q_values
        else:
            return self.network(state)

    def get_action_index(self, row, col, size):
        """
        Convert (row, col, size) to action index

        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            size: Piece size (1-3 for SMALL, MEDIUM, LARGE)

        Returns:
            Action index (0-26)
        """
        # Action index = row * 9 + col * 3 + (size - 1)
        return row * 9 + col * 3 + (size - 1)

    def get_action_from_index(self, action_idx):
        """
        Convert action index to (row, col, size)

        Args:
            action_idx: Action index (0-26)

        Returns:
            Tuple of (row, col, size)
        """
        row = action_idx // 9
        col = (action_idx % 9) // 3
        size = (action_idx % 3) + 1  # 1, 2, or 3
        return row, col, size


class ConvDQNNetwork(nn.Module):
    """
    Convolutional DQN for Tic Tac Pro (alternative architecture)

    This network treats the board as a 3D image and uses convolutional layers
    to extract spatial features before outputting Q-values.
    """

    def __init__(self, action_size=27):
        super(ConvDQNNetwork, self).__init__()

        self.action_size = action_size

        # Input: 2 channels (red and blue) x 3 rows x 3 cols x 3 sizes
        # We'll reshape to (6, 3, 3) where channels are:
        # red-small, red-medium, red-large, blue-small, blue-medium, blue-large

        self.conv1 = nn.Conv2d(6, 32, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0)

        # Calculate conv output size: (3x3 input -> 4x4 after conv1 -> 3x3 after conv2)
        conv_output_size = 64 * 3 * 3  # 576

        # Fully connected layers
        # Add piece count and current player features
        fc_input_size = conv_output_size + 7  # +7 for piece counts and current player

        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, state):
        """
        Forward pass

        Args:
            state: State tensor (batch_size, 61) or (61,)

        Returns:
            Q-values (batch_size, action_size) or (action_size,)
        """
        batch = state.dim() == 2
        if not batch:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]

        # Reshape state to separate board and other features
        # First 54 features are board (27 red + 27 blue)
        board_features = state[:, :54]
        other_features = state[:, 54:]  # Piece counts + current player

        # Reshape board to (batch, 6, 3, 3)
        # 6 channels: red-s, red-m, red-l, blue-s, blue-m, blue-l
        board = board_features.view(batch_size, 2, 3, 3, 3)  # (batch, player, row, col, size)
        board = board.permute(0, 1, 4, 2, 3)  # (batch, player, size, row, col)
        board = board.contiguous().view(batch_size, 6, 3, 3)  # (batch, channels, row, col)

        # Convolutional layers
        x = F.relu(self.conv1(board))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)  # Flatten

        # Concatenate with other features
        x = torch.cat([x, other_features], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        q_values = self.fc3(x)

        if not batch:
            q_values = q_values.squeeze(0)

        return q_values

    def get_action_index(self, row, col, size):
        """Convert (row, col, size) to action index"""
        return row * 9 + col * 3 + (size - 1)

    def get_action_from_index(self, action_idx):
        """Convert action index to (row, col, size)"""
        row = action_idx // 9
        col = (action_idx % 9) // 3
        size = (action_idx % 3) + 1
        return row, col, size


if __name__ == "__main__":
    # Test the networks
    print("Testing DQN Network...")
    net = DQNNetwork()
    dummy_state = torch.randn(4, 61)  # Batch of 4 states
    q_values = net(dummy_state)
    print(f"Input shape: {dummy_state.shape}")
    print(f"Output shape: {q_values.shape}")
    print(f"Q-values sample: {q_values[0, :5]}")

    print("\nTesting Conv DQN Network...")
    conv_net = ConvDQNNetwork()
    q_values_conv = conv_net(dummy_state)
    print(f"Input shape: {dummy_state.shape}")
    print(f"Output shape: {q_values_conv.shape}")
    print(f"Q-values sample: {q_values_conv[0, :5]}")

    print("\nAction index conversion test:")
    print(f"(0, 0, 1) -> {net.get_action_index(0, 0, 1)}")
    print(f"(1, 1, 2) -> {net.get_action_index(1, 1, 2)}")
    print(f"(2, 2, 3) -> {net.get_action_index(2, 2, 3)}")
    print(f"Index 13 -> {net.get_action_from_index(13)}")
