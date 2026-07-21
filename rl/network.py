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
        features = self.feature_layers(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))

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


class GPUDQNNetwork(nn.Module):
    """
    GPU-optimized Dueling DQN for DGX Spark GB10.

    Uses LayerNorm (works at any batch size), orthogonal initialization, and
    hidden sizes that are multiples of 64 for peak Tensor Core occupancy.
    Larger capacity than DQNNetwork for sustained multi-hour training runs.
    """

    def __init__(self, state_size: int = 61, action_size: int = 27,
                 hidden_sizes: tuple = (1024, 512, 256),
                 init_weights: bool = True):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.features = nn.Sequential(
            nn.Linear(state_size, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.LayerNorm(hidden_sizes[2]),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_sizes[2], 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_sizes[2], 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

        # §163: skip in CPU workers that immediately overwrite weights via
        # load_state_dict() — orthogonal init on (512,1024) costs ~4s per call.
        if init_weights:
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # §161: output layers use near-zero init so Q-values start small.
        # Large initial Q-values produce oversized losses and slow early learning.
        nn.init.orthogonal_(self.value_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.value_head[-1].bias)
        nn.init.orthogonal_(self.advantage_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.advantage_head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))

    def get_action_index(self, row: int, col: int, size: int) -> int:
        return row * 9 + col * 3 + (size - 1)

    def get_action_from_index(self, idx: int) -> tuple:
        row = idx // 9
        col = (idx % 9) // 3
        size = (idx % 3) + 1
        return row, col, size


class NumpyDQNInference:
    """
    §164: Pure-numpy inference for CPU workers. Extracts weights from a
    GPUDQNNetwork state-dict and runs forward passes via numpy matrix ops,
    eliminating PyTorch's per-call dispatch overhead (~2.7 ms per nn.Linear
    call). Measured speedup: 19× over GPUDQNNetwork on CPU (1.3 ms vs 24 ms
    per batch of 8 states). Workers run ~18 batched DQN calls per cycle;
    total worker inference time drops from ~432 ms to ~24 ms per cycle.

    Only supports GPUDQNNetwork layout (3 feature layers + LN, dueling heads).
    Immutable after construction — rebuild from a fresh state_dict each cycle.
    """

    _EPS = 1e-5

    def __init__(self, state_dict: dict, hidden_sizes: tuple = (1024, 512, 256)):
        def w(key):
            t = state_dict[key]
            return t.numpy() if hasattr(t, "numpy") else np.array(t)

        # Feature layers: Linear → LayerNorm → ReLU (repeated 3×)
        self._W0  = w("features.0.weight")   # (h0, state_size)
        self._b0  = w("features.0.bias")
        self._LNw0 = w("features.1.weight"); self._LNb0 = w("features.1.bias")

        self._W1  = w("features.3.weight")   # (h1, h0)
        self._b1  = w("features.3.bias")
        self._LNw1 = w("features.4.weight"); self._LNb1 = w("features.4.bias")

        self._W2  = w("features.6.weight")   # (h2, h1)
        self._b2  = w("features.6.bias")
        self._LNw2 = w("features.7.weight"); self._LNb2 = w("features.7.bias")

        # Value head: Linear → ReLU → Linear
        self._Wv1 = w("value_head.0.weight"); self._bv1 = w("value_head.0.bias")
        self._Wv2 = w("value_head.2.weight"); self._bv2 = w("value_head.2.bias")

        # Advantage head: Linear → ReLU → Linear
        self._Wa1 = w("advantage_head.0.weight"); self._ba1 = w("advantage_head.0.bias")
        self._Wa2 = w("advantage_head.2.weight"); self._ba2 = w("advantage_head.2.bias")

    @staticmethod
    def _ln(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        mu  = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        return w * (x - mu) / np.sqrt(var + NumpyDQNInference._EPS) + b

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass. x: float32 array (B, state_size) or (state_size,).
        Returns Q-values: (B, action_size) or (action_size,).
        """
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis]   # (1, state_size)

        h = np.maximum(0, self._ln(x @ self._W0.T + self._b0, self._LNw0, self._LNb0))
        h = np.maximum(0, self._ln(h @ self._W1.T + self._b1, self._LNw1, self._LNb1))
        h = np.maximum(0, self._ln(h @ self._W2.T + self._b2, self._LNw2, self._LNb2))

        value = np.maximum(0, h @ self._Wv1.T + self._bv1) @ self._Wv2.T + self._bv2
        adv   = np.maximum(0, h @ self._Wa1.T + self._ba1) @ self._Wa2.T + self._ba2

        q = value + (adv - adv.mean(axis=-1, keepdims=True))
        return q[0] if squeeze else q

    @staticmethod
    def from_network(net: "GPUDQNNetwork") -> "NumpyDQNInference":
        with torch.no_grad():
            return NumpyDQNInference(net.state_dict())

    def get_action_index(self, row: int, col: int, size: int) -> int:
        return row * 9 + col * 3 + (size - 1)

    def get_action_from_index(self, idx: int) -> tuple:
        row = idx // 9
        col = (idx % 9) // 3
        size = (idx % 3) + 1
        return row, col, size


if __name__ == "__main__":  # pragma: no cover
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

    print("\nTesting GPU DQN Network...")
    gpu_net = GPUDQNNetwork()
    q_values_gpu = gpu_net(dummy_state)
    print(f"Input shape: {dummy_state.shape}")
    print(f"Output shape: {q_values_gpu.shape}")
    print(f"Q-values sample: {q_values_gpu[0, :5]}")

    print("\nAction index conversion test:")
    print(f"(0, 0, 1) -> {net.get_action_index(0, 0, 1)}")
    print(f"(1, 1, 2) -> {net.get_action_index(1, 1, 2)}")
    print(f"(2, 2, 3) -> {net.get_action_index(2, 2, 3)}")
    print(f"Index 13 -> {net.get_action_from_index(13)}")
