"""
DQN Agent with Experience Replay for Tic Tac Pro

This module implements a Deep Q-Learning agent that learns to play Tic Tac Pro
through self-play and experience replay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional
import os

from .network import DQNNetwork, ConvDQNNetwork, GPUDQNNetwork
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


_STATE_SIZE = 61
_ACTION_SIZE = 27


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay using proportional priorities.
    Pre-allocates all storage as numpy arrays for minimal overhead.
    Supports add_batch for efficient bulk insertion after vectorized rollouts.
    """

    def __init__(
        self,
        capacity: int,
        state_size: int = _STATE_SIZE,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 500_000,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_increment = (beta_end - beta_start) / max(beta_steps, 1)

        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.max_priority: float = 1.0
        # Cached cumsum refreshed every _REFRESH_INTERVAL sample() calls.
        # Amortises the O(N) cumsum cost over many gradient steps.
        self._REFRESH_INTERVAL = 16
        self._sample_calls = 0
        self._cumsum_cache: Optional[np.ndarray] = None
        self._raw_cache: Optional[np.ndarray] = None
        self._cache_total: float = 0.0

    def add(self, state, action: int, reward: float, next_state, done: float):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.priorities[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, transitions: list):
        for state, action, reward, next_state, done in transitions:
            self.add(state, action, reward, next_state, done)

    def sample(self, batch_size: int):
        # Rebuild cumsum every _REFRESH_INTERVAL calls to amortise O(N) cost.
        # Each refresh covers ~2.5% buffer changes (negligible distribution drift).
        if self._cumsum_cache is None or self._sample_calls % self._REFRESH_INTERVAL == 0:
            raw = self.priorities[: self.size] ** self.alpha
            self._cumsum_cache = np.cumsum(raw)
            self._raw_cache = raw
            self._cache_total = float(self._cumsum_cache[-1])
        self._sample_calls += 1

        total = self._cache_total
        # Vectorised searchsorted — O(batch*logN) vs O(N) per-sample loop
        r = np.random.uniform(0.0, total, batch_size)
        indices = np.searchsorted(self._cumsum_cache, r, side="left")
        np.clip(indices, 0, self.size - 1, out=indices)

        # Importance-sampling weights — annealed toward unbiased as beta -> 1
        probs = self._raw_cache[indices] / total
        weights = (self.size * probs) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)
        self.beta = min(self.beta_end, self.beta + self.beta_increment)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        np.clip(priorities, 1e-6, None, out=priorities)
        self.priorities[indices] = priorities
        self.max_priority = float(self.priorities[: self.size].max())

    def __len__(self) -> int:
        return self.size


class GPUDQNAgent:
    """
    GPU-optimized Double DQN agent for NVIDIA DGX Spark GB10.

    Improvements over base DQNAgent:
    - Prioritized Experience Replay (PER) with IS weights
    - Double DQN (policy net selects, target net evaluates)
    - BF16 Automatic Mixed Precision via torch.amp.autocast
    - Batched action selection for vectorized rollouts (no per-game GPU call)
    - Pre-allocated numpy replay buffer (minimal Python GC pressure)
    - AdamW + CosineAnnealing LR schedule
    - torch.compile() for Inductor kernel fusion on Blackwell
    """

    def __init__(
        self,
        state_size: int = 61,
        action_size: int = 27,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9999900,
        buffer_size: int = 10_000_000,
        batch_size: int = 16384,
        target_update_freq: int = 5000,
        use_amp: bool = True,
        compile_model: bool = True,
        device=None,
        hidden_sizes: tuple = (1024, 512, 256),
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0
        self.episode_count = 0

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_amp = use_amp and self.device.type == "cuda"

        # Blackwell (compute ≥ 9) supports BF16 natively; skip GradScaler.
        _major = torch.cuda.get_device_properties(0).major if self.device.type == "cuda" else 0
        self._amp_dtype = torch.bfloat16 if _major >= 9 else torch.float16
        self._need_scaler = self._amp_dtype == torch.float16

        self.policy_net = GPUDQNNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_net = GPUDQNNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False

        if compile_model and hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                self.policy_net = torch.compile(self.policy_net, backend="inductor")
                self.target_net = torch.compile(self.target_net, backend="inductor")
                print("torch.compile enabled (Inductor backend)")
            except Exception as e:
                print(f"torch.compile skipped: {e}")

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=learning_rate,
            eps=1e-5,
            weight_decay=1e-5,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=2_000_000, eta_min=1e-5
        )

        if self.use_amp and self._need_scaler:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        self.memory = PrioritizedReplayBuffer(
            buffer_size,
            state_size=state_size,
            beta_steps=500_000,
        )

        self.losses: list = []

        if self.device.type == "cuda":
            total_p = sum(p.numel() for p in self.policy_net.parameters())
            mem_mb = total_p * 4 * 2 / 1024 ** 2  # 2 nets, FP32
            print(
                f"GPUDQNAgent on {self.device} | AMP={self.use_amp} ({self._amp_dtype}) | "
                f"params={total_p:,} | model_mem={mem_mb:.1f}MB | "
                f"buffer={buffer_size:,} | batch={batch_size:,}"
            )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_actions_batch(self, games: list, epsilon: float = None) -> list:
        """
        Select actions for a list of game states using a single batched
        GPU forward pass for greedy actions.

        Returns list of (row, col, size) tuples, same length as `games`.
        Returns None for games with no legal moves.
        """
        if epsilon is None:
            epsilon = self.epsilon

        n = len(games)
        actions: list = [None] * n
        greedy_ids: list = []
        greedy_states: list = []
        legal_cache: list = []

        for i, game in enumerate(games):
            moves = game.get_legal_moves(game.current_player)
            legal_cache.append(moves)
            if not moves:
                continue
            if random.random() < epsilon:
                actions[i] = random.choice(moves)
            else:
                greedy_ids.append(i)
                greedy_states.append(game.get_state_tensor())

        if greedy_states:
            states_np = np.array(greedy_states, dtype=np.float32)
            states_t = torch.from_numpy(states_np).to(self.device, non_blocking=True)

            self.policy_net.eval()
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast("cuda", dtype=self._amp_dtype):
                        qv_t = self.policy_net(states_t)
                    q_np = qv_t.float().cpu().numpy()
                else:
                    q_np = self.policy_net(states_t).cpu().numpy()
            self.policy_net.train()

            for j, gi in enumerate(greedy_ids):
                moves = legal_cache[gi]
                qv = q_np[j]
                mask = np.full(self.action_size, -1e9, dtype=np.float32)
                for row, col, sz in moves:
                    idx = row * 9 + col * 3 + (sz - 1)
                    mask[idx] = qv[idx]
                best = int(np.argmax(mask))
                actions[gi] = (best // 9, (best % 9) // 3, (best % 3) + 1)

        # Fallback for any remaining None with legal moves
        for i, (act, moves) in enumerate(zip(actions, legal_cache)):
            if act is None and moves:
                actions[i] = random.choice(moves)

        return actions

    # Compatibility shim for legacy Trainer
    def get_action(self, game: TicTacPro, player: Player, epsilon: float = None):
        result = self.get_actions_batch([game], epsilon=epsilon)
        return result[0]

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self) -> Optional[float]:
        """Single Double-DQN gradient step with PER importance weights."""
        if len(self.memory) < self.batch_size:
            return None

        s, a, r, ns, d, idxs, w = self.memory.sample(self.batch_size)

        states = torch.from_numpy(s).to(self.device, non_blocking=True)
        actions = torch.from_numpy(a).long().to(self.device, non_blocking=True)
        rewards = torch.from_numpy(r).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(ns).to(self.device, non_blocking=True)
        dones = torch.from_numpy(d).to(self.device, non_blocking=True)
        weights = torch.from_numpy(w).to(self.device, non_blocking=True)

        self.policy_net.train()

        if self.use_amp:
            with torch.amp.autocast("cuda", dtype=self._amp_dtype):
                loss, td_err = self._compute_loss(states, actions, rewards, next_states, dones, weights)
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
                self.optimizer.step()
        else:
            loss, td_err = self._compute_loss(states, actions, rewards, next_states, dones, weights)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
            self.optimizer.step()

        self.scheduler.step()

        td_np = td_err.detach().float().cpu().numpy()
        self.memory.update_priorities(idxs, td_np + 1e-6)

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(
                {k: v for k, v in self.policy_net.state_dict().items()}
            )
            self.target_net.eval()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        loss_val = float(loss.detach().cpu())
        self.losses.append(loss_val)
        return loss_val

    def _compute_loss(self, states, actions, rewards, next_states, dones, weights):
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_acts = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_acts.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q
        td_errors = (current_q - target_q.detach()).abs()
        loss = (weights * F.huber_loss(current_q, target_q.detach(), reduction="none")).mean()
        return loss, td_errors

    # ------------------------------------------------------------------
    # Convenience helpers (keep API compatible with DQNAgent)
    # ------------------------------------------------------------------

    def store_experience(self, state, action: int, reward: float, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def get_q_values(self, game: TicTacPro) -> np.ndarray:
        state_t = torch.FloatTensor(game.get_state_tensor()).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q = self.policy_net(state_t).float().cpu().numpy()[0]
        self.policy_net.train()
        return q

    def get_move_suggestions(self, game: TicTacPro, player: Player, top_k: int = 3):
        moves = game.get_legal_moves(player)
        if not moves:
            return []
        qv = self.get_q_values(game)
        ranked = sorted(
            [((r, c, s), qv[r * 9 + c * 3 + (s - 1)]) for r, c, s in moves],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        # Unwrap compiled module for serialization
        policy_state = getattr(self.policy_net, "_orig_mod", self.policy_net).state_dict()
        target_state = getattr(self.target_net, "_orig_mod", self.target_net).state_dict()
        ckpt = {
            "policy_net": policy_state,
            "target_net": target_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epsilon": self.epsilon,
            "learn_step_counter": self.learn_step_counter,
            "episode_count": self.episode_count,
            "losses": self.losses[-10_000:],
        }
        if self.scaler is not None:
            ckpt["scaler"] = self.scaler.state_dict()
        torch.save(ckpt, filepath)

    def load(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
        raw_policy = getattr(self.policy_net, "_orig_mod", self.policy_net)
        raw_target = getattr(self.target_net, "_orig_mod", self.target_net)
        raw_policy.load_state_dict(ckpt["policy_net"])
        raw_target.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.learn_step_counter = ckpt.get("learn_step_counter", 0)
        self.episode_count = ckpt.get("episode_count", 0)
        self.losses = ckpt.get("losses", [])
        if self.scaler is not None and "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        print(
            f"Loaded {filepath} | eps={self.episode_count:,} "
            f"steps={self.learn_step_counter:,} epsilon={self.epsilon:.4f}"
        )
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
