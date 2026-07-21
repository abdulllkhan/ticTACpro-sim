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

        # Greedy action selection — eval() disables Dropout for deterministic Q-values.
        state = torch.FloatTensor(game.get_state_tensor_normalized()).unsqueeze(0).to(self.device)

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state).cpu().numpy()[0]
        self.policy_net.train()

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

        # Double DQN: policy net selects action, target net evaluates it.
        # §134 negamax: states are normalized to current player's perspective,
        # so next_q is the *opponent's* value — subtract (not add) for zero-sum.
        with torch.no_grad():
            next_acts = self.policy_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(1, next_acts.unsqueeze(1)).squeeze(1)
            target_q_values = rewards - (1 - dones) * self.gamma * next_q_values

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
        state = torch.FloatTensor(game.get_state_tensor_normalized()).unsqueeze(0).to(self.device)

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state).cpu().numpy()[0]
        self.policy_net.train()

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
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

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

    §159+§160: When `device` is a CUDA device:
    - Data arrays (states, actions, rewards, next_states, dones) are stored
      as CUDA tensors — zero-copy path from buffer to training forward pass.
    - Priority array and cumsum live on GPU too — torch.searchsorted replaces
      numpy.searchsorted on a 76MB CPU array (eliminates 200ms CPU bottleneck).
    Result: sample() drops from ~200ms (CPU) to ~2ms (GPU).
    """

    def __init__(
        self,
        capacity: int,
        state_size: int = _STATE_SIZE,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 500_000,
        device=None,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_increment = (beta_end - beta_start) / max(beta_steps, 1)

        self.ptr = 0
        self.size = 0

        # §159: On CUDA, store data arrays as GPU tensors for zero-copy sample().
        import torch as _torch
        _dev = _torch.device(device) if device is not None and not hasattr(device, "type") else device
        self._on_device: bool = (_dev is not None and _dev.type == "cuda")
        self._device = _dev

        if self._on_device:
            self.states      = _torch.zeros((capacity, state_size), dtype=_torch.float32, device=_dev)
            self.actions     = _torch.zeros(capacity, dtype=_torch.int32, device=_dev)
            self.rewards     = _torch.zeros(capacity, dtype=_torch.float32, device=_dev)
            self.next_states = _torch.zeros((capacity, state_size), dtype=_torch.float32, device=_dev)
            self.dones       = _torch.zeros(capacity, dtype=_torch.float32, device=_dev)
            # §160: GPU priorities for torch.searchsorted — eliminates 200ms CPU bottleneck.
            self.priorities_gpu = _torch.ones(capacity, dtype=_torch.float32, device=_dev)
        else:
            self.states      = np.zeros((capacity, state_size), dtype=np.float32)
            self.actions     = np.zeros(capacity, dtype=np.int32)
            self.rewards     = np.zeros(capacity, dtype=np.float32)
            self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
            self.dones       = np.zeros(capacity, dtype=np.float32)

        self.priorities = np.ones(capacity, dtype=np.float32)  # CPU; used for update_priorities
        self.max_priority: float = 1.0
        # Cached cumsum refreshed every _REFRESH_INTERVAL sample() calls.
        self._REFRESH_INTERVAL = 16
        self._sample_calls = 0
        self._cumsum_cache: Optional[np.ndarray] = None       # CPU path
        self._cumsum_gpu = None                                # GPU path (§160)
        self._raw_cache: Optional[np.ndarray] = None
        self._cache_total: float = 0.0

    def add(self, state, action: int, reward: float, next_state, done: float):
        if self._on_device:
            import torch as _torch
            self.states[self.ptr]      = _torch.as_tensor(state, dtype=_torch.float32, device=self._device)
            self.actions[self.ptr]     = int(action)
            self.rewards[self.ptr]     = float(reward)
            self.next_states[self.ptr] = _torch.as_tensor(next_state, dtype=_torch.float32, device=self._device)
            self.dones[self.ptr]       = float(done)
        else:
            self.states[self.ptr]      = state
            self.actions[self.ptr]     = action
            self.rewards[self.ptr]     = reward
            self.next_states[self.ptr] = next_state
            self.dones[self.ptr]       = done
        self.priorities[self.ptr] = self.max_priority
        if self._on_device:
            self.priorities_gpu[self.ptr] = self.max_priority  # §160

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _bulk_write(
        self,
        st: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        nst: np.ndarray,
        don: np.ndarray,
        n: int,
    ) -> None:
        """§157: Shared circular-buffer write logic for add_batch and add_batch_arrays."""
        if n >= self.capacity:
            st  = st[-self.capacity:]
            act = act[-self.capacity:]
            rew = rew[-self.capacity:]
            nst = nst[-self.capacity:]
            don = don[-self.capacity:]
            n   = self.capacity

        tail = self.capacity - self.ptr

        if self._on_device:
            # §159: Copy numpy arrays to GPU tensor slices.
            import torch as _torch
            st_t  = _torch.as_tensor(st,  dtype=_torch.float32)
            act_t = _torch.as_tensor(act, dtype=_torch.int32)
            rew_t = _torch.as_tensor(rew, dtype=_torch.float32)
            nst_t = _torch.as_tensor(nst, dtype=_torch.float32)
            don_t = _torch.as_tensor(don, dtype=_torch.float32)

            if n <= tail:
                sl = slice(self.ptr, self.ptr + n)
                self.states[sl].copy_(st_t,   non_blocking=True)
                self.actions[sl].copy_(act_t, non_blocking=True)
                self.rewards[sl].copy_(rew_t, non_blocking=True)
                self.next_states[sl].copy_(nst_t, non_blocking=True)
                self.dones[sl].copy_(don_t,   non_blocking=True)
                self.priorities[sl] = self.max_priority
                self.priorities_gpu[sl] = self.max_priority       # §160
            else:
                rem = n - tail
                self.states[self.ptr:].copy_(st_t[:tail],  non_blocking=True)
                self.states[:rem].copy_(st_t[tail:],       non_blocking=True)
                self.actions[self.ptr:].copy_(act_t[:tail], non_blocking=True)
                self.actions[:rem].copy_(act_t[tail:],     non_blocking=True)
                self.rewards[self.ptr:].copy_(rew_t[:tail], non_blocking=True)
                self.rewards[:rem].copy_(rew_t[tail:],     non_blocking=True)
                self.next_states[self.ptr:].copy_(nst_t[:tail], non_blocking=True)
                self.next_states[:rem].copy_(nst_t[tail:],      non_blocking=True)
                self.dones[self.ptr:].copy_(don_t[:tail],  non_blocking=True)
                self.dones[:rem].copy_(don_t[tail:],       non_blocking=True)
                self.priorities[self.ptr:] = self.max_priority
                self.priorities[:rem]      = self.max_priority
                self.priorities_gpu[self.ptr:] = self.max_priority  # §160
                self.priorities_gpu[:rem]      = self.max_priority
        else:
            if n <= tail:
                sl = slice(self.ptr, self.ptr + n)
                self.states[sl]      = st
                self.actions[sl]     = act
                self.rewards[sl]     = rew
                self.next_states[sl] = nst
                self.dones[sl]       = don
                self.priorities[sl]  = self.max_priority
            else:
                rem = n - tail
                self.states[self.ptr:]      = st[:tail];    self.states[:rem]      = st[tail:]
                self.actions[self.ptr:]     = act[:tail];   self.actions[:rem]     = act[tail:]
                self.rewards[self.ptr:]     = rew[:tail];   self.rewards[:rem]     = rew[tail:]
                self.next_states[self.ptr:] = nst[:tail];   self.next_states[:rem] = nst[tail:]
                self.dones[self.ptr:]       = don[:tail];   self.dones[:rem]       = don[tail:]
                self.priorities[self.ptr:]  = self.max_priority
                self.priorities[:rem]       = self.max_priority

        self.ptr  = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def add_batch_arrays(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """§157: Direct numpy-array insert — bypasses zip(*tuples) + np.array() overhead."""
        st  = states.astype(np.float32, copy=False)
        act = actions.astype(np.int32,   copy=False)
        rew = rewards.astype(np.float32, copy=False)
        nst = next_states.astype(np.float32, copy=False)
        don = dones.astype(np.float32, copy=False)
        n   = len(st)
        if n == 0:
            return
        self._bulk_write(st, act, rew, nst, don, n)

    def add_batch(self, transitions: list) -> None:
        """
        §138 vectorized bulk insert: replaces per-item Python loop with
        contiguous numpy slice writes — ~100× faster for large augmented batches.
        Handles circular-buffer wrap-around with at most 2 contiguous writes.
        When n > capacity (e.g. symmetry-augmented batches > buffer size) only the
        last `capacity` items are kept, matching the semantics of per-item add().
        """
        n = len(transitions)
        if n == 0:
            return
        sa, aa, ra, nsa, da = zip(*transitions)
        st  = np.array(sa,  dtype=np.float32)
        act = np.array(aa,  dtype=np.int32)
        rew = np.array(ra,  dtype=np.float32)
        nst = np.array(nsa, dtype=np.float32)
        don = np.array(da,  dtype=np.float32)
        self._bulk_write(st, act, rew, nst, don, n)

    def sample(self, batch_size: int):
        self._sample_calls += 1
        _interval_tick = (self._sample_calls - 1) % self._REFRESH_INTERVAL == 0
        if self._on_device:
            need_refresh = self._cumsum_gpu is None or _interval_tick
        else:
            need_refresh = self._cumsum_cache is None or _interval_tick

        if self._on_device:
            # §160: GPU-side priority sampling — torch.searchsorted replaces
            # numpy.searchsorted on a 76MB CPU array (saves ~200ms per call).
            import torch as _torch
            if need_refresh or self._cumsum_gpu is None:
                with _torch.no_grad():
                    raw_gpu = self.priorities_gpu[:self.size].pow(self.alpha)
                    self._cumsum_gpu = _torch.cumsum(raw_gpu, dim=0)
                    self._raw_gpu    = raw_gpu
                    self._cache_total = float(self._cumsum_gpu[-1])

            total = self._cache_total
            r_gpu = _torch.rand(batch_size, device=self._device) * total
            idx_t = _torch.searchsorted(self._cumsum_gpu, r_gpu).clamp_(0, self.size - 1)

            # IS weights on GPU
            raw_at_idx = self._raw_gpu[idx_t]
            probs  = raw_at_idx / total
            w_gpu  = (self.size * probs).pow_(-self.beta)
            w_gpu  = (w_gpu / w_gpu.max()).float()
            self.beta = min(self.beta_end, self.beta + self.beta_increment)

            # §158: Sort for coalesced memory access on GPU
            sort_ord = _torch.argsort(idx_t)
            idx_t  = idx_t[sort_ord]
            w_gpu  = w_gpu[sort_ord]

            indices = idx_t.cpu().numpy()  # for update_priorities (CPU op)
            return (
                self.states[idx_t],
                self.actions[idx_t],
                self.rewards[idx_t],
                self.next_states[idx_t],
                self.dones[idx_t],
                indices,
                w_gpu,
            )

        # CPU path (unchanged)
        if need_refresh or self._cumsum_cache is None:
            raw = self.priorities[: self.size] ** self.alpha
            self._cumsum_cache = np.cumsum(raw)
            self._raw_cache = raw
            self._cache_total = float(self._cumsum_cache[-1])

        total = self._cache_total
        r = np.random.uniform(0.0, total, batch_size)
        indices = np.searchsorted(self._cumsum_cache, r, side="left")
        np.clip(indices, 0, self.size - 1, out=indices)

        probs = self._raw_cache[indices] / total
        weights = (self.size * probs) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)
        self.beta = min(self.beta_end, self.beta + self.beta_increment)

        # §158: Sort indices for cache-friendly gather
        sort_ord = np.argsort(indices, kind="stable")
        indices  = indices[sort_ord]
        weights  = weights[sort_ord]

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
        # §137: O(batch) local max instead of O(capacity) global scan.
        self.max_priority = max(float(self.max_priority), float(priorities.max()))
        if self._on_device:
            # §160: Sync GPU priorities to keep GPU searchsorted accurate.
            import torch as _torch
            with _torch.no_grad():
                idx_t  = _torch.from_numpy(indices).to(self._device)
                prio_t = _torch.from_numpy(priorities).to(self._device)
                self.priorities_gpu[idx_t] = prio_t

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
        tau: float = 0.005,
        beta_steps: int = 500_000,
        mc_returns: bool = False,
        lr_steps: int = 300_000,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        # §162: when mc_returns=True all stored dones=1.0, so skip next-state
        # inference in _compute_loss entirely — no CUDA sync, no wasted forwards.
        self._mc_returns = mc_returns
        # §146: Polyak soft-update coefficient. When τ > 0, target network is
        # updated every gradient step as: target = (1-τ)*target + τ*policy.
        # τ=0.005 gives ~200-step lag (1/τ), far smoother than hard copy every 5000.
        # When τ=0, falls back to hard copy every target_update_freq steps.
        self._tau = tau
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
        # §142/§170: CosineAnnealingLR with configurable T_max.
        # T_max is a half-period: LR decays from lr_max to eta_min over lr_steps,
        # then would rise again. Pass lr_steps ≥ total expected gradient steps to
        # keep LR monotonically decaying for the full run.
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=lr_steps, eta_min=1e-5
        )

        if self.use_amp and self._need_scaler:  # True only on pre-Blackwell (FP16) GPUs
            self.scaler = torch.amp.GradScaler("cuda")  # pragma: no cover
        else:
            self.scaler = None

        self.memory = PrioritizedReplayBuffer(
            buffer_size,
            state_size=state_size,
            beta_steps=beta_steps,
            device=self.device,  # §159: GPU-side buffer when training on CUDA
        )

        self.losses: list = []
        self._last_grad_norm: float = 0.0

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

    def get_actions_batch(self, games: list, epsilon: float = None,
                          precomputed_states: list = None) -> list:
        """
        Select actions for a list of game states using a single batched
        GPU forward pass for greedy actions.

        Returns list of (row, col, size) tuples, same length as `games`.
        Returns None for games with no legal moves.

        precomputed_states: optional list (same length as games) of pre-captured
        state tensors; avoids a redundant get_state_tensor_normalized() call when
        the caller has already captured states for transition storage (§140).
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
                if precomputed_states is not None:
                    greedy_states.append(precomputed_states[i])
                else:
                    greedy_states.append(game.get_state_tensor_normalized())

        if greedy_states:
            states_np = np.array(greedy_states, dtype=np.float32)
            states_t = torch.from_numpy(states_np).to(self.device, non_blocking=True)

            # GPUDQNNetwork has no Dropout/BatchNorm — train/eval modes are
            # equivalent, so we skip the toggle to avoid torch.compile recompiles.
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast("cuda", dtype=self._amp_dtype):
                        qv_t = self.policy_net(states_t)
                    q_np = qv_t.float().cpu().numpy()
                else:
                    q_np = self.policy_net(states_t).cpu().numpy()

            # §152: vectorized legal mask from state tensors — replaces per-game Python
            # loop (O(27) per game) with batched NumPy ops (O(1) per game amortised).
            # _legal_mask_from_state is verified identical to get_legal_moves (§149 test).
            legal_np = self._legal_mask_from_state(states_t).cpu().numpy()  # (n_greedy, 27)
            q_masked = np.where(legal_np, q_np, np.float32(-1e9))           # (n_greedy, 27)
            best_actions = q_masked.argmax(axis=1)                           # (n_greedy,)
            for j, gi in enumerate(greedy_ids):
                best = int(best_actions[j])
                actions[gi] = (best // 9, (best % 9) // 3, (best % 3) + 1)

        # Fallback for any remaining None with legal moves
        for i, (act, moves) in enumerate(zip(actions, legal_cache)):
            if act is None and moves:  # pragma: no cover — greedy pass never leaves None for valid positions
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

        # §159: GPU buffer returns CUDA tensors directly; CPU buffer returns numpy.
        if isinstance(s, np.ndarray):
            states      = torch.from_numpy(s).to(self.device, non_blocking=True)
            actions     = torch.from_numpy(a).long().to(self.device, non_blocking=True)
            rewards     = torch.from_numpy(r).to(self.device, non_blocking=True)
            next_states = torch.from_numpy(ns).to(self.device, non_blocking=True)
            dones       = torch.from_numpy(d).to(self.device, non_blocking=True)
            weights     = torch.from_numpy(w).to(self.device, non_blocking=True)
        else:
            states      = s
            actions     = a.long()
            rewards     = r
            next_states = ns
            dones       = d
            weights     = w

        self.policy_net.train()

        if self.use_amp:
            with torch.amp.autocast("cuda", dtype=self._amp_dtype):
                loss, td_err = self._compute_loss(states, actions, rewards, next_states, dones, weights)
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0))
                self.optimizer.step()
        else:
            loss, td_err = self._compute_loss(states, actions, rewards, next_states, dones, weights)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0))
            self.optimizer.step()
        self._last_grad_norm = grad_norm

        self.scheduler.step()

        td_np = td_err.detach().float().cpu().numpy()
        self.memory.update_priorities(idxs, td_np + 1e-6)

        self.learn_step_counter += 1
        raw_policy = getattr(self.policy_net, "_orig_mod", self.policy_net)
        raw_target = getattr(self.target_net, "_orig_mod", self.target_net)
        if self._tau > 0.0:
            # §146: Polyak EMA — no abrupt target jumps, ~1/τ step lag.
            _one_minus_tau = 1.0 - self._tau
            with torch.no_grad():
                for p_pol, p_tgt in zip(raw_policy.parameters(), raw_target.parameters()):
                    p_tgt.data.mul_(_one_minus_tau).add_(p_pol.data, alpha=self._tau)
        elif self.learn_step_counter % self.target_update_freq == 0:
            raw_target.load_state_dict(raw_policy.state_dict())
            self.target_net.eval()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        loss_val = float(loss.detach().cpu())
        self.losses.append(loss_val)
        return loss_val

    # Precomputed: action index a maps to size-slot a%3 (0=S,1=M,2=L).
    _ACT_SZ_IDX = torch.arange(27) % 3  # (27,) — moved to device in __init__ lazily

    def _legal_mask_from_state(self, states: torch.Tensor) -> torch.Tensor:
        """§149: Derive legal action mask from state tensor — no extra buffer storage.
        A cell (r,c,sz) is legal when:
          (a) neither player has a piece there: states[:, a] + states[:, 27+a] < 0.5
          (b) current player has pieces of that size: states[:, 54 + a%3] > 0
        Returns bool mask (B, 27) where True = legal.
        """
        sz_idx = self._ACT_SZ_IDX.to(states.device)   # (27,)
        not_occupied = (states[:, :27] + states[:, 27:54]) < 0.5  # (B, 27)
        piece_avail  = states[:, 54:57] > 0             # (B, 3)
        return not_occupied & piece_avail[:, sz_idx]    # (B, 27)

    def _compute_loss(self, states, actions, rewards, next_states, dones, weights):
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self._mc_returns:
                # §162: MC returns → all dones=1.0, bootstrap term always zero.
                # Skip both policy and target forward passes on next_states entirely.
                # Static flag avoids CUDA sync that a runtime .any() check would need.
                target_q = rewards
            else:
                # §149: mask illegal next-actions so argmax picks from legal moves only.
                # Without masking, early-training noise can push illegal actions to high Q,
                # biasing the TD target and slowing convergence.
                legal = self._legal_mask_from_state(next_states)  # (B, 27)
                next_q_all = self.policy_net(next_states).masked_fill(~legal, -1e9)
                next_acts = next_q_all.argmax(dim=1)
                next_q = self.target_net(next_states).gather(1, next_acts.unsqueeze(1)).squeeze(1)
                # §134 negamax: next_q is the opponent's best value; subtract.
                target_q = rewards - (1.0 - dones) * self.gamma * next_q
        td_errors = (current_q - target_q.detach()).abs()
        loss = (weights * F.huber_loss(current_q, target_q.detach(), reduction="none")).mean()
        return loss, td_errors

    # ------------------------------------------------------------------
    # Convenience helpers (keep API compatible with DQNAgent)
    # ------------------------------------------------------------------

    def store_experience(self, state, action: int, reward: float, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def get_q_values(self, game: TicTacPro) -> np.ndarray:
        state_t = torch.FloatTensor(game.get_state_tensor_normalized()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state_t).float().cpu().numpy()[0]
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

    def recalibrate_schedules(self, remaining_steps: int) -> None:
        """§184: recalibrate PER beta and LR schedules to the actual run length.

        Called once after the first log interval when measured throughput differs
        from the pre-run estimate.  Ensures beta reaches 1.0 and LR reaches
        eta_min by end of training instead of stalling mid-curve.
        """
        remaining_steps = max(remaining_steps, 1)
        # PER beta: anneal from current value to beta_end over remaining_steps
        self.memory.beta_increment = (
            (self.memory.beta_end - self.memory.beta) / remaining_steps
        )
        # LR: set T_max so cosine decay finishes at end of run.
        # Setting T_max = learn_step_counter + remaining_steps and keeping
        # last_epoch = learn_step_counter gives a smooth continuation that
        # reaches eta_min exactly when training stops.
        self.scheduler.T_max = self.learn_step_counter + remaining_steps

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
            "hidden_sizes": list(self.hidden_sizes),
            "state_size": self.state_size,
            "action_size": self.action_size,
            # §153: preserve PER beta so resuming continues the annealing schedule
            "per_beta": self.memory.beta,
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
        # §153: restore PER beta to continue annealing from where it left off
        if "per_beta" in ckpt:
            self.memory.beta = ckpt["per_beta"]
        saved_hs = ckpt.get("hidden_sizes")
        if saved_hs and tuple(saved_hs) != self.hidden_sizes:
            print(
                f"WARNING: checkpoint hidden_sizes={saved_hs} != agent hidden_sizes={list(self.hidden_sizes)}"
            )
        print(
            f"Loaded {filepath} | eps={self.episode_count:,} "
            f"steps={self.learn_step_counter:,} epsilon={self.epsilon:.4f}"
        )
        return True


if __name__ == "__main__":  # pragma: no cover
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
