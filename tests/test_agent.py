"""Unit tests for PrioritizedReplayBuffer and GPUDQNAgent."""

import pytest
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rl.agent import PrioritizedReplayBuffer, GPUDQNAgent
from game.tictacpro import TicTacPro, Player, PieceSize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer
# ---------------------------------------------------------------------------

class TestPrioritizedReplayBuffer:
    def _make_buf(self, cap=1000):
        return PrioritizedReplayBuffer(capacity=cap, state_size=61)

    def _dummy(self):
        return (
            np.random.randn(61).astype(np.float32),
            int(np.random.randint(0, 27)),
            float(np.random.randn()),
            np.random.randn(61).astype(np.float32),
            0.0,
        )

    def test_len_zero_initially(self):
        buf = self._make_buf()
        assert len(buf) == 0

    def test_add_increments_len(self):
        buf = self._make_buf()
        buf.add(*self._dummy())
        assert len(buf) == 1

    def test_len_capped_at_capacity(self):
        buf = self._make_buf(cap=10)
        for _ in range(20):
            buf.add(*self._dummy())
        assert len(buf) == 10

    def test_sample_shapes(self):
        buf = self._make_buf()
        for _ in range(200):
            buf.add(*self._dummy())
        s, a, r, ns, d, idx, w = buf.sample(64)
        assert s.shape == (64, 61)
        assert a.shape == (64,)
        assert r.shape == (64,)
        assert ns.shape == (64, 61)
        assert d.shape == (64,)
        assert idx.shape == (64,)
        assert w.shape == (64,)

    def test_weights_leq_one(self):
        buf = self._make_buf()
        for _ in range(500):
            buf.add(*self._dummy())
        *_, w = buf.sample(128)
        assert w.max() <= 1.0 + 1e-6
        assert w.min() > 0

    def test_update_priorities_changes_sampling(self):
        buf = self._make_buf(cap=100)
        for _ in range(100):
            buf.add(*self._dummy())
        # Give index 0 very high priority
        buf.update_priorities(np.array([0]), np.array([1e6]))
        counts = np.zeros(100)
        for _ in range(1000):
            *_, idxs, _ = buf.sample(10)
            counts[idxs] += 1
        # Index 0 should be sampled much more often
        assert counts[0] > counts.mean() * 5

    def test_add_batch(self):
        buf = self._make_buf()
        transitions = [self._dummy() for _ in range(50)]
        buf.add_batch(transitions)
        assert len(buf) == 50

    def test_beta_increases_over_samples(self):
        buf = PrioritizedReplayBuffer(1000, beta_start=0.4, beta_end=1.0, beta_steps=10)
        for _ in range(200):
            buf.add(*self._dummy())
        b0 = buf.beta
        buf.sample(10)
        assert buf.beta > b0


# ---------------------------------------------------------------------------
# GPUDQNAgent
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_agent():
    """Lightweight agent for fast tests (no torch.compile, tiny buffer)."""
    return GPUDQNAgent(
        buffer_size=5000,
        batch_size=32,
        hidden_sizes=(64, 64, 32),
        use_amp=False,
        compile_model=False,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
        device=DEVICE,
    )


class TestGPUDQNAgentInit:
    def test_device_assigned(self, small_agent):
        assert small_agent.device == DEVICE

    def test_epsilon_starts_at_one(self, small_agent):
        assert small_agent.epsilon == 1.0

    def test_memory_empty(self, small_agent):
        assert len(small_agent.memory) == 0


class TestGPUDQNAgentActions:
    def test_get_action_returns_valid_move(self, small_agent):
        game = TicTacPro()
        moves = game.get_legal_moves(Player.RED)
        action = small_agent.get_action(game, Player.RED, epsilon=1.0)
        assert action in moves

    def test_get_actions_batch_length(self, small_agent):
        games = [TicTacPro() for _ in range(16)]
        actions = small_agent.get_actions_batch(games, epsilon=1.0)
        assert len(actions) == 16

    def test_get_actions_batch_all_valid(self, small_agent):
        games = [TicTacPro() for _ in range(8)]
        actions = small_agent.get_actions_batch(games, epsilon=0.0)
        for game, act in zip(games, actions):
            assert act is not None
            row, col, size = act
            assert (row, col, size) in game.get_legal_moves(game.current_player)

    def test_greedy_action_is_legal(self, small_agent):
        game = TicTacPro()
        action = small_agent.get_action(game, Player.RED, epsilon=0.0)
        assert action in game.get_legal_moves(Player.RED)


class TestGPUDQNAgentLearning:
    def _fill_buffer(self, agent, n=100):
        game = TicTacPro()
        for _ in range(n):
            state = game.get_state_tensor()
            action = np.random.randint(0, 27)
            reward = float(np.random.randn())
            next_state = game.get_state_tensor()
            agent.memory.add(state, action, reward, next_state, 0.0)

    def test_learn_returns_none_when_buffer_empty(self, small_agent):
        agent = GPUDQNAgent(
            buffer_size=100, batch_size=32, hidden_sizes=(32, 32, 16),
            use_amp=False, compile_model=False, device=DEVICE,
        )
        assert agent.learn() is None

    def test_learn_returns_loss_when_buffer_full(self):
        agent = GPUDQNAgent(
            buffer_size=500, batch_size=32, hidden_sizes=(32, 32, 16),
            use_amp=False, compile_model=False, device=DEVICE,
        )
        for _ in range(100):
            s = np.random.randn(61).astype(np.float32)
            ns = np.random.randn(61).astype(np.float32)
            agent.memory.add(s, int(np.random.randint(27)), float(np.random.randn()), ns, 0.0)
        loss = agent.learn()
        assert loss is not None
        assert np.isfinite(loss)

    def test_loss_decreases_over_many_steps(self):
        """Loss should trend downward with a trivially learnable task."""
        agent = GPUDQNAgent(
            buffer_size=2000, batch_size=64, hidden_sizes=(64, 64, 32),
            use_amp=False, compile_model=False,
            epsilon_decay=1.0,  # don't decay
            device=DEVICE,
        )
        # Fill with uniform reward=1.0 transitions
        for _ in range(500):
            s = np.random.randn(61).astype(np.float32)
            ns = np.random.randn(61).astype(np.float32)
            agent.memory.add(s, 0, 1.0, ns, 1.0)

        early_losses, late_losses = [], []
        for i in range(200):
            l = agent.learn()
            if l is not None:
                (early_losses if i < 50 else late_losses).append(l)

        if early_losses and late_losses:
            # Late losses should be lower (allow generous margin)
            assert np.mean(late_losses) < np.mean(early_losses) * 2

    def test_epsilon_decays_on_learn(self):
        agent = GPUDQNAgent(
            buffer_size=500, batch_size=32, hidden_sizes=(32, 32, 16),
            use_amp=False, compile_model=False,
            epsilon_start=1.0, epsilon_decay=0.9, device=DEVICE,
        )
        for _ in range(50):
            s = np.random.randn(61).astype(np.float32)
            ns = np.random.randn(61).astype(np.float32)
            agent.memory.add(s, int(np.random.randint(27)), 0.0, ns, 0.0)
        agent.learn()
        assert agent.epsilon < 1.0

    def test_target_network_syncs(self):
        freq = 5
        agent = GPUDQNAgent(
            buffer_size=500, batch_size=32, hidden_sizes=(32, 32, 16),
            use_amp=False, compile_model=False,
            target_update_freq=freq, device=DEVICE,
        )
        for _ in range(50):
            s = np.random.randn(61).astype(np.float32)
            ns = np.random.randn(61).astype(np.float32)
            agent.memory.add(s, int(np.random.randint(27)), 0.0, ns, 0.0)

        # Initially policy != target (random init but same weights from load_state_dict)
        # Do freq - 1 steps then freeze to observe the sync at exactly step `freq`
        for _ in range(freq):
            agent.learn()

        # At learn_step_counter == freq the target is synced to policy
        raw_policy = getattr(agent.policy_net, "_orig_mod", agent.policy_net)
        raw_target = getattr(agent.target_net, "_orig_mod", agent.target_net)
        # Target should now match policy (sync happened on step `freq`)
        for k in raw_policy.state_dict():
            diff = (raw_policy.state_dict()[k] - raw_target.state_dict()[k]).abs().max()
            assert diff < 1e-5, f"Param {k} not synced at step {freq}: max diff {diff}"


class TestGPUDQNAgentSaveLoad:
    def test_save_load_roundtrip(self, tmp_path, small_agent):
        path = str(tmp_path / "test_ckpt.pt")
        # Do a few learn steps so there's something non-trivial to save
        for _ in range(50):
            s = np.random.randn(61).astype(np.float32)
            ns = np.random.randn(61).astype(np.float32)
            small_agent.memory.add(s, int(np.random.randint(27)), 0.0, ns, 0.0)
        small_agent.learn()

        small_agent.epsilon = 0.42
        small_agent.save(path)

        agent2 = GPUDQNAgent(
            buffer_size=5000, batch_size=32, hidden_sizes=(64, 64, 32),
            use_amp=False, compile_model=False, device=DEVICE,
        )
        assert agent2.load(path)
        assert abs(agent2.epsilon - 0.42) < 1e-6

    def test_q_values_preserved(self, tmp_path):
        agent = GPUDQNAgent(
            buffer_size=500, batch_size=32, hidden_sizes=(32, 32, 16),
            use_amp=False, compile_model=False, device=DEVICE,
        )
        path = str(tmp_path / "qv_ckpt.pt")
        agent.save(path)
        game = TicTacPro()
        qv1 = agent.get_q_values(game).copy()

        agent2 = GPUDQNAgent(
            buffer_size=500, batch_size=32, hidden_sizes=(32, 32, 16),
            use_amp=False, compile_model=False, device=DEVICE,
        )
        agent2.load(path)
        qv2 = agent2.get_q_values(game)
        assert np.allclose(qv1, qv2, atol=1e-5)
