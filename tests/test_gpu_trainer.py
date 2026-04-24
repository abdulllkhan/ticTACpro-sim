"""Unit tests for collect_episodes_vectorized and GPUTrainer."""

import pytest
import numpy as np
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from rl.agent import GPUDQNAgent
from rl.gpu_trainer import collect_episodes_vectorized, GPUTrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def agent():
    return GPUDQNAgent(
        buffer_size=5000,
        batch_size=32,
        hidden_sizes=(32, 32, 16),
        use_amp=False,
        compile_model=False,
        epsilon_start=1.0,
        epsilon_decay=1.0,  # freeze epsilon for deterministic tests
        device=DEVICE,
    )


# ---------------------------------------------------------------------------
# collect_episodes_vectorized
# ---------------------------------------------------------------------------

class TestCollectEpisodesVectorized:
    def test_returns_transitions_and_stats(self, agent):
        transitions, stats = collect_episodes_vectorized(agent, n_envs=4)
        assert isinstance(transitions, list)
        assert isinstance(stats, dict)

    def test_n_games_in_stats(self, agent):
        _, stats = collect_episodes_vectorized(agent, n_envs=8)
        assert stats["n_games"] == 8

    def test_win_counts_sum_to_n_games(self, agent):
        _, stats = collect_episodes_vectorized(agent, n_envs=16)
        assert stats["red_wins"] + stats["blue_wins"] + stats["draws"] == 16

    def test_transitions_have_correct_shape(self, agent):
        transitions, _ = collect_episodes_vectorized(agent, n_envs=4)
        assert len(transitions) > 0
        for s, a, r, ns, d in transitions:
            assert s.shape == (61,)
            assert ns.shape == (61,)
            assert isinstance(int(a), int)
            assert r in (-1.0, 0.0, 1.0)
            assert d in (0.0, 1.0)

    def test_action_indices_in_range(self, agent):
        transitions, _ = collect_episodes_vectorized(agent, n_envs=8)
        for _, a, *_ in transitions:
            assert 0 <= int(a) < 27

    def test_rewards_are_valid(self, agent):
        transitions, _ = collect_episodes_vectorized(agent, n_envs=8)
        for _, _, r, *_ in transitions:
            assert r in (-1.0, 0.0, 1.0)

    def test_states_are_float32(self, agent):
        transitions, _ = collect_episodes_vectorized(agent, n_envs=4)
        for s, _, _, ns, _ in transitions:
            assert s.dtype == np.float32
            assert ns.dtype == np.float32

    def test_avg_moves_positive(self, agent):
        _, stats = collect_episodes_vectorized(agent, n_envs=8)
        assert stats["avg_moves"] > 0

    def test_large_batch(self, agent):
        transitions, stats = collect_episodes_vectorized(agent, n_envs=64)
        assert stats["n_games"] == 64
        assert len(transitions) >= 64  # At least 1 transition per game


# ---------------------------------------------------------------------------
# GPUTrainer
# ---------------------------------------------------------------------------

class TestGPUTrainer:
    @pytest.fixture
    def trainer_agent(self):
        return GPUDQNAgent(
            buffer_size=2000,
            batch_size=32,
            hidden_sizes=(32, 32, 16),
            use_amp=False,
            compile_model=False,
            epsilon_start=1.0,
            epsilon_decay=1.0,
            device=DEVICE,
        )

    def test_trainer_creates_dirs(self, trainer_agent, tmp_path):
        ckpt_dir = str(tmp_path / "ckpts")
        log_dir = str(tmp_path / "logs")
        GPUTrainer(
            trainer_agent,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir,
            n_envs=4,
        )
        assert os.path.isdir(ckpt_dir)
        assert os.path.isdir(log_dir)

    def test_short_training_run(self, trainer_agent, tmp_path):
        """Train for 0.002 h (~7 s) and verify checkpoints are saved."""
        trainer = GPUTrainer(
            trainer_agent,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
            n_envs=8,
            train_steps_per_collect=2,
            log_interval=1,
            checkpoint_interval=2,
            state_file=str(tmp_path / "state.json"),
        )
        trainer.train(duration_hours=0.001)  # ~3.6 s
        assert os.path.exists(str(tmp_path / "ckpts" / "final.pt"))

    def test_state_file_written(self, trainer_agent, tmp_path):
        state_file = str(tmp_path / "state.json")
        trainer = GPUTrainer(
            trainer_agent,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
            n_envs=4,
            train_steps_per_collect=1,
            log_interval=1,
            checkpoint_interval=1,
            state_file=state_file,
        )
        trainer.train(duration_hours=0.001)
        import json
        assert os.path.exists(state_file)
        with open(state_file) as fh:
            state = json.load(fh)
        assert "total_episodes" in state

    def test_evaluate(self, trainer_agent, tmp_path):
        trainer = GPUTrainer(
            trainer_agent,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
            n_envs=8,
        )
        result = trainer.evaluate(n_games=32)
        assert abs(result["red_win_rate"] + result["blue_win_rate"] + result["draw_rate"] - 1.0) < 1e-5
        assert result["avg_moves"] > 0

    def test_buffer_fills_during_training(self, trainer_agent, tmp_path):
        trainer = GPUTrainer(
            trainer_agent,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
            n_envs=16,
            train_steps_per_collect=1,
        )
        trainer.train(duration_hours=0.001)
        assert len(trainer_agent.memory) > 0
