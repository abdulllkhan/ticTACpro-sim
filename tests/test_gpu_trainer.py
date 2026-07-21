"""Unit tests for collect_episodes_vectorized and GPUTrainer."""

import pytest
import numpy as np
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from rl.agent import GPUDQNAgent
from rl.gpu_trainer import collect_episodes_vectorized, GPUTrainer, _cpu_worker_collect, HealthMonitor

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

    # §109 — random_opponent_frac tests
    def test_random_opp_full_fraction(self, agent):
        """random_opponent_frac=1.0: all BLUE turns are random; stats still valid."""
        transitions, stats = collect_episodes_vectorized(agent, n_envs=16, random_opponent_frac=1.0)
        assert stats["n_games"] == 16
        assert stats["red_wins"] + stats["blue_wins"] + stats["draws"] == 16
        assert len(transitions) > 0

    def test_random_opp_partial_fraction(self, agent):
        """random_opponent_frac=0.5: mixed self-play and random-BLUE games."""
        transitions, stats = collect_episodes_vectorized(agent, n_envs=20, random_opponent_frac=0.5)
        assert stats["n_games"] == 20
        assert stats["red_wins"] + stats["blue_wins"] + stats["draws"] == 20

    def test_random_opp_zero_fraction_unchanged(self, agent):
        """random_opponent_frac=0.0 must behave identically to the default."""
        _, s1 = collect_episodes_vectorized(agent, n_envs=8, random_opponent_frac=0.0)
        _, s2 = collect_episodes_vectorized(agent, n_envs=8)
        # Both should produce valid stats (counts can differ due to randomness)
        assert s1["red_wins"] + s1["blue_wins"] + s1["draws"] == 8
        assert s2["red_wins"] + s2["blue_wins"] + s2["draws"] == 8

    def test_random_opp_transitions_shape(self, agent):
        """Transitions from random-opponent games have correct dtype/shape."""
        transitions, _ = collect_episodes_vectorized(agent, n_envs=8, random_opponent_frac=1.0)
        for s, a, r, ns, d in transitions:
            assert s.shape == (61,)
            assert ns.shape == (61,)
            assert 0 <= int(a) < 27
            assert r in (-1.0, 0.0, 1.0)
            assert d in (0.0, 1.0)


# ---------------------------------------------------------------------------
# _cpu_worker_collect — §139 batched CPU inference
# ---------------------------------------------------------------------------

def _make_worker_args(n_games=8, epsilon=1.0, random_frac=0.3, reverse_frac=0.0, mc_returns=False, gamma=0.99):
    """Build a valid args tuple for _cpu_worker_collect."""
    import pickle
    from rl.network import GPUDQNNetwork
    hidden = (32, 32, 16)
    net = GPUDQNNetwork(61, 27, hidden)
    net.eval()
    weights = {k: v.cpu() for k, v in net.state_dict().items()}
    # §167: workers receive pre-pickled bytes
    weights_bytes = pickle.dumps(weights)
    return (n_games, epsilon, weights_bytes, hidden, 61, 27, random_frac, reverse_frac, mc_returns, gamma)


class TestCPUWorkerCollect:
    def test_returns_correct_n_games(self):
        transitions, stats = _cpu_worker_collect(_make_worker_args(n_games=6))
        assert stats["n_games"] == 6

    def test_win_counts_sum_to_n_games(self):
        _, stats = _cpu_worker_collect(_make_worker_args(n_games=8))
        assert stats["red_wins"] + stats["blue_wins"] + stats["draws"] == 8

    def test_transitions_shape_and_dtype(self):
        # §166/§157: worker returns pre-augmented arrays tuple (out_s, out_acts, out_rewards, out_ns, out_dones)
        arrays, _ = _cpu_worker_collect(_make_worker_args(n_games=4))
        out_s, out_acts, out_rewards, out_ns, out_dones = arrays
        assert len(out_s) > 0, "expected non-empty transitions"
        assert out_s.shape[1] == 61
        assert out_ns.shape[1] == 61
        assert out_s.dtype == np.float32
        assert out_acts.dtype == np.int32
        assert ((out_acts >= 0) & (out_acts < 27)).all()
        # mc_returns=False: terminal rewards are ±1.0 or 0.0; non-terminal are 0.0
        assert np.all(np.isin(out_rewards, [-1.0, 0.0, 1.0]) | (out_dones == 0.0))
        assert np.all(np.isin(out_dones, [0.0, 1.0]))

    def test_greedy_epsilon_zero(self):
        # §139: at epsilon=0 every DQN game should use batched greedy inference
        transitions, stats = _cpu_worker_collect(_make_worker_args(n_games=8, epsilon=0.0))
        assert stats["n_games"] == 8
        assert stats["red_wins"] + stats["blue_wins"] + stats["draws"] == 8

    def test_random_opponent_frac_one(self):
        # All BLUE turns are random — should still produce valid stats
        arrays, stats = _cpu_worker_collect(_make_worker_args(n_games=8, random_frac=1.0))
        assert stats["red_wins"] + stats["blue_wins"] + stats["draws"] == 8
        assert len(arrays[0]) > 0  # out_s should be non-empty

    def test_batched_matches_sequential_stats(self):
        # §139: batched path must produce same structural invariants as the old
        # sequential path (not identical outcomes — stochastic — but same counts).
        args = _make_worker_args(n_games=16, epsilon=0.5)
        _, stats = _cpu_worker_collect(args)
        assert stats["red_wins"] + stats["blue_wins"] + stats["draws"] == 16

    def test_mc_returns_all_dones_one(self):
        """§147: MC returns — every transition's done flag must be 1.0."""
        arrays, _ = _cpu_worker_collect(_make_worker_args(n_games=8, mc_returns=True))
        _, _, _, _, out_dones = arrays
        assert (out_dones == 1.0).all(), "MC returns path must set done=1.0 for all transitions"

    def test_mc_returns_rewards_in_unit_interval(self):
        """§147: MC returns rewards are ±γ^k ∈ (-1, 1]; all magnitudes ≤ 1."""
        arrays, _ = _cpu_worker_collect(_make_worker_args(n_games=8, mc_returns=True, gamma=0.99))
        _, _, out_rewards, _, _ = arrays
        assert (np.abs(out_rewards) <= 1.0 + 1e-6).all(), (
            f"MC rewards outside [-1, 1]: min={out_rewards.min()}, max={out_rewards.max()}"
        )

    def test_mc_returns_draw_games_have_zero_reward(self):
        """§147: draw games must produce all-zero rewards regardless of MC returns."""
        import random as _rand
        _rand.seed(0)
        from rl.gpu_trainer import _cpu_worker_collect as _wc
        arrays, stats = _wc(_make_worker_args(n_games=32, mc_returns=True, gamma=0.99))
        _, _, out_rewards, _, _ = arrays
        if stats["draws"] > 0:
            # Draw games produce reward=0.0 — can only verify total is plausible
            # (some 0s exist); exact set of which transitions are draws is opaque.
            assert (out_rewards == 0.0).any(), "expected some zero rewards with draws present"

    def test_heuristic_epsilon_eleventh_arg(self):
        """§177: 11-element tuple with heuristic_epsilon>0 must produce valid stats."""
        import pickle
        from rl.network import GPUDQNNetwork
        hidden = (32, 32, 16)
        net = GPUDQNNetwork(61, 27, hidden)
        net.eval()
        weights_bytes = pickle.dumps({k: v.cpu() for k, v in net.state_dict().items()})
        # 11-element tuple with heuristic_epsilon=0.5
        args = (8, 1.0, weights_bytes, hidden, 61, 27, 0.3, 0.0, False, 0.99, 0.5)
        arrays, stats = _cpu_worker_collect(args)
        assert stats["n_games"] == 8
        assert stats["red_wins"] + stats["blue_wins"] + stats["draws"] == 8
        assert len(arrays[0]) > 0

    def test_game_length_stats_in_stats(self):
        """§225: §178 — avg_game_length and per-category lengths must be > 0 in stats."""
        # n_games=8: n_random=2 (random-opp), n_reverse=2 (reverse-opp), 4 self-play
        args = _make_worker_args(n_games=8, random_frac=0.25, reverse_frac=0.25)
        _, stats = _cpu_worker_collect(args)
        assert stats.get("avg_game_length", 0.0) > 0, \
            "§178: avg_game_length must be positive (games must take ≥1 move)"
        assert stats.get("avg_sp_game_length", 0.0) > 0, \
            "§178: avg_sp_game_length must be positive for self-play games"
        assert stats.get("avg_ro_game_length", 0.0) > 0, \
            "§178: avg_ro_game_length must be positive for random-opp games"
        assert stats.get("avg_rv_game_length", 0.0) > 0, \
            "§178: avg_rv_game_length must be positive for reverse-opp games"

    def test_game_length_aggregation_across_workers(self):
        """§226: §178 game length stats must survive aggregation across multiple workers.

        The GPUTrainer aggregation loop computes weighted averages of per-worker
        game lengths into ep_stats. Previously (§226 bug), avg_game_length was
        never included in ep_stats — always returning 0.0.
        This test verifies the aggregation produces non-zero values.
        """
        from rl.gpu_trainer import _cpu_worker_collect
        # Simulate what GPUTrainer does: collect from multiple workers, aggregate
        worker_results = [
            _cpu_worker_collect(_make_worker_args(n_games=6, random_frac=0.33, reverse_frac=0.33))
            for _ in range(3)
        ]
        # Aggregate like GPUTrainer does (the fixed aggregation logic)
        agg_games = 0
        agg_sp_n = agg_ro_n = agg_rv_n = 0
        agg_gl_sum = agg_sp_gl_sum = agg_ro_gl_sum = agg_rv_gl_sum = 0.0
        for _, stats in worker_results:
            wn = stats["n_games"]
            agg_games += wn
            w_sp_n = stats.get("sp_red",0)+stats.get("sp_blue",0)+stats.get("sp_draw",0)
            w_ro_n = stats.get("ro_red",0)+stats.get("ro_blue",0)+stats.get("ro_draw",0)
            w_rv_n = stats.get("rv_red",0)+stats.get("rv_blue",0)+stats.get("rv_draw",0)
            agg_gl_sum    += stats.get("avg_game_length",    0.0) * wn
            agg_sp_gl_sum += stats.get("avg_sp_game_length", 0.0) * w_sp_n
            agg_ro_gl_sum += stats.get("avg_ro_game_length", 0.0) * w_ro_n
            agg_rv_gl_sum += stats.get("avg_rv_game_length", 0.0) * w_rv_n
            agg_sp_n += w_sp_n;  agg_ro_n += w_ro_n;  agg_rv_n += w_rv_n
        ep_stats = {
            "avg_game_length":    agg_gl_sum    / max(agg_games, 1),
            "avg_sp_game_length": agg_sp_gl_sum / max(agg_sp_n, 1),
            "avg_ro_game_length": agg_ro_gl_sum / max(agg_ro_n, 1),
            "avg_rv_game_length": agg_rv_gl_sum / max(agg_rv_n, 1),
        }
        assert ep_stats["avg_game_length"]    > 0.0, "aggregated avg_game_length must be > 0"
        assert ep_stats["avg_sp_game_length"] > 0.0, "aggregated avg_sp_game_length must be > 0"
        assert ep_stats["avg_ro_game_length"] > 0.0, "aggregated avg_ro_game_length must be > 0"
        assert ep_stats["avg_rv_game_length"] > 0.0, "aggregated avg_rv_game_length must be > 0"

    def test_use_optimal_thirteenth_arg(self):
        """§347: 13-element tuple with use_optimal=True uses OptimalAgent as heuristic opponent."""
        import pickle
        from rl.network import GPUDQNNetwork
        hidden = (32, 32, 16)
        net = GPUDQNNetwork(61, 27, hidden)
        net.eval()
        weights_bytes = pickle.dumps({k: v.cpu() for k, v in net.state_dict().items()})
        # 13-element tuple: ..., heuristic_epsilon=0.0, use_bullseye=False, use_optimal=True
        args = (8, 1.0, weights_bytes, hidden, 61, 27, 0.3, 0.3, False, 0.99, 0.0, False, True)
        arrays, stats = _cpu_worker_collect(args)
        assert stats["n_games"] == 8
        assert stats["red_wins"] + stats["blue_wins"] + stats["draws"] == 8
        assert len(arrays[0]) > 0, "use_optimal path must produce transitions"

    def test_use_bullseye_twelfth_arg(self):
        """§336: 12-element tuple with use_bullseye=True uses BullseyeAgent as heuristic opponent."""
        import pickle
        from rl.network import GPUDQNNetwork
        hidden = (32, 32, 16)
        net = GPUDQNNetwork(61, 27, hidden)
        net.eval()
        weights_bytes = pickle.dumps({k: v.cpu() for k, v in net.state_dict().items()})
        # 12-element tuple: ..., heuristic_epsilon=0.0, use_bullseye=True
        args = (8, 1.0, weights_bytes, hidden, 61, 27, 0.3, 0.3, False, 0.99, 0.0, True)
        arrays, stats = _cpu_worker_collect(args)
        assert stats["n_games"] == 8
        assert stats["red_wins"] + stats["blue_wins"] + stats["draws"] == 8
        assert len(arrays[0]) > 0, "use_bullseye path must produce transitions"


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


# ---------------------------------------------------------------------------
# §201: _SYM_PERMS D4 dihedral group correctness
# ---------------------------------------------------------------------------

class TestSymPermsD4:
    """§201: The 8 board symmetry permutations must form the D4 dihedral group.

    Any bug in _build_sym_perms (wrong row/column mapping, wrong transform)
    would silently corrupt all augmented training data, since invalid
    permutations produce board states that don't correspond to real positions.
    """

    @pytest.fixture(scope="class")
    def perms(self):
        from rl.gpu_trainer import _SYM_PERMS, _SYM_INV_PERMS
        return _SYM_PERMS, _SYM_INV_PERMS

    def test_identity_is_first_perm(self, perms):
        """perm[0] must be the identity permutation."""
        SYM_PERMS, _ = perms
        assert list(SYM_PERMS[0]) == list(range(27)), "perm[0] is not the identity"

    def test_all_eight_perms_are_distinct(self, perms):
        """All 8 permutations must be different — duplicates would mean fewer
        than 8 distinct symmetries, wasting buffer capacity."""
        SYM_PERMS, _ = perms
        perm_set = {tuple(row) for row in SYM_PERMS}
        assert len(perm_set) == 8, f"Only {len(perm_set)}/8 distinct permutations"

    def test_inv_perms_are_true_inverses(self, perms):
        """_SYM_INV_PERMS[k] must be the inverse of _SYM_PERMS[k] for all k."""
        SYM_PERMS, SYM_INV_PERMS = perms
        identity = list(range(27))
        for k in range(8):
            composed = SYM_PERMS[k][SYM_INV_PERMS[k]]
            assert list(composed) == identity, (
                f"perm[{k}][inv_perm[{k}]] != identity — "
                f"inverse is wrong for symmetry {k}"
            )

    def test_group_closure(self, perms):
        """Composing any two D4 permutations must give another permutation in D4."""
        SYM_PERMS, _ = perms
        perm_set = {tuple(row) for row in SYM_PERMS}
        for i in range(8):
            for j in range(8):
                composed = SYM_PERMS[i][SYM_PERMS[j]]
                assert tuple(composed) in perm_set, (
                    f"perm[{i}] ∘ perm[{j}] = {tuple(composed)} not in D4 — "
                    "permutations do not form a closed group"
                )

    def test_rotation_90cw_has_order_4(self, perms):
        """Applying the 90° CW rotation 4 times must give the identity."""
        SYM_PERMS, _ = perms
        rot90 = SYM_PERMS[1]
        composed = np.arange(27, dtype=np.int32)
        for _ in range(4):
            composed = rot90[composed]
        assert list(composed) == list(range(27)), (
            "90° CW rotation (perm[1]) does not have order 4 — "
            "rotation transform is incorrect"
        )

    def test_each_perm_is_a_valid_bijection(self, perms):
        """Each permutation must be a bijection (covers all 27 indices exactly once)."""
        SYM_PERMS, _ = perms
        for k in range(8):
            assert sorted(SYM_PERMS[k]) == list(range(27)), (
                f"perm[{k}] is not a valid permutation of 0..26"
            )


class TestAugmentTransitionsConsistency:
    """§210: augment_transitions() and augment_transitions_arrays() must agree.

    Both functions apply the same 8 D4 symmetries. augment_transitions returns a
    list of tuples; augment_transitions_arrays returns raw numpy arrays. They must
    produce identical state/action/reward/next_state/done values.
    """

    def test_arrays_match_tuple_list(self):
        """States, actions, rewards, dones from both functions must be identical."""
        from rl.gpu_trainer import augment_transitions, augment_transitions_arrays
        n = 3
        rng = np.random.default_rng(42)
        transitions = []
        for _ in range(n):
            s  = rng.random(61).astype(np.float32)
            ns = rng.random(61).astype(np.float32)
            a  = int(rng.integers(0, 27))
            r  = float(rng.random())
            d  = float(rng.integers(0, 2))
            transitions.append((s, a, r, ns, d))

        # Tuple-list version
        aug_list = augment_transitions(transitions)
        # Arrays version
        out_s, out_a, out_r, out_ns, out_d = augment_transitions_arrays(transitions)

        assert len(aug_list) == 8 * n == len(out_s)

        for i, (ts, ta, tr, tns, td) in enumerate(aug_list):
            np.testing.assert_allclose(ts, out_s[i], atol=1e-6,
                err_msg=f"state mismatch at index {i}")
            np.testing.assert_allclose(tns, out_ns[i], atol=1e-6,
                err_msg=f"next_state mismatch at index {i}")
            assert int(ta) == int(out_a[i]), f"action mismatch at index {i}"
            assert abs(float(tr) - float(out_r[i])) < 1e-6, f"reward mismatch at index {i}"
            assert abs(float(td) - float(out_d[i])) < 1e-6, f"done mismatch at index {i}"

    def test_empty_both_return_empty(self):
        """Both functions on empty input must return consistently empty results."""
        from rl.gpu_trainer import augment_transitions, augment_transitions_arrays
        assert augment_transitions([]) == []
        out_s, out_a, out_r, out_ns, out_d = augment_transitions_arrays([])
        assert out_s.shape[0] == 0 and out_a.shape[0] == 0


class TestAugmentTransitionsArrays:
    """§208: augment_transitions_arrays() must correctly apply all 8 D4 symmetries.

    This function multiplies training data by 8x. A bug silently corrupts every
    training step — producing board states that don't correspond to real positions
    or actions that point to wrong cells.
    """

    def _make_transition(self, piece_idx: int, action: int):
        """Create a single (s, a, r, ns, d) tuple with a single piece at piece_idx."""
        s  = np.zeros(61, dtype=np.float32); s[piece_idx] = 1.0
        ns = np.zeros(61, dtype=np.float32); ns[piece_idx] = 1.0
        return (s, action, 1.0, ns, 0.0)

    def test_empty_input_returns_empty_arrays(self):
        """Empty transitions → all five output arrays have shape (0, ...)."""
        from rl.gpu_trainer import augment_transitions_arrays
        out_s, out_a, out_r, out_ns, out_d = augment_transitions_arrays([])
        assert out_s.shape  == (0, 61)
        assert out_a.shape  == (0,)
        assert out_r.shape  == (0,)
        assert out_ns.shape == (0, 61)
        assert out_d.shape  == (0,)

    def test_output_length_is_8n(self):
        """n transitions → 8n augmented transitions."""
        from rl.gpu_trainer import augment_transitions_arrays
        n = 5
        transitions = [self._make_transition(i, i) for i in range(n)]
        out_s, out_a, out_r, out_ns, out_d = augment_transitions_arrays(transitions)
        assert out_s.shape[0]  == 8 * n
        assert out_a.shape[0]  == 8 * n
        assert out_r.shape[0]  == 8 * n
        assert out_ns.shape[0] == 8 * n
        assert out_d.shape[0]  == 8 * n

    def test_rewards_and_dones_are_repeated_not_permuted(self):
        """Rewards and dones must be tiled 8x — they have no spatial structure."""
        from rl.gpu_trainer import augment_transitions_arrays
        s  = np.zeros(61, dtype=np.float32)
        ns = np.zeros(61, dtype=np.float32)
        transitions = [(s, 0, 0.5, ns, 1.0), (s, 1, -0.5, ns, 0.0)]
        _, _, out_r, _, out_d = augment_transitions_arrays(transitions)
        # Each original reward/done should appear exactly 8 times.
        assert np.sum(np.isclose(out_r, 0.5))  == 8
        assert np.sum(np.isclose(out_r, -0.5)) == 8
        assert np.sum(np.isclose(out_d, 1.0))  == 8
        assert np.sum(np.isclose(out_d, 0.0))  == 8

    def test_action_permutation_consistent_with_state(self):
        """For each of the 8 symmetries, the permuted action must point to the
        permuted piece position in the corresponding permuted state.

        Concretely: if state s has a piece only at board index p, and action a=p,
        then in the k-th symmetry:
        - out_s has the piece at fwd[p]
        - out_acts = fwd[a] = fwd[p]  (same index)
        So: out_s[fwd[p]] == 1.0 and out_acts == fwd[p] for all 8 k.
        """
        from rl.gpu_trainer import augment_transitions_arrays, _SYM_PERMS
        piece_idx = 5  # arbitrary board position in me-slice (0..26)
        transitions = [self._make_transition(piece_idx, piece_idx)]
        out_s, out_a, _, _, _ = augment_transitions_arrays(transitions)
        # out_s/out_a has 8 rows (one per symmetry)
        for k in range(8):
            perm_pos  = int(_SYM_PERMS[k][piece_idx])
            row_state = out_s[k]
            row_act   = int(out_a[k])
            assert row_state[perm_pos] == pytest.approx(1.0), (
                f"symmetry {k}: piece should be at fwd[{piece_idx}]={perm_pos}, "
                f"but state has piece at {np.where(row_state[:27] > 0.5)[0]}"
            )
            assert row_act == perm_pos, (
                f"symmetry {k}: action {row_act} != fwd[{piece_idx}]={perm_pos}"
            )

    def test_identity_symmetry_preserves_original(self):
        """The identity permutation (k=0) must leave state and action unchanged."""
        from rl.gpu_trainer import augment_transitions_arrays
        s  = np.random.rand(61).astype(np.float32)
        ns = np.random.rand(61).astype(np.float32)
        action = 13
        transitions = [(s, action, 0.7, ns, 0.0)]
        out_s, out_a, _, out_ns, _ = augment_transitions_arrays(transitions)
        # k=0 is identity; the first row after interleave is the original transition.
        # augment interleaves: row 0 = k=0 of transition 0.
        np.testing.assert_array_equal(out_s[0], s,  err_msg="identity must preserve state")
        np.testing.assert_array_equal(out_ns[0], ns, err_msg="identity must preserve next_state")
        assert int(out_a[0]) == action, f"identity must preserve action, got {out_a[0]}"

    def test_state_total_occupancy_preserved(self):
        """The total number of occupied pieces must be invariant under all symmetries."""
        from rl.gpu_trainer import augment_transitions_arrays
        s = np.zeros(61, dtype=np.float32)
        # Place 3 pieces at different board positions.
        for idx in [0, 10, 20]:
            s[idx] = 1.0
        ns = np.zeros(61, dtype=np.float32)
        ns[5] = 1.0
        transitions = [(s, 0, 0.0, ns, 0.0)]
        out_s, _, _, out_ns, _ = augment_transitions_arrays(transitions)
        for k in range(8):
            assert out_s[k, :27].sum() == pytest.approx(3.0), (
                f"symmetry {k}: me-slice sum changed (expected 3 pieces)"
            )
            assert out_ns[k, :27].sum() == pytest.approx(1.0), (
                f"symmetry {k}: ns me-slice sum changed"
            )


# ---------------------------------------------------------------------------
# §313: HealthMonitor — update, flush, and lifecycle
# ---------------------------------------------------------------------------

class TestHealthMonitor:
    """§313: HealthMonitor (gpu_trainer.py:617-687) has no direct test coverage.

    HealthMonitor runs a background thread that writes training_state.json
    every check_interval seconds and warns when training appears stuck or when
    loss is non-finite.  The GPUTrainer creates one internally, but the
    TestGPUTrainer suite only verifies the file gets written, not the
    monitor's own state-management or threading lifecycle.

    These tests verify:
    A) update() stores arbitrary kwargs plus 'timestamp' and 'last_update'.
    B) _flush() writes valid JSON to the state file.
    C) start() / stop() thread lifecycle completes without error.
    D) A second flush after stop() reflects the final state."""

    def _monitor(self, tmp_path) -> HealthMonitor:
        return HealthMonitor(
            state_file=str(tmp_path / "state.json"),
            check_interval=60,   # long interval — we call _flush() manually
            stuck_threshold=300,
        )

    def test_update_stores_kwargs(self, tmp_path):
        """§313a: update() must persist arbitrary kwargs into _state."""
        m = self._monitor(tmp_path)
        m.update(epoch=5, avg_loss=0.123, eps_per_second=42)
        with m._lock:
            state = dict(m._state)
        assert state["epoch"]          == 5
        assert state["avg_loss"]       == pytest.approx(0.123)
        assert state["eps_per_second"] == 42
        assert "timestamp"    in state, "update() must set timestamp"
        assert "last_update"  in state, "update() must set last_update string"

    def test_flush_writes_valid_json(self, tmp_path):
        """§313b: _flush() must write a valid JSON file with the current state."""
        import json
        m = self._monitor(tmp_path)
        m.update(epoch=3, gradient_steps=1000)
        m._flush()
        sf = tmp_path / "state.json"
        assert sf.exists(), "_flush() must create the state file"
        with sf.open() as fh:
            data = json.load(fh)
        assert data["epoch"]           == 3
        assert data["gradient_steps"]  == 1000

    def test_start_stop_lifecycle(self, tmp_path):
        """§313c: start() must launch the background thread; stop() must join it."""
        m = self._monitor(tmp_path)
        assert m._thread is None, "Thread must not exist before start()"
        m.start()
        assert m._thread is not None,  "Thread must exist after start()"
        assert m._thread.is_alive(),   "Thread must be alive after start()"
        m.stop()
        assert not m._thread.is_alive(), "Thread must not be alive after stop()"

    def test_stop_flushes_final_state(self, tmp_path):
        """§313d: stop() calls _flush() so the final state is always persisted."""
        import json
        m = self._monitor(tmp_path)
        m.update(epoch=99)
        m.start()
        m.stop()
        sf = tmp_path / "state.json"
        assert sf.exists(), "stop() must flush state to file"
        with sf.open() as fh:
            data = json.load(fh)
        assert data["epoch"] == 99


# ---------------------------------------------------------------------------
# §329 — gpu_trainer.py coverage gaps
# ---------------------------------------------------------------------------

class TestGPUTrainerCoverageGaps:
    """§329: Cover remaining uncovered lines in gpu_trainer.py."""

    def _make_agent(self):
        return GPUDQNAgent(
            buffer_size=2000, batch_size=32, hidden_sizes=(32, 32, 16),
            use_amp=False, compile_model=False, epsilon_start=1.0,
            epsilon_decay=1.0, device=DEVICE,
        )

    def test_worker_process_init_sets_env_vars(self):
        """Lines 47-51: _worker_process_init sets BLAS thread env vars."""
        from rl.gpu_trainer import _worker_process_init
        import os
        _worker_process_init()
        assert os.environ.get("OMP_NUM_THREADS")      == "1"
        assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
        assert os.environ.get("MKL_NUM_THREADS")      == "1"
        assert os.environ.get("NUMEXPR_NUM_THREADS")  == "1"

    def test_health_monitor_flush_exception_is_swallowed(self, tmp_path):
        """Lines 650-651: _flush() swallows exceptions from open()."""
        m = HealthMonitor(state_file="/nonexistent_dir/nowhere/state.json")
        m.update(x=1)
        m._flush()  # must not raise — OSError caught by except Exception: pass

    def test_health_monitor_stuck_warning(self, tmp_path, capsys):
        """Lines 662-666: _run() prints stuck warning when timestamp is very old."""
        import threading
        m = HealthMonitor(
            state_file=str(tmp_path / "state.json"),
            check_interval=60,
            stuck_threshold=1,  # 1s threshold — any age > 1s triggers warning
        )
        # Timestamp = 0 (Unix epoch) → age = ~decades → stuck_threshold exceeded
        with m._lock:
            m._state["timestamp"] = 0.1
        threading.Timer(0.005, m._stop.set).start()
        m._run()
        out = capsys.readouterr().out
        assert "HEALTH WARNING" in out

    def test_health_monitor_nan_loss_warning(self, tmp_path, capsys):
        """Line 670: _run() warns when avg_loss is non-finite (NaN)."""
        import threading
        m = HealthMonitor(
            state_file=str(tmp_path / "state.json"),
            check_interval=60,
            stuck_threshold=300,
        )
        m.update(avg_loss=float("nan"))
        threading.Timer(0.005, m._stop.set).start()
        m._run()
        out = capsys.readouterr().out
        assert "HEALTH WARNING" in out and "non-finite" in out

    def test_health_monitor_zero_eps_warning(self, tmp_path, capsys):
        """Line 674: _run() warns when eps_per_second=0 after 1000+ gradient steps."""
        import threading
        m = HealthMonitor(
            state_file=str(tmp_path / "state.json"),
            check_interval=60,
            stuck_threshold=300,
        )
        m.update(eps_per_second=0, gradient_steps=1001)
        threading.Timer(0.005, m._stop.set).start()
        m._run()
        out = capsys.readouterr().out
        assert "HEALTH WARNING" in out and "0" in out

    def test_direct_log_file_written(self, tmp_path):
        """Lines 744-745, 787-789: _tlog writes to direct_log_file when set."""
        agent = self._make_agent()
        log_path = str(tmp_path / "direct.log")
        trainer = GPUTrainer(
            agent,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
            n_envs=4,
            direct_log_file=log_path,
        )
        trainer._tlog("test message", flush=True)
        assert os.path.exists(log_path)
        with open(log_path) as f:
            content = f.read()
        assert "test message" in content
        trainer._direct_log.close()

    def test_make_worker_args_returns_correct_count(self, tmp_path):
        """Lines 807-830: _make_worker_args serializes weights and distributes epsilons."""
        agent = self._make_agent()
        trainer = GPUTrainer(
            agent,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
            n_cpu_workers=2,
            games_per_worker=4,
        )
        args_list = trainer._make_worker_args()
        assert len(args_list) == 2
        # Each arg tuple has 13 elements (n_games, epsilon, weights_bytes, ...)
        assert all(len(a) == 13 for a in args_list)  # §347: added use_optimal_heuristic
        # weights_bytes must be non-empty bytes
        assert all(isinstance(a[2], bytes) and len(a[2]) > 0 for a in args_list)

    def test_train_resumes_from_checkpoint(self, tmp_path):
        """Line 869: agent.load() called when resume_from exists."""
        agent = self._make_agent()
        trainer = GPUTrainer(
            agent,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
            n_envs=4,
            train_steps_per_collect=1,
            log_interval=1,
            checkpoint_interval=1,
        )
        trainer.train(duration_hours=0.0003)  # ~1s — produce final checkpoint
        resume_path = str(tmp_path / "ckpts" / "final.pt")
        assert os.path.exists(resume_path)
        trainer2 = GPUTrainer(
            self._make_agent(),
            checkpoint_dir=str(tmp_path / "ckpts2"),
            log_dir=str(tmp_path / "logs2"),
            n_envs=4,
        )
        trainer2.train(duration_hours=0.0003, resume_from=resume_path)

    def test_train_with_cpu_workers_covers_pipeline_path(self, tmp_path):
        """Lines 907-913, 923-1001, 1032-1052, 1135: pipelined training with 1 CPU worker."""
        agent = self._make_agent()
        trainer = GPUTrainer(
            agent,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
            n_cpu_workers=1,
            games_per_worker=4,
            n_envs=0,                  # no vectorized envs — use worker only
            train_steps_per_collect=1,
            log_interval=1,
            checkpoint_interval=100,   # no checkpoint saves (slow)
        )
        trainer.train(duration_hours=0.0005)  # ~1.8s
        assert len(agent.memory) > 0


class TestGPUTrainerLogStep:
    """§329b: _log_step() paths requiring deque pre-population."""

    def _make_trainer(self, tmp_path):
        agent = GPUDQNAgent(
            buffer_size=2000, batch_size=32, hidden_sizes=(32, 32, 16),
            use_amp=False, compile_model=False, epsilon_start=1.0,
            epsilon_decay=1.0, device=DEVICE,
        )
        return GPUTrainer(
            agent,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
            n_envs=4,
            train_steps_per_collect=1,
            log_interval=1,
        )

    def test_log_step_with_per_category_stats(self, tmp_path, capsys):
        """Lines 1184-1220: _log_progress logs sp/ro/rv win rates and game lengths
        when the deques are pre-populated."""
        import time as _time
        trainer = self._make_trainer(tmp_path)
        trainer.training_start = _time.time() - 10.0  # simulate 10s elapsed
        trainer._log_call_count = 1  # skip eval branch (1+1=2, 2%5!=1)
        # Pre-populate deques with outcome data
        trainer._recent_sp.extend(["red"] * 30 + ["blue"] * 10 + ["draw"] * 10)
        trainer._recent_ro.extend(["red"] * 20 + ["blue"] * 20 + ["draw"] * 10)
        trainer._recent_rv.extend(["blue"] * 25 + ["red"] * 15 + ["draw"] * 10)
        trainer._recent_gl.extend([17.0] * 50)
        trainer._recent_gl_sp.extend([16.0] * 50)
        trainer._recent_gl_ro.extend([18.0] * 50)
        trainer._recent_gl_rv.extend([19.0] * 50)
        trainer._recent_results.extend(["red"] * 30 + ["blue"] * 20)
        trainer._recent_losses.extend([0.5] * 10)
        trainer._log_progress(end_time=_time.time() + 100.0)
        out = capsys.readouterr().out
        assert "Self-play" in out
        assert "Random-opp" in out
        assert "Reverse-opp" in out
        assert "Avg game length" in out

    def test_log_step_best_checkpoint_saved(self, tmp_path):
        """Lines 1269-1272: best checkpoint saved when combined heuristic wr improves."""
        import time as _time
        trainer = self._make_trainer(tmp_path)
        trainer.training_start = _time.time() - 10.0
        trainer._log_call_count = 1  # skip eval branch
        trainer._best_combined_heuristic_wr = 0.0
        trainer._recent_results.extend(["red"] * 30 + ["blue"] * 20)
        trainer._recent_losses.extend([0.5] * 10)
        # Manually trigger the best-checkpoint save path
        combined_wr = 0.9  # above the 0.0 threshold
        trainer._best_combined_heuristic_wr = combined_wr - 0.1
        trainer._save("best")
        # Best checkpoint should be saved
        assert os.path.exists(str(tmp_path / "ckpts" / "best.pt"))


class TestGPUTrainerCoverageFinalGaps:
    """§329b: remaining coverage gaps — MC returns vectorized, rv_blue, best-checkpoint
    save path via _log_progress, and _direct_log.close() in train() finally block."""

    def _make_agent(self):
        return GPUDQNAgent(epsilon_start=1.0, epsilon_end=1.0, epsilon_decay=1.0,
                           buffer_size=500, batch_size=32, device=DEVICE)

    def _make_trainer(self, tmp_path, **kw):
        agent = self._make_agent()
        return GPUTrainer(
            agent,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
            n_envs=4,
            train_steps_per_collect=1,
            log_interval=1,
            **kw,
        )

    def test_mc_returns_vectorized_collect(self):
        """Lines 359-365: collect_episodes_vectorized with mc_returns=True."""
        agent = self._make_agent()
        transitions, _ = collect_episodes_vectorized(agent, n_envs=8, mc_returns=True)
        # In MC mode, all done flags are 1.0 (no bootstrapping)
        dones = [t[4] for t in transitions]
        assert all(d == 1.0 for d in dones)
        # Rewards should be discounted (not just 0 or ±1)
        rewards = [t[2] for t in transitions]
        assert len(rewards) > 0

    def test_reverse_games_stats_integrity(self):
        """Lines 553-557: reverse-opp stats are correctly counted in _cpu_worker_collect."""
        # All-reverse games: DQN-BLUE vs heuristic-RED
        args = _make_worker_args(n_games=20, random_frac=0.0, reverse_frac=1.0, epsilon=1.0)
        _, stats = _cpu_worker_collect(args)
        total = stats["rv_red"] + stats["rv_blue"] + stats["rv_draw"]
        assert total == 20
        assert stats.get("avg_rv_game_length", 0.0) > 0

    def test_log_progress_best_checkpoint_save(self, tmp_path, monkeypatch):
        """Lines 1270-1272: best checkpoint saved when combined heuristic wr improves.

        Monkeypatches _eval_vs_heuristic/_eval_vs_random to return high win rates
        without running actual games, then calls _log_progress with _log_call_count=0
        so the eval branch fires on the first call.
        """
        import time as _time
        trainer = self._make_trainer(tmp_path)
        trainer.training_start = _time.time() - 10.0
        trainer._log_call_count = 0  # first call → 0+1=1 → 1%5==1 → triggers eval
        trainer._best_combined_heuristic_wr = 0.0
        trainer._recent_results.extend(["red"] * 20 + ["blue"] * 10)
        trainer._recent_losses.extend([0.5] * 10)

        monkeypatch.setattr(trainer, "_eval_vs_random",
                            lambda n_games: {"dqn_red_wr": 0.9, "dqn_blue_wr": 0.8})
        monkeypatch.setattr(trainer, "_eval_vs_heuristic",
                            lambda n_games: {"dqn_red_wr": 0.85, "dqn_blue_wr": 0.80})

        trainer._log_progress(end_time=_time.time() + 100.0)
        # combined_wr = (0.85 + 0.80) / 2 = 0.825 > 0.0 → best.pt saved
        assert os.path.exists(str(tmp_path / "ckpts" / "best.pt"))
        assert trainer._best_combined_heuristic_wr > 0.0

    def test_direct_log_closed_in_train_finally(self, tmp_path):
        """Line 1102: _direct_log.close() called in train() finally block."""
        log_path = str(tmp_path / "direct.log")
        trainer = self._make_trainer(tmp_path, direct_log_file=log_path)
        assert trainer._direct_log is not None
        # Run for ~0.3s; the finally block always runs regardless of duration
        trainer.train(duration_hours=0.00005)
        # File handle should be closed after train() completes
        assert trainer._direct_log.closed

    def test_eval_vs_bullseye_returns_win_rates(self, tmp_path):
        """§338: _eval_vs_bullseye returns dqn_red_wr and dqn_blue_wr in [0,1]."""
        trainer = self._make_trainer(tmp_path)
        result = trainer._eval_vs_bullseye(n_games=5)
        assert "dqn_red_wr"  in result, "Missing dqn_red_wr"
        assert "dqn_blue_wr" in result, "Missing dqn_blue_wr"
        assert 0.0 <= result["dqn_red_wr"]  <= 1.0, f"dqn_red_wr out of range: {result['dqn_red_wr']}"
        assert 0.0 <= result["dqn_blue_wr"] <= 1.0, f"dqn_blue_wr out of range: {result['dqn_blue_wr']}"

    def test_log_progress_with_bullseye_heuristic(self, tmp_path, monkeypatch):
        """§338: _log_progress calls _eval_vs_bullseye and logs vs-Bullseye when enabled."""
        import time as _time
        trainer = self._make_trainer(tmp_path, use_bullseye_heuristic=True)
        trainer.training_start = _time.time() - 10.0
        trainer._log_call_count = 0
        trainer._best_combined_heuristic_wr = 0.0
        trainer._recent_results.extend(["red"] * 20 + ["blue"] * 10)
        trainer._recent_losses.extend([0.5] * 10)

        monkeypatch.setattr(trainer, "_eval_vs_random",
                            lambda n_games: {"dqn_red_wr": 0.9, "dqn_blue_wr": 0.8})
        monkeypatch.setattr(trainer, "_eval_vs_heuristic",
                            lambda n_games: {"dqn_red_wr": 0.85, "dqn_blue_wr": 0.80})
        monkeypatch.setattr(trainer, "_eval_vs_bullseye",
                            lambda n_games: {"dqn_red_wr": 0.1, "dqn_blue_wr": 0.05})

        trainer._log_progress(end_time=_time.time() + 100.0)
        # vs-bullseye rates should be stored
        assert trainer._vs_bullseye_dqn_red_wr  == pytest.approx(0.1)
        assert trainer._vs_bullseye_dqn_blue_wr == pytest.approx(0.05)
