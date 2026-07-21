"""
GPU-optimized trainer for Tic Tac Pro on NVIDIA DGX Spark GB10.

Design:
  - Vectorized game collection: n_envs games run in lock-step, one batched GPU
    forward pass per move-step saturates Tensor Cores.
  - Parallel CPU workers (ProcessPoolExecutor): separate processes run game
    simulation using a periodically-synced CPU copy of the policy so all 20
    ARM cores stay busy while the GPU trains.
  - Time-based loop: runs for `duration_hours`, not a fixed episode count.
  - HealthMonitor thread: writes training_state.json every 60 s and warns if
    training appears stuck (no update for >5 min, NaN loss, etc.).
  - Tensorboard logging throughout.
"""

from __future__ import annotations

import json
import multiprocessing as _mp
import os
import pickle
import random
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Must use "spawn" so worker processes start fresh without inheriting the
# parent's CUDA context (forked CUDA contexts hang in child processes).
_spawn_ctx = _mp.get_context("spawn")


def _worker_process_init():
    """
    §164a: Initializer run in each spawned worker process before any task.
    Forces single-threaded BLAS so 16 parallel workers don't spawn 16×N_THREADS
    threads on N CPUs (e.g. 16×20=320 threads on 20 cores = 16× slowdown).
    Must run before numpy/BLAS is first used — ProcessPoolExecutor's initializer
    runs before any task is dispatched, so this fires at process creation time.
    """
    import os as _os
    _os.environ["OMP_NUM_THREADS"]      = "1"
    _os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"]      = "1"
    _os.environ["NUMEXPR_NUM_THREADS"]  = "1"

from game.tictacpro import Player, TicTacPro, pick_rollout_move
from rl.agent import GPUDQNAgent
from rl.network import GPUDQNNetwork, NumpyDQNInference


# ---------------------------------------------------------------------------
# §135  Symmetry augmentation — 8-fold dihedral group on the 3×3 board
# ---------------------------------------------------------------------------
#
# TicTacPro's 3×3 board has 8 symmetries: 4 rotations × 2 reflections.
# Applying all 8 to each collected transition gives 8× data for free, which:
#   1. Improves data efficiency (fills the buffer with more diverse positions)
#   2. Forces the network to learn rotation/reflection invariance
#   3. Reduces self-play overfitting (same game seen from 8 perspectives)
#
# State layout: buf[0:27] = current-player board, buf[27:54] = opp board,
#               both encoded as flat r*9+c*3+s (s=0,1,2 for S/M/L).
# Action index = r*9 + c*3 + (size−1), same encoding as the board.
# Piece-count features (buf[54:60]) and color feature (buf[60]) are invariant
# under spatial symmetry — rotation/flip doesn't change who moves next.
# The 8 permutations are precomputed once at module load.

def _build_sym_perms() -> np.ndarray:
    """Return (8, 27) int32 array: perm[k, src] = dst after symmetry k."""
    transforms = [
        lambda r, c: (r,     c    ),   # identity
        lambda r, c: (c,     2-r  ),   # 90° CW
        lambda r, c: (2-r,   2-c  ),   # 180°
        lambda r, c: (2-c,   r    ),   # 270° CW
        lambda r, c: (r,     2-c  ),   # horizontal flip
        lambda r, c: (2-r,   c    ),   # vertical flip
        lambda r, c: (c,     r    ),   # transpose (main diagonal)
        lambda r, c: (2-c,   2-r  ),   # anti-diagonal flip
    ]
    perms = np.zeros((8, 27), dtype=np.int32)
    for k, tf in enumerate(transforms):
        for r in range(3):
            for c in range(3):
                for s in range(3):
                    nr, nc = tf(r, c)
                    perms[k, r*9 + c*3 + s] = nr*9 + nc*3 + s
    return perms

_SYM_PERMS: np.ndarray = _build_sym_perms()  # shape (8, 27)
# Inverse permutations for §156 vectorized gather: inv_perm[k] = argsort(perm[k]).
# scatter new_s[perm[k]] = old_s ↔ gather new_s = old_s[inv_perm[k]]
_SYM_INV_PERMS: np.ndarray = np.argsort(_SYM_PERMS, axis=1).astype(np.int32)  # (8, 27)


def augment_transitions(transitions: List[Tuple]) -> List[Tuple]:
    """
    §156: Vectorized 8-fold symmetry augmentation.

    Replaces the nested Python loop (N × 8 iterations, 2 np.empty() allocs each)
    with 8 bulk numpy fancy-index operations on (N, 27) slabs.  For N=8704
    (16 workers × 32 games × ~17 moves), this reduces Python overhead from
    ~140k iterations to 8 numpy calls — ~10–30× faster.

    Input/output format: list of (state, action_idx, reward, next_state, done).
    All 8 symmetries (including identity at k=0) are returned.
    """
    if not transitions:
        return []

    n = len(transitions)
    # Unpack in one pass — avoids repeated tuple attribute access.
    states_arr    = np.empty((n, 61), dtype=np.float32)
    nstates_arr   = np.empty((n, 61), dtype=np.float32)
    actions_arr   = np.empty(n, dtype=np.int32)
    rewards_arr   = np.empty(n, dtype=np.float32)
    dones_arr     = np.empty(n, dtype=np.float32)

    for i, (s, a, r, ns, d) in enumerate(transitions):
        states_arr[i]  = s
        nstates_arr[i] = ns
        actions_arr[i] = a
        rewards_arr[i] = r
        dones_arr[i]   = d

    # Board slabs: (n, 27) each
    s_me  = states_arr[:, :27]   # current-player board
    s_opp = states_arr[:, 27:54]
    s_rest = states_arr[:, 54:]  # (n, 7): piece counts + bias — invariant

    ns_me  = nstates_arr[:, :27]
    ns_opp = nstates_arr[:, 27:54]
    ns_rest = nstates_arr[:, 54:]

    # Output arrays for all 8 × n augmented transitions.
    out_s  = np.empty((8 * n, 61), dtype=np.float32)
    out_ns = np.empty((8 * n, 61), dtype=np.float32)
    out_acts = np.empty(8 * n, dtype=np.int32)

    for k in range(8):
        fwd  = _SYM_PERMS[k]      # (27,): scatter perm — new_board[fwd[i]] = old_board[i]
        inv  = _SYM_INV_PERMS[k]  # (27,): gather perm — new_board = old_board[inv]
        sl   = slice(k * n, (k + 1) * n)

        # §156: scatter new_s[fwd] = s_me ↔ gather new_s = s_me[inv].
        # Using gather (fancy index on last axis) enables bulk numpy ops on (n, 27) slabs.
        out_s[sl, :27]   = s_me[:, inv]
        out_s[sl, 27:54] = s_opp[:, inv]
        out_s[sl, 54:]   = s_rest

        out_ns[sl, :27]   = ns_me[:, inv]
        out_ns[sl, 27:54] = ns_opp[:, inv]
        out_ns[sl, 54:]   = ns_rest

        out_acts[sl] = fwd[actions_arr]  # action index transforms via forward perm

    # Reorder from k-major (8, n, ...) → transition-major (n, 8, ...)
    # to match original ordering: [t0_k0, t0_k1, ..., t0_k7, t1_k0, ...].
    out_s    = out_s.reshape(8, n, 61).transpose(1, 0, 2).reshape(8 * n, 61)
    out_ns   = out_ns.reshape(8, n, 61).transpose(1, 0, 2).reshape(8 * n, 61)
    out_acts = out_acts.reshape(8, n).T.reshape(8 * n)

    # Tile rewards and dones interleaved: [r0,r0,...,r0, r1,r1,...] → repeat each n 8 times.
    out_rewards = np.repeat(rewards_arr, 8)   # (8n,)
    out_dones   = np.repeat(dones_arr, 8)     # (8n,)

    # Return as list of tuples for backward compatibility with add_batch.
    return list(zip(out_s, out_acts.tolist(), out_rewards.tolist(),
                    out_ns, out_dones.tolist()))


def augment_transitions_arrays(
    transitions: List[Tuple],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    §157: Same as augment_transitions() but returns five raw numpy arrays
    (states, actions, rewards, next_states, dones) instead of a list of tuples.

    Skips the final list(zip(...)) call that creates ~70K Python objects for a
    typical 8704-transition batch × 8 symmetries.  Feed directly into
    PrioritizedReplayBuffer.add_batch_arrays() for a full zero-copy path from
    game collection to buffer insertion.

    Returns:
        out_s       (8n, 61) float32
        out_acts    (8n,)    int32
        out_rewards (8n,)    float32
        out_ns      (8n, 61) float32
        out_dones   (8n,)    float32
    """
    if not transitions:
        empty61 = np.empty((0, 61), dtype=np.float32)
        empty1  = np.empty(0, dtype=np.float32)
        return empty61, np.empty(0, dtype=np.int32), empty1, empty61, empty1

    n = len(transitions)
    states_arr  = np.empty((n, 61), dtype=np.float32)
    nstates_arr = np.empty((n, 61), dtype=np.float32)
    actions_arr = np.empty(n, dtype=np.int32)
    rewards_arr = np.empty(n, dtype=np.float32)
    dones_arr   = np.empty(n, dtype=np.float32)

    for i, (s, a, r, ns, d) in enumerate(transitions):
        states_arr[i]  = s
        nstates_arr[i] = ns
        actions_arr[i] = a
        rewards_arr[i] = r
        dones_arr[i]   = d

    s_me   = states_arr[:, :27];   s_opp  = states_arr[:, 27:54];  s_rest  = states_arr[:, 54:]
    ns_me  = nstates_arr[:, :27];  ns_opp = nstates_arr[:, 27:54]; ns_rest = nstates_arr[:, 54:]

    out_s    = np.empty((8 * n, 61), dtype=np.float32)
    out_ns   = np.empty((8 * n, 61), dtype=np.float32)
    out_acts = np.empty(8 * n, dtype=np.int32)

    for k in range(8):
        fwd = _SYM_PERMS[k]; inv = _SYM_INV_PERMS[k]
        sl  = slice(k * n, (k + 1) * n)
        out_s[sl, :27]   = s_me[:, inv];  out_s[sl, 27:54]  = s_opp[:, inv];  out_s[sl, 54:]   = s_rest
        out_ns[sl, :27]  = ns_me[:, inv]; out_ns[sl, 27:54] = ns_opp[:, inv]; out_ns[sl, 54:]  = ns_rest
        out_acts[sl] = fwd[actions_arr]

    out_s    = out_s.reshape(8, n, 61).transpose(1, 0, 2).reshape(8 * n, 61)
    out_ns   = out_ns.reshape(8, n, 61).transpose(1, 0, 2).reshape(8 * n, 61)
    out_acts = out_acts.reshape(8, n).T.reshape(8 * n)
    out_rewards = np.repeat(rewards_arr, 8)
    out_dones   = np.repeat(dones_arr, 8)

    return out_s, out_acts, out_rewards, out_ns, out_dones


# ---------------------------------------------------------------------------
# Vectorized episode collection (main-process, batched GPU inference)
# ---------------------------------------------------------------------------

def collect_episodes_vectorized(
    agent: GPUDQNAgent,
    n_envs: int,
    random_opponent_frac: float = 0.0,
    mc_returns: bool = False,
    gamma: float = 0.99,
) -> Tuple[List[Tuple], Dict]:
    """
    Collect n_envs complete episodes.

    All active games share a single batched GPU forward pass per move-step,
    which keeps Tensor Core utilisation high for large n_envs.

    random_opponent_frac: fraction of games where the BLUE (second) player is
        replaced by a random agent.  Exposes the DQN to diverse game states and
        reduces self-play overfitting without adding inference overhead on those
        turns (the random games are excluded from the batched forward pass when
        it is BLUE's turn).

    mc_returns: §147 — if True, store discounted Monte Carlo returns for every
        transition (done=1.0 for all, reward=±γ^{T-1-t}).  Bypasses the
        bootstrapping chain: every transition directly carries the full game
        outcome, so the Q-function learns the true value without waiting for
        Bellman backpropagation across 17 iterations.

    Returns:
        transitions: list of (state, action_idx, reward, next_state, done)
        stats: dict with aggregate episode statistics
    """
    games = [TicTacPro() for _ in range(n_envs)]
    # per-game: list of (state, action_idx, player, next_state, game_done)
    ep_data: List[List] = [[] for _ in range(n_envs)]
    done = np.zeros(n_envs, dtype=bool)

    # Games in [0, n_random) have a random BLUE opponent (§109).
    n_random = int(n_envs * random_opponent_frac)
    _BLUE = Player.BLUE

    while not done.all():
        active_ids = np.where(~done)[0]

        # Separate games that need DQN inference from random-BLUE games on BLUE's turn.
        dqn_ids: List[int] = []
        rand_ids: List[int] = []
        for i in active_ids:
            if i < n_random and games[i].current_player == _BLUE:
                rand_ids.append(i)
            else:
                dqn_ids.append(i)

        # Pre-capture states before inference so get_actions_batch can reuse
        # them (§140 — avoids a second get_state_tensor_normalized() per step).
        dqn_states = [games[i].get_state_tensor_normalized() for i in dqn_ids]
        state_for  = dict(zip(dqn_ids, dqn_states))

        # Batched DQN inference — pass pre-captured states to avoid redundant call.
        dqn_actions = (
            agent.get_actions_batch(
                [games[i] for i in dqn_ids],
                precomputed_states=dqn_states,
            )
            if dqn_ids else []
        )

        for j, i in enumerate(dqn_ids):
            game = games[i]
            act  = dqn_actions[j]

            if act is None or game.game_over:  # pragma: no cover — agent always returns valid action
                done[i] = True
                continue

            row, col, size = act
            action_idx = row * 9 + col * 3 + (size - 1)
            state  = state_for[i]
            player = game.current_player

            game.make_move(row, col, size)
            next_state = game.get_state_tensor_normalized()

            ep_data[i].append((state, action_idx, player, next_state, game.game_over))
            if game.game_over:
                done[i] = True

        # Random moves for BLUE in random-opponent games.
        # §141: do NOT store these transitions — the action was random, not DQN-
        # controlled, so the (state, random_action) pair is noise that pollutes
        # the Q-function for BLUE positions.  RED's preceding transition still
        # gets the correct bootstrap via the negamax Bellman target (§134).
        for i in rand_ids:
            game  = games[i]
            moves = game.get_legal_moves(_BLUE)
            if not moves or game.game_over:  # pragma: no cover — TicTacPro always has legal moves
                done[i] = True
                continue

            act = random.choice(moves)
            row, col, size = act
            game.make_move(row, col, size)
            if game.game_over:
                done[i] = True

    transitions: List[Tuple] = []
    win_counts = {Player.RED: 0, Player.BLUE: 0, Player.NONE: 0}
    move_lengths: List[int] = []

    for i, (game, data) in enumerate(zip(games, ep_data)):
        winner = game.winner
        win_counts[winner] += 1
        T = len(data)
        move_lengths.append(T)

        if mc_returns:
            # §147: Monte Carlo returns — every transition carries the full discounted
            # game outcome.  reward_t = ±γ^{T-1-t} from player_t's perspective.
            # done=1.0 for all (no bootstrapping); next_state stored but unused in target.
            for t, (state, action_idx, player, next_state, _) in enumerate(data):
                if winner == Player.NONE:  # pragma: no cover — draws extremely rare in TicTacPro
                    reward = 0.0
                else:
                    disc = gamma ** (T - 1 - t)
                    reward = disc if winner == player else -disc
                transitions.append((state, action_idx, reward, next_state, 1.0))
        else:
            for state, action_idx, player, next_state, is_done in data:
                # §134 terminal-only reward: non-terminal transitions get 0.0 so that
                # the Bellman target reduces to -γ*next_q (negamax bootstrap).
                if is_done:
                    if winner == Player.NONE:
                        reward = 0.0
                    elif winner == player:
                        reward = 1.0
                    else:  # pragma: no cover — only the winning player makes the done=True move
                        reward = -1.0
                else:
                    reward = 0.0
                transitions.append((state, action_idx, reward, next_state, float(is_done)))

    stats = {
        "n_games": n_envs,
        "red_wins": win_counts[Player.RED],
        "blue_wins": win_counts[Player.BLUE],
        "draws": win_counts[Player.NONE],
        "avg_moves": float(np.mean(move_lengths)) if move_lengths else 0.0,
        "n_transitions": len(transitions),
    }
    return transitions, stats


# ---------------------------------------------------------------------------
# CPU worker (spawned by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _cpu_worker_collect(args: Tuple) -> Tuple[List[Tuple], Dict]:
    """
    Collect `n_games` complete episodes using a CPU copy of the policy.
    Runs in a separate process — no GPU access needed.
    §167: receives weights as pre-pickled bytes (serialized once on main process)
    so each of the 16 per-worker pickle calls just copies bytes, not a full dict.
    Games in [0, n_random) have a heuristic BLUE opponent (§109/§165).
    NOTE: "random_opponent" naming is historical; BLUE uses pick_rollout_move
    (heuristic), not random.choice.  This differs from the GPU-vectorized path
    which uses truly random BLUE.

    Returns (arrays_tuple, stats) where arrays_tuple is the 5-array pre-augmented
    result from augment_transitions_arrays(), so main process only needs np.concatenate.
    """
    import pickle as _pickle
    # §177: heuristic_epsilon is an optional 11th element; default 0.0 for
    # backward compatibility with tests that still pass 10-element tuples.
    # §336: use_bullseye is an optional 12th element; when True, BullseyeAgent
    # replaces pick_rollout_move as the heuristic opponent, forcing DQN to
    # learn bullseye blocking and counter-strategy.
    # §347: use_optimal is an optional 13th element; when True, OptimalAgent
    # replaces pick_rollout_move as the heuristic opponent, teaching DQN-BLUE
    # to counter the anti-diagonal strategy.
    if len(args) < 11:
        _args = args + (0.0, False, False)
    elif len(args) < 12:
        _args = args + (False, False)
    elif len(args) < 13:
        _args = args + (False,)
    else:
        _args = args
    n_games, epsilon, weights_bytes, hidden_sizes, state_size, action_size, random_opponent_frac, reverse_opponent_frac, mc_returns, gamma, heuristic_epsilon, use_bullseye, use_optimal = _args

    # §164a: Force single-threaded BLAS before any matmul.  OpenBLAS defaults to
    # multi-threaded, which causes thread-pool contention when 16 workers run in
    # parallel on 20 cores (each numpy @ spawns N threads → 16*N > 20 → thrash).
    # Setting these env vars at function entry (each spawn is a fresh process) pins
    # each worker to 1 BLAS thread, dropping per-call cost 1.28ms → 0.28ms (4.6×)
    # and eliminating the 30× parallel-execution slowdown.
    import os as _os
    _os.environ["OMP_NUM_THREADS"]     = "1"
    _os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"]     = "1"
    _os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # §167: deserialize weights from pre-pickled bytes. This costs ~2ms (same as
    # the old dict deserialization) but saves 35ms on the MAIN PROCESS side:
    # pickle.dumps(dict) × 16 → pickle.dumps(bytes) × 16 (bytes is a memcpy).
    weights_cpu = _pickle.loads(weights_bytes)
    net = NumpyDQNInference(weights_cpu, tuple(hidden_sizes))

    n_random  = int(n_games * random_opponent_frac)
    # §168: reverse heuristic — games in [n_random, n_random+n_reverse) have
    # heuristic RED so the DQN trains as BLUE. RED transitions are skipped
    # (skip_red=True) to avoid noisy heuristic-side signal.
    n_reverse = int(n_games * reverse_opponent_frac)
    _BLUE = Player.BLUE
    _RED  = Player.RED

    games = [TicTacPro() for _ in range(n_games)]
    ep_data: List[List] = [[] for _ in range(n_games)]
    done = [False] * n_games
    move_counts = [0] * n_games  # §178: total moves per game (both players)

    while not all(done):
        # --- Pass 1: classify action type; capture pre-move state tensor ---
        # §139: collect DQN states for a single batched forward pass instead of
        # one serial net(st) call per game — reduces ~575 forward passes/round
        # to ~18 batched calls (~1 per move-step across all active games).
        # §141: heuristic BLUE turns skip state capture and storage (noisy signal).
        # §168: reverse games (n_random ≤ i < n_random+n_reverse) skip RED turns.
        pending = []  # (game_idx, moves, state_or_None, needs_dqn)
        for i, game in enumerate(games):
            if done[i] or game.game_over:
                done[i] = True
                continue
            moves = game.get_legal_moves(game.current_player)
            if not moves:  # pragma: no cover — TicTacPro always has legal moves before game_over
                done[i] = True
                continue
            is_heuristic_blue = (i < n_random and game.current_player == _BLUE)
            is_reverse_game   = (n_random <= i < n_random + n_reverse)
            is_heuristic_red  = (is_reverse_game and game.current_player == _RED)
            if is_heuristic_blue or is_heuristic_red:
                # §141/§168: heuristic opponent move — don't store transition.
                # state=None signals Pass 2 to use pick_rollout_move.
                pending.append((i, moves, None, False))
            else:
                needs_dqn = random.random() >= epsilon
                pending.append((i, moves, game.get_state_tensor_normalized(), needs_dqn))

        if not pending:  # pragma: no cover — at least one active game always has pending moves
            break

        # --- Batched DQN inference (§139, §164) ---
        dqn_ks = [k for k, (_, _, _, nd) in enumerate(pending) if nd]
        q_map: dict = {}
        if dqn_ks:
            # §164: pass numpy array directly — NumpyDQNInference needs no
            # torch.from_numpy() wrapper, avoiding all tensor allocation overhead.
            batch_np = np.stack([pending[k][2] for k in dqn_ks])
            q_batch = net(batch_np)
            for k, qv in zip(dqn_ks, q_batch):
                q_map[pending[k][0]] = qv

        # --- Pass 2: apply moves ---
        for game_idx, moves, state, needs_dqn in pending:
            game = games[game_idx]
            player = game.current_player
            if needs_dqn:
                qv = q_map[game_idx]
                mask = np.full(action_size, -1e9, dtype=np.float32)
                for r, c, s in moves:
                    idx = r * 9 + c * 3 + (s - 1)
                    mask[idx] = qv[idx]
                best = int(np.argmax(mask))
                act = (best // 9, (best % 9) // 3, (best % 3) + 1)
            elif state is None:
                # §165: heuristic opponent — win > block > threat > rest.
                # §177: with heuristic_epsilon, randomly skip the heuristic to
                # expose DQN-BLUE to non-deterministic RED play and prevent
                # overfitting to fixed opening patterns.
                # §336: use_bullseye replaces pick_rollout_move with BullseyeAgent,
                # forcing DQN to learn bullseye blocking and counter-strategy.
                if heuristic_epsilon > 0.0 and random.random() < heuristic_epsilon:
                    act = random.choice(moves)
                elif use_optimal:
                    # §347: OptimalAgent — anti-diagonal rule-based agent.
                    # Teaches DQN-BLUE to counter the anti-diagonal Small strategy.
                    from game.optimal_agent import OptimalAgent as _OptimalAgent
                    if not hasattr(_cpu_worker_collect, '_oa'):
                        _cpu_worker_collect._oa = _OptimalAgent()
                    heuristic_act = _cpu_worker_collect._oa.get_action(game)
                    act = heuristic_act if heuristic_act is not None else random.choice(moves)
                elif use_bullseye:
                    from game.bullseye_agent import BullseyeAgent as _BullseyeAgent
                    if not hasattr(_cpu_worker_collect, '_bsa'):
                        _cpu_worker_collect._bsa = _BullseyeAgent()
                    heuristic_act = _cpu_worker_collect._bsa.get_action(game)
                    act = heuristic_act if heuristic_act is not None else random.choice(moves)
                else:
                    heuristic_act, _ = pick_rollout_move(game)
                    act = heuristic_act if heuristic_act is not None else random.choice(moves)
            else:
                # epsilon-random exploration in self-play games
                act = random.choice(moves)
            row, col, size = act
            action_idx = row * 9 + col * 3 + (size - 1)
            game.make_move(row, col, size)
            move_counts[game_idx] += 1  # §178: count every move (DQN + random + heuristic)
            if state is not None:  # §141: skip heuristic-opponent transitions
                ep_data[game_idx].append(
                    (state, action_idx, player, game.get_state_tensor_normalized(), game.game_over)
                )
            if game.game_over:
                done[game_idx] = True

    transitions: List[Tuple] = []
    red_wins = blue_wins = draws = 0
    # §176: per-category win counters — self-play, random-opp, reverse-opp.
    # Enables separate win-rate display so DQN-RED vs DQN-BLUE progress can
    # be tracked independently from random/heuristic baseline wins.
    sp_red = sp_blue = sp_draw = 0    # self-play (both players DQN)
    ro_red = ro_blue = ro_draw = 0    # random-opp (DQN-RED vs random-BLUE)
    rv_red = rv_blue = rv_draw = 0    # reverse-opp (heuristic-RED vs DQN-BLUE)
    # §178: game length accumulators (total moves, both players, per category)
    all_lengths: List[int] = []
    sp_lengths: List[int] = []
    ro_lengths: List[int] = []
    rv_lengths: List[int] = []
    for i, (game, data) in enumerate(zip(games, ep_data)):
        winner = game.winner
        if winner == Player.RED:
            red_wins += 1
        elif winner == Player.BLUE:
            blue_wins += 1
        else:
            draws += 1
        # §176: assign to category based on game index; §178: accumulate lengths
        gl = move_counts[i]
        all_lengths.append(gl)
        if i < n_random:
            if winner == Player.RED:   ro_red  += 1
            elif winner == Player.BLUE: ro_blue += 1
            else:                       ro_draw += 1  # pragma: no cover — draws extremely rare vs heuristic
            ro_lengths.append(gl)
        elif i < n_random + n_reverse:
            if winner == Player.RED:   rv_red  += 1
            elif winner == Player.BLUE: rv_blue += 1  # pragma: no cover — requires skilled DQN; untrained rarely beats heuristic-RED
            else:                       rv_draw += 1  # pragma: no cover — draws extremely rare vs heuristic
            rv_lengths.append(gl)
        else:
            if winner == Player.RED:   sp_red  += 1
            elif winner == Player.BLUE: sp_blue += 1
            else:                       sp_draw += 1
            sp_lengths.append(gl)
        T = len(data)
        if mc_returns:
            # §147: MC returns — ±γ^{T-1-t} from each player's perspective.
            for t, (state, action_idx, player, next_state, _) in enumerate(data):
                if winner == Player.NONE:
                    reward = 0.0
                else:
                    disc = gamma ** (T - 1 - t)
                    reward = disc if winner == player else -disc
                transitions.append((state, action_idx, reward, next_state, 1.0))
        else:
            for state, action_idx, player, next_state, is_done in data:
                # §134 terminal-only reward: see collect_episodes_vectorized above.
                if is_done:
                    reward = (
                        0.0
                        if winner == Player.NONE
                        else (1.0 if winner == player else -1.0)
                    )
                else:
                    reward = 0.0
                transitions.append((state, action_idx, reward, next_state, float(is_done)))

    stats = {
        "n_games": n_games,
        "red_wins": red_wins,
        "blue_wins": blue_wins,
        "draws": draws,
        "n_transitions": len(transitions),
        # §176: per-category breakdown
        "sp_red": sp_red, "sp_blue": sp_blue, "sp_draw": sp_draw,
        "ro_red": ro_red, "ro_blue": ro_blue, "ro_draw": ro_draw,
        "rv_red": rv_red, "rv_blue": rv_blue, "rv_draw": rv_draw,
        "n_random": n_random, "n_reverse": n_reverse,
        # §178: actual game lengths (total moves, both players) per category
        "avg_game_length":    float(np.mean(all_lengths)) if all_lengths else 0.0,
        "avg_sp_game_length": float(np.mean(sp_lengths))  if sp_lengths  else 0.0,
        "avg_ro_game_length": float(np.mean(ro_lengths))  if ro_lengths  else 0.0,
        "avg_rv_game_length": float(np.mean(rv_lengths))  if rv_lengths  else 0.0,
    }
    # §166: Augment here (in the worker) rather than on the main process.
    # Moving the 8-fold symmetry expansion into the worker eliminates the ~62ms
    # serial augment_transitions_arrays() call from the main-process critical path.
    # Each worker augments its own ~544 transitions in ~4ms, running in parallel
    # with GPU training (~820ms) so the cost is entirely hidden.
    # Main process just concatenates pre-augmented arrays from all workers (~1ms).
    out_s, out_acts, out_rewards, out_ns, out_dones = augment_transitions_arrays(transitions)
    return (out_s, out_acts, out_rewards, out_ns, out_dones), stats


# ---------------------------------------------------------------------------
# Health monitor (background thread)
# ---------------------------------------------------------------------------

class HealthMonitor:
    """
    Writes `training_state.json` every `check_interval` seconds and logs a
    warning if no update has arrived in `stuck_threshold` seconds or if the
    loss becomes non-finite.
    """

    def __init__(
        self,
        state_file: str = "training_state.json",
        check_interval: int = 60,
        stuck_threshold: int = 300,
    ):
        self.state_file = state_file
        self.check_interval = check_interval
        self.stuck_threshold = stuck_threshold
        self._state: Dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def update(self, **kwargs):
        with self._lock:
            self._state.update(kwargs)
            self._state["timestamp"] = time.time()
            self._state["last_update"] = datetime.now().strftime("%H:%M:%S")

    def _flush(self):
        with self._lock:
            snapshot = dict(self._state)
        try:
            with open(self.state_file, "w") as fh:
                json.dump(snapshot, fh, indent=2, default=str)
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            self._flush()
            with self._lock:
                snapshot = dict(self._state)

            ts = snapshot.get("timestamp", 0.0)
            age = time.time() - ts
            if ts > 0 and age > self.stuck_threshold:
                print(
                    f"\n[HEALTH WARNING] No training update for {age:.0f}s "
                    f"(threshold {self.stuck_threshold}s) — training may be stuck!"
                )
                print(f"  Last known state: {snapshot}")

            avg_loss = snapshot.get("avg_loss")
            if avg_loss is not None and not np.isfinite(avg_loss):
                print(f"\n[HEALTH WARNING] Loss is non-finite: {avg_loss}")

            eps_sec = snapshot.get("eps_per_second", 0)
            if snapshot and eps_sec == 0 and snapshot.get("gradient_steps", 0) > 1000:
                print("\n[HEALTH WARNING] Episodes/sec reported as 0 — check worker health")

            self._stop.wait(self.check_interval)
        self._flush()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="HealthMonitor")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._flush()


# ---------------------------------------------------------------------------
# Main GPU trainer
# ---------------------------------------------------------------------------

class GPUTrainer:
    """
    GPU trainer for Tic Tac Pro.

    Two collection modes selectable via `n_cpu_workers`:
      0  — vectorized sequential collection in main process (simpler)
      >0 — parallel CPU workers feed the main-process GPU trainer
           (uses all available ARM cores on the DGX Spark)

    Training loop per iteration:
      1. Collect n_envs games → add transitions to PER buffer
      2. Run `train_steps_per_collect` Double-DQN gradient steps
      3. Log every `log_interval` seconds
      4. Checkpoint every `checkpoint_interval` seconds
    """

    def __init__(
        self,
        agent: GPUDQNAgent,
        checkpoint_dir: str = "checkpoints_spark",
        log_dir: str = "logs_spark",
        n_envs: int = 512,
        train_steps_per_collect: int = 8,
        log_interval: int = 60,
        checkpoint_interval: int = 600,
        state_file: str = "training_state.json",
        n_cpu_workers: int = 0,
        games_per_worker: int = 64,
        random_opponent_frac: float = 0.3,
        reverse_opponent_frac: float = 0.0,
        heuristic_epsilon: float = 0.1,
        mc_returns: bool = False,
        direct_log_file: Optional[str] = None,
        use_bullseye_heuristic: bool = False,  # §336: use BullseyeAgent as opponent
        use_optimal_heuristic: bool = False,   # §347: use OptimalAgent as opponent
    ):
        self.agent = agent
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.n_envs = n_envs
        self.train_steps_per_collect = train_steps_per_collect
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.n_cpu_workers = n_cpu_workers
        self.games_per_worker = games_per_worker
        self.random_opponent_frac = random_opponent_frac
        self.reverse_opponent_frac = reverse_opponent_frac
        self.heuristic_epsilon = heuristic_epsilon  # §177
        self.use_bullseye_heuristic = use_bullseye_heuristic  # §336
        self.use_optimal_heuristic = use_optimal_heuristic    # §347
        self.mc_returns = mc_returns
        # §155: direct file log bypasses the stdout→pipe→tee buffering path
        self._direct_log: Optional[object] = None
        if direct_log_file:
            os.makedirs(os.path.dirname(direct_log_file) if os.path.dirname(direct_log_file) else ".", exist_ok=True)
            self._direct_log = open(direct_log_file, "a", buffering=1)  # line-buffered

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(log_dir, f"spark_{run_tag}"))

        self.total_episodes = 0
        self.total_gradient_steps = 0
        self.training_start: Optional[float] = None

        self.monitor = HealthMonitor(state_file)

        self._recent_results: deque = deque(maxlen=2000)
        self._recent_losses: deque = deque(maxlen=2000)
        self._recent_grad_norms: deque = deque(maxlen=2000)  # §174
        self._recent_collect_ms: deque = deque(maxlen=200)
        self._recent_train_ms: deque = deque(maxlen=200)
        # §176: per-category result deques — self-play / random-opp / reverse-opp
        self._recent_sp: deque = deque(maxlen=2000)   # self-play (DQN vs DQN)
        self._recent_ro: deque = deque(maxlen=2000)   # random-opp (DQN-RED vs random)
        self._recent_rv: deque = deque(maxlen=2000)   # reverse-opp (DQN-BLUE vs heuristic)
        # §178: game length deques — actual total moves (both players) per category
        self._recent_gl:    deque = deque(maxlen=2000)  # overall average game length
        self._recent_gl_sp: deque = deque(maxlen=2000)  # self-play
        self._recent_gl_ro: deque = deque(maxlen=2000)  # random-opp
        self._recent_gl_rv: deque = deque(maxlen=2000)  # reverse-opp
        self._log_call_count: int = 0
        # §148/§173: cached vs-random win rates (updated every 5 log intervals)
        self._vs_random_dqn_red_wr: float = float("nan")
        self._vs_random_dqn_blue_wr: float = float("nan")
        # §169: cached vs-heuristic win rates (updated every 5 log intervals)
        self._vs_heuristic_dqn_red_wr: float = float("nan")
        self._vs_heuristic_dqn_blue_wr: float = float("nan")
        # §173: best combined heuristic win rate for best.pt checkpoint saving
        self._best_combined_heuristic_wr: float = 0.0
        # §338: cached vs-bullseye win rates (updated every 5 log intervals when enabled)
        self._vs_bullseye_dqn_red_wr: float = float("nan")
        self._vs_bullseye_dqn_blue_wr: float = float("nan")
        # §347: cached vs-optimal win rates (updated every 5 log intervals when enabled)
        self._vs_optimal_dqn_red_wr: float = float("nan")
        self._vs_optimal_dqn_blue_wr: float = float("nan")
        self._best_optimal_blue_wr: float = 0.0  # §347: for best_vs_optimal.pt saving

    def _tlog(self, msg: str, flush: bool = False):
        """§155: Print to stdout AND direct log file (bypasses pipe buffering)."""
        print(msg, flush=flush)
        if self._direct_log is not None:
            self._direct_log.write(msg + "\n")
            if flush:
                self._direct_log.flush()

    # ------------------------------------------------------------------

    def _make_worker_args(self) -> List[Tuple]:
        """Serialize current policy weights for CPU worker processes.

        §167: pre-pickle the weight dict ONCE, then share the same bytes object
        in all 16 worker argument tuples. ProcessPoolExecutor pickles each tuple
        separately — if the dict were included directly, it would be serialized 16×
        (35ms measured for a 3MB GPUDQNNetwork). Passing pre-pickled bytes reduces
        per-tuple pickling to a bare memcpy, cutting the total from ~35ms to ~3ms.

        §171: Ape-X-style epsilon diversity — different epsilon values across
        worker groups produce a richer replay buffer than all workers sharing the
        same epsilon. Three tiers: greedy (ε=0, high-quality games), standard
        (ε=current), and exploratory (ε=min(1, current*4), novel state coverage).
        """
        raw_net = getattr(self.agent.policy_net, "_orig_mod", self.agent.policy_net)
        weights_cpu = {k: v.cpu() for k, v in raw_net.state_dict().items()}
        # §167: serialize once; all worker tuples share the same bytes object.
        weights_bytes = pickle.dumps(weights_cpu)

        # §171: assign epsilon tiers to workers
        # NOTE: greedy tier uses eps_low = max(0.02, eps*0.1) rather than 0.0.
        # With ε=0 and a deterministic network, all 32 games in a worker start
        # from the same empty board → same deterministic Q-argmax chain → all
        # 32 games are IDENTICAL. That wastes 31 of 32 game slots per greedy
        # worker and bloats the buffer with duplicate transitions. ε=0.02 is
        # small enough to be nearly greedy while ensuring diverse game trees.
        n = self.n_cpu_workers
        n_greedy  = max(1, n // 4)                    # 25% near-greedy
        n_explore = max(1, n // 4)                    # 25% high-exploration
        n_standard = n - n_greedy - n_explore         # 50% standard
        eps = self.agent.epsilon
        eps_low  = max(0.02, eps * 0.1)               # near-greedy (not pure 0)
        eps_high = min(1.0,  eps * 4.0)
        epsilons = ([eps_low]  * n_greedy +
                    [eps]      * n_standard +
                    [eps_high] * n_explore)

        return [
            (
                self.games_per_worker,
                epsilons[i],
                weights_bytes,
                self.agent.hidden_sizes,
                self.agent.state_size,
                self.agent.action_size,
                self.random_opponent_frac,
                self.reverse_opponent_frac,
                self.mc_returns,
                self.agent.gamma,
                self.heuristic_epsilon,          # §177
                self.use_bullseye_heuristic,     # §336
                self.use_optimal_heuristic,      # §347
            )
            for i in range(n)
        ]

    def _run_train_steps(self, n_steps: int) -> Tuple[List[float], float]:
        """Run n_steps gradient updates. Returns (losses, elapsed_ms)."""
        t0 = time.time()
        losses = []
        for _ in range(n_steps):
            loss = self.agent.learn()
            if loss is not None:
                losses.append(loss)
                self.total_gradient_steps += 1
                self._recent_losses.append(loss)
                self._recent_grad_norms.append(self.agent._last_grad_norm)  # §174
        return losses, (time.time() - t0) * 1000

    def train(self, duration_hours: float = 2.0, resume_from: Optional[str] = None):
        """
        Train for `duration_hours` hours.

        With n_cpu_workers > 0: CPU workers collect next batch concurrently
        while the GPU trains on the current batch (pipelined).
        With n_cpu_workers == 0: sequential vectorized collect + train.
        """
        if resume_from and os.path.exists(resume_from):
            self.agent.load(resume_from)

        self.total_episodes = self.agent.episode_count
        self.total_gradient_steps = self.agent.learn_step_counter
        _steps_at_run_start = self.total_gradient_steps  # §184: baseline for rate measurement

        end_time = time.time() + duration_hours * 3600
        self.training_start = time.time()

        mode = (
            f"pipelined ({self.n_cpu_workers} CPU workers × {self.games_per_worker} games)"
            if self.n_cpu_workers > 0
            else f"vectorized ({self.n_envs} envs)"
        )

        print("\n" + "=" * 72)
        print(f"  DGX Spark Training — {duration_hours:.1f} hours")
        print(f"  Device    : {self.agent.device}")
        print(f"  AMP dtype : {self.agent._amp_dtype}")
        print(f"  Mode      : {mode}")
        print(f"  Batch     : {self.agent.batch_size:,}  |  Buffer: {self.agent.memory.capacity:,}")
        print(f"  Train steps / collect: {self.train_steps_per_collect}")
        print(f"  ETA       : {datetime.now() + timedelta(hours=duration_hours):%Y-%m-%d %H:%M:%S}")
        print("=" * 72 + "\n", flush=True)

        self.monitor.start()

        last_log = self.training_start
        last_ckpt = self.training_start
        _schedule_calibrated = False  # §184: fire once after first log interval
        executor = None

        try:
            if self.n_cpu_workers > 0:
                # spawn context: workers start fresh, no inherited CUDA state.
                # §164a: initializer pins BLAS to 1 thread per worker before any
                # numpy matmul — prevents 16 workers × 20 BLAS threads = 320
                # threads on 20 cores from thrashing (was 16× parallel slowdown).
                executor = ProcessPoolExecutor(
                    max_workers=self.n_cpu_workers,
                    mp_context=_spawn_ctx,
                    initializer=_worker_process_init,
                )
                # Kick off the first collection round before training starts.
                pending_futs = [
                    executor.submit(_cpu_worker_collect, a)
                    for a in self._make_worker_args()
                ]

            while time.time() < end_time:
                if self.n_cpu_workers > 0:
                    # --- Harvest previous collection -------------------------
                    # §166: workers return pre-augmented arrays — no serial
                    # augment_transitions_arrays() needed on the main process.
                    t_c0 = time.time()
                    w_s: List[np.ndarray]   = []
                    w_a: List[np.ndarray]   = []
                    w_r: List[np.ndarray]   = []
                    w_ns: List[np.ndarray]  = []
                    w_d: List[np.ndarray]   = []
                    agg_red = agg_blue = agg_draws = agg_games = agg_trans = 0
                    # §176: per-category aggregators
                    agg_sp_red = agg_sp_blue = agg_sp_draw = 0
                    agg_ro_red = agg_ro_blue = agg_ro_draw = 0
                    agg_rv_red = agg_rv_blue = agg_rv_draw = 0
                    agg_n_random = agg_n_reverse = 0
                    # §178: game length accumulators (unnormalized sums for weighted average)
                    agg_gl_sum = agg_sp_gl_sum = agg_ro_gl_sum = agg_rv_gl_sum = 0.0
                    agg_sp_n = agg_ro_n = agg_rv_n = 0
                    for fut in as_completed(pending_futs):
                        worker_arrays, worker_stats = fut.result()
                        w_s.append(worker_arrays[0]);  w_a.append(worker_arrays[1])
                        w_r.append(worker_arrays[2]);  w_ns.append(worker_arrays[3])
                        w_d.append(worker_arrays[4])
                        agg_red   += worker_stats["red_wins"]
                        agg_blue  += worker_stats["blue_wins"]
                        agg_draws += worker_stats["draws"]
                        wn = worker_stats["n_games"]
                        agg_games += wn
                        agg_trans += worker_stats["n_transitions"]
                        # §176: accumulate per-category counts
                        w_sp_red  = worker_stats.get("sp_red",  0)
                        w_sp_blue = worker_stats.get("sp_blue", 0)
                        w_sp_draw = worker_stats.get("sp_draw", 0)
                        w_ro_red  = worker_stats.get("ro_red",  0)
                        w_ro_blue = worker_stats.get("ro_blue", 0)
                        w_ro_draw = worker_stats.get("ro_draw", 0)
                        w_rv_red  = worker_stats.get("rv_red",  0)
                        w_rv_blue = worker_stats.get("rv_blue", 0)
                        w_rv_draw = worker_stats.get("rv_draw", 0)
                        agg_sp_red  += w_sp_red;  agg_sp_blue += w_sp_blue;  agg_sp_draw += w_sp_draw
                        agg_ro_red  += w_ro_red;  agg_ro_blue += w_ro_blue;  agg_ro_draw += w_ro_draw
                        agg_rv_red  += w_rv_red;  agg_rv_blue += w_rv_blue;  agg_rv_draw += w_rv_draw
                        agg_n_random  += worker_stats.get("n_random",  0)
                        agg_n_reverse += worker_stats.get("n_reverse", 0)
                        # §178: weighted sum of per-category game lengths
                        w_sp_n = w_sp_red + w_sp_blue + w_sp_draw
                        w_ro_n = w_ro_red + w_ro_blue + w_ro_draw
                        w_rv_n = w_rv_red + w_rv_blue + w_rv_draw
                        agg_gl_sum    += worker_stats.get("avg_game_length",    0.0) * wn
                        agg_sp_gl_sum += worker_stats.get("avg_sp_game_length", 0.0) * w_sp_n
                        agg_ro_gl_sum += worker_stats.get("avg_ro_game_length", 0.0) * w_ro_n
                        agg_rv_gl_sum += worker_stats.get("avg_rv_game_length", 0.0) * w_rv_n
                        agg_sp_n += w_sp_n;  agg_ro_n += w_ro_n;  agg_rv_n += w_rv_n
                    collect_ms = (time.time() - t_c0) * 1000

                    # --- Submit NEXT collection immediately (pipeline) -------
                    pending_futs = [
                        executor.submit(_cpu_worker_collect, a)
                        for a in self._make_worker_args()
                    ]

                    ep_stats = {
                        "n_games":       agg_games,
                        "red_wins":      agg_red,
                        "blue_wins":     agg_blue,
                        "draws":         agg_draws,
                        "avg_moves":     agg_trans / max(agg_games, 1),
                        "n_transitions": agg_trans,
                        # §176: per-category
                        "sp_red": agg_sp_red, "sp_blue": agg_sp_blue, "sp_draw": agg_sp_draw,
                        "ro_red": agg_ro_red, "ro_blue": agg_ro_blue, "ro_draw": agg_ro_draw,
                        "rv_red": agg_rv_red, "rv_blue": agg_rv_blue, "rv_draw": agg_rv_draw,
                        "n_random": agg_n_random, "n_reverse": agg_n_reverse,
                        # §178: weighted-average game lengths across all workers
                        "avg_game_length":    agg_gl_sum    / max(agg_games, 1),
                        "avg_sp_game_length": agg_sp_gl_sum / max(agg_sp_n, 1),
                        "avg_ro_game_length": agg_ro_gl_sum / max(agg_ro_n, 1),
                        "avg_rv_game_length": agg_rv_gl_sum / max(agg_rv_n, 1),
                    }
                    # §166: concatenate pre-augmented arrays (~1ms) instead of
                    # running augment_transitions_arrays (~62ms) on main process.
                    self.agent.memory.add_batch_arrays(
                        np.concatenate(w_s,  axis=0),
                        np.concatenate(w_a,  axis=0),
                        np.concatenate(w_r,  axis=0),
                        np.concatenate(w_ns, axis=0),
                        np.concatenate(w_d,  axis=0),
                    )
                else:
                    # --- Sequential vectorized collect -----------------------
                    t_c0 = time.time()
                    transitions, ep_stats = collect_episodes_vectorized(
                        self.agent, self.n_envs, self.random_opponent_frac,
                        mc_returns=self.mc_returns, gamma=self.agent.gamma,
                    )
                    collect_ms = (time.time() - t_c0) * 1000
                    # Sequential path: augment on main process (no workers to parallelize with).
                    self.agent.memory.add_batch_arrays(
                        *augment_transitions_arrays(transitions)
                    )
                self._recent_collect_ms.append(collect_ms)

                self.total_episodes += ep_stats["n_games"]
                self.agent.episode_count = self.total_episodes
                self._recent_results.extend(["red"]  * ep_stats.get("red_wins",  0))
                self._recent_results.extend(["blue"] * ep_stats.get("blue_wins", 0))
                self._recent_results.extend(["draw"] * ep_stats.get("draws",     0))
                # §176: per-category deques — only populated by pipelined worker path
                sp_n = ep_stats.get("sp_red", 0) + ep_stats.get("sp_blue", 0) + ep_stats.get("sp_draw", 0)
                ro_n = ep_stats.get("ro_red", 0) + ep_stats.get("ro_blue", 0) + ep_stats.get("ro_draw", 0)
                rv_n = ep_stats.get("rv_red", 0) + ep_stats.get("rv_blue", 0) + ep_stats.get("rv_draw", 0)
                if sp_n > 0:
                    self._recent_sp.extend(["red"]  * ep_stats.get("sp_red",  0))
                    self._recent_sp.extend(["blue"] * ep_stats.get("sp_blue", 0))
                    self._recent_sp.extend(["draw"] * ep_stats.get("sp_draw", 0))
                if ro_n > 0:
                    self._recent_ro.extend(["red"]  * ep_stats.get("ro_red",  0))
                    self._recent_ro.extend(["blue"] * ep_stats.get("ro_blue", 0))
                    self._recent_ro.extend(["draw"] * ep_stats.get("ro_draw", 0))
                if rv_n > 0:  # pragma: no cover — dead in sequential path; collect_episodes_vectorized has no reverse games
                    self._recent_rv.extend(["red"]  * ep_stats.get("rv_red",  0))
                    self._recent_rv.extend(["blue"] * ep_stats.get("rv_blue", 0))
                    self._recent_rv.extend(["draw"] * ep_stats.get("rv_draw", 0))
                # §178: accumulate actual game lengths per category
                n_all = ep_stats["n_games"]
                if n_all > 0 and ep_stats.get("avg_game_length", 0.0) > 0:
                    self._recent_gl.append(ep_stats["avg_game_length"])
                if sp_n > 0 and ep_stats.get("avg_sp_game_length", 0.0) > 0:
                    self._recent_gl_sp.append(ep_stats["avg_sp_game_length"])
                if ro_n > 0 and ep_stats.get("avg_ro_game_length", 0.0) > 0:
                    self._recent_gl_ro.append(ep_stats["avg_ro_game_length"])
                if rv_n > 0 and ep_stats.get("avg_rv_game_length", 0.0) > 0:  # pragma: no cover — dead in sequential path
                    self._recent_gl_rv.append(ep_stats["avg_rv_game_length"])

                # --- GPU training (concurrent with next CPU collection) -----
                step_losses, train_ms = self._run_train_steps(self.train_steps_per_collect)
                self._recent_train_ms.append(train_ms)

                if step_losses:
                    avg_l = float(np.mean(step_losses))
                    self.writer.add_scalar("Loss/train", avg_l, self.total_gradient_steps)
                    self.writer.add_scalar("Training/epsilon", self.agent.epsilon, self.total_gradient_steps)
                    self.writer.add_scalar("Training/buffer_fill", len(self.agent.memory), self.total_gradient_steps)
                    self.writer.add_scalar("Training/lr", self.agent.optimizer.param_groups[0]["lr"], self.total_gradient_steps)
                    # §174: per-step grad_norm kept in _recent_grad_norms; avg/max logged in _log_progress

                now = time.time()
                if now - last_log >= self.log_interval:
                    # §184: after first log interval, measure actual throughput and
                    # recalibrate PER beta and LR schedules so they complete by run end.
                    if not _schedule_calibrated and self.total_gradient_steps > _steps_at_run_start:
                        elapsed_s = now - self.training_start
                        remaining_s = max(end_time - now, 1.0)
                        steps_this_run = self.total_gradient_steps - _steps_at_run_start
                        rate = steps_this_run / max(elapsed_s, 1.0)
                        remaining_steps = int(rate * remaining_s)
                        if remaining_steps > 0:
                            self.agent.recalibrate_schedules(remaining_steps)
                            print(
                                f"  §184 schedule recal: {rate * 3600:.0f} steps/h measured "
                                f"→ remaining ~{remaining_steps:,} steps "
                                f"(β-inc={self.agent.memory.beta_increment:.2e}, "
                                f"LR T_max={self.agent.scheduler.T_max:,})",
                                flush=True,
                            )
                        _schedule_calibrated = True
                    self._log_progress(end_time)
                    last_log = now

                if now - last_ckpt >= self.checkpoint_interval:
                    self._save("latest")
                    last_ckpt = now

        except KeyboardInterrupt:  # pragma: no cover
            print("\nTraining interrupted by user.")
        finally:
            if executor:
                executor.shutdown(wait=False)
            self._save("final")
            self.monitor.stop()
            self.writer.close()
            if self._direct_log is not None:
                self._direct_log.close()

        elapsed = time.time() - self.training_start
        print(f"\n{'='*72}")
        print(f"  Training complete: {elapsed/3600:.2f} h")
        print(f"  Total episodes     : {self.total_episodes:,}")
        print(f"  Total grad steps   : {self.total_gradient_steps:,}")
        print(f"  Final epsilon      : {self.agent.epsilon:.4f}")
        if self._recent_losses:
            recent = list(self._recent_losses)[-200:]
            print(f"  Final avg loss     : {np.mean(recent):.4f}")
        print(f"  Checkpoints in     : {self.checkpoint_dir}/")
        print("=" * 72)

    def _log_progress(self, end_time: float):
        now = time.time()
        elapsed = now - self.training_start
        remaining = max(0.0, end_time - now)

        n = len(self._recent_results)
        red_r = self._recent_results.count("red") / max(n, 1)
        blue_r = self._recent_results.count("blue") / max(n, 1)
        draw_r = self._recent_results.count("draw") / max(n, 1)

        recent_losses = list(self._recent_losses)[-200:]
        avg_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")

        c_ms = float(np.mean(self._recent_collect_ms)) if self._recent_collect_ms else 0
        t_ms = float(np.mean(self._recent_train_ms)) if self._recent_train_ms else 0
        # §144: pipelined mode overlaps collect and train, so the bottleneck
        # is max(collect, train), not the sum. Sequential mode (no CPU workers)
        # runs them serially, so the sum is correct there.
        if self.n_cpu_workers > 0:
            iter_ms = max(c_ms, t_ms)
        else:
            iter_ms = c_ms + t_ms
        games_per_round = (
            self.n_cpu_workers * self.games_per_worker if self.n_cpu_workers > 0
            else self.n_envs
        )
        eps_per_sec = (games_per_round * 1000 / iter_ms) if iter_ms > 0 else 0.0

        gpu_mem_gb = 0.0
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.memory_allocated() / 1024 ** 3

        lr_now = self.agent.optimizer.param_groups[0]["lr"]

        self._tlog(
            f"\n[{datetime.now():%H:%M:%S}] "
            f"Elapsed {elapsed/3600:.2f}h | Remaining {remaining/3600:.2f}h",
            flush=True,
        )
        self._tlog(
            f"  Episodes: {self.total_episodes:>10,} | "
            f"Grad steps: {self.total_gradient_steps:>10,} | "
            f"Buffer: {len(self.agent.memory):>10,}"
        )
        per_beta = self.agent.memory.beta
        # §174: gradient norm diagnostics — avg and max over recent 2000 steps
        gn_arr = np.array(self._recent_grad_norms, dtype=np.float32) if self._recent_grad_norms else np.array([0.0])
        avg_gn  = float(gn_arr.mean())
        max_gn  = float(gn_arr.max())
        clip_warn = "  ⚠ NEAR CLIP" if max_gn > 8.0 else ""
        self.writer.add_scalar("Training/grad_norm_avg", avg_gn, self.total_gradient_steps)
        self.writer.add_scalar("Training/grad_norm_max", max_gn, self.total_gradient_steps)
        self._tlog(
            f"  Epsilon: {self.agent.epsilon:.4f} | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {lr_now:.2e} | "
            f"β: {per_beta:.3f} | "
            f"‖g‖: {avg_gn:.2f} (max {max_gn:.2f}){clip_warn}"
        )
        self._tlog(
            f"  Win rates (last {n:,}): "
            f"Red={red_r:.1%}  Blue={blue_r:.1%}  Draw={draw_r:.1%}"
        )
        # §176: per-category win rates — only shown when worker data is present
        sp_n = len(self._recent_sp)
        ro_n = len(self._recent_ro)
        rv_n = len(self._recent_rv)
        if sp_n > 0:
            sp_red_r  = self._recent_sp.count("red")  / sp_n
            sp_blue_r = self._recent_sp.count("blue") / sp_n
            self._tlog(
                f"  Self-play     (last {sp_n:,}): RED={sp_red_r:.1%}  BLUE={sp_blue_r:.1%}"
            )
            self.writer.add_scalar("Eval/selfplay_red_wr",  sp_red_r,  self.total_gradient_steps)
            self.writer.add_scalar("Eval/selfplay_blue_wr", sp_blue_r, self.total_gradient_steps)
        if ro_n > 0:
            ro_red_r = self._recent_ro.count("red") / ro_n
            self._tlog(
                f"  Random-opp    (last {ro_n:,}): DQN-RED={ro_red_r:.1%}"
            )
            self.writer.add_scalar("Eval/random_opp_red_wr", ro_red_r, self.total_gradient_steps)
        if rv_n > 0:
            rv_blue_r = self._recent_rv.count("blue") / rv_n
            self._tlog(
                f"  Reverse-opp   (last {rv_n:,}): DQN-BLUE={rv_blue_r:.1%}"
            )
            self.writer.add_scalar("Eval/reverse_opp_blue_wr", rv_blue_r, self.total_gradient_steps)
        # §178: game length display — longer games signal harder, more defensive play
        if self._recent_gl:
            avg_gl    = float(np.mean(self._recent_gl))
            avg_gl_sp = float(np.mean(self._recent_gl_sp)) if self._recent_gl_sp else float("nan")
            avg_gl_ro = float(np.mean(self._recent_gl_ro)) if self._recent_gl_ro else float("nan")
            avg_gl_rv = float(np.mean(self._recent_gl_rv)) if self._recent_gl_rv else float("nan")
            parts = [f"Avg game length: {avg_gl:.1f} moves"]
            if np.isfinite(avg_gl_sp): parts.append(f"SP={avg_gl_sp:.1f}")
            if np.isfinite(avg_gl_ro): parts.append(f"RO={avg_gl_ro:.1f}")
            if np.isfinite(avg_gl_rv): parts.append(f"RV={avg_gl_rv:.1f}")
            self._tlog("  " + "  ".join(parts))
            self.writer.add_scalar("Eval/avg_game_length",    avg_gl,    self.total_gradient_steps)
            if np.isfinite(avg_gl_sp):
                self.writer.add_scalar("Eval/avg_game_length_sp", avg_gl_sp, self.total_gradient_steps)
            if np.isfinite(avg_gl_ro):
                self.writer.add_scalar("Eval/avg_game_length_ro", avg_gl_ro, self.total_gradient_steps)
            if np.isfinite(avg_gl_rv):
                self.writer.add_scalar("Eval/avg_game_length_rv", avg_gl_rv, self.total_gradient_steps)
        self._tlog(
            f"  Speed: {eps_per_sec:.0f} ep/s | "
            f"Collect: {c_ms:.0f}ms | Train: {t_ms:.0f}ms | "
            f"GPU mem: {gpu_mem_gb:.2f} GB",
            flush=True,
        )

        self.monitor.update(
            total_episodes=self.total_episodes,
            gradient_steps=self.total_gradient_steps,
            buffer_size=len(self.agent.memory),
            epsilon=self.agent.epsilon,
            avg_loss=avg_loss if np.isfinite(avg_loss) else None,
            red_win_rate=red_r,
            blue_win_rate=blue_r,
            draw_rate=draw_r,
            eps_per_second=eps_per_sec,
            gpu_memory_gb=gpu_mem_gb,
            elapsed_hours=elapsed / 3600,
            remaining_hours=remaining / 3600,
            status="running",
        )

        self.writer.add_scalar("Eval/red_win_rate", red_r, self.total_gradient_steps)
        self.writer.add_scalar("Eval/blue_win_rate", blue_r, self.total_gradient_steps)
        self.writer.add_scalar("Eval/draw_rate", draw_r, self.total_gradient_steps)
        self.writer.add_scalar("Training/per_beta", per_beta, self.total_gradient_steps)  # §172

        # §148/§169/§173: evaluate vs random/heuristic every 5 log intervals
        self._log_call_count += 1
        if self._log_call_count % 5 == 1:  # first call, then every 5th
            # §148/§173: evaluate DQN as both RED and BLUE vs random
            vs = self._eval_vs_random(n_games=100)
            self._vs_random_dqn_red_wr  = vs["dqn_red_wr"]
            self._vs_random_dqn_blue_wr = vs["dqn_blue_wr"]
            self.writer.add_scalar("Eval/vs_random_dqn_red_wr",  vs["dqn_red_wr"],  self.total_gradient_steps)
            self.writer.add_scalar("Eval/vs_random_dqn_blue_wr", vs["dqn_blue_wr"], self.total_gradient_steps)

            # §169: vs-heuristic eval — meaningful signal, unlike vs-random (saturates at ~100%)
            vh = self._eval_vs_heuristic(n_games=50)
            self._vs_heuristic_dqn_red_wr  = vh["dqn_red_wr"]
            self._vs_heuristic_dqn_blue_wr = vh["dqn_blue_wr"]
            self.writer.add_scalar("Eval/vs_heuristic_dqn_red_wr",  vh["dqn_red_wr"],  self.total_gradient_steps)
            self.writer.add_scalar("Eval/vs_heuristic_dqn_blue_wr", vh["dqn_blue_wr"], self.total_gradient_steps)

            # §173: save best.pt when combined heuristic win rate improves
            combined_wr = (vh["dqn_red_wr"] + vh["dqn_blue_wr"]) / 2.0
            self.writer.add_scalar("Eval/combined_heuristic_wr", combined_wr, self.total_gradient_steps)
            if combined_wr > self._best_combined_heuristic_wr:
                self._best_combined_heuristic_wr = combined_wr
                self._save("best")
                self._tlog(f"  ★ New best checkpoint: combined_heuristic_wr={combined_wr:.1%}")

            # §338: vs-bullseye eval — measures whether DQN learns to handle bullseye threats
            if self.use_bullseye_heuristic:
                vb = self._eval_vs_bullseye(n_games=20)
                self._vs_bullseye_dqn_red_wr  = vb["dqn_red_wr"]
                self._vs_bullseye_dqn_blue_wr = vb["dqn_blue_wr"]
                self.writer.add_scalar("Eval/vs_bullseye_dqn_red_wr",  vb["dqn_red_wr"],  self.total_gradient_steps)
                self.writer.add_scalar("Eval/vs_bullseye_dqn_blue_wr", vb["dqn_blue_wr"], self.total_gradient_steps)

            # §347: vs-optimal eval — measures whether DQN-BLUE learns to counter anti-diagonal
            if self.use_optimal_heuristic:
                vo = self._eval_vs_optimal(n_games=20)
                self._vs_optimal_dqn_red_wr  = vo["dqn_red_wr"]
                self._vs_optimal_dqn_blue_wr = vo["dqn_blue_wr"]
                self.writer.add_scalar("Eval/vs_optimal_dqn_red_wr",  vo["dqn_red_wr"],  self.total_gradient_steps)
                self.writer.add_scalar("Eval/vs_optimal_dqn_blue_wr", vo["dqn_blue_wr"], self.total_gradient_steps)
                if vo["dqn_blue_wr"] > self._best_optimal_blue_wr:
                    self._best_optimal_blue_wr = vo["dqn_blue_wr"]
                    self._save("best_vs_optimal")
                    self._tlog(f"  ★ New best_vs_optimal.pt: DQN-BLUE={vo['dqn_blue_wr']:.1%} vs OptimalAgent-RED")

        if not np.isnan(self._vs_random_dqn_red_wr):
            self._tlog(
                f"  vs Random     (100g): "
                f"DQN-RED={self._vs_random_dqn_red_wr:.1%}  DQN-BLUE={self._vs_random_dqn_blue_wr:.1%}"
            )
        if not np.isnan(self._vs_heuristic_dqn_red_wr):
            self._tlog(
                f"  vs Heuristic  ( 50g): "
                f"DQN-RED={self._vs_heuristic_dqn_red_wr:.1%}  DQN-BLUE={self._vs_heuristic_dqn_blue_wr:.1%}"
            )
        if not np.isnan(self._vs_bullseye_dqn_red_wr):
            self._tlog(
                f"  vs Bullseye   ( 20g): "
                f"DQN-RED={self._vs_bullseye_dqn_red_wr:.1%}  DQN-BLUE={self._vs_bullseye_dqn_blue_wr:.1%}"
            )
        if not np.isnan(self._vs_optimal_dqn_red_wr):
            self._tlog(
                f"  vs Optimal    ( 20g): "
                f"DQN-RED={self._vs_optimal_dqn_red_wr:.1%}  DQN-BLUE={self._vs_optimal_dqn_blue_wr:.1%}"
            )

    def _eval_vs_random(self, n_games: int = 100) -> dict:
        """§148/§173: Greedy DQN vs random for both color roles.

        DQN-as-RED: existing §148 metric — use collect_episodes_vectorized fast path.
        DQN-as-BLUE: §173 new — diagnoses color asymmetry; uses same batched forward
        pass pattern as _eval_vs_heuristic.  Typical runtime: ~200ms for n_games=100.
        """
        saved_eps = self.agent.epsilon
        self.agent.epsilon = 0.0

        # DQN as RED vs random BLUE (original §148 fast path)
        _, stats = collect_episodes_vectorized(
            self.agent, n_games,
            random_opponent_frac=1.0,
            mc_returns=False,
            gamma=self.agent.gamma,
        )
        dqn_red_wr = stats["red_wins"] / stats["n_games"]

        # §173: DQN as BLUE vs random RED — batched forward pass
        games = [TicTacPro() for _ in range(n_games)]
        done  = np.zeros(n_games, dtype=bool)
        _BLUE = Player.BLUE
        _RED  = Player.RED

        while not done.all():
            active   = [i for i in range(n_games) if not done[i]]
            dqn_ids  = [i for i in active if not games[i].game_over
                        and games[i].current_player == _BLUE]
            rand_ids = [i for i in active if not games[i].game_over
                        and games[i].current_player == _RED]

            if dqn_ids:
                actions = self.agent.get_actions_batch([games[i] for i in dqn_ids])
                for j, i in enumerate(dqn_ids):
                    act = actions[j]
                    if act is None or games[i].game_over:  # pragma: no cover
                        done[i] = True
                        continue
                    games[i].make_move(*act)
                    if games[i].game_over:
                        done[i] = True

            for i in rand_ids:
                g = games[i]
                if g.game_over:  # pragma: no cover
                    done[i] = True
                    continue
                moves = g.get_legal_moves()
                if not moves:  # pragma: no cover — TicTacPro always has legal moves
                    done[i] = True
                    continue
                g.make_move(*random.choice(moves))
                if g.game_over:
                    done[i] = True

            for i in active:
                if games[i].game_over:
                    done[i] = True

        dqn_blue_wr = sum(1 for g in games if g.winner == _BLUE) / n_games

        self.agent.epsilon = saved_eps
        return {"dqn_red_wr": dqn_red_wr, "dqn_blue_wr": dqn_blue_wr}

    def _eval_vs_heuristic(self, n_games: int = 50) -> dict:
        """§169: Batched greedy DQN vs pick_rollout_move opponent.

        Runs two scenarios back-to-back:
          1. DQN plays RED, heuristic plays BLUE  → dqn_red_wr
          2. DQN plays BLUE, heuristic plays RED  → dqn_blue_wr

        Uses the same batched forward-pass pattern as collect_episodes_vectorized
        so GPU utilisation stays high: ~2×avg_moves forward passes of size n_games
        rather than n_games×avg_moves single-item passes.
        """
        saved_eps = self.agent.epsilon
        self.agent.epsilon = 0.0
        dqn_red_wr = dqn_blue_wr = float("nan")

        for dqn_is_red in (True, False):
            dqn_color  = Player.RED  if dqn_is_red else Player.BLUE
            heur_color = Player.BLUE if dqn_is_red else Player.RED

            games = [TicTacPro() for _ in range(n_games)]
            done  = np.zeros(n_games, dtype=bool)

            while not done.all():
                active = [i for i in range(n_games) if not done[i]]
                dqn_ids  = [i for i in active if games[i].current_player == dqn_color  and not games[i].game_over]
                heur_ids = [i for i in active if games[i].current_player == heur_color and not games[i].game_over]

                if dqn_ids:
                    actions = self.agent.get_actions_batch([games[i] for i in dqn_ids])
                    for j, i in enumerate(dqn_ids):
                        act = actions[j]
                        if act is None or games[i].game_over:  # pragma: no cover
                            done[i] = True
                            continue
                        games[i].make_move(*act)
                        if games[i].game_over:  # pragma: no cover
                            done[i] = True

                # pick_rollout_move is 0.027ms/game — sequential is fine
                for i in heur_ids:
                    g = games[i]
                    if g.game_over:  # pragma: no cover
                        done[i] = True
                        continue
                    act, _ = pick_rollout_move(g)
                    if act is None:  # pragma: no cover — pick_rollout_move always returns a move
                        done[i] = True
                        continue
                    g.make_move(*act)
                    if g.game_over:
                        done[i] = True

                # Mark any remaining active but game_over games done
                for i in active:
                    if games[i].game_over:
                        done[i] = True

            win_counts = {Player.RED: 0, Player.BLUE: 0, Player.NONE: 0}
            for g in games:
                win_counts[g.winner] += 1

            if dqn_is_red:
                dqn_red_wr  = win_counts[Player.RED]  / n_games
            else:
                dqn_blue_wr = win_counts[Player.BLUE] / n_games

        self.agent.epsilon = saved_eps
        return {"dqn_red_wr": dqn_red_wr, "dqn_blue_wr": dqn_blue_wr}

    def _eval_vs_bullseye(self, n_games: int = 20) -> dict:
        """§338: Greedy DQN vs BullseyeAgent for both color roles.

        Measures whether DQN training against BullseyeAgent (use_bullseye_heuristic)
        is teaching the DQN to recognize and block bullseye threats.  Sequential
        (not batched) since BullseyeAgent is a Python loop and n_games is small.
        """
        from game.bullseye_agent import BullseyeAgent as _BullseyeAgent
        bsa = _BullseyeAgent()
        saved_eps = self.agent.epsilon
        self.agent.epsilon = 0.0
        dqn_red_wr = dqn_blue_wr = float("nan")

        for dqn_is_red in (True, False):
            dqn_color  = Player.RED  if dqn_is_red else Player.BLUE
            bsa_color  = Player.BLUE if dqn_is_red else Player.RED
            wins = 0
            for _ in range(n_games):
                game = TicTacPro()
                for _step in range(100):
                    if game.game_over:
                        break
                    if game.current_player == dqn_color:
                        acts = self.agent.get_actions_batch([game])
                        act  = acts[0]
                    else:
                        act = bsa.get_action(game, bsa_color)
                    if act is None:
                        break
                    game.make_move(*act)
                if game.winner == dqn_color:
                    wins += 1
            if dqn_is_red:
                dqn_red_wr  = wins / n_games
            else:
                dqn_blue_wr = wins / n_games

        self.agent.epsilon = saved_eps
        return {"dqn_red_wr": dqn_red_wr, "dqn_blue_wr": dqn_blue_wr}

    def _eval_vs_optimal(self, n_games: int = 20) -> dict:
        """§347: Greedy DQN vs OptimalAgent for both color roles.

        Measures whether training against OptimalAgent teaches DQN-BLUE to
        counter the anti-diagonal Small strategy. Sequential (not batched) since
        OptimalAgent is a fast Python rule-based agent and n_games is small.
        """
        from game.optimal_agent import OptimalAgent as _OptimalAgent
        oa = _OptimalAgent()
        saved_eps = self.agent.epsilon
        self.agent.epsilon = 0.0
        dqn_red_wr = dqn_blue_wr = float("nan")

        for dqn_is_red in (True, False):
            dqn_color = Player.RED  if dqn_is_red else Player.BLUE
            oa_color  = Player.BLUE if dqn_is_red else Player.RED
            wins = 0
            for _ in range(n_games):
                game = TicTacPro()
                for _step in range(100):
                    if game.game_over:
                        break
                    if game.current_player == dqn_color:
                        acts = self.agent.get_actions_batch([game])
                        act  = acts[0]
                    else:
                        act = oa.get_action(game, oa_color)
                    if act is None:
                        break
                    game.make_move(*act)
                if game.winner == dqn_color:
                    wins += 1
            if dqn_is_red:
                dqn_red_wr  = wins / n_games
            else:
                dqn_blue_wr = wins / n_games

        self.agent.epsilon = saved_eps
        return {"dqn_red_wr": dqn_red_wr, "dqn_blue_wr": dqn_blue_wr}

    def _save(self, tag: str):
        path = os.path.join(self.checkpoint_dir, f"{tag}.pt")
        self.agent.save(path)
        print(f"  Checkpoint saved: {path}")

    def evaluate(self, n_games: int = 500) -> Dict:
        """Greedy policy evaluation. Returns win-rate dict."""
        saved_eps = self.agent.epsilon
        self.agent.epsilon = 0.0
        agg = {"red_wins": 0, "blue_wins": 0, "draws": 0, "total": 0, "move_counts": []}
        remaining = n_games
        while remaining > 0:
            batch = min(self.n_envs, remaining)
            _, stats = collect_episodes_vectorized(self.agent, batch)
            agg["red_wins"] += stats["red_wins"]
            agg["blue_wins"] += stats["blue_wins"]
            agg["draws"] += stats["draws"]
            agg["total"] += stats["n_games"]
            agg["move_counts"].append(stats["avg_moves"])
            remaining -= batch
        self.agent.epsilon = saved_eps
        t = agg["total"]
        return {
            "red_win_rate": agg["red_wins"] / t,
            "blue_win_rate": agg["blue_wins"] / t,
            "draw_rate": agg["draws"] / t,
            "avg_moves": float(np.mean(agg["move_counts"])),
        }
