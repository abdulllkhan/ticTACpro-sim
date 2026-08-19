"""
Parallel MCTS agent for TicTacPro — root parallelization.

Spawns N independent MCTSAgent workers (one per CPU core) and aggregates
their visit counts to pick the best move.  Each worker sees the same root
position but explores independently, so the effective simulation budget
scales almost linearly with the number of workers.

Speedup profile (GB10 DGX Spark, 20 cores, after §59 critical fix, under training load):
  1 worker  ~   96,128 sims/2s  (depth-6 tree, real MCTS)
  4 workers ~  821,952 sims/2s  (~8.5× single; fresh subprocesses bypass GIL)
  16 workers ~ 2,520,832 sims/2s (~26× single; ~6.5× vs 4 workers)

Design notes:
  - Workers are pre-spawned once; reused across get_action() calls.
  - Game state is pickled (5.7µs) and sent to each worker; Pool.map()
    overhead is ~0.5ms/call regardless of N workers.
  - No tree reuse across moves (each worker starts fresh per move).
    The ~16× simulation gain more than compensates.
  - Module-level _worker_run is required for multiprocessing pickle.
"""

import multiprocessing as mp
import random
from typing import Optional, Tuple

from game.tictacpro import TicTacPro, Player, _would_win_bf, _WIN_LINES, _CELL_SLOT, _SIZES_IDX, _CELLS_IDX
from rl.mcts_agent import MCTSAgent

# ── Worker function (module-level for pickle) ─────────────────────────────────

def _worker_run(args):
    """Run a single MCTS search in a worker process.

    Returns dict of {move: visit_count} for the root's children.
    """
    g, player_int, time_limit, max_sims, seed = args
    player = Player(player_int)
    # §179: seed Python random so workers diverge even when C rollout is deterministic.
    # randomize_expansion shuffles non-winning expansion order so each worker explores
    # a different subset of the game tree rather than all N workers building one tree.
    if seed is not None:
        random.seed(seed)
    agent = MCTSAgent(time_limit=time_limit, max_simulations=max_sims,
                      randomize_expansion=(seed is not None))
    root = agent._build_tree(g, player)
    return {child.move: child.visits for child in root.children}


# ── Parallel MCTS Agent ───────────────────────────────────────────────────────

class ParallelMCTSAgent:
    """
    Root-parallelized MCTS: N independent trees, aggregated vote.

    Args:
        n_workers:       Number of parallel worker processes.
                         Default: min(16, cpu_count - 2).
        time_limit:      Wall-clock seconds per move.
        max_simulations: Per-worker hard cap on simulations.
    """

    def __init__(self,
                 n_workers: int = None,
                 time_limit: float = 2.0,
                 max_simulations: int = 10_000_000):
        self.n_workers       = n_workers or min(16, max(1, mp.cpu_count() - 2))
        self.time_limit      = time_limit
        self.max_simulations = max_simulations
        self._last_sims      = 0
        # Pre-spawn workers — survives across get_action() calls.
        # Use spawn context to avoid forking a CUDA context (if CUDA was initialized
        # before this Pool is created, fork would copy the context into workers that
        # never use it — spawn starts clean processes with no CUDA state).
        self._pool = mp.get_context("spawn").Pool(self.n_workers)

    def __del__(self):
        try:
            self._pool.terminate()
        except Exception:
            pass

    def reset(self):
        """No-op for API compatibility (no tree reuse in parallel mode)."""
        pass

    # ── Public API ────────────────────────────────────────────────────────────

    def get_action(self, game: TicTacPro, player: Player = None,
                   epsilon: float = 0.0) -> Optional[Tuple]:
        if game.game_over:
            return None
        if player is None:
            player = game.current_player

        # Fast path: immediate winning move or forced block.
        # §333/§346: dual-track bull_block vs line_block so bullseye threats
        # take priority over same-size line threats, matching OptimalAgent.
        p_int      = int(player)
        opp_int    = 3 - p_int
        opp_player = Player.BLUE if player == Player.RED else Player.RED
        opp_pr     = game.pieces_remaining[opp_player]
        pr         = game.pieces_remaining[player]
        bf         = game.board.tobytes()
        bull_block_mv = None   # §346: bullseye block (i1//3==i2//3) — highest priority
        line_block_mv = None   # §346: standard same-size line block
        for sz_idx, sz in _SIZES_IDX:
            if pr[sz] == 0:
                continue
            csl_sz     = _CELL_SLOT[sz_idx]
            wl_sz      = _WIN_LINES[sz_idx]
            has_opp_sz = opp_pr[sz] > 0
            for ci, (r, c) in _CELLS_IDX:
                if bf[csl_sz[ci]]:
                    continue
                for i1, i2 in wl_sz[ci]:
                    v1 = bf[i1]
                    if v1 == p_int and bf[i2] == p_int:
                        self._last_sims = 0
                        return (r, c, sz)
                    if has_opp_sz and v1 == opp_int and bf[i2] == opp_int:
                        if i1 // 3 == i2 // 3:      # same cell → bullseye threat
                            if bull_block_mv is None:
                                bull_block_mv = (r, c, sz)
                        elif line_block_mv is None:
                            line_block_mv = (r, c, sz)
        block_mv = bull_block_mv if bull_block_mv is not None else line_block_mv
        if block_mv is not None:
            self._last_sims = 0
            return block_mv

        # Send shallow clone (no history — saves pickle bytes)
        g = game.clone(copy_history=False)
        # §179: unique seed per worker so randomize_expansion produces different
        # expansion orderings across the pool — each worker explores a distinct
        # subset of non-forced branches instead of all N building identical trees.
        base_seed = random.randint(0, (1 << 32) - 1)
        args = [(g, int(player), self.time_limit, self.max_simulations,
                 base_seed + i)
                for i in range(self.n_workers)]
        results = self._pool.map(_worker_run, args)

        # Aggregate visit counts across workers
        total = {}
        for move_visits in results:
            for move, v in move_visits.items():
                total[move] = total.get(move, 0) + v

        if not total:
            moves = game.get_legal_moves()
            return moves[0] if moves else None

        self._last_sims = sum(total.values())
        return max(total, key=total.get)

    def get_info(self, game: TicTacPro, player: Player = None):
        """Returns (action, (total_simulations, win_rate_estimate))."""
        action = self.get_action(game, player)
        return action, (self._last_sims, 0.0)
