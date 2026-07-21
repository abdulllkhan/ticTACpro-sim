"""
Neural MCTS agent for TicTacPro.

Architecture (v6 — sorted-once expansion with deferred DQN inference):

  Selection   — UCT as usual
  Expansion   — sorted-once: on FIRST visit to a node, sort ALL untried moves in
                  priority order and cache. Subsequent expansions just pop the front.
                  Priority order:
                    1. Immediate winning move (guaranteed correct)
                    2. Blocking opponent's immediate win (guaranteed correct)
                    3. DQN Q-value ordering using perspective-normalised tensor
  Evaluation  — win-then-block greedy rollout (same as MCTSAgent)
  Backprop    — standard win/visit counts

Rationale for sorted-once expansion (v5 → v6):

  v5 called DQN on the SAME parent state N times when expanding N untried moves.
  v6 sorts all untried_moves ONCE on the first visit (one DQN call per node) and
  pre-sorts newly created child nodes immediately at creation time.  This reduces
  DQN calls from O(N) to O(1) per node, yielding ~3-10× more simulations per
  second without changing the search semantics.

Rationale for win-detection in expansion (v4 → v5):

  Previously, winning/blocking moves might go unexplored for many simulations
  while other tree branches accumulated visits.  Adding explicit detection
  ensures critical moves are expanded FIRST, guaranteeing the tree never
  misses a forced win or blocks a forced loss.

Rationale for the perspective normalisation (v3 → v4):

  The DQN was trained with 100% RED self-play wins, so raw Q-values for BLUE
  positions give no useful ordering signal.  get_state_tensor_normalized()
  always maps the current player's pieces to slots 0-26 (as if it were RED
  to move), giving reliable relative Q-ordering for both sides.
"""

import math
import time
import random
import operator
import numpy as np
import torch
from typing import Optional, Tuple

from game.tictacpro import (TicTacPro, Player, PieceSize, would_win, pick_rollout_move,
                             _WIN_LINES, _SWITCH_PLAYER)
from rl.mcts_agent import (_push_undo, _push_known_undo, _pop_undo,
                            _apply_move, _LAZY, _INV_SQRT, _SQRT_LOG, _INV, _get_visits)
from game.rollout_c_wrapper import (fast_rollout_c, get_legal_moves_fast,
                                     get_sorted_legal_ex_c, C_ROLLOUT_AVAILABLE)

PIECES_PER_SIZE = 3
UCT_C = math.sqrt(2)


# ── Tree node ─────────────────────────────────────────────────────────────────
class NeuralMCTSNode:
    __slots__ = ("move", "move_fidx", "move_sz", "parent", "children",
                 "wins", "visits", "winrate", "uct_scale", "untried_moves",
                 "player_who_moved", "untried_sorted", "move_is_win", "_n_non_winning",
                 "ch_wr", "ch_us", "ch_n", "child_idx")

    def __init__(self, move=None, parent=None, untried_moves=None,
                 player_who_moved=None, untried_sorted=False, move_is_win=False,
                 n_non_winning=-1, move_fidx: int = 0, move_sz: int = 0):
        self.move             = move
        self.move_fidx        = move_fidx       # §53: r*9+c*3+sz-1 pre-computed flat board index
        self.move_sz          = move_sz
        self.parent           = parent
        self.children         = []
        self.wins             = 0.0
        self.visits           = 0
        self.winrate          = 0.0
        self.uct_scale        = 0.0
        self.untried_moves    = untried_moves or []
        self.player_who_moved = player_who_moved
        self.untried_sorted   = untried_sorted
        self.move_is_win      = move_is_win
        self._n_non_winning   = n_non_winning
        self.ch_wr     = None  # §57: float64 array of child winrates (lazy alloc on first child)
        self.ch_us     = None  # §57: float64 array of child uct_scales
        self.ch_n      = 0     # §57: count of expanded children (== len(children))
        self.child_idx = 0     # §57: this node's index in parent's ch_wr/ch_us arrays

    def uct_select(self) -> "NeuralMCTSNode":
        explore_base = UCT_C * _SQRT_LOG[self.visits if self.visits <= 65536 else 65536]
        n = self.ch_n
        if n > 0 and self.ch_wr is not None:
            # §57: numpy vectorized UCT — 874ns vs Python loop 1851ns for n=27
            return self.children[(self.ch_wr[:n] + explore_base * self.ch_us[:n]).argmax()]
        best_score = -1.0
        best_child = self.children[0]
        for c in self.children:
            score = c.winrate + explore_base * c.uct_scale
            if score > best_score:
                best_score = score
                best_child = c
        return best_child

    def best_child(self) -> "NeuralMCTSNode":
        return max(self.children, key=_get_visits)


# ── Agent ─────────────────────────────────────────────────────────────────────
class NeuralMCTSAgent:
    """
    MCTS agent with DQN-guided expansion ordering and win-greedy rollout evaluation.

    Args:
        policy_net:       Trained GPUDQNNetwork (already on device, in eval mode).
        device:           torch.device for inference.
        time_limit:       Wall-clock seconds per move.
        max_simulations:  Hard cap on simulations per move.
        use_amp:          Use BF16 AMP for inference (matches training regime).
        amp_dtype:        AMP dtype (bfloat16 for Blackwell, float16 otherwise).
        max_dqn_depth:    Depth limit for DQN-guided expansion. Nodes deeper than
                          this use C-sorted legal moves (win/block/rest order without
                          Q-value guidance). Default 1: only root + its children
                          (~28 DQN calls vs 1000+ without the cap). Beyond depth 1,
                          UCT already guides the search to good branches; Q-value
                          ordering adds negligible benefit but consumes ~93% of time.
        use_dqn_eval:     §154: Replace rollout evaluation with DQN max-Q → win
                          probability: raw_p = clamp((max_q+1)/2, 0, 1). Fewer
                          simulations/s than C rollout but potentially more accurate
                          value signal. Default False (use C rollout).
    """

    def __init__(self,
                 policy_net,
                 device,
                 time_limit: float = 2.0,
                 max_simulations: int = 10_000_000,
                 use_amp: bool = True,
                 amp_dtype=torch.bfloat16,
                 max_dqn_depth: int = 1,
                 use_dqn_eval: bool = False):
        self.net              = policy_net
        self.device           = device
        self.time_limit       = time_limit
        self.max_simulations  = max_simulations
        self.use_amp          = use_amp and (device.type == "cuda")
        self.amp_dtype        = amp_dtype
        self.max_dqn_depth    = max_dqn_depth
        self.use_dqn_eval     = use_dqn_eval  # §154: DQN leaf eval instead of rollout
        self._last_sims       = 0
        self._reuse_node: Optional[NeuralMCTSNode] = None

        self.net.eval()

    def reset(self):
        """Discard cached tree (call when starting a new game)."""
        self._reuse_node = None

    # ── Public API ────────────────────────────────────────────────────────────
    def get_action(self, game: TicTacPro, player: Player = None,
                   epsilon: float = 0.0) -> Optional[Tuple]:
        if game.game_over:
            return None
        if player is None:
            player = game.current_player
        # Fast path: immediate win or forced block (§199 — phantom-block guard matches §143).
        # §337: dual-track bullseye vs line blocks (same §333 fix as OptimalAgent) to ensure
        # bullseye blocks are preferred over line blocks when both are present.
        p_int      = int(player)
        opp_int    = 3 - p_int
        opp_player = Player.BLUE if player == Player.RED else Player.RED
        opp_pr     = game.pieces_remaining[opp_player]
        bf         = game.board.tobytes()
        bullseye_block = None
        line_block     = None
        for mv in game.get_legal_moves():
            r, c, sz = mv
            sz_idx = sz - 1
            ci     = r * 3 + c
            opp_has_sz = opp_pr[sz] > 0
            for i1, i2 in _WIN_LINES[sz_idx][ci]:
                v1 = bf[i1]; v2 = bf[i2]
                if v1 == p_int and v2 == p_int:
                    self._reuse_node = None
                    return mv  # immediate win — always take it
                if opp_has_sz and v1 == opp_int and v2 == opp_int:
                    if i1 // 3 == i2 // 3:   # same cell → bullseye threat
                        if bullseye_block is None:
                            bullseye_block = mv
                    elif line_block is None:
                        line_block = mv
        block_mv = bullseye_block if bullseye_block is not None else line_block
        if block_mv is not None:
            self._reuse_node = None
            return block_mv
        g    = game.clone()
        root = self._build_tree(g, player)
        if not root.children:
            moves = game.get_legal_moves()
            self._reuse_node = None
            return moves[0] if moves else None
        best = root.best_child()
        self._reuse_node = best
        return best.move

    def get_info(self, game: TicTacPro, player: Player = None):
        if game.game_over:
            return None, (0, 0.0)
        if player is None:
            player = game.current_player
        p_int      = int(player)
        opp_int    = 3 - p_int
        opp_player = Player.BLUE if player == Player.RED else Player.RED
        opp_pr     = game.pieces_remaining[opp_player]
        bf             = game.board.tobytes()
        bullseye_block = None
        line_block     = None
        for mv in game.get_legal_moves():
            r, c, sz = mv
            sz_idx = sz - 1
            ci     = r * 3 + c
            opp_has_sz = opp_pr[sz] > 0
            for i1, i2 in _WIN_LINES[sz_idx][ci]:
                v1 = bf[i1]; v2 = bf[i2]
                if v1 == p_int and v2 == p_int:
                    self._reuse_node = None
                    return mv, (0, 1.0)
                if opp_has_sz and v1 == opp_int and v2 == opp_int:
                    if i1 // 3 == i2 // 3:
                        if bullseye_block is None:
                            bullseye_block = mv
                    elif line_block is None:
                        line_block = mv
        block_mv = bullseye_block if bullseye_block is not None else line_block
        if block_mv is not None:
            self._reuse_node = None
            return block_mv, (0, 0.5)
        g    = game.clone()
        root = self._build_tree(g, player)
        if not root.children:
            moves = game.get_legal_moves()
            return (moves[0] if moves else None), (0, 0.0)
        best    = root.best_child()
        wr      = best.wins / best.visits if best.visits else 0.0
        return best.move, (self._last_sims, wr)

    # ── Tree building ─────────────────────────────────────────────────────────
    def _build_tree(self, g: TicTacPro, root_player: Player) -> NeuralMCTSNode:
        # Undo-based: push/undo onto g directly — no per-simulation clone.
        root = None
        if self._reuse_node is not None and g.move_history:
            last_move = g.move_history[-1][:3]
            for child in self._reuse_node.children:
                if child.move == last_move:
                    root = child
                    root.parent = None
                    break
        if root is None:
            # §56: positional args — kwarg dict matching adds ~720ns/call overhead
            root = NeuralMCTSNode(None, None, g.get_legal_moves(), None)
        self._reuse_node = None
        deadline = time.perf_counter() + self.time_limit
        sims     = 0
        _check = 64  # batch time checks to reduce perf_counter() syscall overhead
        # Pre-allocated fixed-size undo stack (same rationale as MCTSAgent._build_tree).
        _toks   = [None] * 40
        _toks_d = 0
        # Bind hot module-level names as locals — LOAD_FAST vs LOAD_GLOBAL per iteration
        _perf_counter = time.perf_counter
        _pundo  = _push_undo
        _lazy    = _LAZY
        _inv     = _INV
        _inv_sqrt = _INV_SQRT
        _Player_NONE  = Player.NONE
        _NeuralMCTSNode = NeuralMCTSNode
        _get_legal    = get_legal_moves_fast if C_ROLLOUT_AVAILABLE else None
        _get_sorted_ex = get_sorted_legal_ex_c if C_ROLLOUT_AVAILABLE else None
        _c_rollout    = fast_rollout_c if C_ROLLOUT_AVAILABLE else None
        _max_dqn_depth = self.max_dqn_depth  # §62: cap DQN calls to shallow nodes
        _use_dqn_eval  = self.use_dqn_eval   # §154: DQN leaf evaluation
        _eval_dqn      = self._eval_leaf_dqn if self.use_dqn_eval else None
        # _pcs_left is synced to g once per sim before fast_rollout_c.
        # _winner/_game_over caches eliminate all g.winner/g.game_over LOAD/STORE_ATTR
        # in the hot loops — neither C rollout nor selection/undo code reads g.* for these.
        _pr = g.pieces_remaining
        _prs = [None, _pr[Player.RED], _pr[Player.BLUE]]  # §51: list for O(1) outer lookup
        _pcs_left  = g._pieces_left
        _winner    = g.winner
        _game_over = g.game_over
        _board = g.board.ravel()  # flat view: avoids 3-tuple creation on every setitem
        _UCT_C    = UCT_C
        _sqrt_log = _SQRT_LOG
        _cp_root  = int(g.current_player)  # §49: plain int avoids IntEnum overhead throughout
        _np_zeros = np.zeros  # §57: numpy array allocation for ch_wr/ch_us

        while sims < self.max_simulations:
            if sims % _check == 0 and _perf_counter() >= deadline:
                break
            node = root
            _toks_d = 0
            _cp = _cp_root

            # 1. Selection — inlined + numpy-vectorized UCT.
            # §57: (ch_wr[:n] + eb * ch_us[:n]).argmax() is 2.1× faster than Python
            # for-loop for n=27 (874ns vs 1851ns). ch_wr/ch_us maintained in backprop.
            while not node.untried_moves and node.children:
                _eb  = _UCT_C * _sqrt_log[node.visits if node.visits <= 65536 else 65536]
                # §57: numpy vectorized UCT — dominant bottleneck at root (n=27 children)
                _n   = node.ch_n
                node = node.children[(node.ch_wr[:_n] + _eb * node.ch_us[:_n]).argmax()]
                _sz   = node.move_sz                    # LOAD_ATTR slot: needed for _prs[_pl][_sz]
                _fidx = node.move_fidx                  # §53: pre-computed flat index, no arithmetic
                _pl   = _cp
                _pw = _winner; _pgo = _game_over
                _board[_fidx] = _pl
                _prs[_pl][_sz] -= 1                     # §51: list outer lookup
                _pcs_left -= 1
                if node.move_is_win:
                    _winner = _pl; _game_over = True
                elif _pcs_left == 0:
                    _game_over = True
                else:
                    _cp = 3 - _pl                       # §49: arithmetic vs dict (~10ns faster)
                _toks[_toks_d] = (_fidx, _pl, _sz, _pw, _pgo); _toks_d += 1

            # 2. Expansion — sorted-once: call _sort_moves (DQN) once per node.
            # §62: DQN is only used for nodes at depth ≤ max_dqn_depth; deeper nodes
            # use get_sorted_legal_ex_c (C win/block/rest order, no GPU call). This
            # caps DQN calls at ~28 per move (1 root + 27 children) vs 1000+ without
            # the limit, recovering ~93% of time spent on GPU inference while
            # preserving Q-value guidance where it matters most (first 2 levels).
            # _LAZY sentinel: truthy (stops selection), replaced on first expansion.
            g.current_player = _cp  # sync once for C helpers and _sort_moves
            if node.untried_moves:
                if node.untried_moves is _lazy:
                    if _game_over:
                        node.untried_moves = []
                    elif _toks_d > _max_dqn_depth and _get_sorted_ex:
                        # §62: deep node — use C sort (no DQN), set sorted flag here
                        _dm, _dnw = _get_sorted_ex(g)
                        node.untried_moves  = _dm
                        node._n_non_winning = len(_dm) - _dnw
                        node.untried_sorted = True
                    else:
                        node.untried_moves = _get_legal(g) if _get_legal else g.get_legal_moves()
                if node.untried_moves:
                    if not node.untried_sorted:
                        # §62: shallow node (depth ≤ max_dqn_depth) — use DQN ordering
                        node.untried_moves, nw = self._sort_moves(g, node.untried_moves)
                        node._n_non_winning = len(node.untried_moves) - nw
                        node.untried_sorted = True
                    _nnw = node._n_non_winning
                    is_win = _nnw >= 0 and len(node.untried_moves) > _nnw
                    mv    = node.untried_moves.pop()
                    mover = _cp
                    r, c, sz = mv
                    if _nnw >= 0:
                        # Inline _push_known_undo for expansion
                        _sz1 = sz - 1
                        _pw = _winner; _pgo = _game_over   # LOAD_FAST vs LOAD_ATTR
                        _fidx = r*9 + c*3 + _sz1
                        _board[_fidx] = mover
                        _prs[mover][sz] -= 1                   # §51: list outer lookup
                        _pcs_left -= 1
                        if is_win:
                            _winner = mover; _game_over = True  # cache only
                        elif _pcs_left == 0:
                            _game_over = True
                        else:
                            g.current_player = 3 - mover
                        _toks[_toks_d] = (_fidx, mover, sz, _pw, _pgo); _toks_d += 1
                    else:  # pragma: no cover — _n_non_winning is always ≥0 after sort (lines 321,329)
                        g._pieces_left = _pcs_left    # §59: prevent drift (see mcts_agent.py)
                        g.game_over    = _game_over   # §59: prevent stale True capture
                        _toks[_toks_d] = _pundo(g, r, c, sz); _toks_d += 1
                        _pcs_left  = g._pieces_left   # re-sync: _pundo decremented g.*
                        _winner    = g.winner
                        _game_over = g.game_over
                        is_win = _game_over and _winner != _Player_NONE
                        _fidx = r*9 + c*3 + sz - 1  # §53: pre-compute for move_fidx
                    # §57: maintain parent's numpy arrays for vectorized UCT selection.
                    # Allocate on first child (lazy): size = remaining + 1 = total possible.
                    _cidx = node.ch_n
                    if node.ch_wr is None:
                        _nmax = _cidx + len(node.untried_moves) + 1
                        node.ch_wr = _np_zeros(_nmax)
                        node.ch_us = _np_zeros(_nmax)
                    node.ch_n = _cidx + 1
                    # §56: positional args — kwarg dict matching adds ~720ns/call overhead
                    # order: move, parent, untried_moves, player_who_moved, untried_sorted,
                    #        move_is_win, n_non_winning, move_fidx, move_sz
                    child = _NeuralMCTSNode(mv, node, _lazy, mover, False, is_win, -1, _fidx, sz)
                    child.child_idx = _cidx
                    node.children.append(child)
                    node = child

            # 3. Evaluation — sync _pcs_left once (fast_rollout_c reads g._pieces_left).
            # g.winner/g.game_over stay stale; use local caches throughout.
            g._pieces_left = _pcs_left
            if _game_over:
                _w = _winner
                result = 1.0 if _w == root_player else (0.5 if _w == _Player_NONE else 0.0)
            elif _use_dqn_eval:
                # §154: DQN leaf evaluation — convert max Q-value to win probability.
                # g.current_player is already set to next-to-move by expansion.
                result = _eval_dqn(g, root_player)
            elif _c_rollout is not None:
                _w = _c_rollout(g)
                result = 1.0 if _w == root_player else (0.5 if _w == _Player_NONE else 0.0)
            else:
                result = self._rollout(g, root_player)

            # 4. Backpropagation — alternating result avoids per-step player comparison.
            result_opp = 1.0 - result
            if node.player_who_moved == root_player:
                _ra, _rb = result, result_opp
            else:
                _ra, _rb = result_opp, result
            while node is not None:
                v = node.visits + 1
                node.visits = v
                # §55: local w avoids second LOAD_ATTR for node.wins in winrate computation
                w = node.wins + _ra
                node.wins  = w
                v_cap = v if v <= 65536 else 65536
                wr = w * _inv[v_cap]
                us = _inv_sqrt[v_cap]
                node.winrate   = wr
                node.uct_scale = us
                _par = node.parent
                if _par is not None:
                    # §57: keep parent's numpy arrays in sync for vectorized UCT
                    _par.ch_wr[node.child_idx] = wr
                    _par.ch_us[node.child_idx] = us
                node = _par
                _ra, _rb = _rb, _ra

            # 5. Undo: depth-counter walk avoids reversed() iterator allocation per sim.
            # Token layout: (flat_idx, player, sz, prev_winner, prev_game_over)
            # Restore _winner/_game_over caches; g.winner/g.game_over stay stale (never read).
            while _toks_d > 0:
                _toks_d -= 1
                # §54: UNPACK_SEQUENCE replaces 5 separate tuple subscripts (~10ns/step saved)
                _tf, _tp, _ts, _winner, _game_over = _toks[_toks_d]
                _board[_tf] = 0
                _prs[_tp][_ts] += 1                     # §51: list outer lookup
                _pcs_left += 1
                # §50: g.current_player NOT restored here (next sim sets it via g.current_player=_cp)

            sims += 1

        self._last_sims = sims
        return root

    # ── §154 / §181: DQN leaf evaluation ─────────────────────────────────────
    def _eval_leaf_dqn(self, g: TicTacPro, root_player: Player) -> float:
        """
        §207: Evaluate a leaf node using softmax-weighted average Q over legal moves.

        §154 used max Q-value, which was overconfident (MCTS saw most positions as
        near-certain wins, starving exploration).  Tournament v19 confirmed this:
        neural_dqn_eval lost 20-0 to plain random-rollout MCTS.

        Fix: softmax-weighted average over legal Q-values with temperature=1.0.
        Average is far less optimistic than max and better approximates the expected
        return under a mixed policy — closer to what rollouts actually measure.

        §181: mask to LEGAL actions only. Illegal slots have untrained Q-values.
        """
        try:
            sv = torch.from_numpy(
                g.get_state_tensor_normalized()).unsqueeze(0).to(
                self.device, non_blocking=True)
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                        q_raw = self.net(sv)
                    q_raw = q_raw.float()
                else:
                    q_raw = self.net(sv)
            # §181 / §207: restrict to legal actions.
            legal_moves = g.get_legal_moves()
            q_vals = q_raw[0]
            if legal_moves:
                legal_idx = [r * 9 + c * 3 + (sz - 1) for r, c, sz in legal_moves]
                q_legal = q_vals[legal_idx]
                # Softmax-weighted average — less optimistic than max, reduces
                # overconfident leaf values that starved MCTS exploration.
                weights = torch.softmax(q_legal, dim=0)
                avg_q = float((weights * q_legal).sum())
            else:
                avg_q = float(q_vals.mean())
            raw_p = max(0.0, min(1.0, (avg_q + 1.0) / 2.0))
            return raw_p if g.current_player == root_player else 1.0 - raw_p
        except Exception:
            return 0.5

    # ── DQN-guided move sorting ───────────────────────────────────────────────
    def _sort_moves(self, state: TicTacPro, moves: list) -> list:
        """
        Return moves in reversed-priority order for pop() access.
        §341: layout [rest, line_blocking, bullseye_blocking, winning] — pop()
        returns winning first, then bullseye blocks, then line blocks, then rest.
        DQN Q-values sort the 'rest' bucket; blocking/winning are exact detections.
        Called once per node (lazy on first expansion); O(1) pops thereafter.
        """
        if len(moves) <= 1:
            return list(moves), 0

        bf     = state.board.tobytes()
        cur    = int(state.current_player)
        opp    = 3 - cur
        opp_pr = state.pieces_remaining[_SWITCH_PLAYER[state.current_player]]

        winning      = []
        bull_blocking = []   # §341: bullseye threats — same cell i1//3==i2//3
        line_blocking = []   # §341: standard 3-in-a-row threats
        rest         = []
        for m in moves:
            sz_idx     = m[2] - 1    # PieceSize is IntEnum — avoids int() cast overhead
            sz         = m[2]
            ci         = m[0] * 3 + m[1]
            wl         = _WIN_LINES[sz_idx][ci]
            is_win     = False
            is_bull    = False
            is_line    = False
            opp_has_sz = opp_pr[sz] > 0
            for i1, i2 in wl:
                v1 = bf[i1]
                if v1 == cur:
                    if bf[i2] == cur:
                        is_win = True; break
                elif v1 == opp and bf[i2] == opp and opp_has_sz:
                    if i1 // 3 == i2 // 3:   # same cell → bullseye threat
                        is_bull = True
                    else:
                        is_line = True
            if is_win:
                winning.append(m)
            elif is_bull:
                bull_blocking.append(m)
            elif is_line:
                line_blocking.append(m)
            else:
                rest.append(m)

        if rest:
            sv = torch.from_numpy(state.get_state_tensor_normalized()).unsqueeze(0).to(
                    self.device, non_blocking=True)
            try:
                with torch.no_grad():
                    if self.use_amp:
                        with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                            features = self.net.features(sv)
                            v_raw    = self.net.value_head(features)
                            adv      = self.net.advantage_head(features)
                        v_raw = v_raw.float()
                        adv   = adv.float()
                    else:
                        features = self.net.features(sv)
                        v_raw    = self.net.value_head(features)
                        adv      = self.net.advantage_head(features)
                    q_np = (v_raw + adv - adv.mean(dim=-1, keepdim=True))[0].cpu().numpy()
                # Sort ascending so pop() (from end) returns highest-Q first.
                rest.sort(key=lambda m: q_np[m[0] * 9 + m[1] * 3 + m[2] - 1])
            except Exception:
                pass  # keep rest in original order on failure

        # Layout: [rest, line_blocking, bullseye_blocking, winning] — pop() → winning first.
        return rest + line_blocking + bull_blocking + winning, len(winning)

    # ── Win-then-block greedy rollout (clone-based) ──────────────────────────
    def _rollout(self, g: TicTacPro, root_player: Player) -> float:
        """
        Rollout using a lightweight clone (~0.7µs) to avoid undo-token overhead.
        Returns 1.0 if root_player wins, 0.5 for draw, 0.0 for loss.

        Only called when C_ROLLOUT_AVAILABLE is False.  g.game_over / g.winner
        may be stale (§59: local caches are authoritative in the simulation loop).
        Callers guarantee _game_over is False when this is invoked, so we do NOT
        read g.game_over here — instead we clone and let the clone's own tracking
        drive the rollout.  The clone must be told game is active because
        g.game_over may read True from an earlier simulation round.
        """
        gc = g.clone()
        gc.game_over = False   # g.game_over may be stale; caller guarantees game is active
        gc.winner    = Player.NONE
        _prm = pick_rollout_move  # LOAD_FAST > LOAD_GLOBAL in tight loop (~1.5×)
        _am  = _apply_move
        while not gc.game_over:
            mv, is_win = _prm(gc)
            if mv is None:
                break
            r, c, sz = mv
            _am(gc, r, c, sz, is_win)
        return 1.0 if gc.winner == root_player else (0.5 if gc.winner == Player.NONE else 0.0)
