"""
Monte Carlo Tree Search agent for TicTacPro.

Uses UCT (Upper Confidence bounds for Trees) with:
- Configurable simulation budget (by time or count)
- Win-then-block priority in both EXPANSION and ROLLOUT (no-rollout_agent mode):
    expansion: winning/blocking moves expanded first (not random), so critical
               lines are explored before time expires
    rollout:   (1) take immediate win, (2) block opponent win, (3) random
- Optional RL Q-value guidance for rollout policy (bypasses win detection)
"""

import math
import time
import random
import operator
import numpy as np
from typing import Optional, Tuple

from game.tictacpro import (TicTacPro, Player, PieceSize, would_win, pick_rollout_move,
                             _would_win_bf, _WIN_LINES, _CELL_SLOT, _CELLS, _SWITCH_PLAYER)
from game.rollout_c_wrapper import (fast_rollout_c, get_legal_moves_fast,
                                     get_sorted_legal_c, get_sorted_legal_ex_c,
                                     C_ROLLOUT_AVAILABLE)

# UCT exploration constant (√2 is standard; tune higher for more exploration)
UCT_C = math.sqrt(2)

# Lookup tables for UCT hot path — ~2× faster than calling math functions for integer args.
# _INV_SQRT[v] = 1/sqrt(v) — child score denominator (updated in backprop, read in UCT)
# _SQRT_LOG[v] = sqrt(log(v)) — parent exploration base (computed once per uct_select call)
_INV_SQRT = [0.0] + [1.0 / math.sqrt(i) for i in range(1, 65537)]
_SQRT_LOG  = [0.0] + [math.sqrt(math.log(i)) for i in range(1, 65537)]
_INV       = [0.0] + [1.0 / i              for i in range(1, 65537)]

# Positional priority for "rest" moves in _sort_untried.
# Anti-diagonal Small (TR-S, CC-S, BL-S) is the dominant winning mechanism;
# these moves get the highest priority in expansion ordering.
# pop() returns the LAST element, so highest priority goes at the end (sort ascending).
# Stored as a flat list indexed by (r*3+c)*3 + (sz-1) for O(1) list lookup.
_AD = frozenset([(0, 2), (1, 1), (2, 0)])
_REST_PRIORITY_FLAT: list = [0] * 27   # 9 cells × 3 sizes
for _r in range(3):
    for _c in range(3):
        for _sz in (1, 2, 3):   # Small=1, Medium=2, Large=3
            _p = 0
            if (_r, _c) in _AD:
                _p += 40 if _sz == 1 else 20
            if (_r, _c) == (1, 1):
                _p += 30
            elif _r in (0, 2) and _c in (0, 2):
                _p += 10
            _p += 6 - _sz * 2   # Small:+4, Medium:+2, Large:0
            _REST_PRIORITY_FLAT[(_r * 3 + _c) * 3 + (_sz - 1)] = _p
del _r, _c, _sz, _p, _AD

# Dict-keyed move priority — avoids index arithmetic in sort key lambda.
# Dict.__getitem__ is a C-level callable: 2× faster than lambda in sort benchmarks.
_MOVE_PRIORITY: dict = {
    (r, c, sz): _REST_PRIORITY_FLAT[(r * 3 + c) * 3 + (int(sz) - 1)]
    for r in range(3) for c in range(3)
    for sz in (PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE)
}


def _sort_untried(state: TicTacPro, moves: list, guided: bool) -> list:
    """
    Return moves sorted so pop() yields winning moves first, then bullseye blocks,
    then line blocks, then rest (sorted by _MOVE_PRIORITY). Skipped when guided=True.
    Layout: [rest, line_blocking, bull_blocking, winning] — pop() → winning first.
    """
    if guided or len(moves) <= 1:
        return moves
    bf  = state.board.tobytes()
    cur = int(state.current_player)
    opp = 3 - cur
    opp_pr = state.pieces_remaining[_SWITCH_PLAYER[state.current_player]]
    winning, bull_blocking, line_blocking, rest = [], [], [], []
    for m in moves:
        sz_idx = m[2] - 1      # PieceSize is IntEnum — arithmetic avoids int() cast overhead
        sz     = m[2]
        ci     = m[0] * 3 + m[1]
        wl     = _WIN_LINES[sz_idx][ci]
        is_win = False
        is_bull = False
        is_line = False
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
    # Sort rest ascending so pop() yields highest-priority moves first.
    # Anti-diagonal Small (TR-S, CC-S, BL-S) at the end → explored before edges.
    # Flat-list lookup avoids dict overhead: index = (r*3+c)*3 + (sz-1).
    rest.sort(key=_MOVE_PRIORITY.__getitem__)
    # §341c: bullseye blocks after line blocks → pop() returns bullseye first.
    return rest + line_blocking + bull_blocking + winning


# ── Move application helpers ──────────────────────────────────────────────────

def _apply_move(g: TicTacPro, r: int, c: int, size: PieceSize, is_win: bool) -> None:
    """Apply move to g without returning an undo token (for clone-based rollouts)."""
    player = g.current_player
    sz_idx = size - 1
    g.board[r, c, sz_idx] = player
    g.pieces_remaining[player][size] -= 1
    g._pieces_left -= 1
    if is_win:
        g.winner    = player
        g.game_over = True
    elif g._pieces_left == 0:
        g.game_over = True
    else:
        g.current_player = _SWITCH_PLAYER[player]


# ── Undo-capable push/pop helpers ─────────────────────────────────────────────
# Used by _build_tree to avoid per-simulation cloning (~3× speedup).
# Token layout: (flat_idx, player, size, prev_winner, prev_game_over)
# flat_idx = r*9 + c*3 + sz_idx — pre-computed at push time so undo avoids
# recomputing 2 multiplications + 2 additions + 2 extra subscripts per step.

def _push_undo(g: TicTacPro, r: int, c: int, size: PieceSize) -> tuple:
    """Push move onto g without validation; return undo token."""
    player = g.current_player
    sz_idx = size - 1          # PieceSize is IntEnum: arithmetic works directly
    wins   = _would_win_bf(g.board.tobytes(), r, c, sz_idx, player)
    prev_winner    = g.winner
    prev_game_over = g.game_over
    g.board[r, c, sz_idx] = player
    g.pieces_remaining[player][size] -= 1
    g._pieces_left -= 1
    if wins:
        g.winner    = player
        g.game_over = True
    elif g._pieces_left == 0:
        g.game_over = True
    else:
        g.current_player = _SWITCH_PLAYER[player]
    return (r*9 + c*3 + sz_idx, player, size, prev_winner, prev_game_over)


def _push_known_undo(g: TicTacPro, r: int, c: int, size: PieceSize, is_win: bool) -> tuple:
    """push_known() with undo token — caller already knows is_win."""
    player = g.current_player
    sz_idx = size - 1
    prev_winner    = g.winner
    prev_game_over = g.game_over
    g.board[r, c, sz_idx] = player
    g.pieces_remaining[player][size] -= 1
    g._pieces_left -= 1
    if is_win:
        g.winner    = player
        g.game_over = True
    elif g._pieces_left == 0:
        g.game_over = True
    else:
        g.current_player = _SWITCH_PLAYER[player]
    return (r*9 + c*3 + sz_idx, player, size, prev_winner, prev_game_over)


def _pop_undo(g: TicTacPro, tok: tuple) -> None:
    """Restore g to pre-push state using token from _push_undo / _push_known_undo."""
    flat_idx, player, size, prev_winner, prev_game_over = tok
    g.board.ravel()[flat_idx] = 0
    g.pieces_remaining[player][size] += 1
    g._pieces_left     += 1
    g.winner            = prev_winner
    g.game_over         = prev_game_over
    g.current_player    = player  # player is the mover; restoring undoes the switch


# Sentinel for lazily-computed untried_moves: truthy (stops selection loop, triggers expansion),
# replaced by get_legal_moves() the first time a node is selected for expansion.
# Saves get_legal_moves() for leaf nodes that are created but never expanded from (~60-70%
# of created nodes in typical 26K-sim searches). Must be a truthy singleton.
_LAZY = object()

# C-level attrgetter — avoids lambda overhead in best_child() max() call
_get_visits = operator.attrgetter("visits")


# ── Tree node ─────────────────────────────────────────────────────────────────
class MCTSNode:
    __slots__ = ("move", "move_fidx", "move_sz", "parent", "children",
                 "wins", "visits", "winrate", "uct_scale", "untried_moves",
                 "player_who_moved", "untried_sorted", "move_is_win", "_n_non_winning",
                 "ch_wr", "ch_us", "ch_n", "child_idx")

    def __init__(self, move=None, parent=None, untried_moves=None,
                 player_who_moved=None, move_is_win=False, n_non_winning=-1,
                 move_fidx: int = 0, move_sz: int = 0):
        self.move             = move            # (r, c, sz) that led here
        self.move_fidx        = move_fidx       # r*9+c*3+sz-1 pre-computed flat board index
        self.move_sz          = move_sz         # sz cached to avoid tuple unpack in selection
        self.parent           = parent
        self.children         = []
        self.wins             = 0.0
        self.visits           = 0
        self.winrate          = 0.0
        self.uct_scale        = 0.0
        self.untried_moves    = untried_moves
        self.player_who_moved = player_who_moved
        self.untried_sorted   = False
        self.move_is_win      = move_is_win
        self._n_non_winning   = n_non_winning
        self.ch_wr     = None  # §57: float64 array of child winrates (lazy alloc on first child)
        self.ch_us     = None  # §57: float64 array of child uct_scales
        self.ch_n      = 0     # §57: count of expanded children (== len(children))
        self.child_idx = 0     # §57: this node's index in parent's ch_wr/ch_us arrays

    def uct_select(self) -> "MCTSNode":
        """Select child with highest UCT score."""
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

    def best_child(self) -> "MCTSNode":
        """Select most-visited child (used at root for final decision)."""
        return max(self.children, key=_get_visits)


# ── MCTS agent ────────────────────────────────────────────────────────────────
class MCTSAgent:
    """
    MCTS agent.  Compatible with the same get_action(game, player, epsilon)
    interface used by GPUDQNAgent and MinimaxAgent.
    """

    def __init__(self,
                 time_limit: float = 2.0,
                 max_simulations: int = 10_000_000,
                 rollout_agent=None,
                 randomize_expansion: bool = False):
        """
        Args:
            time_limit:            Wall-clock seconds per move.
            max_simulations:       Hard cap on simulations per move.
            rollout_agent:         Optional agent for guided rollouts.
            randomize_expansion:   §179: shuffle non-winning/blocking expansion
                                   order so parallel workers diverge rather than
                                   building identical trees under deterministic C
                                   sorted expansion + deterministic C rollout.
        """
        self.time_limit          = time_limit
        self.max_simulations     = max_simulations
        self.rollout_agent       = rollout_agent
        self._randomize_exp      = randomize_expansion  # §179
        self._last_sims          = 0
        # Tree reuse: keep best-child subtree across calls to reuse statistics
        self._reuse_node: Optional[MCTSNode] = None

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
        # Fast path: return any immediate winning move without building a tree.
        p_int = int(player)
        bf    = game.board.tobytes()
        for mv in game.get_legal_moves():
            if _would_win_bf(bf, mv[0], mv[1], mv[2] - 1, p_int):
                self._reuse_node = None
                return mv
        g = game.clone()
        root  = self._build_tree(g, player)
        if not root.children:
            moves = game.get_legal_moves()
            self._reuse_node = None
            return moves[0] if moves else None
        best = root.best_child()
        # Save best child's subtree for next call (our opponent will move next)
        self._reuse_node = best
        return best.move

    def get_info(self, game: TicTacPro, player: Player = None):
        """Returns (action, (simulations, win_rate))."""
        if game.game_over:
            return None, (0, 0.0)
        if player is None:
            player = game.current_player
        p_int = int(player)
        bf    = game.board.tobytes()
        for mv in game.get_legal_moves():
            if _would_win_bf(bf, mv[0], mv[1], mv[2] - 1, p_int):
                self._reuse_node = None
                return mv, (0, 1.0)
        g = game.clone()
        root = self._build_tree(g, player)
        if not root.children:
            moves = game.get_legal_moves()
            return (moves[0] if moves else None), (0, 0.0)
        best = root.best_child()
        win_rate = best.wins / best.visits if best.visits else 0
        return best.move, (self._last_sims, win_rate)

    # ── Tree building ─────────────────────────────────────────────────────────
    def _build_tree(self, g: TicTacPro, root_player: Player) -> MCTSNode:
        # Undo-based approach: push/undo onto g directly — no per-simulation clone.
        # All simulations share the same TicTacPro instance; state is restored after
        # each simulation via the undo token stack.
        root = None
        if self._reuse_node is not None and g.move_history:
            last_move = g.move_history[-1][:3]  # (row, col, size)
            for child in self._reuse_node.children:
                if child.move == last_move:
                    root = child
                    root.parent = None  # detach from old tree
                    break
        if root is None:
            # §56: positional args — kwarg dict matching adds ~720ns/call overhead
            root = MCTSNode(None, None, g.get_legal_moves(), None)
        self._reuse_node = None
        deadline = time.perf_counter() + self.time_limit
        sims = 0
        _guided = self.rollout_agent is not None
        _check = 64  # batch time checks to reduce perf_counter() syscall overhead
        # Pre-allocated fixed-size undo stack: avoids list.append() resize checks,
        # list.clear() decref loop, and reversed() iterator allocation per sim.
        # Max game depth = 36 (6 pieces × 2 players × 3 sizes); 40 slots is safe.
        _toks   = [None] * 40
        _toks_d = 0  # depth counter; reset per sim instead of calling .clear()
        # Bind hot module-level names as locals — LOAD_FAST vs LOAD_GLOBAL per iteration
        _perf_counter = time.perf_counter
        _pundo  = _push_undo
        _lazy   = _LAZY
        _inv    = _INV
        _inv_sqrt = _INV_SQRT
        _Player_NONE = Player.NONE
        _MCTSNode = MCTSNode
        # §64: always use C-level win/block sorting when available — guided flag only
        # controlled rollout phase, but fast_rollout_c takes priority anyway so the
        # rollout_agent is only used as fallback. Expansion always benefits from sorted moves.
        _get_sorted_ex = (get_sorted_legal_ex_c if C_ROLLOUT_AVAILABLE else None)
        _get_legal     = (get_legal_moves_fast  if C_ROLLOUT_AVAILABLE and _guided else None)
        # §179: randomize_expansion shuffles non-winning moves so parallel workers
        # explore different branches instead of building identical trees.
        _randomize_exp = self._randomize_exp
        # Bind rollout function as local — eliminates method dispatch + LOAD_GLOBAL per sim
        _rollout_agent = self.rollout_agent
        _c_rollout = fast_rollout_c if C_ROLLOUT_AVAILABLE else None
        _pr = g.pieces_remaining
        # §51: pre-cache inner player dicts as a list for O(1) list subscript vs dict lookup.
        # _pl is a plain int (§49), so list[plain_int] (~9ns) beats dict[plain_int] (~22ns).
        # Inner dicts are shared refs — all mutations reflect in g.pieces_remaining directly.
        _prs = [None, _pr[Player.RED], _pr[Player.BLUE]]
        _pcs_left = g._pieces_left
        _winner    = g.winner
        _game_over = g.game_over
        # Flat 1D view of g.board: avoids 3-tuple creation for every numpy setitem.
        # ravel() returns a view (g.board is C-contiguous) — mutations are reflected in g.board.
        _board = g.board.ravel()
        # UCT constants as LOAD_FAST: eliminates LOAD_GLOBAL per uct_select call (~20ns each).
        _UCT_C    = UCT_C
        _sqrt_log = _SQRT_LOG
        # Cache root current_player as plain int — LOAD_FAST at sim start; int avoids
        # IntEnum method overhead on every numpy setitem, arithmetic switch (3-p), and
        # dict lookup. Player.RED=1, Player.BLUE=2: 3-1=2, 3-2=1.
        _cp_root  = int(g.current_player)
        _np_zeros = np.zeros  # §57: numpy array allocation for ch_wr/ch_us
        # §58: fast-path detection flag (set once when root fully expands + all terminal)
        _root_all_terminal = False
        _terminal_checked  = False

        while sims < self.max_simulations:
            if sims % _check == 0 and _perf_counter() >= deadline:
                break
            node = root
            _toks_d = 0  # reset depth (no .clear() decref loop)
            _cp = _cp_root  # LOAD_FAST reset — undo always restores g to root state

            # 1. Selection — UCT walk. Inlined + numpy-vectorized for n-child nodes.
            # §57: (ch_wr[:n] + eb * ch_us[:n]).argmax() is 2.1× faster than Python
            # for-loop for n=27 (874ns vs 1851ns). ch_wr/ch_us maintained in backprop.
            # _cp tracks g.current_player as a LOAD_FAST local; synced once before expansion.
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
                _prs[_pl][_sz] -= 1                     # §51: list outer lookup vs dict
                _pcs_left -= 1
                if node.move_is_win:
                    _winner = _pl; _game_over = True
                elif _pcs_left == 0:
                    _game_over = True
                else:
                    _cp = 3 - _pl                       # §49: arithmetic vs dict (~10ns faster)
                _toks[_toks_d] = (_fidx, _pl, _sz, _pw, _pgo); _toks_d += 1

            # 2. Expansion — lazy sort on first expansion (sort once, pop() forever).
            # Non-guided: get_sorted_legal_ex_c generates+categorizes+returns n_winning in one C call.
            # Guided:     get_legal_moves_fast + _sort_untried (rollout_agent handles order).
            # Sync g.current_player once here — C helpers and _sort_untried read it.
            g.current_player = _cp
            if node.untried_moves:
                if node.untried_moves is _lazy:
                    if _game_over:
                        node.untried_moves = []
                    elif _get_sorted_ex:
                        moves, nw = _get_sorted_ex(g)
                        nnw = len(moves) - nw
                        # §179: shuffle non-winning prefix so parallel workers diverge.
                        # Winning moves stay at end (highest priority, still popped first).
                        if _randomize_exp and nnw > 1:
                            sub = moves[:nnw]
                            random.shuffle(sub)
                            moves[:nnw] = sub
                        node.untried_moves   = moves
                        node._n_non_winning  = nnw
                        node.untried_sorted  = True
                    else:
                        node.untried_moves = _get_legal(g) if _get_legal else g.get_legal_moves()
                if node.untried_moves:
                    if not node.untried_sorted:
                        node.untried_moves = _sort_untried(g, node.untried_moves, _guided)
                        node.untried_sorted = True
                    _nnw = node._n_non_winning
                    is_win = _nnw >= 0 and len(node.untried_moves) > _nnw
                    mv    = node.untried_moves.pop()
                    mover = _cp                         # LOAD_FAST vs LOAD_ATTR (~20ns saved)
                    r, c, sz = mv
                    if _nnw >= 0:
                        # Inline _push_known_undo for expansion too
                        _sz1 = sz - 1
                        _pw = _winner; _pgo = _game_over   # LOAD_FAST vs LOAD_ATTR
                        _fidx = r*9 + c*3 + _sz1            # pre-compute flat index
                        _board[_fidx] = mover
                        _prs[mover][sz] -= 1            # §51: list outer lookup
                        _pcs_left -= 1
                        if is_win:
                            _winner = mover; _game_over = True  # cache only
                        elif _pcs_left == 0:
                            _game_over = True
                        else:
                            g.current_player = 3 - mover
                        _toks[_toks_d] = (_fidx, mover, sz, _pw, _pgo); _toks_d += 1
                    else:
                        # §59: sync g.* from locals BEFORE _push_undo reads them.
                        # _push_undo captures prev_game_over=g.game_over and decrements
                        # g._pieces_left.  Without this sync, g._pieces_left drifts down
                        # by 1 each sim (never restored by the undo loop), reaching 0 on
                        # sim 18 of a fresh game → g.game_over=True is captured into every
                        # subsequent token → all depth-1 LAZY resolutions see _game_over=True
                        # → untried_moves=[] → tree never expands past depth 1.
                        g._pieces_left = _pcs_left    # prevent drift
                        g.game_over    = _game_over   # prevent stale True capture
                        _toks[_toks_d] = _pundo(g, r, c, sz); _toks_d += 1  # guided path
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
                    child = _MCTSNode(mv, node, _lazy, mover, is_win, -1, _fidx, sz)
                    child.child_idx = _cidx
                    node.children.append(child)
                    node = child

            # 3. Rollout — sync _pcs_left once (fast_rollout_c reads g._pieces_left).
            # g.winner/g.game_over are NOT synced — neither fast_rollout_c nor the
            # selection/undo loops read them; we use the local caches throughout.
            # §345: _rollout_undo is the exception — it reads g.game_over directly.
            # The expansion else-branch already syncs g.game_over=_game_over before
            # each _pundo call, so g.game_over is correct by the time we reach here.
            # The explicit re-sync below is defensive: it makes the invariant visible
            # in the code and guards against future refactors removing the else-branch
            # sync.  Cost is 2 STORE_ATTRs on an already-slow guided-no-C path.
            g._pieces_left = _pcs_left
            if _game_over:           # local cache, no LOAD_ATTR
                _w = _winner         # local cache, no LOAD_ATTR
                result = 1.0 if _w == root_player else (0.5 if _w == _Player_NONE else 0.0)
            elif _c_rollout is not None:
                _w = _c_rollout(g)
                result = 1.0 if _w == root_player else (0.5 if _w == _Player_NONE else 0.0)
            elif _rollout_agent is not None:
                g.game_over = _game_over   # §345: defensive sync for _rollout_undo
                g.winner    = _winner      # §345: defensive sync for _rollout_undo
                result = self._rollout_undo(g, root_player)  # rare: guided path
            else:
                gc = g.clone()
                _prm = pick_rollout_move
                _am  = _apply_move
                while not gc.game_over:
                    mv_r, is_win_r = _prm(gc)
                    if mv_r is None: break
                    _am(gc, mv_r[0], mv_r[1], mv_r[2], is_win_r)
                _w = gc.winner
                result = 1.0 if _w == root_player else (0.5 if _w == _Player_NONE else 0.0)

            # 4. Backpropagation — alternating result avoids per-step player comparison.
            # Players alternate up the tree: swap _ra/_rb each step instead of checking
            # node.player_who_moved (saves 1 LOAD_ATTR + COMPARE_OP per backprop step).
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

            # 5. Undo selection + expansion: depth-counter walk avoids reversed() iterator.
            # Token layout: (flat_idx, player, sz, prev_winner, prev_game_over)
            # flat_idx = r*9+c*3+sz_idx pre-computed at push — avoids 2 MUL+2 ADD per undo step.
            # Restore _winner/_game_over caches; g.winner/g.game_over stay stale (never read).
            # §50: g.current_player is NOT restored here — the next sim sets it via
            # `g.current_player = _cp` before expansion, and selection uses _cp (LOAD_FAST).
            # Removing it saves 1 STORE_ATTR (~7ns) per undo step.
            while _toks_d > 0:
                _toks_d -= 1
                # §54: UNPACK_SEQUENCE replaces 5 separate tuple subscripts (~10ns/step saved)
                _tf, _tp, _ts, _winner, _game_over = _toks[_toks_d]
                _board[_tf] = 0
                _prs[_tp][_ts] += 1                     # §51: list outer lookup
                _pcs_left += 1

            sims += 1
            # §58: one-time check — once root is fully expanded, test if all children are
            # terminal wins. If so, break to the fast path loop (no move application needed).
            if not _terminal_checked:
                if not root.untried_moves and root.children:
                    _terminal_checked = True
                    if all(c.move_is_win for c in root.children):
                        _root_all_terminal = True
                        break

        # §58 fast path: all root children are terminal (move_is_win=True).
        # Every sim: numpy UCT at root → terminal child → result → 2-step backprop.
        # Skips: board write, pieces decrement, undo tokens, game-state sync (~400ns/sim).
        # result = 1.0 always: depth-1 mover IS root_player (root_player's turn) and wins.
        if _root_all_terminal:
            _n     = root.ch_n
            _rch   = root.children   # LOAD_FAST: eliminates LOAD_ATTR per sim
            _rchwr = root.ch_wr      # LOAD_FAST: eliminates LOAD_ATTR in UCT + backprop
            _rchus = root.ch_us      # LOAD_FAST: eliminates LOAD_ATTR in UCT + backprop
            _rv    = root.visits     # running cache: avoids LOAD_ATTR in UCT eb + backprop
            while sims < self.max_simulations:
                if sims % _check == 0 and _perf_counter() >= deadline:
                    break
                # UCT at root (no game state modification)
                _eb  = _UCT_C * _sqrt_log[_rv if _rv <= 65536 else 65536]
                node = _rch[(_rchwr[:_n] + _eb * _rchus[:_n]).argmax()]
                # Backprop: leaf (terminal child) — always result=1.0 (root_player wins)
                v = node.visits + 1; node.visits = v
                w = node.wins + 1.0;  node.wins  = w
                v_cap = v if v <= 65536 else 65536
                wr = w * _inv[v_cap]; us = _inv_sqrt[v_cap]
                node.winrate = wr; node.uct_scale = us
                _rchwr[node.child_idx] = wr; _rchus[node.child_idx] = us
                # Backprop: root (visits+uct_scale only; wins/winrate not used for decisions)
                _rv += 1; root.visits = _rv
                v_cap = _rv if _rv <= 65536 else 65536
                root.uct_scale = _inv_sqrt[v_cap]
                sims += 1

        self._last_sims = sims
        return root

    def _rollout_undo(self, g: TicTacPro, root_player: Player) -> float:
        """
        Rollout from current game state. Returns 1.0/0.5/0.0 for win/draw/loss.

        Uses a lightweight clone for the rollout path: clone() costs ~0.7µs and avoids
        allocating undo tokens (7-tuple per move) + _pop_undo calls, saving ~2.7µs/rollout.
        The rollout agent path still uses undo (rollout_agent calls are infrequent).
        """
        if g.game_over:
            w = g.winner
            return 1.0 if w == root_player else (0.5 if w == Player.NONE else 0.0)

        if self.rollout_agent is not None:
            rtoks = []
            while not g.game_over:
                mv = self.rollout_agent.get_action(g, g.current_player)
                if mv is None:
                    break
                r, c, sz = mv
                rtoks.append(_push_undo(g, r, c, sz))
            winner = g.winner
            for tok in reversed(rtoks):
                _pop_undo(g, tok)
            return 1.0 if winner == root_player else (0.5 if winner == Player.NONE else 0.0)

        if C_ROLLOUT_AVAILABLE:
            winner = fast_rollout_c(g)
            return 1.0 if winner == root_player else (0.5 if winner == Player.NONE else 0.0)
        gc = g.clone()
        _prm = pick_rollout_move  # LOAD_FAST > LOAD_GLOBAL in tight loop (~1.5×)
        _am  = _apply_move
        while not gc.game_over:
            mv, is_win = _prm(gc)
            if mv is None:
                break
            r, c, sz = mv
            _am(gc, r, c, sz, is_win)
        return 1.0 if gc.winner == root_player else (0.5 if gc.winner == Player.NONE else 0.0)
