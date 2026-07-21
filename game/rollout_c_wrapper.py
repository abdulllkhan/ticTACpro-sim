"""
C-accelerated rollout and legal-move generation for TicTacPro.

Loads rollout_c.so (built with: gcc -O3 -march=native -fPIC -shared -o game/rollout_c.so game/rollout_c.c)

Exports:
    fast_rollout_c(g)        -> Player  — full greedy rollout in one C call (~4.3× vs Python)
    get_legal_moves_fast(g)  -> list    — legal moves via C (~8× vs Python get_legal_moves)
    C_ROLLOUT_AVAILABLE      -> bool    — False if .so is missing (graceful fallback)
"""

import ctypes
import os

from game.tictacpro import Player, _CELLS, _SIZES, PieceSize, _SWITCH_PLAYER

# ── Load shared library ───────────────────────────────────────────────────────

_lib = None
_LIB_PATH = os.path.join(os.path.dirname(__file__), "rollout_c.so")

try:
    _lib = ctypes.CDLL(_LIB_PATH)

    # run_rollout_c: entire rollout in one C call — returns winner int
    _lib.run_rollout_c.restype = ctypes.c_int
    _lib.run_rollout_c.argtypes = [
        ctypes.POINTER(ctypes.c_char),    # bf (27-byte board, modified in place)
        ctypes.c_int,                     # cur player (1 or 2)
        ctypes.POINTER(ctypes.c_int),     # pr_flat[6] (modified in place)
        ctypes.c_int,                     # pieces_left
    ]

    # get_legal_moves_c: returns legal move indices (0-26) for current player
    _lib.get_legal_moves_c.restype = ctypes.c_int
    _lib.get_legal_moves_c.argtypes = [
        ctypes.POINTER(ctypes.c_char),   # bf
        ctypes.POINTER(ctypes.c_int),    # pr_cur[3]: current player's [S, M, L] counts
        ctypes.POINTER(ctypes.c_ubyte),  # out_moves[27]
    ]

    # sort_legal_moves_c: generate moves AND categorize [rest..., blocking..., winning...]
    _lib.sort_legal_moves_c.restype = ctypes.c_int
    _lib.sort_legal_moves_c.argtypes = [
        ctypes.POINTER(ctypes.c_char),   # bf
        ctypes.POINTER(ctypes.c_int),    # pr_cur[3]
        ctypes.c_int,                    # cur
        ctypes.c_int,                    # opp
        ctypes.POINTER(ctypes.c_ubyte),  # out_moves[27]
        ctypes.POINTER(ctypes.c_int),    # pr_opp[3] — prevents phantom block classification
    ]

    # sort_legal_moves_ex_c: same as sort_legal_moves_c + writes n_winning to output ptr
    _lib.sort_legal_moves_ex_c.restype = ctypes.c_int
    _lib.sort_legal_moves_ex_c.argtypes = [
        ctypes.POINTER(ctypes.c_char),   # bf
        ctypes.POINTER(ctypes.c_int),    # pr_cur[3]
        ctypes.c_int,                    # cur
        ctypes.c_int,                    # opp
        ctypes.POINTER(ctypes.c_ubyte),  # out_moves[27]
        ctypes.POINTER(ctypes.c_int),    # out_n_winning
        ctypes.POINTER(ctypes.c_int),    # pr_opp[3] — prevents phantom block classification
    ]

    # pick_rollout_move_c: single-move selector (kept for compatibility)
    _lib.pick_rollout_move_c.restype = None
    _lib.pick_rollout_move_c.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    C_ROLLOUT_AVAILABLE = True
except (OSError, AttributeError):  # pragma: no cover — fires only when C extension fails to load
    C_ROLLOUT_AVAILABLE = False

# ── Pre-allocated buffers (not thread-safe — one per process) ────────────────
_pr_arr        = (ctypes.c_int * 6)()
_pr_cur_arr    = (ctypes.c_int * 3)()
_pr_opp_arr    = (ctypes.c_int * 3)()   # opponent's piece counts for sort functions
_moves_buf     = (ctypes.c_ubyte * 27)()
_sort_buf      = (ctypes.c_ubyte * 27)()
_n_winning_buf = (ctypes.c_int * 1)()   # output buffer for sort_legal_moves_ex_c
_c_char_27     = ctypes.c_char * 27

# Lookup: flat C move index (0-26) → Python (r, c, PieceSize) tuple
# Index = sz_idx * 9 + ci where ci = r*3+c
_MOVE_BY_IDX = [
    (r, c, sz)
    for sz_idx, sz in enumerate(_SIZES)
    for ci, (r, c) in enumerate(_CELLS)
]

_PLAYER_RED  = Player.RED
_PLAYER_BLUE = Player.BLUE
_PLAYER_NONE = Player.NONE


def fast_rollout_c(g) -> Player:
    """
    C-accelerated greedy rollout (win > block > first-available).

    Runs the entire game to completion in a single C call — no Python loop,
    no clone of the game object. ~4.3× faster than the Python rollout.

    Returns the winner as a Player enum. Does NOT modify g.
    Not thread-safe (shared ctypes piece-count buffer).
    """
    pr = g.pieces_remaining
    pr1 = pr[1]; pr2 = pr[2]   # cache inner dicts: saves 4 dict lookups vs pr[1][k] × 6
    _pr_arr[0] = pr1[1]; _pr_arr[1] = pr1[2]; _pr_arr[2] = pr1[3]
    _pr_arr[3] = pr2[1]; _pr_arr[4] = pr2[2]; _pr_arr[5] = pr2[3]
    bf_arr = _c_char_27.from_buffer_copy(g.board)   # skip tobytes(): 0.17µs vs 0.41µs
    winner_int = _lib.run_rollout_c(bf_arr, g.current_player, _pr_arr, g._pieces_left)
    return _PLAYER_RED if winner_int == 1 else (_PLAYER_BLUE if winner_int == 2 else _PLAYER_NONE)


def get_sorted_legal_c(g) -> list:
    """
    C-accelerated legal move generation with win/block/rest categorization.

    Returns moves pre-sorted for pop() access: [rest..., blocking..., winning...]
    so pop() returns winning moves first, then blocking, then rest.

    Replaces both get_legal_moves() + _sort_untried() in the MCTS expansion path.
    ~8× faster than the Python equivalent for non-guided MCTS.
    Not thread-safe.
    """
    player = g.current_player
    pr_p = g.pieces_remaining[player]
    pr_o = g.pieces_remaining[_SWITCH_PLAYER[player]]
    _pr_cur_arr[0] = pr_p[1]; _pr_cur_arr[1] = pr_p[2]; _pr_cur_arr[2] = pr_p[3]
    _pr_opp_arr[0] = pr_o[1]; _pr_opp_arr[1] = pr_o[2]; _pr_opp_arr[2] = pr_o[3]
    cur = int(player)
    bf_arr = _c_char_27.from_buffer_copy(g.board)
    n = _lib.sort_legal_moves_c(bf_arr, _pr_cur_arr, cur, 3 - cur, _sort_buf, _pr_opp_arr)
    return [_MOVE_BY_IDX[_sort_buf[i]] for i in range(n)]


def get_sorted_legal_ex_c(g):
    """
    Like get_sorted_legal_c but also returns n_winning (count of winning moves at tail).

    Returns (moves, n_winning) where n_winning is the count of winning moves at the
    end of the list. Allows callers to use _push_known_undo instead of _push_undo by
    pre-detecting whether the popped move is a win.
    Not thread-safe.
    """
    player = g.current_player
    pr_p = g.pieces_remaining[player]
    pr_o = g.pieces_remaining[_SWITCH_PLAYER[player]]
    _pr_cur_arr[0] = pr_p[1]; _pr_cur_arr[1] = pr_p[2]; _pr_cur_arr[2] = pr_p[3]
    _pr_opp_arr[0] = pr_o[1]; _pr_opp_arr[1] = pr_o[2]; _pr_opp_arr[2] = pr_o[3]
    cur = int(player)
    bf_arr = _c_char_27.from_buffer_copy(g.board)
    n = _lib.sort_legal_moves_ex_c(bf_arr, _pr_cur_arr, cur, 3 - cur, _sort_buf, _n_winning_buf, _pr_opp_arr)
    return [_MOVE_BY_IDX[_sort_buf[i]] for i in range(n)], _n_winning_buf[0]


def get_legal_moves_fast(g) -> list:
    """
    C-accelerated legal move generation for current player.

    ~8× faster than TicTacPro.get_legal_moves(). Returns the same list format:
    [(r, c, PieceSize), ...] ordered by size then cell (SMALL first).
    Not thread-safe (shared ctypes buffers).
    """
    player = g.current_player
    pr_p = g.pieces_remaining[player]
    _pr_cur_arr[0] = pr_p[1]; _pr_cur_arr[1] = pr_p[2]; _pr_cur_arr[2] = pr_p[3]
    bf_arr = _c_char_27.from_buffer_copy(g.board)
    n = _lib.get_legal_moves_c(bf_arr, _pr_cur_arr, _moves_buf)
    return [_MOVE_BY_IDX[_moves_buf[i]] for i in range(n)]
