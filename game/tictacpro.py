"""
Tic Tac Pro Game Engine

This module implements the complete game logic for Tic Tac Pro by Brass Monkey.
The game features:
- 3x3 board
- Each player has 9 pieces: 3 small, 3 medium, 3 large
- Pieces can be stacked (larger pieces can cover smaller ones)
- Win conditions:
  1. Three pieces of the SAME SIZE in a row (horizontal/vertical/diagonal)
  2. A "bullseye" - all three sizes of your color stacked in one cell
"""

import numpy as np
from enum import IntEnum
from typing import List, Tuple, Optional, Set


class PieceSize(IntEnum):
    """Piece sizes - larger values can stack on smaller values"""
    EMPTY = 0
    SMALL = 1
    MEDIUM = 2
    LARGE = 3


class Player(IntEnum):
    """Player identifiers"""
    NONE = 0
    RED = 1
    BLUE = 2


# Pre-computed iteration helpers — avoids per-call attribute lookups in hot loops.
_CELLS = [(r, c) for r in range(3) for c in range(3)]  # all 9 board cells
_SIZES = [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]

# 6.7× faster than Player(3 - int(p)) in the switch-player hot path.
_SWITCH_PLAYER = {Player.RED: Player.BLUE, Player.BLUE: Player.RED}


class TicTacPro:
    """
    Tic Tac Pro game implementation

    State representation:
    - board: 3x3x3 numpy array
      - First dimension: rows
      - Second dimension: columns
      - Third dimension: stack levels (small=0, medium=1, large=2)
      - Values: Player.NONE, Player.RED, or Player.BLUE
    """

    BOARD_SIZE = 3
    NUM_PIECE_SIZES = 3  # small, medium, large
    PIECES_PER_SIZE = 3  # 3 of each size per player

    def __init__(self):
        # Board state: [row, col, size_level] -> Player
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE, self.NUM_PIECE_SIZES), dtype=np.int8)

        # Track remaining pieces for each player
        # {Player.RED: {PieceSize.SMALL: 3, PieceSize.MEDIUM: 3, PieceSize.LARGE: 3}}
        self.pieces_remaining = {
            Player.RED: {
                PieceSize.SMALL: self.PIECES_PER_SIZE,
                PieceSize.MEDIUM: self.PIECES_PER_SIZE,
                PieceSize.LARGE: self.PIECES_PER_SIZE
            },
            Player.BLUE: {
                PieceSize.SMALL: self.PIECES_PER_SIZE,
                PieceSize.MEDIUM: self.PIECES_PER_SIZE,
                PieceSize.LARGE: self.PIECES_PER_SIZE
            }
        }

        self.current_player = Player.RED
        self.winner = Player.NONE
        self.game_over = False
        self.move_history = []
        self._pieces_left = 18  # 3 sizes × 3 pieces × 2 players

    def get_visible_board(self) -> np.ndarray:
        """
        Get the visible board state (only top pieces visible)
        Returns: 3x3 array with (player, size) tuples for each cell
        """
        visible = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE, 2), dtype=np.int8)

        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                # Find the topmost (largest) piece in this cell
                for size in reversed(range(self.NUM_PIECE_SIZES)):
                    if self.board[row, col, size] != Player.NONE:
                        visible[row, col, 0] = self.board[row, col, size]  # player
                        visible[row, col, 1] = size + 1  # piece size (1=small, 2=medium, 3=large)
                        break

        return visible

    def get_legal_moves(self, player: Player = None) -> List[Tuple[int, int, PieceSize]]:
        """
        Get all legal moves for the current player.
        Returns: List of (row, col, piece_size) tuples.
        """
        if player is None:
            player = self.current_player

        if self.game_over:
            return []

        legal_moves = []
        pr = self.pieces_remaining[player]
        bf = self.board.tobytes()

        for sz_idx, size in _SIZES_IDX:
            if pr[size] == 0:
                continue
            csl_sz = _CELL_SLOT[sz_idx]
            for ci, (r, c) in _CELLS_IDX:
                if not bf[csl_sz[ci]]:
                    legal_moves.append((r, c, size))

        return legal_moves

    def _is_valid_move(self, row: int, col: int, size: PieceSize) -> bool:
        """Check if placing a piece at (row, col) of given size is valid"""
        size_idx = size - 1  # Convert to 0-indexed

        # Check if this size slot is empty
        if self.board[row, col, size_idx] != Player.NONE:
            return False

        # Each size has its own independent slot — no gobbling.
        return True

    def make_move(self, row: int, col: int, size: PieceSize, player: Player = None) -> bool:
        """
        Make a move on the board
        Returns: True if move was successful, False otherwise
        """
        if player is None:
            player = self.current_player

        if self.game_over:
            return False

        # Validate move
        if not (0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE):
            return False

        if size not in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]:
            return False

        if self.pieces_remaining[player][size] == 0:
            return False

        if not self._is_valid_move(row, col, size):
            return False

        # Make the move
        size_idx = size - 1
        wins = would_win(self.board, row, col, size_idx, int(player))
        self.board[row, col, size_idx] = player
        self.pieces_remaining[player][size] -= 1
        self._pieces_left -= 1
        self.move_history.append((row, col, size, player))

        if wins:
            self.winner = player
            self.game_over = True
        elif self._pieces_left == 0:
            # Theorem: a player always has legal moves while they have pieces
            # (each size has 9 slots; ≤6 can be filled → always ≥3 empty).
            # Draw ↔ all 18 pieces placed (both players have 0 pieces left).
            self.game_over = True
        else:
            # Switch players
            self.current_player = _SWITCH_PLAYER[player]

        return True

    def get_state_tensor(self) -> np.ndarray:
        """
        Get the game state as a tensor for neural network input
        Returns: Flattened state representation
        Shape: (3*3*3*2 + 6 + 1) = 61 features
        - 54 features for board (3x3x3 for each player)
        - 6 features for remaining pieces (3 sizes * 2 players)
        - 1 feature for current player
        """
        buf   = np.empty(61, dtype=np.float32)
        bflat = self.board.reshape(-1)
        np.equal(bflat, int(Player.RED),  out=buf[:27])
        np.equal(bflat, int(Player.BLUE), out=buf[27:54])
        pr = self.pieces_remaining
        buf[54] = pr[Player.RED][PieceSize.SMALL]    * _PIECES_INV
        buf[55] = pr[Player.RED][PieceSize.MEDIUM]   * _PIECES_INV
        buf[56] = pr[Player.RED][PieceSize.LARGE]    * _PIECES_INV
        buf[57] = pr[Player.BLUE][PieceSize.SMALL]   * _PIECES_INV
        buf[58] = pr[Player.BLUE][PieceSize.MEDIUM]  * _PIECES_INV
        buf[59] = pr[Player.BLUE][PieceSize.LARGE]   * _PIECES_INV
        buf[60] = 1.0 if self.current_player == Player.RED else 0.0
        return buf

    def get_state_tensor_normalized(self) -> np.ndarray:
        """
        State tensor normalized to current player's perspective.
        Current player's pieces always in slots 0-26, opponent in 27-53.
        Slot 60: §180 first-mover indicator — 1.0 if current player is RED
        (moves first, has first-mover advantage), 0.0 if BLUE (moves second).
        This lets the DQN learn distinct RED-aggressive vs BLUE-defensive strategies.
        NOTE: Checkpoints trained before §180 used buf[60]=1.0 always and are
        incompatible with this representation.
        """
        me_int  = int(self.current_player)
        opp_int = 3 - me_int
        me      = self.current_player
        opp     = Player.BLUE if me == Player.RED else Player.RED
        buf     = np.empty(61, dtype=np.float32)
        bflat   = self.board.reshape(-1)   # flat view — no copy
        # np.equal with out= writes 0.0/1.0 directly into float32 buffer — avoids
        # creating intermediate bool array + .astype(float32) + concatenate.
        np.equal(bflat, me_int,  out=buf[:27])
        np.equal(bflat, opp_int, out=buf[27:54])
        pr = self.pieces_remaining
        buf[54] = pr[me][PieceSize.SMALL]    * _PIECES_INV
        buf[55] = pr[me][PieceSize.MEDIUM]   * _PIECES_INV
        buf[56] = pr[me][PieceSize.LARGE]    * _PIECES_INV
        buf[57] = pr[opp][PieceSize.SMALL]   * _PIECES_INV
        buf[58] = pr[opp][PieceSize.MEDIUM]  * _PIECES_INV
        buf[59] = pr[opp][PieceSize.LARGE]   * _PIECES_INV
        # §180: 1.0 if current player is RED (first mover), 0.0 if BLUE.
        buf[60] = 1.0 if me == Player.RED else 0.0
        return buf

    def clone(self, copy_history: bool = True):
        """Create a deep copy of the game state.

        Args:
            copy_history: copy move_history (default True). Pass False when the
                          caller never reads it (e.g. MCTS rollout clones) to
                          skip list-copy overhead that grows as the game lengthens.
        """
        # object.__new__ skips __init__ (avoids allocating a fresh board/dicts
        # that would immediately be overwritten).
        new_game = object.__new__(TicTacPro)
        new_game.board = self.board.copy()
        new_game.pieces_remaining = {
            Player.RED:  dict(self.pieces_remaining[Player.RED]),
            Player.BLUE: dict(self.pieces_remaining[Player.BLUE]),
        }
        new_game.current_player = self.current_player
        new_game.winner = self.winner
        new_game.game_over = self.game_over
        new_game.move_history = self.move_history.copy() if copy_history else []
        new_game._pieces_left = self._pieces_left
        return new_game

    def push(self, row: int, col: int, size: PieceSize) -> None:
        """
        Make a move without validation — for search/rollout callers that already
        verified legality (move came from get_legal_moves or pick_rollout_move).
        Also skips move_history.append (search code doesn't read it in clones).
        """
        player   = self.current_player
        size_idx = int(size) - 1
        wins = would_win(self.board, row, col, size_idx, int(player))
        self.board[row, col, size_idx] = player
        self.pieces_remaining[player][size] -= 1
        self._pieces_left -= 1
        if wins:
            self.winner    = player
            self.game_over = True
        elif self._pieces_left == 0:
            self.game_over = True
        else:
            self.current_player = _SWITCH_PLAYER[player]

    def push_known(self, row: int, col: int, size: PieceSize, is_win: bool) -> None:
        """
        push() variant for rollouts: caller already knows is_win from pick_rollout_move,
        so we skip the would_win() check (which was 5.7% of MCTS time).
        """
        player   = self.current_player
        size_idx = int(size) - 1
        self.board[row, col, size_idx] = player
        self.pieces_remaining[player][size] -= 1
        self._pieces_left -= 1
        if is_win:
            self.winner    = player
            self.game_over = True
        elif self._pieces_left == 0:
            self.game_over = True
        else:
            self.current_player = _SWITCH_PLAYER[player]

    def reset(self):
        """Reset the game to initial state"""
        self.__init__()

    def __str__(self) -> str:
        """String representation of the visible board"""
        visible = self.get_visible_board()
        symbols = {
            Player.NONE: '.',
            Player.RED: 'R',
            Player.BLUE: 'B'
        }
        size_symbols = {0: ' ', 1: 's', 2: 'm', 3: 'l'}

        lines = []
        lines.append("  0   1   2")
        for row in range(self.BOARD_SIZE):
            row_str = f"{row} "
            for col in range(self.BOARD_SIZE):
                player = visible[row, col, 0]
                size = visible[row, col, 1]
                cell = symbols[player] + size_symbols[size]
                row_str += f"{cell:2} "
            lines.append(row_str)

        lines.append(f"\nCurrent player: {self.current_player.name}")
        lines.append(f"Red pieces: S={self.pieces_remaining[Player.RED][PieceSize.SMALL]} "
                    f"M={self.pieces_remaining[Player.RED][PieceSize.MEDIUM]} "
                    f"L={self.pieces_remaining[Player.RED][PieceSize.LARGE]}")
        lines.append(f"Blue pieces: S={self.pieces_remaining[Player.BLUE][PieceSize.SMALL]} "
                    f"M={self.pieces_remaining[Player.BLUE][PieceSize.MEDIUM]} "
                    f"L={self.pieces_remaining[Player.BLUE][PieceSize.LARGE]}")

        if self.game_over:
            if self.winner != Player.NONE:
                lines.append(f"\n{self.winner.name} WINS!")
            else:
                lines.append("\nDRAW!")

        return '\n'.join(lines)



# ── Fast win-check helper (used by MCTS rollouts) ────────────────────────────

_LINES = [
    [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],
    [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],
    [(0,0),(1,1),(2,2)], [(0,2),(1,1),(2,0)],
]
_CELL_LINES: list = [[] for _ in range(9)]
for _li, _ln in enumerate(_LINES):
    for _r, _c in _ln:
        _CELL_LINES[_r * 3 + _c].append(_li)

# For each cell, pre-compute the OTHER 2 cells in each line through it.
# _OTHERS[r*3+c] = list of ((r1,c1),(r2,c2)) — eliminates the inner loop in would_win.
_OTHERS: list = [[] for _ in range(9)]
for _li, _ln in enumerate(_LINES):
    for _i, (_r, _c) in enumerate(_ln):
        _rest = [(_ln[j][0], _ln[j][1]) for j in range(3) if j != _i]
        _OTHERS[_r * 3 + _c].append(_rest)

# For each size index, the other two size indices — avoids a generator in bullseye check.
_OTHER_SIZES = [[1, 2], [0, 2], [0, 1]]

# Precomputed flat-index tables for pick_rollout_move — eliminates arithmetic in inner loops.
#
# _OTHERS_FLAT[sz_idx][cell_idx] = [(flat1, flat2), ...]
#   Each pair is the flat board indices of the other two cells in a winning line,
#   for the given size slice. Replaces r1*9+c1*3+sz_idx arithmetic inside the hot loop.
#
# _BULLSEYE_FLAT[sz_idx][cell_idx] = (bi0, bi1)
#   Flat indices of the two other size slots in the same cell (bullseye win check).
#
# _CELL_SLOT[sz_idx][cell_idx] = flat board index of this cell+size slot.
#   Used for the "occupied?" check: bf[_CELL_SLOT[sz_idx][ci]] != 0.
#
# _CELLS_IDX = list(enumerate(_CELLS)) — avoids enumerate() overhead inside the hot loop.

_OTHERS_FLAT: list = []
_BULLSEYE_FLAT: list = []
_CELL_SLOT: list = []
for _sz in range(3):
    _s0, _s1 = _OTHER_SIZES[_sz]
    _of_row, _bf_row, _cs_row = [], [], []
    for _ci, (_r, _c) in enumerate(_CELLS):
        _base = _r * 9 + _c * 3
        _of_row.append([(_ln[0][0]*9+_ln[0][1]*3+_sz, _ln[1][0]*9+_ln[1][1]*3+_sz)
                        for _ln in _OTHERS[_r*3+_c]])
        _bf_row.append((_base + _s0, _base + _s1))
        _cs_row.append(_base + _sz)
    _OTHERS_FLAT.append(_of_row)
    _BULLSEYE_FLAT.append(_bf_row)
    _CELL_SLOT.append(_cs_row)

_CELLS_IDX: list = list(enumerate(_CELLS))  # [(0,(0,0)), (1,(0,1)), ...]

# _WIN_LINES[sz_idx][cell_idx] = [(idx1, idx2), ...] — all winning-line pairs for this slot,
# including the bullseye pair (the two other size slots at the same cell).
# Merging _OTHERS_FLAT + _BULLSEYE_FLAT into one list eliminates a second lookup and
# a separate conditional in pick_rollout_move and _sort_moves hot paths.
_WIN_LINES: list = []
for _sz in range(3):
    _wl_row = []
    for _ci in range(9):
        _wl_row.append(list(_OTHERS_FLAT[_sz][_ci]) + [_BULLSEYE_FLAT[_sz][_ci]])
    _WIN_LINES.append(_wl_row)

# Precomputed enumerate(_SIZES) — avoids creating a new enumerate object every call.
_SIZES_IDX: list = list(enumerate(_SIZES))

# Precomputed inverse PIECES_PER_SIZE (float32) for get_state_tensor_normalized.
_PIECES_INV = np.float32(1.0 / 3.0)


def _would_win_bf(bf: bytes, r: int, c: int, sz_idx: int, player: int) -> bool:
    """would_win variant accepting pre-computed board.tobytes() — avoids repeated tobytes call."""
    ci = r * 3 + c
    for i1, i2 in _WIN_LINES[sz_idx][ci]:
        if bf[i1] == player and bf[i2] == player:
            return True
    return False


def would_win(board, r: int, c: int, sz_idx: int, player: int) -> bool:
    """Return True if placing `player` at board[r,c,sz_idx] would win. No side effects."""
    return _would_win_bf(board.tobytes(), r, c, sz_idx, player)


def pick_rollout_move(state: "TicTacPro"):
    """
    Fast rollout move selector: win > block > threat > first-available (deterministic).

    §175: Added "threat" tier between block and first-available. A threat move is one
    that creates a 2-in-a-row (needing one more piece to win the line), making the
    heuristic proactively set up win threats rather than falling back to positional order.
    This forces DQN RED opponents to deal with genuine multi-threat setups.

    Returns (move, is_win) where is_win=True iff the move immediately wins the game.
    Callers can pass is_win to _push_known_undo() to skip the redundant would_win check.

    Returns (None, False) if no legal moves remain.
    """
    bf    = state.board.tobytes()
    cur   = int(state.current_player)
    opp   = 3 - cur
    pr    = state.pieces_remaining[state.current_player]
    opp_pr = state.pieces_remaining[_SWITCH_PLAYER[state.current_player]]

    bull_block_mv = None   # §343: bullseye 2/3 block (higher urgency)
    line_block_mv = None   # §343: standard same-size line block
    threat_mv     = None   # §175: first 2-in-a-row setup found
    rest_pick     = None

    for sz_idx, size in _SIZES_IDX:
        if pr[size] == 0:
            continue
        opp_has_size = opp_pr[size] > 0  # skip block if opp can't threaten this size
        wl  = _WIN_LINES[sz_idx]
        csl = _CELL_SLOT[sz_idx]

        for ci, (r, c) in _CELLS_IDX:
            if bf[csl[ci]]:
                continue

            if bull_block_mv is None:
                # Full scan: wins, bullseye blocks, line blocks, threats, rest.
                is_bull   = False
                is_line   = False
                is_threat = False
                for idx1, idx2 in wl[ci]:
                    v1 = bf[idx1]
                    if v1 == cur:
                        if bf[idx2] == cur:
                            return (r, c, size), True   # WIN
                        elif bf[idx2] == 0:
                            is_threat = True            # §175: 2-in-a-row setup
                    elif v1 == opp:
                        if bf[idx2] == opp and opp_has_size:
                            if idx1 // 3 == idx2 // 3:
                                is_bull = True          # §343: same cell → bullseye threat
                            else:
                                is_line = True
                    else:                               # v1 == 0
                        if bf[idx2] == cur:
                            is_threat = True            # §175: 2-in-a-row setup
                if is_bull:
                    bull_block_mv = (r, c, size)
                elif is_line and line_block_mv is None:
                    line_block_mv = (r, c, size)
                elif is_threat and threat_mv is None:
                    threat_mv = (r, c, size)            # §175
                elif rest_pick is None:
                    rest_pick = (r, c, size)
            else:
                # Bullseye block already found — only scan for wins.
                for idx1, idx2 in wl[ci]:
                    if bf[idx1] == cur and bf[idx2] == cur:
                        return (r, c, size), True

    if bull_block_mv is not None:
        return bull_block_mv, False
    if line_block_mv is not None:
        return line_block_mv, False
    if threat_mv is not None:                           # §175
        return threat_mv, False
    if rest_pick is not None:
        return rest_pick, False
    return None, False


if __name__ == "__main__":  # pragma: no cover
    # Quick test
    game = TicTacPro()
    print(game)
    print(f"\nLegal moves: {len(game.get_legal_moves())}")

    # Test some moves
    game.make_move(0, 0, PieceSize.SMALL)
    game.make_move(1, 1, PieceSize.SMALL)
    game.make_move(0, 1, PieceSize.SMALL)
    print("\n" + str(game))
