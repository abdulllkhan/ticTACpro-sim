"""
BullseyeAgent — hardcoded BLUE-optimal strategy for TicTacPro.

§332: TicTacPro is a second-player win. This agent implements BLUE's proven
winning strategy — the "sequential bullseye fork" discovered via minimax analysis:
1. Block RED's anti-diagonal plan (occupy CC-S early)
2. Accumulate 2 pieces in a chosen cell to create a bullseye threat
3. Force RED to spend a piece blocking the bullseye
4. Shift to a fresh cell and repeat — RED eventually runs out of blocks

Decision priority:
1. Win immediately (any 3-in-a-row or bullseye completion)
2. Block opponent's immediate winning move (bullseye blocks prioritized over line blocks)
3. Deny opponent's anti-diagonal S cells (if 2/3 occupied)
4. Complete own bullseye (cell where I have 2 pieces — play the 3rd size)
5. Build 2nd piece in a cell I already have 1 piece (no opponent pieces in that cell)
6. Start new bullseye in a fresh cell (prefer corners/center for forking potential)
7. Anti-diagonal S offense
8. Any legal move
"""

from game.tictacpro import (TicTacPro, Player, PieceSize,
                            _CELL_SLOT, _CELLS, _CELLS_IDX, _SIZES, _SIZES_IDX,
                            _WIN_LINES)

# Anti-diagonal cells (BLUE uses same strategy as RED)
_ANTIDIAG_S = [(0, 2, PieceSize.SMALL), (1, 1, PieceSize.SMALL), (2, 0, PieceSize.SMALL)]

# Cell priority for starting a new bullseye: center > corners > edges
# (0,0)=TL, (0,1)=TC, (0,2)=TR, (1,0)=ML, (1,1)=CC, (1,2)=MR, (2,0)=BL, (2,1)=BC, (2,2)=BR
_CELL_PRIORITY = [
    (1, 1),  # CC — center (highest forking value)
    (1, 0),  # ML
    (0, 2),  # TR
    (2, 0),  # BL
    (0, 0),  # TL
    (2, 2),  # BR
    (0, 1),  # TC
    (1, 2),  # MR
    (2, 1),  # BC
]

# Size order for building a bullseye: LARGE first, then MEDIUM, then SMALL.
# §334b: playing L first forces opponent to spend their L piece to block, while
# keeping S pieces available to block anti-diagonal threats or complete a late bullseye.
# The minimax PV uses TR-L → TR-M → TR-S for the winning bullseye completion.
_SIZES_ORDER = [PieceSize.LARGE, PieceSize.MEDIUM, PieceSize.SMALL]


class BullseyeAgent:
    """
    Rule-based agent implementing BLUE's sequential bullseye fork strategy.
    Works for both players but optimized for BLUE (second-player winning strategy).
    """

    def get_action(self, game: TicTacPro, player=None, epsilon: float = 0.0) -> tuple:
        me      = game.current_player
        me_int  = int(me)
        opp_int = 3 - me_int
        opp     = Player.BLUE if me_int == 1 else Player.RED

        board  = game.board
        bf     = board.tobytes()
        pr     = game.pieces_remaining[me]
        pr_opp = game.pieces_remaining[opp]

        # ── Pass 1: single scan for wins, blocks (bullseye-prioritized), and legal list ──
        bullseye_block = None  # block opponent completing a bullseye
        line_block     = None  # block opponent completing a 3-in-a-row
        legal          = []

        for sz_idx, size in _SIZES_IDX:
            if pr[size] == 0:
                continue
            csl_sz     = _CELL_SLOT[sz_idx]
            wl_sz      = _WIN_LINES[sz_idx]
            has_opp_sz = pr_opp[size] > 0
            for ci, (r, c) in _CELLS_IDX:
                if bf[csl_sz[ci]]:
                    continue
                legal.append((r, c, size))
                for i1, i2 in wl_sz[ci]:
                    v1 = bf[i1]
                    v2 = bf[i2]
                    if v1 == me_int and v2 == me_int:
                        return (r, c, size)   # immediate win
                    if has_opp_sz and v1 == opp_int and v2 == opp_int:
                        if i1 // 3 == i2 // 3:     # same cell → bullseye threat
                            if bullseye_block is None:
                                bullseye_block = (r, c, size)
                        elif line_block is None:
                            line_block = (r, c, size)

        if not legal:  # pragma: no cover — defensive guard
            return None

        block_mv = bullseye_block if bullseye_block is not None else line_block
        if block_mv is not None:
            return block_mv

        legal_set = set(legal)

        # ── Step 3: Deny anti-diagonal S cells when opponent has 2/3 ──
        opp_antidiag = (board[0,2,0]==opp_int) + (board[1,1,0]==opp_int) + (board[2,0,0]==opp_int)
        if opp_antidiag >= 2 and pr_opp[PieceSize.SMALL] > 0:
            for r, c, sz in _ANTIDIAG_S:  # pragma: no cover — pre-empted by step 2
                if (r, c, sz) in legal_set and board[r, c, 0] == 0:
                    return (r, c, sz)

        # ── Pass 2: cell analysis for bullseye building ──
        # Count pieces per cell for me and opponent
        my_count  = [0] * 9  # how many of my pieces in each cell
        opp_count = [0] * 9  # how many of opponent's pieces in each cell
        my_sizes  = [set() for _ in range(9)]   # which sizes I occupy per cell
        for ci, (r, c) in _CELLS_IDX:
            base = r * 9 + c * 3
            for sz in range(3):
                v = bf[base + sz]
                if v == me_int:
                    my_count[ci] += 1
                    my_sizes[ci].add(sz)
                elif v == opp_int:
                    opp_count[ci] += 1

        # ── Step 4: Complete own bullseye (I have 2 in a cell) ──
        for r, c in _CELL_PRIORITY:
            ci = r * 3 + c
            if my_count[ci] == 2 and opp_count[ci] == 0:
                # Find the missing size
                for sz_idx, size in _SIZES_IDX:
                    if pr[size] > 0 and sz_idx not in my_sizes[ci]:
                        mv = (r, c, size)
                        if mv in legal_set:
                            return mv

        # ── Step 5: Build 2nd piece in a cell I already started (no opponent) ──
        for r, c in _CELL_PRIORITY:
            ci = r * 3 + c
            if my_count[ci] == 1 and opp_count[ci] == 0:
                # Play the smallest available size I don't already have here
                for size in _SIZES_ORDER:
                    sz_idx = int(size) - 1
                    if pr[size] > 0 and sz_idx not in my_sizes[ci]:
                        mv = (r, c, size)
                        if mv in legal_set:
                            return mv

        # ── Step 6: Start fresh bullseye in a cell opponent hasn't touched ──
        for r, c in _CELL_PRIORITY:
            ci = r * 3 + c
            if my_count[ci] == 0 and opp_count[ci] == 0:
                # Play smallest available size in this cell
                for size in _SIZES_ORDER:
                    mv = (r, c, size)
                    if mv in legal_set:
                        return mv

        # ── Step 7: Anti-diagonal offense ──
        for r, c, sz in _ANTIDIAG_S:
            if (r, c, sz) in legal_set:
                return (r, c, sz)

        # ── Step 8: Any legal move ──
        return legal[0]  # pragma: no cover — steps 4-7 cover all realistic game states
