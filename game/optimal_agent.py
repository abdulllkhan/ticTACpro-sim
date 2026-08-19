"""
Rule-based agent for TicTacPro (heuristic — NOT game-theoretically optimal).

§332: TicTacPro is a second-player win; BLUE wins with perfect play.
This agent implements a strong RED anti-diagonal heuristic that beats random and
weak opponents, but loses to minimax/MCTS BLUE because the anti-diagonal plan
can be countered by a sequential bullseye fork strategy.

Uses rule-based logic: win > block (bullseye priority) > plan > fallback.
No neural network inference required.

Usage:
    from game.optimal_agent import OptimalAgent
    agent = OptimalAgent()
    action = agent.get_action(game)  # returns (row, col, PieceSize)
"""

from game.tictacpro import (TicTacPro, Player, PieceSize,
                            _CELL_SLOT, _CELLS_IDX, _SIZES_IDX, _WIN_LINES)

# Anti-diagonal cells for the Small-piece win (winning line)
_ANTIDIAG_S = [(0, 2, PieceSize.SMALL), (1, 1, PieceSize.SMALL), (2, 0, PieceSize.SMALL)]
# Just the cell positions (without size)
_ANTIDIAG_CELLS = [(0, 2), (1, 1), (2, 0)]

# RED's planned sequence in priority order (on-plan moves)
_RED_PLAN = [
    (0, 1, PieceSize.LARGE),   # TC-L  — opening
    (0, 1, PieceSize.MEDIUM),  # TC-M
    (2, 0, PieceSize.MEDIUM),  # BL-M
    (1, 1, PieceSize.LARGE),   # CC-L
    (2, 2, PieceSize.MEDIUM),  # BR-M
    (2, 2, PieceSize.LARGE),   # BR-L
    (1, 1, PieceSize.SMALL),   # CC-S  ← anti-diag 1/3
    (0, 2, PieceSize.SMALL),   # TR-S  ← anti-diag 2/3
    (2, 0, PieceSize.SMALL),   # BL-S  ← anti-diag 3/3  WIN
]


class OptimalAgent:
    """
    Rule-based agent implementing the anti-diagonal Small-piece win.
    Works correctly for both RED (first) and BLUE (second) player.

    Decision priority:
    1. Win immediately if a winning move is available
    2. Block opponent's immediate winning move — bullseye blocks take priority
       over line blocks (§333: previous single-block-mv missed bullseye threats
       when a lower-priority line block was found first in scan order)
    3. Deny anti-diagonal S cells to the opponent if they have 2/3 already
       (preemptive blocking of the dominant 3-step threat)
    4. Execute the anti-diagonal plan (CC-S, TR-S, BL-S)
    5. Follow the pre-planned sequence (_RED_PLAN)
    6. Any legal move

    NOTE: §332 — TicTacPro is a second-player win. This agent loses to perfect
    BLUE play (minimax/MCTS) due to the sequential bullseye fork strategy.
    """

    def get_action(self, game: TicTacPro, player=None, epsilon: float = 0.0) -> tuple:
        me     = game.current_player
        me_int = int(me)
        opp_int = 3 - me_int
        opp = Player.BLUE if me_int == 1 else Player.RED

        board  = game.board
        bf     = board.tobytes()  # single tobytes for the whole method
        pr     = game.pieces_remaining[me]
        pr_opp = game.pieces_remaining[opp]

        # Single-pass scan: find first win (immediate return), all blocks, and legal list.
        # §333: track bullseye blocks (same-cell, i1//3==i2//3) separately from line blocks.
        # Bullseye blocks take priority because they prevent the "sequential bullseye fork"
        # strategy used by minimax BLUE — line blocks miss bullseye threats found later in
        # scan order when block_mv is already set.
        bullseye_block = None  # highest priority block: opponent completing all 3 sizes in a cell
        line_block     = None  # standard 3-in-a-row block
        legal          = []

        for sz_idx, size in _SIZES_IDX:
            if pr[size] == 0:
                continue
            csl_sz = _CELL_SLOT[sz_idx]
            wl_sz  = _WIN_LINES[sz_idx]
            has_opp_sz = pr_opp[size] > 0
            for ci, (r, c) in _CELLS_IDX:
                if bf[csl_sz[ci]]:
                    continue  # slot occupied
                legal.append((r, c, size))
                for i1, i2 in wl_sz[ci]:
                    v1 = bf[i1]
                    v2 = bf[i2]
                    if v1 == me_int and v2 == me_int:
                        return (r, c, size)
                    if has_opp_sz and v1 == opp_int and v2 == opp_int:
                        if i1 // 3 == i2 // 3:  # same cell → bullseye threat
                            if bullseye_block is None:
                                bullseye_block = (r, c, size)
                        elif line_block is None:
                            line_block = (r, c, size)

        if not legal:  # pragma: no cover — defensive guard; game always has legal moves before end
            return None
        block_mv = bullseye_block if bullseye_block is not None else line_block
        if block_mv is not None:
            return block_mv

        # Steps 3-5 need set membership — build lazily (only reached when no win/block)
        legal_set = set(legal)

        # 3. Deny anti-diagonal S cells when opponent has 2/3 already
        # Guard: only block if opponent still has Small pieces to complete the threat.
        opp_antidiag = (board[0,2,0]==opp_int) + (board[1,1,0]==opp_int) + (board[2,0,0]==opp_int)
        if opp_antidiag >= 2 and pr_opp[PieceSize.SMALL] > 0:
            for r, c, sz in _ANTIDIAG_S:  # pragma: no cover — pre-empted by step 2 immediate block
                if (r, c, sz) in legal_set and board[r, c, 0] == 0:
                    return (r, c, sz)

        # 4. Anti-diagonal Small pieces (the winning plan)
        for r, c, sz in _ANTIDIAG_S:
            if (r, c, sz) in legal_set:
                return (r, c, sz)

        # 5. Follow the planned sequence in order
        for r, c, sz in _RED_PLAN:
            if (r, c, sz) in legal_set:
                return (r, c, sz)

        # 6. Any legal move
        return legal[0]  # pragma: no cover — 9-move plan covers all piece types; unreachable in practice
