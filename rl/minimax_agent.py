"""
Minimax agent with alpha-beta pruning and transposition table for TicTacPro.

Performance-focused: uses numpy for fast evaluation, lightweight push/undo,
and iterative deepening so any time budget produces the best move found so far.
"""

import time
import numpy as np
from typing import Optional, Tuple, Dict

from game.tictacpro import TicTacPro, Player, PieceSize, would_win, _would_win_bf, _SWITCH_PLAYER, _SIZES

WIN  =  1_000_000
LOSS = -1_000_000
DRAW = 0

EXACT = 0
LOWER = 1
UPPER = 2

# ── Pre-computed line structures ──────────────────────────────────────────────
_LINES_LIST = [
    [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],
    [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],
    [(0,0),(1,1),(2,2)], [(0,2),(1,1),(2,0)],
]
# Numpy indices: shape (8, 3, 2) — [line_idx, cell_idx, (row/col)]
_LINE_RC  = np.array([[(r,c) for r,c in line] for line in _LINES_LIST], dtype=np.int8)
_LINE_R   = _LINE_RC[:, :, 0]   # (8,3)
_LINE_C   = _LINE_RC[:, :, 1]   # (8,3)

# Per-size line-threat weights for _evaluate (§110).
# Small pieces participate in more winning mechanisms (diagonal, anti-diagonal,
# rows, cols) and are scarcer (3 per player), so 2-of-3 Small threats are worth
# more than Medium or Large.  The diagonal bonuses (500/300 pts) are additive.
_SZ_THREAT_W  = np.array([70, 50, 30], dtype=np.int32)  # 2-of-3: [S, M, L]
_SZ_THREAT_W1 = np.array([14, 10,  6], dtype=np.int32)  # 1-of-3: [S, M, L]

# ── Push / undo (in-place, avoids clone) ─────────────────────────────────────
def _push(g: TicTacPro, r: int, c: int, sz: PieceSize, _bf: bytes = None) -> tuple:
    cur    = g.current_player
    player = int(cur)
    sz_idx = sz - 1

    prev = (g.winner, g.game_over, cur)
    if _bf is None:
        _bf = g.board.tobytes()
    wins = _would_win_bf(_bf, r, c, sz_idx, player)   # check BEFORE placement
    g.board[r, c, sz_idx] = player
    g.pieces_remaining[cur][sz] -= 1
    g._pieces_left -= 1

    if wins:
        g.winner    = cur
        g.game_over = True
    elif g._pieces_left == 0:
        g.game_over = True
    g.current_player = _SWITCH_PLAYER[cur]  # always switch (negamax invariant)

    return (r, c, sz_idx, player, prev)


def _pop(g: TicTacPro, tok: tuple) -> None:
    r, c, sz_idx, player, (prev_winner, prev_game_over, prev_player) = tok
    g.board[r, c, sz_idx] = 0
    g.pieces_remaining[prev_player][_SIZES[sz_idx]] += 1   # prev_player == the mover
    g._pieces_left += 1
    g.winner         = prev_winner
    g.game_over      = prev_game_over
    g.current_player = prev_player


# Anti-diagonal cells — the empirically dominant winning line
_ANTIDIAG = [(0, 2), (1, 1), (2, 0)]    # TR, CC, BL
_MAINDIAG  = [(0, 0), (1, 1), (2, 2)]   # TL, CC, BR


# ── Vectorised evaluation (v2 — strategy-aware) ───────────────────────────────
def _evaluate(g: TicTacPro, current_player: int) -> int:
    """
    Heuristic from current_player's perspective (§110 — size-weighted threats).

    Line threat scores are weighted by piece size (Small > Medium > Large):
      - Small 2-of-3: 70 pts  (scarcer, participates in more winning lines)
      - Medium 2-of-3: 50 pts
      - Large 2-of-3: 30 pts
    Anti-diagonal S and main-diagonal S get additional large bonuses (additive).
    Bullseye setup and fork penalties retained.
    """
    board = g.board           # (3,3,3) int8
    opp   = 3 - current_player

    score = 0

    # 1. General same-size line threats (all 8 lines × 3 sizes, batched).
    # board[_LINE_R, _LINE_C] → (8,3,3): [line, cell_in_line, size]
    # me_cnt / them_cnt shape: (8,3) — [line, size].
    lines_all = board[_LINE_R, _LINE_C]
    me_cnt    = (lines_all == current_player).sum(axis=1)  # (8,3)
    them_cnt  = (lines_all == opp).sum(axis=1)             # (8,3)
    no_them   = them_cnt == 0
    no_me     = me_cnt   == 0

    me_line_2   = (me_cnt   == 2) & no_them
    them_line_2 = (them_cnt == 2) & no_me
    me_open_1   = (me_cnt   == 1) & no_them
    them_open_1 = (them_cnt == 1) & no_me

    # §110: per-size weights — Small/Medium/Large broadcasted over (8,3) bool arrays.
    score += int((me_line_2   * _SZ_THREAT_W ).sum())
    score += int((me_open_1   * _SZ_THREAT_W1).sum())
    score -= int((them_line_2 * _SZ_THREAT_W ).sum())
    score -= int((them_open_1 * _SZ_THREAT_W1).sum())

    # 2. Anti-diagonal S (line 7 in _LINES_LIST) and Main-diagonal S (line 6).
    # Reuse lines_all already computed above — avoids redundant board accesses.
    ad_small = lines_all[7, :, 0]   # (3,): anti-diag small pieces
    md_small = lines_all[6, :, 0]   # (3,): main-diag small pieces
    ad_me  = int((ad_small == current_player).sum())
    ad_opp = int((ad_small == opp).sum())
    md_me  = int((md_small == current_player).sum())
    md_opp = int((md_small == opp).sum())

    if ad_opp == 0:
        score += (500 if ad_me == 2 else 100 if ad_me == 1 else 0)
    if ad_me == 0:
        score -= (500 if ad_opp == 2 else 100 if ad_opp == 1 else 0)
    if md_opp == 0:
        score += (300 if md_me == 2 else 60 if md_me == 1 else 0)
    if md_me == 0:
        score -= (300 if md_opp == 2 else 60 if md_opp == 1 else 0)

    # 4. Bullseye setup — vectorized
    cell_me  = (board == current_player).sum(axis=2)  # (3,3)
    cell_opp = (board == opp).sum(axis=2)             # (3,3)
    me_bull_2   = (cell_me  == 2) & (cell_opp == 0)
    them_bull_2 = (cell_opp == 2) & (cell_me  == 0)
    me_bull_1   = (cell_me  == 1) & (cell_opp == 0)   # early beachhead
    them_bull_1 = (cell_opp == 1) & (cell_me  == 0)
    score += int(80 * me_bull_2.sum())
    score -= int(80 * them_bull_2.sum())
    score += int(25 * me_bull_1.sum())
    score -= int(25 * them_bull_1.sum())

    # 5. Fork detection — ≥2 simultaneous 2-of-3 threats is decisive.
    #    me_fork checked FIRST: if current player has a fork they win next move
    #    regardless of opponent threats.  them_fork checked second: if opponent
    #    has a fork after current player moves, current player can only block one.
    #    Early return (§124) overrides the rest of the heuristic; without this,
    #    large diagonal bonuses (500 pts) mask opponent forks (300 pts penalty).
    me_fork   = int(me_line_2.sum())   + int(me_bull_2.sum())
    them_fork = int(them_line_2.sum()) + int(them_bull_2.sum())
    if me_fork >= 2:
        return 200_000    # current player has fork; strong but not falsely decisive (§128)
    if them_fork >= 2:
        return -200_000   # opponent fork; allow search to see counter-play

    # 6. Pre-fork beachhead penalty — opponent occupying 2+ uncontested cells
    #    with 1-of-3 pieces signals a dual-fork setup within 2 moves.  Each
    #    extra beachhead beyond the first adds a moderate penalty so the search
    #    actively avoids letting the opponent build multiple simultaneous threats
    #    (§120: detected BL-M+TL-M dual-beachhead as a known forced-win setup).
    n_them_bull1 = int(them_bull_1.sum())
    n_me_bull1   = int(me_bull_1.sum())
    if n_them_bull1 >= 2:
        score -= 45 * (n_them_bull1 - 1)
    if n_me_bull1 >= 2:
        score += 45 * (n_me_bull1 - 1)

    # 7. Multi-cell line control: opponent has pieces (any size) in 2+ cells of a
    #    winning line while I have 0 in that line — latent fork threat.  Distinct from
    #    §1 (same-size 2-of-3): here, different-size pieces across adjacent cells signal
    #    that the opponent can stack sizes later to create simultaneous threats.
    #    (§124: CC-L→TL-M→BL-S gave RED pieces in TL and BL of col-0 at +35 pts each.)
    me_any   = (board == current_player).any(axis=2)           # (3,3) bool
    opp_any  = (board == opp).any(axis=2)                      # (3,3) bool
    me_lc    = me_any[_LINE_R,  _LINE_C].sum(axis=1)           # (8,) cells with ≥1 me piece
    opp_lc   = opp_any[_LINE_R, _LINE_C].sum(axis=1)           # (8,) cells with ≥1 opp piece
    opp_2cl  = int(((opp_lc >= 2) & (me_lc == 0)).sum())
    me_2cl   = int(((me_lc  >= 2) & (opp_lc == 0)).sum())
    score += 40 * me_2cl
    score -= 40 * opp_2cl

    return score


# ── Move ordering ─────────────────────────────────────────────────────────────
# Center > corners > edges; Large > Medium > Small
_POS_SCORE = {(1,1):40, (0,0):20,(0,2):20,(2,0):20,(2,2):20}
_SZ_SCORE  = {PieceSize.SMALL:15, PieceSize.MEDIUM:10, PieceSize.LARGE:5}  # small wins more lines
# Anti-diagonal S cells — flat byte indices for board[r,c,0] = bf[r*9+c*3].
# Using bytes indexing avoids numpy scalar creation overhead vs board[r,c,0].
_AD_CELLS       = {(0,2), (1,1), (2,0)}
_AD_SMALL_IDX   = frozenset(r*9+c*3 for r, c in _AD_CELLS)  # flat byte idx of AD Small slots

def _order_moves(g: TicTacPro, moves, player: Player, tt_move=None, killers=None,
                 _bf: bytes = None):
    p_int      = int(player)
    opp        = 3 - p_int
    opp_player = _SWITCH_PLAYER[player]
    bf         = _bf if _bf is not None else g.board.tobytes()
    # Count how many AD Small slots already belong to the current player — bytes
    # indexing (~10ns) is faster than numpy board[r,c,0] (~60ns per access).
    ad_me   = sum(1 for fi in _AD_SMALL_IDX if bf[fi] == p_int)
    scored  = []
    for mv in moves:
        r, c, sz = mv
        sz_idx = sz - 1
        if tt_move and mv == tt_move:
            pri = 10_000_000
        elif _would_win_bf(bf, r, c, sz_idx, p_int):
            pri = 9_000_000
        elif (g.pieces_remaining[opp_player][sz] > 0
              and _would_win_bf(bf, r, c, sz_idx, opp)):
            pri = 8_000_000
        elif killers and mv in killers:
            pri = 7_500_000
        else:
            base = _POS_SCORE.get((r, c), 0) + _SZ_SCORE.get(sz, 0)
            antidiag_bonus = 0
            fi_s = r*9 + c*3   # flat byte index for this cell's Small slot
            if sz_idx == 0 and fi_s in _AD_SMALL_IDX and bf[fi_s] == 0:
                antidiag_bonus = 35 * (ad_me + 1)
            # §129 anti-bullseye: contest a cell where opponent already has ≥1 piece.
            # Occupying the remaining slot denies opponent's bullseye completion.
            # §133: only apply when I have NO piece in this cell yet — if I already
            # occupy any slot there, the bullseye is already denied for the opponent
            # and the bonus is double-counted (e.g. CC-S getting +200 when BLUE already
            # has CC-M, making CC-L unreachable for RED).
            fi0 = r*9 + c*3
            opp_in_cell = (bf[fi0]==opp) + (bf[fi0+1]==opp) + (bf[fi0+2]==opp)
            me_in_cell  = (bf[fi0]==p_int) + (bf[fi0+1]==p_int) + (bf[fi0+2]==p_int)
            anti_bull = (200 * opp_in_cell
                         if opp_in_cell > 0 and me_in_cell == 0 and bf[fi0+sz_idx]==0
                         else 0)
            pri = base + antidiag_bonus + anti_bull
        scored.append((pri, mv))
    scored.sort(key=lambda x: -x[0])
    return [mv for _, mv in scored]


# ── State hash ────────────────────────────────────────────────────────────────
def _hash_bf(bf: bytes, player_int: int) -> int:
    return hash((bf, player_int))


def _hash(g) -> int:
    return hash((g.board.tobytes(), int(g.current_player)))


# ── Opening book — first N moves following the anti-diagonal plan ─────────────
# Format: (board_pieces_placed, player) → preferred move in order
# The book is consulted when total pieces on board ≤ BOOK_DEPTH.
RED_BOOK_DEPTH  = 10  # RED gets book for first 6 moves (pp 0, 2, 4, 6, 8, 10)
BLUE_BOOK_DEPTH = 1   # BLUE gets book for first move only (pp 1); search handles pp≥3

# RED opening: exact 6-move winning sequence from TC-L, extracted via exact game-tree solve.
# Win mechanism: TL-S (pp=6) threatens col-0-S (with BL-S already at pp=4), forcing BLUE to block ML-S.
# Then TL-M (pp=8) creates a dual fork: bullseye at TL (TL-S+TL-M+TL-L) vs main-diag-M (TL-M+CC-M+BR-M).
# BLUE can only block one; RED wins the other on pp=10.
# Used when BLUE does NOT open with CC-L.
_RED_BOOK_PLAN = [
    (0, 1, PieceSize.LARGE),    # TC-L  (pp=0)  — strong column-1 opening
    (1, 1, PieceSize.MEDIUM),   # CC-M  (pp=2)  — center M; sets up main-diag-M fork
    (2, 0, PieceSize.SMALL),    # BL-S  (pp=4)  — anti-diag anchor; col-0-S 1/3
    (0, 0, PieceSize.SMALL),    # TL-S  (pp=6)  — col-0-S 2/3 threat; forces BLUE to block ML-S
    (0, 0, PieceSize.MEDIUM),   # TL-M  (pp=8)  — creates fork: bullseye TL vs main-diag-M
    (0, 0, PieceSize.LARGE),    # TL-L  (pp=10) — bullseye win at TL (if BLUE blocked main-diag-M)
    (2, 2, PieceSize.MEDIUM),   # BR-M  — fallback: main-diag-M win (TL-M+CC-M+BR-M)
    (0, 2, PieceSize.SMALL),    # TR-S  — fallback
    (2, 2, PieceSize.LARGE),    # BR-L  — fallback
]

# RED counter-plan when BLUE opens with CC-L (e.g. MCTS with corrected rollout).
# Exact solve from TC-L(RED) → CC-L(BLUE) proves RED wins at pp=8 via top-row-L fork.
# TC-S (pp=2) threatens TC bullseye (TC-L+TC-S+TC-M), forcing BLUE to place TC-M.
# TR-S (pp=4) anchors TR cell. TR-L (pp=6) creates dual fork: TR bullseye (needs TR-M)
# vs top-row-L (TC-L+TR-L, needs TL-L). BLUE blocks one; RED wins with the other at pp=8.
_RED_BOOK_PLAN_VS_CCL = [
    (0, 1, PieceSize.LARGE),    # TC-L  (pp=0)  — opening
    (0, 1, PieceSize.SMALL),    # TC-S  (pp=2)  — threatens TC bullseye; forces BLUE TC-M
    (0, 2, PieceSize.SMALL),    # TR-S  (pp=4)  — anchors TR; builds top-row-S
    (0, 2, PieceSize.LARGE),    # TR-L  (pp=6)  — dual fork: TR bullseye vs top-row-L
    (0, 0, PieceSize.LARGE),    # TL-L  (pp=8)  — top-row-L win (TL-L+TC-L+TR-L)
]

# RED counter-plan when BLUE opens with CC-M.
# Exact solve from TC-L(RED) → CC-M(BLUE) proves RED wins at pp=8 via TC bullseye.
# CC-S (pp=2) + CC-L (pp=4) build TC cell anchor. TC-S (pp=6) threatens TC bullseye
# (TC-L+TC-S already placed). TC-M (pp=8) wins if BLUE hasn't blocked; or the
# immediate-win check fires earlier if BLUE misses either TC-M or TC-S block.
_RED_BOOK_PLAN_VS_CCM = [
    (0, 1, PieceSize.LARGE),    # TC-L  (pp=0)  — opening
    (1, 1, PieceSize.SMALL),    # CC-S  (pp=2)  — contests CC; builds CC anchor
    (1, 1, PieceSize.LARGE),    # CC-L  (pp=4)  — CC anchor (CC-S+CC-L); forces BLUE to focus
    (0, 1, PieceSize.SMALL),    # TC-S  (pp=6)  — threatens TC bullseye (TC-L+TC-S+TC-M)
    (0, 1, PieceSize.MEDIUM),   # TC-M  (pp=8)  — TC bullseye win (TC-S+TC-M+TC-L)
]

# BLUE counter-book: contest CC immediately, then pursue anti-diagonal.
# Priority order: CC-S (best), CC-M (if RED took CC-S — blocks CC bullseye), CC-L (if both taken).
# Contesting CC prevents MCTS from building a CC bullseye win (S+M+L same cell, same player).
# After CC is contested, anti-diagonal gives strong winning threats.
_BLUE_BOOK_PLAN = [
    (1, 1, PieceSize.SMALL),    # CC-S  ← take center + block both diagonals for RED
    (1, 1, PieceSize.MEDIUM),   # CC-M  ← if RED took CC-S: poisons CC bullseye (RED can't win there)
    (1, 1, PieceSize.LARGE),    # CC-L  ← if RED took CC-S + CC-M: still deny the cell
    (0, 2, PieceSize.SMALL),    # TR-S  ← anti-diag 2/3 (threatens BL-S for win)
    (2, 0, PieceSize.SMALL),    # BL-S  ← anti-diag win
    (0, 0, PieceSize.SMALL),    # TL-S  ← start main-diag
    (2, 2, PieceSize.SMALL),    # BR-S  ← main-diag win
    (0, 1, PieceSize.LARGE),    # TC-L  — column staking
    (2, 1, PieceSize.LARGE),    # BC-L  — bottom row control
]

# BLUE counter-book against CC-L opener (§124).
# MCTS consistently opens CC-L and follows with a BL-M fork: TL-M → BL-S → TR-M (force TC-M
# block) → BL-M (creates BL-bullseye 2/3 + col-0-M 2/3 simultaneously — two unblockable threats).
# Fix: BLUE poisons TL-bullseye with TL-S, then occupies BL-M to block the fork entirely.
#   CC-S (contest center) → TL-S (poison TL bullseye + block col-0-S) →
#   BL-M (block col-0-M + BL bullseye) → TR-S (anti-diagonal) → BL-S → ...
_BLUE_BOOK_VS_CCL = [
    (1, 1, PieceSize.MEDIUM),   # CC-M  — poisons CC bullseye AND denies center Medium (§127)
                                #  Search-verified: CC-M gives score≈-122 vs CC-S gives -500000.
    (0, 2, PieceSize.SMALL),    # TR-S  — anti-diagonal anchor (fallback if CC-M taken)
    (2, 0, PieceSize.SMALL),    # BL-S  — anti-diagonal win
    (0, 0, PieceSize.SMALL),    # TL-S  — corner / main-diag
    (2, 2, PieceSize.SMALL),    # BR-S  — corner / main-diag
    (0, 1, PieceSize.LARGE),    # TC-L  — column staking
    (2, 1, PieceSize.LARGE),    # BC-L  — bottom row control
]


def _book_move(g: TicTacPro, player: int) -> Optional[Tuple]:
    """
    Return the next in-book move for `player`.
    Priority: immediate win → forced 1-move block → plan.
    """
    legal = set(g.get_legal_moves())
    bf  = g.board.tobytes()
    opp = 3 - player
    # Immediate win
    for mv in legal:
        r, c, sz = mv
        if _would_win_bf(bf, r, c, sz - 1, player):
            return mv
    # Forced 1-move block — only if opponent still has pieces of that size
    opp_player = _SWITCH_PLAYER[Player(player)]
    for mv in legal:
        r, c, sz = mv
        if g.pieces_remaining[opp_player][sz] > 0 and _would_win_bf(bf, r, c, sz - 1, opp):
            return mv
    # Player-specific book plan
    board    = g.board
    blue_int = int(Player.BLUE)
    red_int  = int(Player.RED)
    if player == red_int:
        # All counter-plans below were designed for the TC-L opener (pp=0).
        # §132: if RED never played TC-L, the plans don't apply — fall to search.
        red_has_tcl = (board[0, 1, 2] == red_int)
        # Select counter-plan based on what BLUE played at CC, guarding against
        # switching plans mid-sequence by checking RED has not yet committed.
        if board[1, 1, 2] == blue_int and board[1, 1, 1] != red_int:
            # BLUE has CC-L and RED hasn't played CC-M → use top-row-L fork plan
            if not red_has_tcl:
                return None
            plan = _RED_BOOK_PLAN_VS_CCL
        elif board[1, 1, 1] == blue_int:
            # BLUE has CC-M → TC-bullseye plan (only valid when RED opened TC-L)
            if not red_has_tcl:
                return None
            plan = _RED_BOOK_PLAN_VS_CCM
        else:
            # §131: _RED_BOOK_PLAN's fork (TL-bullseye + main-diag-M) is only proven
            # when BLUE stays passive at non-center cells.  Against any non-CC BLUE
            # opener, the book's CC-M at pp=2 scores -999,993 (loses) while search
            # finds wins (+999,993) via CC-L/TC-S for every tested BLUE opener.
            # §136: BLUE CC-S (center-Small) does NOT disrupt the plan — CC-M remains
            # correct since CC-M slot is free and the TL-bullseye + diag-M fork is
            # intact.  Only fall to search when BLUE occupies a non-center cell.
            if int((g.board != 0).sum()) > 0 and board[1, 1, 0] != blue_int:
                return None
            plan = _RED_BOOK_PLAN
    else:
        pp = int((g.board != 0).sum())
        # §124: CC-L counter-book — if RED has CC-L anywhere in the game, switch to
        # defensive plan that poisons TL-bullseye and blocks the BL-M fork.
        if board[1, 1, 2] == red_int:
            plan = _BLUE_BOOK_VS_CCL
        # §119: if RED's first move is a corner with Medium, contest that corner's Small
        # slot immediately to poison the bullseye and deny the corner-M fork.
        elif pp == 1:
            for cr, cc in ((0, 0), (0, 2), (2, 0), (2, 2)):
                if board[cr, cc, 1] == red_int:   # RED opened corner with Medium
                    counter = (cr, cc, PieceSize.SMALL)
                    if counter in legal:
                        return counter
            plan = _BLUE_BOOK_PLAN
        elif pp == 3:
            # §125: Diagonal endpoint fork prevention — if RED has pieces at both
            # main- or anti-diagonal corner endpoints, block Medium at the endpoint
            # that holds only a Small.  RED plans to stack M there to create
            # bullseye 2/3 + same-size diagonal 2/3 simultaneously (a double threat
            # that BLUE cannot block in one move once established at pp=5).
            for (r1, c1), (r2, c2) in [((0, 0), (2, 2)), ((0, 2), (2, 0))]:
                red1 = any(board[r1, c1, si] == red_int for si in range(3))
                red2 = any(board[r2, c2, si] == red_int for si in range(3))
                if not (red1 and red2):
                    continue
                # Find which endpoint holds only Small (ready to grow into fork)
                only_s1 = (board[r1,c1,0]==red_int and board[r1,c1,1]==0 and board[r1,c1,2]==0)
                only_s2 = (board[r2,c2,0]==red_int and board[r2,c2,1]==0 and board[r2,c2,2]==0)
                target = None
                if only_s2 and not only_s1:
                    target = (r2, c2)
                elif only_s1 and not only_s2:
                    target = (r1, c1)
                if target:
                    counter_m = (target[0], target[1], PieceSize.MEDIUM)
                    if counter_m in legal:
                        return counter_m
            plan = _BLUE_BOOK_PLAN
        else:
            plan = _BLUE_BOOK_PLAN
    for mv in plan:
        if mv not in legal:
            continue
        # For BLUE: skip CC-L (and any non-S center move) if BLUE already has CC-S,
        # so the 3rd book slot advances to TR-S rather than re-stacking the center.
        r, c, sz = mv
        if (player == blue_int and (r, c) == (1, 1)
                and sz != PieceSize.SMALL
                and board[1, 1, 0] == blue_int):
            continue
        return mv
    return None


# ── Minimax agent ─────────────────────────────────────────────────────────────
class MinimaxAgent:
    """
    Iterative-deepening negamax with alpha-beta pruning and transposition table.

    Uses an opening book for the first BOOK_DEPTH moves (anti-diagonal plan),
    then switches to full search.  This gives strong play in phases where the
    branching factor makes exhaustive search infeasible.

    API: get_action(game, player=None, epsilon=0.0) → (row, col, PieceSize).
    """

    def __init__(self, time_limit: float = 2.0, max_depth: int = 18,
                 use_book: bool = True):
        self.time_limit = time_limit
        self.max_depth  = max_depth
        self.use_book   = use_book

    def get_action(self, game: TicTacPro, player: Player = None,
                   epsilon: float = 0.0) -> Optional[Tuple]:
        if game.game_over:
            return None
        if player is None:
            player = game.current_player
        # Opening book for both colors
        if self.use_book:
            pieces_placed = int((game.board != 0).sum())
            p_int = int(player)
            depth = RED_BOOK_DEPTH if p_int == int(Player.RED) else BLUE_BOOK_DEPTH
            if pieces_placed <= depth:
                bm = _book_move(game, p_int)
                if bm is not None:
                    return bm
        g = game.clone(copy_history=False)
        action, _ = self._id_search(g, player)
        return action

    def get_info(self, game: TicTacPro, player: Player = None):
        if game.game_over:
            return None, (0, DRAW, 0, 0)
        if player is None:
            player = game.current_player
        if self.use_book:
            pieces_placed = int((game.board != 0).sum())
            p_int = int(player)
            depth = RED_BOOK_DEPTH if p_int == int(Player.RED) else BLUE_BOOK_DEPTH
            if pieces_placed <= depth:
                bm = _book_move(game, p_int)
                if bm is not None:
                    return bm, (0, 0, 0, 0)
        g = game.clone(copy_history=False)
        return self._id_search(g, player)

    # ── Iterative deepening ───────────────────────────────────────────────────
    _ASP_DELTA = 500  # aspiration window half-width; covers heuristic eval range for draws

    def _id_search(self, g, max_player):
        self._tt:      Dict = {}
        self._killers: list = [[] for _ in range(self.max_depth + 2)]
        self._nodes = 0
        self._hits  = 0
        self._dl    = time.perf_counter() + self.time_limit
        self._mp    = int(max_player)

        moves = g.get_legal_moves()
        if not moves:           return None, (0, DRAW, 0, 0)
        if len(moves) == 1:     return moves[0], (1, 0, 1, 0)

        best_action = moves[0]
        best_score  = LOSS
        depth_done  = 0
        prev_score  = DRAW

        for depth in range(1, self.max_depth + 1):
            try:
                if depth <= 2 or abs(prev_score) >= WIN // 2:
                    score, action = self._root(g, depth, LOSS - 1, WIN + 1)
                else:
                    # Aspiration window: search narrow band around previous score
                    a = max(LOSS - 1, prev_score - self._ASP_DELTA)
                    b = min(WIN + 1,  prev_score + self._ASP_DELTA)
                    score, action = self._root(g, depth, a, b)
                    if score <= a:   # fail-low: widen down
                        score, action = self._root(g, depth, LOSS - 1, b)
                    elif score >= b:  # fail-high: widen up
                        score, action = self._root(g, depth, a, WIN + 1)
                best_action = action
                best_score  = score
                prev_score  = score
                depth_done  = depth
                if abs(score) >= WIN - 100_000:  # only for actual terminal wins, not fork heuristic
                    break
            except _TimeUp:
                break

        return best_action, (depth_done, best_score, self._nodes, self._hits)

    def _root(self, g, depth, alpha=LOSS-1, beta=WIN+1):
        best_score  = LOSS-1
        best_action = None
        bf_root     = g.board.tobytes()   # shared by TT lookup, _order_moves, and all _push calls at root
        h_root      = _hash_bf(bf_root, int(g.current_player))
        tt_move     = self._tt.get(h_root, (0,0,0,None))[3]
        moves       = _order_moves(g, g.get_legal_moves(), g.current_player, tt_move,
                                   _bf=bf_root)
        orig_alpha  = alpha
        for mv in moves:
            if time.perf_counter() > self._dl:
                raise _TimeUp()
            tok = _push(g, *mv, bf_root)
            score = -self._negamax(g, depth-1, -beta, -alpha, 1)
            _pop(g, tok)
            if score > best_score:
                best_score  = score
                best_action = mv
            alpha = max(alpha, score)
            # Cut off when aspiration window is exceeded (fail-high) or a forced win
            # is found — no remaining move can improve further.
            if alpha >= beta or best_score >= WIN - 100_000:  # only for terminal wins
                break

        # Store root result in TT so the next ID iteration uses previous best for ordering.
        if best_action is not None:
            flag = UPPER if best_score <= orig_alpha else (LOWER if best_score >= beta else EXACT)
            self._tt[h_root] = (best_score, flag, depth, best_action)
        return best_score, best_action

    # LMR constants: apply reduction to moves after the first _LMR_IDX at depth >= _LMR_DEPTH.
    _LMR_IDX   = 3   # first 3 moves (win/block/TT/killer) always searched at full depth
    _LMR_DEPTH = 3   # minimum depth for reduction

    def _negamax(self, g, depth, alpha, beta, ply: int) -> int:
        self._nodes += 1

        if g.game_over:
            if g.winner != Player.NONE:
                return LOSS + ply   # current_player is the loser (always switched)
            return DRAW

        # Evaluate at leaf without computing legal moves (saves O(27) work per leaf node).
        if depth == 0:
            return _evaluate(g, int(g.current_player))

        # No moves for current player but game not flagged over (stuck player)
        moves = g.get_legal_moves()
        if not moves:
            return DRAW

        bf_node     = g.board.tobytes()   # computed once: shared by TT hash, _order_moves, and _push calls
        h           = _hash_bf(bf_node, int(g.current_player))
        tt_move = None
        if h in self._tt:
            self._hits += 1
            ts, tf, td, tt_move = self._tt[h]
            if td >= depth:
                if tf == EXACT: return ts
                if tf == LOWER: alpha = max(alpha, ts)
                if tf == UPPER: beta  = min(beta,  ts)
                if alpha >= beta: return ts

        # Throttled time check
        if self._nodes % 10_000 == 0 and time.perf_counter() > self._dl:
            raise _TimeUp()
        moves_ord   = _order_moves(g, moves, g.current_player, tt_move,
                                   self._killers[ply] if ply < len(self._killers) else None,
                                   _bf=bf_node)
        orig_alpha  = alpha
        best        = LOSS - 1
        best_mv     = None
        do_lmr      = depth >= self._LMR_DEPTH

        for i, mv in enumerate(moves_ord):
            tok = _push(g, *mv, bf_node)
            if do_lmr and i >= self._LMR_IDX and abs(best) < WIN // 2:
                # LMR: null-window search at reduced depth; re-search full depth on cutthrough.
                score = -self._negamax(g, depth - 2, -alpha - 1, -alpha, ply + 1)
                if score > alpha:
                    score = -self._negamax(g, depth - 1, -beta, -alpha, ply + 1)
            else:
                score = -self._negamax(g, depth - 1, -beta, -alpha, ply + 1)
            _pop(g, tok)
            if score > best:
                best    = score
                best_mv = mv
            alpha = max(alpha, score)
            if alpha >= beta:
                if abs(score) < WIN // 2 and ply < len(self._killers):
                    k = self._killers[ply]
                    if mv not in k:
                        k.insert(0, mv)
                        del k[2:]
                break

        flag = UPPER if best <= orig_alpha else (LOWER if best >= beta else EXACT)
        self._tt[h] = (best, flag, depth, best_mv)
        return best


class _TimeUp(Exception):
    pass
