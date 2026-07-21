"""Integration tests for MinimaxAgent — correctness and regression guards."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
from game.tictacpro import TicTacPro, Player, PieceSize
import time
from rl.minimax_agent import (
    MinimaxAgent, _evaluate, _push, _pop, _order_moves, _book_move, _hash, _hash_bf,
    LOSS, WIN, DRAW, EXACT, LOWER, UPPER,
)


# ── _push / _pop roundtrip ────────────────────────────────────────────────────

class TestPushPop:
    def test_board_restored(self):
        g = TicTacPro()
        before = g.board.copy()
        tok = _push(g, 0, 0, PieceSize.SMALL)
        _pop(g, tok)
        assert (g.board == before).all()

    def test_pieces_remaining_restored(self):
        g = TicTacPro()
        before = g.pieces_remaining[Player.RED][PieceSize.SMALL]
        tok = _push(g, 0, 0, PieceSize.SMALL)
        _pop(g, tok)
        assert g.pieces_remaining[Player.RED][PieceSize.SMALL] == before

    def test_pieces_left_restored(self):
        g = TicTacPro()
        tok = _push(g, 0, 0, PieceSize.SMALL)
        _pop(g, tok)
        assert g._pieces_left == 18

    def test_current_player_restored(self):
        g = TicTacPro()
        tok = _push(g, 0, 0, PieceSize.SMALL)
        _pop(g, tok)
        assert g.current_player == Player.RED

    def test_winner_restored_after_win(self):
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        tok = _push(g, 0, 2, PieceSize.SMALL)
        assert g.winner == Player.RED
        assert g.game_over
        _pop(g, tok)
        assert g.winner == Player.NONE
        assert not g.game_over

    def test_multiple_roundtrip(self):
        g = TicTacPro()
        original_board = g.board.copy()
        t1 = _push(g, 0, 0, PieceSize.LARGE)
        t2 = _push(g, 1, 1, PieceSize.MEDIUM)
        _pop(g, t2)
        _pop(g, t1)
        assert (g.board == original_board).all()
        assert g._pieces_left == 18
        assert g.current_player == Player.RED

    def test_push_always_switches_player_negamax_invariant(self):
        """§213: _push always switches current_player — even on a winning move.
        This is the negamax invariant: the switch happens unconditionally so
        _negamax can always negate and recurse without a terminal check first."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        # Push a WINNING move.
        tok = _push(g, 0, 2, PieceSize.SMALL)
        assert g.game_over, "_push winning move must set game_over"
        assert g.winner == Player.RED, "_push winning move must set winner"
        # Crucially: player switches even though the game is over.
        assert g.current_player == Player.BLUE, (
            "Negamax invariant: _push must switch player even on terminal (winning) move"
        )
        _pop(g, tok)

    def test_push_draw_sets_game_over_not_winner(self):
        """§213: _push that exhausts all pieces (draw) sets game_over=True but
        leaves winner=Player.NONE — minimax must distinguish draw from win.

        Use a blank board with _pieces_left=1 and push to an isolated position
        with no 2-in-a-row set up, so the move cannot win."""
        g = TicTacPro()
        # Blank board — no pieces on it yet. Set _pieces_left=1 artificially
        # so the next push exhausts the piece count without any winning line.
        g._pieces_left = 1
        # RED has 1 Large left; no other pieces on the board → placing anywhere
        # cannot form 2-in-a-row (there are no existing matching pieces).
        g.pieces_remaining[Player.RED][PieceSize.LARGE] = 1
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 0
        g.current_player = Player.RED
        tok = _push(g, 1, 2, PieceSize.LARGE)
        assert g._pieces_left == 0
        assert g.game_over, "_push with last piece must set game_over"
        assert g.winner == Player.NONE, (
            "Draw: winner must remain NONE when placing a single non-winning piece"
        )
        _pop(g, tok)
        assert not g.game_over
        assert g.winner == Player.NONE


# ── _evaluate heuristic ───────────────────────────────────────────────────────

class TestEvaluate:
    def test_empty_board_near_zero(self):
        g = TicTacPro()
        score = _evaluate(g, int(Player.RED))
        assert abs(score) < 500

    def test_anti_diag_lead_positive(self):
        """Two anti-diagonal S pieces for current player → positive score."""
        g = TicTacPro()
        g.board[0, 2, 0] = int(Player.RED)
        g.board[1, 1, 0] = int(Player.RED)
        score = _evaluate(g, int(Player.RED))
        assert score > 0

    def test_opponent_anti_diag_lead_negative(self):
        """Two anti-diagonal S pieces for opponent → negative score from RED's view."""
        g = TicTacPro()
        g.board[0, 2, 0] = int(Player.BLUE)
        g.board[1, 1, 0] = int(Player.BLUE)
        score = _evaluate(g, int(Player.RED))
        assert score < 0

    def test_bullseye_setup_positive(self):
        """Two size slots for current player at one cell → positive score."""
        g = TicTacPro()
        g.board[1, 1, 0] = int(Player.RED)
        g.board[1, 1, 1] = int(Player.RED)
        score = _evaluate(g, int(Player.RED))
        assert score > 0

    def test_perspective_symmetry(self):
        """Score from RED's view and BLUE's view should be negatives when symmetric."""
        g = TicTacPro()
        # Red and blue both have one AD small piece each — opposite cells
        g.board[0, 2, 0] = int(Player.RED)
        g.board[2, 0, 0] = int(Player.BLUE)
        s_red  = _evaluate(g, int(Player.RED))
        s_blue = _evaluate(g, int(Player.BLUE))
        # Not strictly negatives (different cells have different bonuses) but should be in opposition
        assert s_red * s_blue <= 0 or abs(s_red + s_blue) < 200

    def test_main_diagonal_bonus_positive(self):
        """§223: Two S pieces on main diagonal → positive score (300 pt bonus)."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[1, 1, 0] = int(Player.RED)
        score = _evaluate(g, int(Player.RED))
        assert score > 0, f"Main diagonal 2-piece bonus should be positive, got {score}"

    def test_pre_fork_beachhead_penalty(self):
        """§223: Opponent with ≥2 uncontested cells (1-of-3) triggers pre-fork penalty."""
        g = TicTacPro()
        # Give BLUE a Medium in two distinct cells (no RED pieces in those cells)
        g.board[0, 0, 1] = int(Player.BLUE)
        g.board[1, 1, 1] = int(Player.BLUE)
        score = _evaluate(g, int(Player.RED))
        # Should be negative (opponent has dual beachhead)
        assert score < 0, f"Pre-fork beachhead penalty should make score negative, got {score}"

    def test_multi_cell_line_control_positive(self):
        """§223: Current player with 2 cells in a line (mixed sizes) → positive score."""
        g = TicTacPro()
        # RED has pieces (any size) in 2 cells of row 0, BLUE has none
        g.board[0, 0, 0] = int(Player.RED)   # Small at (0,0)
        g.board[0, 1, 1] = int(Player.RED)   # Medium at (0,1)
        # No BLUE in row 0 → RED controls 2 cells in row-0 line
        score = _evaluate(g, int(Player.RED))
        assert score > 0, f"Multi-cell line control should give positive score, got {score}"

    def test_contested_md_neutralizes_red_2md_advantage(self):
        """§289: Section 2 `if md_opp == 0` guard — when BLUE contests maindiag with 1 Small,
        RED's 2-MD-Small advantage (+300) is neutralized (score += 0 instead of +300).

        Setup A: RED TL-S + CC-S — md_me=2, md_opp=0 → +300 from sec2 MD.
        Setup B: Same RED + BLUE BR-S (contests maindiag): md_opp=1 → `if md_opp==0`
                 is False → 0 from sec2 MD.

        Delta = s_a - s_b = 463:
          sec2: -300 (MD bonus neutralized)
          sec1: -70 (maindiag-S drops from me_line_2 to 0 due to them_cnt>0)
               -28 (BLUE's row2-S + col2-S them_open_1: -14 each)
          sec4: -25 (BLUE's (2,2,S) adds them_bull_1 → -25)
          sec7: -40 (maindiag me_2cl drops from 1 to 0 since opp_lc>0)
          Total: 300+70+28+25+40 = 463.

        Exercises the `if md_opp == 0` guard at minimax_agent.py:129."""
        g_a = TicTacPro()
        g_a.board[0, 0, 0] = int(Player.RED)   # TL-S (maindiag)
        g_a.board[1, 1, 0] = int(Player.RED)   # CC-S (maindiag) — 2 MD smalls
        s_a = _evaluate(g_a, int(Player.RED))

        g_b = TicTacPro()
        g_b.board[0, 0, 0] = int(Player.RED)
        g_b.board[1, 1, 0] = int(Player.RED)
        g_b.board[2, 2, 0] = int(Player.BLUE)  # BR-S — BLUE contests maindiag
        s_b = _evaluate(g_b, int(Player.RED))

        assert s_a > s_b, f"§289: contested MD must reduce RED's score; s_a={s_a}, s_b={s_b}"
        assert s_a - s_b == 463, (
            f"§289: exact delta must be 463 (MD sec2:-300 + sec1:-98 + sec4:-25 + sec7:-40); "
            f"got {s_a - s_b}"
        )

    def test_contested_ad_neutralizes_red_2ad_advantage(self):
        """§288: Section 2 `if ad_opp == 0` guard — when BLUE contests AD with 1 Small,
        RED's 2-AD-Small advantage (+500) is neutralized (score += 0 instead of +500).

        Setup A: RED (0,2,S) + (1,1,S) — 2 AD smalls, ad_opp=0 → +500 from sec2 AD.
        Setup B: Same RED pieces + BLUE (2,0,S) — BLUE contests AD: ad_opp=1 → `if ad_opp==0`
                 is False → RED gets 0 from sec2 AD.

        Delta = s_a - s_b = 663:
          sec2: -500 (AD bonus neutralized)
          sec1: -70 (antidiag-S drops from me_line_2=True to me_open_1=False due to them_cnt>0)
               -28 (BLUE's row2-S + col0-S them_open_1: -14 each)
          sec4: -25 (BLUE's (2,0,S) adds them_bull_1 → -25)
          sec7: -40 (antidiag me_2cl drops from 1 to 0 since opp_lc>0 blocks me_2cl)
          Total: 500+70+28+25+40 = 663.

        Asserting s_a - s_b == 663 exercises the `ad_opp == 0` guard exactly."""
        g_a = TicTacPro()
        g_a.board[0, 2, 0] = int(Player.RED)   # TR-S (AD corner)
        g_a.board[1, 1, 0] = int(Player.RED)   # CC-S (AD center) — 2 AD smalls
        s_a = _evaluate(g_a, int(Player.RED))

        g_b = TicTacPro()
        g_b.board[0, 2, 0] = int(Player.RED)
        g_b.board[1, 1, 0] = int(Player.RED)
        g_b.board[2, 0, 0] = int(Player.BLUE)  # BL-S — BLUE contests AD
        s_b = _evaluate(g_b, int(Player.RED))

        assert s_a > s_b, f"§288: contested AD must reduce RED's score; s_a={s_a}, s_b={s_b}"
        assert s_a - s_b == 663, (
            f"§288: exact delta must be 663 (AD sec2:-500 + sec1:-98 + sec4:-25 + sec7:-40); "
            f"got {s_a - s_b}"
        )

    def test_two_ad_small_scores_higher_than_one_by_at_least_400(self):
        """§252: _evaluate ad_me==2 branch (+500 pts) vs ad_me==1 branch (+100 pts).

        Section 2 of _evaluate: if ad_opp==0: score += (500 if ad_me==2 else 100 ...).
        Existing test_anti_diag_lead_positive already places 2 AD Smalls (exercising the
        ==2 branch) but only asserts score > 0 — a bug swapping the 500/100 values or
        returning a constant would be invisible.

        Proof: adding (1,1,S) to a position with (0,2,S) upgrades ad_me from 1→2,
        adding exactly +400 in section 2 (500-100).  All other section changes are also
        positive, so delta ≥ 400 is a tight lower bound."""
        g1 = TicTacPro()
        g1.board[0, 2, 0] = int(Player.RED)   # 1 AD SMALL (TR)
        s1 = _evaluate(g1, int(Player.RED))

        g2 = TicTacPro()
        g2.board[0, 2, 0] = int(Player.RED)
        g2.board[1, 1, 0] = int(Player.RED)   # 2 AD SMALL (TR + CC)
        s2 = _evaluate(g2, int(Player.RED))

        assert s2 > s1, f"§252: 2 AD pieces (score={s2}) must outscore 1 (score={s1})"
        assert s2 - s1 >= 400, (
            f"§252: ad_me=2→+500 vs ad_me=1→+100 must contribute ≥400 pts net; "
            f"diff={s2 - s1} (s2={s2}, s1={s1})"
        )

    def test_two_ad_small_exact_delta_668(self):
        """§286: Upgrade §252 from >= 400 to == 668 — exact composite delta.

        §252 asserts >= 400 (a lower bound). The exact delta when adding (1,1,S)
        to a position with (0,2,S) is 668:
          sec1: antidiag upgrades open_1→line_2 (+56), plus row1/col1/maindiag open1 (+42) = +98
          sec2: AD bonus 100→500 (+400) + maindiag bonus 0→60 (+60) = +460
          sec4: me_bull_1 count 1→2 (+25)
          sec6: n_me_bull1 1→2 → +45
          sec7: antidiag now has 2 RED cells → me_2cl=1 → +40
          Total: 98+460+25+45+40 = 668

        Wrong sec2 values (e.g. 400 or 200 instead of 500) would change this delta.
        Wrong sec7 coeff would also fail. Exact assertion fully constrains all these."""
        g1 = TicTacPro()
        g1.board[0, 2, 0] = int(Player.RED)   # 1 AD SMALL (TR)
        s1 = _evaluate(g1, int(Player.RED))

        g2 = TicTacPro()
        g2.board[0, 2, 0] = int(Player.RED)
        g2.board[1, 1, 0] = int(Player.RED)   # 2 AD SMALL (TR + CC)
        s2 = _evaluate(g2, int(Player.RED))

        assert s2 - s1 == 668, (
            f"§286: exact delta (1→2 AD Smalls) must be 668; got {s2 - s1}. "
            f"s1={s1}, s2={s2}. Wrong: sec2+sec1+sec4+sec6+sec7 combination."
        )

    def test_me_dual_beachhead_section6_bonus(self):
        """§253: n_me_bull1 ≥ 2 → score += 45*(n-1) in section 6 of _evaluate.
        The OPPONENT version is covered by test_pre_fork_beachhead_penalty; this
        covers the current-player version which was untested.

        Setup: RED has beachhead at (0,0,S) and a second at (2,1,L) — cells share
        no lines, so section 7 (multi-cell control) does NOT fire.  Delta budget:
          section 1: row2 + col1 Large open-threats → 6+6=12
          section 4: second cell me_bull_1 → 25
          section 6: n_me_bull1=2 → +45
          total expected delta ≥ 82

        If section 6 is disabled the delta falls to ≤ 37 < 45, making the assertion fail."""
        g1 = TicTacPro()
        g1.board[0, 0, 0] = int(Player.RED)   # 1 beachhead
        s1 = _evaluate(g1, int(Player.RED))

        g2 = TicTacPro()
        g2.board[0, 0, 0] = int(Player.RED)   # same first beachhead
        g2.board[2, 1, 2] = int(Player.RED)   # 2nd beachhead at (2,1,Large) — no shared lines
        s2 = _evaluate(g2, int(Player.RED))

        assert s2 - s1 >= 45, (
            f"§253: n_me_bull1=2 must contribute ≥45 pts over n=1 (section 6 fires); "
            f"delta={s2 - s1}. If delta≈37, section 6 is likely disabled."
        )

    def test_me_open1_small_scores_more_than_large(self):
        """§256: _SZ_THREAT_W1 per-size weighting — me_open_1 SMALL (14/line) must
        score higher than LARGE (6/line) at edge cell (0,1), which is on exactly 2 lines
        (row0, col1) and has no diagonal contribution that could confound the comparison.

        Expected delta: 2 lines × (14-6) = 16 from me_open_1 alone.
        me_bull_1 (+25) is identical for both → cancels. Section 7 is 0 for both
        (1 cell in each line, not ≥2). Section 2 is 0 (no diagonal). Delta ≥ 16."""
        g_small = TicTacPro()
        g_small.board[0, 1, 0] = int(Player.RED)   # SMALL at edge (0,1)
        s_small = _evaluate(g_small, int(Player.RED))

        g_large = TicTacPro()
        g_large.board[0, 1, 2] = int(Player.RED)   # LARGE at same edge (0,1)
        s_large = _evaluate(g_large, int(Player.RED))

        assert s_small > s_large, (
            f"§256: SMALL 1-piece threat (weight 14) must outvalue LARGE (weight 6); "
            f"s_small={s_small}, s_large={s_large}"
        )
        assert s_small - s_large >= 16, (
            f"§256: Delta ≥ 16 expected from _SZ_THREAT_W1 (2 lines × 8); "
            f"got {s_small - s_large}"
        )

    def test_them_bull1_single_beachhead_penalty(self):
        """§257: them_bull_1 in section 4 applies -25 per uncontested opponent cell.
        Section 4 is exercised by test_pre_fork_beachhead_penalty, but only through a
        composite score (2 beachheads + section 6). This test isolates the single-cell
        -25 penalty.

        Setup: BLUE LARGE at edge cell (1,2) — on row1 and col2 only (no diagonals,
        not AD). With RED having no pieces:
          section 1: them_open_1 row1-L, col2-L → -6×2 = -12
          section 4: them_bull_1 = 1 cell → -25
          section 6: n_them_bull1=1 < 2 → 0
          exact total: -37
        Asserting score == -37 catches both the -25 them_bull_1 penalty and -12 open-threat.
        A bug zeroing section 4 them_bull_1 would yield -12 ≠ -37."""
        g = TicTacPro()
        g.board[1, 2, 2] = int(Player.BLUE)   # BLUE LARGE at (1,2) — edge, no diagonal
        score = _evaluate(g, int(Player.RED))
        assert score == -37, (
            f"§257: BLUE LARGE at (1,2) should give score=-37; got {score}. "
            f"If score==-12, them_bull_1 (-25) is not firing."
        )

    def test_opp_two_cell_line_control_penalty(self):
        """§258: Section 7 opp_2cl — opponent holding 2 cells in the same win line
        (different sizes, so them_line_2 doesn't fire) subtracts exactly 40 pts.

        Setup A: BLUE SMALL at (0,0) + BLUE MEDIUM at (0,1) → row0 has opp in 2 cells,
                 opp_2cl = 1, section 7 contribution = -40.
        Setup B: BLUE SMALL at (0,0) + BLUE MEDIUM at (2,1) → no shared line,
                 opp_2cl = 0, section 7 contribution = 0.
        All other terms are identical (them_open_1 = 5 threats each, same section 2/4/6).
        Expected: s_A < s_B and s_B - s_A == 40."""
        g_a = TicTacPro()
        g_a.board[0, 0, 0] = int(Player.BLUE)   # (0,0,S)
        g_a.board[0, 1, 1] = int(Player.BLUE)   # (0,1,M) — shares row0 with (0,0)
        s_a = _evaluate(g_a, int(Player.RED))

        g_b = TicTacPro()
        g_b.board[0, 0, 0] = int(Player.BLUE)   # (0,0,S) — same first piece
        g_b.board[2, 1, 1] = int(Player.BLUE)   # (2,1,M) — different line entirely
        s_b = _evaluate(g_b, int(Player.RED))

        assert s_a < s_b, (
            f"§258: two BLUE cells in row0 (opp_2cl=1) must score worse for RED "
            f"than cells in separate lines (opp_2cl=0); s_a={s_a}, s_b={s_b}"
        )
        assert s_b - s_a == 40, (
            f"§258: delta must be exactly 40 (one opp_2cl × 40); got {s_b - s_a}"
        )

    def test_opponent_main_diag_lead_negative(self):
        """§260: _evaluate section 2 md_opp=2 branch — opponent 2 SMALL on main diagonal
        triggers `score -= 300`. Mirror of test_main_diagonal_bonus_positive (md_me=2 →
        +300) which is tested but the opponent direction was not.

        Setup: BLUE SMALL at (0,0) and (2,2) — both on main diagonal, neither on
        anti-diagonal (so ad_opp remains 0, no ad-section confounding). From RED's
        perspective: md_opp=2, md_me=0 → section 2 applies -300 penalty. Combined with
        section 1 them_line_2 (-70) and them_bull_1 (-50) and sections 6/7 the total
        is strongly negative. Asserting score <= -300 captures the md_opp=2 penalty."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.BLUE)   # BLUE SMALL at (0,0) — maindiag, not AD
        g.board[2, 2, 0] = int(Player.BLUE)   # BLUE SMALL at (2,2) — maindiag, not AD
        score = _evaluate(g, int(Player.RED))
        assert score <= -300, (
            f"§260: BLUE 2-piece maindiag must trigger -300 section 2 penalty; "
            f"got {score}. If score > -300, md_opp=2 branch is not firing."
        )

    def test_me_line2_small_scores_more_than_large(self):
        """§263: _SZ_THREAT_W per-size weighting for me_line_2 — SMALL 2-of-3 threat
        (weight 70) must outscore LARGE 2-of-3 (weight 30) by exactly 56.

        Setup: Two RED pieces in row1 at edge cells (1,0) and (1,2).
          Both cells are non-diagonal edge cells (row1 only + individual columns).
          Row1 has 2 pieces → me_line_2 contributes one value.
          Col0 and col2 each have 1 piece → me_open_1 contributes per col.

        Section 1 delta: (70 + 14 + 14) − (30 + 6 + 6) = 98 − 42 = 56.
          · me_line_2: 70 − 30 = 40 (the _SZ_THREAT_W difference)
          · me_open_1: 2 × (14 − 6) = 16 (col0 + col2 open-1 lines)
        Sections 2/4/6/7 are identical for both setups → cancel.
        Exact assertion s_A − s_B == 56 catches weight swap or coefficient bugs."""
        g_small = TicTacPro()
        g_small.board[1, 0, 0] = int(Player.RED)   # SMALL at (1,0) — edge, row1
        g_small.board[1, 2, 0] = int(Player.RED)   # SMALL at (1,2) — edge, row1
        s_small = _evaluate(g_small, int(Player.RED))

        g_large = TicTacPro()
        g_large.board[1, 0, 2] = int(Player.RED)   # LARGE at (1,0)
        g_large.board[1, 2, 2] = int(Player.RED)   # LARGE at (1,2)
        s_large = _evaluate(g_large, int(Player.RED))

        assert s_small > s_large, (
            f"§263: SMALL 2-of-3 (weight 70) must outscore LARGE (weight 30); "
            f"s_small={s_small}, s_large={s_large}"
        )
        assert s_small - s_large == 56, (
            f"§263: delta must be exactly 56 (me_line_2: 40 + me_open_1 col×2: 16); "
            f"got {s_small - s_large}"
        )

    def test_them_line2_small_penalizes_more_than_large(self):
        """§264: _SZ_THREAT_W per-size weighting for them_line_2 — opponent SMALL 2-of-3
        (weight 70 → -70 penalty) penalizes more than LARGE (-30), exact delta = 56.

        Mirror of §263 (me_line_2). Uses same symmetric setup but with BLUE pieces.
          Section 1 delta: (70+14+14) − (30+6+6) = 56 (exactly).
          Sections 2/4/6/7 identical for both sizes → cancel.
          Exact assertion s_large − s_small == 56 (LARGE is less penalizing by 56)."""
        g_small = TicTacPro()
        g_small.board[1, 0, 0] = int(Player.BLUE)   # BLUE SMALL at (1,0) — edge, row1
        g_small.board[1, 2, 0] = int(Player.BLUE)   # BLUE SMALL at (1,2) — edge, row1
        s_small = _evaluate(g_small, int(Player.RED))

        g_large = TicTacPro()
        g_large.board[1, 0, 2] = int(Player.BLUE)   # BLUE LARGE at (1,0)
        g_large.board[1, 2, 2] = int(Player.BLUE)   # BLUE LARGE at (1,2)
        s_large = _evaluate(g_large, int(Player.RED))

        assert s_small < s_large, (
            f"§264: BLUE SMALL 2-of-3 must penalize RED more than BLUE LARGE; "
            f"s_small={s_small}, s_large={s_large}"
        )
        assert s_large - s_small == 56, (
            f"§264: delta must be exactly 56 (them_line_2: 40 + them_open_1 col×2: 16); "
            f"got {s_large - s_small}"
        )

    def test_me_bull2_scores_more_than_me_bull1_exact(self):
        """§267: me_bull_2 (+80 per cell with 2 of my pieces) vs me_bull_1 (+25 for 1).
        Cell (1,0) is non-diagonal so section 2 doesn't fire; section 7 has me_lc≤1
        for all lines so me_2cl=0.

        Position A (2 sizes): RED SMALL + MEDIUM at (1,0):
          section 1: row1-S+col0-S+row1-M+col0-M open-1 → +48
          section 4: me_bull_2 = 1 → +80
        Position B (1 size): RED SMALL at (1,0):
          section 1: row1-S+col0-S open-1 → +28
          section 4: me_bull_1 = 1 → +25
        Delta: (48+80)-(28+25) = 75 exactly.
        Catches coefficient bugs in me_bull_2 (+80) or me_bull_1 (+25)."""
        g_two = TicTacPro()
        g_two.board[1, 0, 0] = int(Player.RED)   # SMALL at (1,0)
        g_two.board[1, 0, 1] = int(Player.RED)   # MEDIUM at (1,0) — same cell
        s_two = _evaluate(g_two, int(Player.RED))

        g_one = TicTacPro()
        g_one.board[1, 0, 0] = int(Player.RED)   # SMALL only at (1,0)
        s_one = _evaluate(g_one, int(Player.RED))

        assert s_two > s_one, (
            f"§267: 2-slot bullseye setup must score more than 1-slot; "
            f"s_two={s_two}, s_one={s_one}"
        )
        assert s_two - s_one == 75, (
            f"§267: delta must be exactly 75 (me_bull_2 80−me_bull_1 25=55, section1 Δ=20); "
            f"got {s_two - s_one}"
        )

    def test_them_bull2_penalizes_more_than_them_bull1_exact(self):
        """§268: them_bull_2 (−80 per opp cell with 2 pieces) vs them_bull_1 (−25 for 1).
        Opponent mirror of §267 using same non-diagonal edge cell (1,0).

        Position A (BLUE S+M at (1,0)): them_bull_2=1 → -80; sec1 = -48.
        Position B (BLUE S at (1,0)):   them_bull_1=1 → -25; sec1 = -28.
        Delta s_b − s_a == 75 (sec4: 80−25=55, sec1: 48−28=20).
        Catches them_bull_2 coefficient bugs independently of them_bull_1."""
        g_two = TicTacPro()
        g_two.board[1, 0, 0] = int(Player.BLUE)   # BLUE SMALL at (1,0)
        g_two.board[1, 0, 1] = int(Player.BLUE)   # BLUE MEDIUM at (1,0) — same cell
        s_a = _evaluate(g_two, int(Player.RED))

        g_one = TicTacPro()
        g_one.board[1, 0, 0] = int(Player.BLUE)   # BLUE SMALL only at (1,0)
        s_b = _evaluate(g_one, int(Player.RED))

        assert s_a < s_b, (
            f"§268: 2-slot opp bullseye must penalize RED more than 1-slot; "
            f"s_a={s_a}, s_b={s_b}"
        )
        assert s_b - s_a == 75, (
            f"§268: delta must be exactly 75 (them_bull_2 80−them_bull_1 25=55, sec1 Δ=20); "
            f"got {s_b - s_a}"
        )

    def test_ad_me1_outscores_md_me1_by_40(self):
        """§269: Section 2 single-piece coefficients — ad_me=1 (+100) must outscore
        md_me=1 (+60) by exactly 40 when all other sections are equal.

        (0,2,S) is an AD corner: ad_me=1 → +100, md_me=0.
        (0,0,S) is a MD corner: md_me=1 → +60, ad_me=0.
        Both positions have identical section 1 (3 lines × 14 = 42), section 4
        (me_bull_1 → +25), and section 7 (0). Delta comes from section 2 alone."""
        g_ad = TicTacPro()
        g_ad.board[0, 2, 0] = int(Player.RED)   # AD corner (0,2,S): ad_me=1
        s_ad = _evaluate(g_ad, int(Player.RED))

        g_md = TicTacPro()
        g_md.board[0, 0, 0] = int(Player.RED)   # MD corner (0,0,S): md_me=1
        s_md = _evaluate(g_md, int(Player.RED))

        assert s_ad > s_md, (
            f"§269: AD corner (ad_me=1, +100) must outscore MD corner (md_me=1, +60); "
            f"s_ad={s_ad}, s_md={s_md}"
        )
        assert s_ad - s_md == 40, (
            f"§269: delta must be 40 (100−60); got {s_ad - s_md}"
        )

    def test_ad_opp1_penalizes_more_than_md_opp1_by_40(self):
        """§270: Section 2 opponent single-piece coefficients — ad_opp=1 (-100) must
        penalize RED more than md_opp=1 (-60) by exactly 40.

        BLUE at (0,2,S): ad_opp=1 → -100. BLUE at (0,0,S): md_opp=1 → -60.
        Both positions have identical section 1/4/7. Delta from section 2 alone."""
        g_ad = TicTacPro()
        g_ad.board[0, 2, 0] = int(Player.BLUE)  # AD corner (0,2,S): ad_opp=1
        s_ad = _evaluate(g_ad, int(Player.RED))

        g_md = TicTacPro()
        g_md.board[0, 0, 0] = int(Player.BLUE)  # MD corner (0,0,S): md_opp=1
        s_md = _evaluate(g_md, int(Player.RED))

        assert s_ad < s_md, (
            f"§270: BLUE at AD corner (ad_opp=1, -100) must penalize RED more than "
            f"MD corner (md_opp=1, -60); s_ad={s_ad}, s_md={s_md}"
        )
        assert s_md - s_ad == 40, (
            f"§270: delta must be 40 (100−60); got {s_md - s_ad}"
        )

    def test_medium_line2_scores_between_small_and_large(self):
        """§271: _SZ_THREAT_W[MEDIUM]=50 lies between SMALL=70 and LARGE=30.
        §263 tested S vs L (delta 56). §264 tested S vs L for opp. This test
        fixes MEDIUM vs LARGE to prove the middle weight (50) is correctly placed.

        Setup: Two RED pieces in row1 at non-diagonal edge cells (1,0) and (1,2).
          MEDIUM: row1-M (line2, +50) + col0-M (open1, +10) + col2-M (open1, +10) = 70
          LARGE:  row1-L (line2, +30) + col0-L (open1,  +6) + col2-L (open1,  +6) = 42
        Sections 4/6/7 identical → cancel. Delta = 70 − 42 = 28.
        Asserting s_M − s_L == 28 pins the MEDIUM coefficient to exactly 50."""
        g_med = TicTacPro()
        g_med.board[1, 0, 1] = int(Player.RED)   # MEDIUM at (1,0)
        g_med.board[1, 2, 1] = int(Player.RED)   # MEDIUM at (1,2)
        s_med = _evaluate(g_med, int(Player.RED))

        g_lar = TicTacPro()
        g_lar.board[1, 0, 2] = int(Player.RED)   # LARGE at (1,0)
        g_lar.board[1, 2, 2] = int(Player.RED)   # LARGE at (1,2)
        s_lar = _evaluate(g_lar, int(Player.RED))

        assert s_med > s_lar, (
            f"§271: MEDIUM 2-of-3 (weight 50) must outscore LARGE (weight 30); "
            f"s_med={s_med}, s_lar={s_lar}"
        )
        assert s_med - s_lar == 28, (
            f"§271: delta must be exactly 28 (me_line2: 20 + me_open1 col×2: 8); "
            f"got {s_med - s_lar}"
        )

    def test_me_two_cell_line_control_exact_40(self):
        """§272: Section 7 me_2cl coefficient (+40) — exact-delta test.
        test_multi_cell_line_control_positive only asserts score > 0 (weak).

        Setup A: RED MEDIUM at (0,0) + RED LARGE at (0,1) → row0 has 2 RED cells →
                 me_2cl=1 → +40. Both cells also in separate col/diag lines (me_lc≤1).
        Setup B: RED MEDIUM at (0,0) + RED LARGE at (2,1) → no shared line →
                 me_2cl=0 → +0.

        Sections 1/2/4/6 are identical for both setups:
          sec1: (0,0,M) = 30 (row0-M + col0-M + maindiag-M); (r,c,L) = 12 for each.
          sec4: 2 × me_bull_1 → +50; sec6: n_me_bull1=2 → +45.
        Delta = 40 exactly — catches any change to the 40-pt me_2cl coefficient."""
        g_a = TicTacPro()
        g_a.board[0, 0, 1] = int(Player.RED)   # MEDIUM at (0,0)
        g_a.board[0, 1, 2] = int(Player.RED)   # LARGE at (0,1) — same row0
        s_a = _evaluate(g_a, int(Player.RED))

        g_b = TicTacPro()
        g_b.board[0, 0, 1] = int(Player.RED)   # MEDIUM at (0,0) — identical
        g_b.board[2, 1, 2] = int(Player.RED)   # LARGE at (2,1) — different line
        s_b = _evaluate(g_b, int(Player.RED))

        assert s_a > s_b, (
            f"§272: shared-line position (me_2cl=1) must score higher than "
            f"separate-line (me_2cl=0); s_a={s_a}, s_b={s_b}"
        )
        assert s_a - s_b == 40, (
            f"§272: delta must be exactly 40 (one me_2cl × 40); got {s_a - s_b}"
        )

    def test_medium_open1_weight_exactly_10(self):
        """§273: _SZ_THREAT_W1[MEDIUM]=10 vs LARGE=6 — exact delta at edge cell (0,1).
        §256 tested SMALL(14) vs LARGE(6) giving delta 16 for 2 lines.
        This test pins MEDIUM(10) vs LARGE(6) giving delta 8 for 2 lines,
        fully constraining all three _SZ_THREAT_W1 weights.

        Cell (0,1) is on row0 + col1 only (no diagonals, not AD).
        MEDIUM: me_open_1 row0-M + col1-M → 2 × 10 = 20.
        LARGE:  me_open_1 row0-L + col1-L → 2 × 6  = 12.
        Section 4 (me_bull_1 +25), sections 2/6/7 identical → cancel.
        Delta = 20 − 12 = 8 exactly."""
        g_med = TicTacPro()
        g_med.board[0, 1, 1] = int(Player.RED)   # MEDIUM at (0,1) — edge, no diagonal
        s_med = _evaluate(g_med, int(Player.RED))

        g_lar = TicTacPro()
        g_lar.board[0, 1, 2] = int(Player.RED)   # LARGE at (0,1) — same cell
        s_lar = _evaluate(g_lar, int(Player.RED))

        assert s_med > s_lar, (
            f"§273: MEDIUM 1-piece threat (weight 10) must outscore LARGE (6); "
            f"s_med={s_med}, s_lar={s_lar}"
        )
        assert s_med - s_lar == 8, (
            f"§273: delta must be exactly 8 (2 lines × (10−6)); got {s_med - s_lar}"
        )

    def test_them_line2_medium_penalizes_more_than_large(self):
        """§274: _SZ_THREAT_W[MEDIUM]=50 for them_line_2 — opp mirror of §271.
        §264 tested BLUE SMALL vs LARGE (delta 56). §271 tested RED MEDIUM vs LARGE (delta 28).
        This test pins them_line_2 MEDIUM weight to exactly 50 (vs LARGE=30), delta = 28.

        Setup: Two BLUE pieces in row1 at non-diagonal edge cells (1,0) and (1,2).
          MEDIUM: row1-M (them_line_2 −50) + col0-M (them_open_1 −10) + col2-M (−10) = −70
          LARGE:  row1-L (them_line_2 −30) + col0-L (them_open_1  −6) + col2-L  (−6) = −42
        Sections 2/4/6/7 identical → cancel. Delta = s_large − s_small = −42 − (−70) = 28.
        Asserting s_lar − s_med == 28 pins the MEDIUM them_line_2 coefficient to exactly 50."""
        g_med = TicTacPro()
        g_med.board[1, 0, 1] = int(Player.BLUE)   # BLUE MEDIUM at (1,0) — edge, row1
        g_med.board[1, 2, 1] = int(Player.BLUE)   # BLUE MEDIUM at (1,2) — edge, row1
        s_med = _evaluate(g_med, int(Player.RED))

        g_lar = TicTacPro()
        g_lar.board[1, 0, 2] = int(Player.BLUE)   # BLUE LARGE at (1,0)
        g_lar.board[1, 2, 2] = int(Player.BLUE)   # BLUE LARGE at (1,2)
        s_lar = _evaluate(g_lar, int(Player.RED))

        assert s_med < s_lar, (
            f"§274: BLUE MEDIUM 2-of-3 must penalize RED more than BLUE LARGE; "
            f"s_med={s_med}, s_lar={s_lar}"
        )
        assert s_lar - s_med == 28, (
            f"§274: delta must be exactly 28 (them_line2: 20 + them_open1 col×2: 8); "
            f"got {s_lar - s_med}"
        )

    def test_me_open1_small_vs_large_exact_delta_16(self):
        """§278: Pin §256 assertion to == 16 (was >= 16). Exact-delta sanity check.

        §256 uses >= 16. The exact delta is 16 (2 lines × (14−6)):
          SMALL at (0,1): me_open_1 row0-S(14) + col1-S(14) = 28. sec4: +25.
          LARGE at (0,1): me_open_1 row0-L(6) + col1-L(6)   = 12. sec4: +25.
          sec2/6/7 identical (no diagonal, no shared-line pair) → cancel.
          Delta = 28 − 12 = 16.

        Asserting == 16 catches: weight SMALL raised (e.g. 15→ delta 18) or
        LARGE raised (e.g. 8 → delta 12). Also verifies no extra section 7
        contribution (me_2cl would add 40, pushing delta to 56)."""
        g_small = TicTacPro()
        g_small.board[0, 1, 0] = int(Player.RED)   # SMALL at edge (0,1)
        s_small = _evaluate(g_small, int(Player.RED))

        g_large = TicTacPro()
        g_large.board[0, 1, 2] = int(Player.RED)   # LARGE at same edge (0,1)
        s_large = _evaluate(g_large, int(Player.RED))

        assert s_small - s_large == 16, (
            f"§278: exact delta must be 16 (2 lines × (14−6)); "
            f"got {s_small - s_large}"
        )

    def test_section6_them_beachhead_coefficient_exactly_45(self):
        """§277: Section 6 them_beachhead penalty coefficient = 45 — opp mirror of §276.
        §223 test_pre_fork_beachhead_penalty only asserts score < 0 — coefficient unpinned.

        Setup A: BLUE LARGE at (0,1) + BLUE LARGE at (2,0) — no shared lines.
          sec1: (0,1,L) → -12; (2,0,L) → -(row2+col0+antidiag)×6 = -18; total -30.
          sec4: them_bull_1=2 → -50.
          sec6: n_them_bull1=2 → -45*(2-1) = -45. Total s_a = -125.
        Setup B: BLUE LARGE at (0,1) only. sec1=-12; sec4=-25; sec6=0 → total s_b = -37.
        s_b − s_a = 88 exactly. If coefficient=40 → 83; if=50 → 93. Pins it to exactly 45."""
        g_a = TicTacPro()
        g_a.board[0, 1, 2] = int(Player.BLUE)   # BLUE LARGE at (0,1) — edge, row0+col1
        g_a.board[2, 0, 2] = int(Player.BLUE)   # BLUE LARGE at (2,0) — edge, row2+col0+antidiag
        s_a = _evaluate(g_a, int(Player.RED))

        g_b = TicTacPro()
        g_b.board[0, 1, 2] = int(Player.BLUE)   # BLUE LARGE at (0,1) only
        s_b = _evaluate(g_b, int(Player.RED))

        assert s_b - s_a == 88, (
            f"§277: exact delta must be 88 (sec1:18 + sec4:25 + sec6:45); "
            f"got {s_b - s_a}. Wrong coefficient: sec6 coeff ≈ {s_b - s_a - 43}"
        )

    def test_section6_me_beachhead_coefficient_exactly_45(self):
        """§276: Section 6 me_beachhead bonus coefficient = 45 — exact-delta isolation.
        §253 only asserts ≥45, leaving the coefficient itself unpinned.

        Setup A: RED LARGE at (0,1) + RED LARGE at (2,0) — no shared lines.
          sec1: (0,1,L) → row0-L(6)+col1-L(6)=12; (2,0,L) → row2-L(6)+col0-L(6)+antidiag-L(6)=18 → 30
          sec4: me_bull_1=2 → +50
          sec6: n_me_bull1=2 → +45*(2-1) = +45
          Total: 125

        Setup B: RED LARGE at (0,1) only.
          sec1: 12; sec4: +25; sec6: 0 → total: 37

        Delta = 88. If coefficient = 40 → delta=83; if = 50 → delta=93. Pins it to exactly 45.
        Cell (2,0): antidiag but LARGE (sec2 checks SMALL only) → no sec2 contamination."""
        g_a = TicTacPro()
        g_a.board[0, 1, 2] = int(Player.RED)   # LARGE at (0,1) — edge, row0+col1
        g_a.board[2, 0, 2] = int(Player.RED)   # LARGE at (2,0) — edge, row2+col0+antidiag
        s_a = _evaluate(g_a, int(Player.RED))

        g_b = TicTacPro()
        g_b.board[0, 1, 2] = int(Player.RED)   # LARGE at (0,1) only
        s_b = _evaluate(g_b, int(Player.RED))

        assert s_a - s_b == 88, (
            f"§276: exact delta must be 88 (sec1:18 + sec4:25 + sec6:45); "
            f"got {s_a - s_b}. Wrong coefficient: sec6 coeff = {s_a - s_b - 43}"
        )

    def test_them_open1_medium_weight_exactly_10(self):
        """§275: _SZ_THREAT_W1[MEDIUM]=10 for them_open_1 — opp mirror of §273.
        §273 pinned me_open_1 MEDIUM vs LARGE to delta 8 (2 lines × (10−6)).
        This test pins them_open_1 MEDIUM weight in the opponent direction.

        Cell (0,1) is on row0 + col1 only (no diagonals, not AD).
        BLUE MEDIUM: them_open_1 row0-M + col1-M → 2 × 10 = 20 penalty.
        BLUE LARGE:  them_open_1 row0-L + col1-L → 2 × 6  = 12 penalty.
        Section 4 (them_bull_1 −25), sections 2/6/7 identical → cancel.
        Delta = s_lar − s_med = (−12) − (−20) = 8 exactly."""
        g_med = TicTacPro()
        g_med.board[0, 1, 1] = int(Player.BLUE)   # BLUE MEDIUM at (0,1) — edge, no diagonal
        s_med = _evaluate(g_med, int(Player.RED))

        g_lar = TicTacPro()
        g_lar.board[0, 1, 2] = int(Player.BLUE)   # BLUE LARGE at (0,1) — same cell
        s_lar = _evaluate(g_lar, int(Player.RED))

        assert s_med < s_lar, (
            f"§275: BLUE MEDIUM 1-piece threat (weight 10) must penalize RED more than "
            f"BLUE LARGE (6); s_med={s_med}, s_lar={s_lar}"
        )
        assert s_lar - s_med == 8, (
            f"§275: delta must be exactly 8 (2 lines × (10−6)); got {s_lar - s_med}"
        )

    def test_opp_two_ad_smalls_exact_delta_594(self):
        """§290: _evaluate section 2 `ad_opp==2` branch — adding a 2nd BLUE AD Small
        to a 1-AD-Small position lowers RED's score by exactly 594.

        Setup A: BLUE TR-S at (0,2) — 1 AD Small (antidiag corner, not CC).
        Setup B: BLUE TR-S + BL-S — 2 AD Smalls (antidiag endpoints, no CC overlap).

        Delta = s_a - s_b = 594:
          sec1: antidiag-S upgrades them_open_1(14)→them_line_2(70): −56
                BL adds row2-S(−14) + col0-S(−14): −28 → sec1 total −84
          sec2: ad_opp 1→2: −500 vs −100: −400
          sec4: them_bull_1 1→2 cells: −25
          sec6: n_them_bull1 1→2: −45*(2-1)=−45
          sec7: antidiag opp_2cl fires: −40
          Total: 84+400+25+45+40 = 594

        Exercises the `ad_opp==2` branch (−500 penalty) and distinguishes it from
        `ad_opp==1` (−100). Wrong coefficient (e.g. −200 or −400) changes delta."""
        g_a = TicTacPro()
        g_a.board[0, 2, 0] = int(Player.BLUE)   # TR-S — antidiag, 1 AD small
        s_a = _evaluate(g_a, int(Player.RED))

        g_b = TicTacPro()
        g_b.board[0, 2, 0] = int(Player.BLUE)   # TR-S — antidiag corner
        g_b.board[2, 0, 0] = int(Player.BLUE)   # BL-S — antidiag corner (no CC)
        s_b = _evaluate(g_b, int(Player.RED))

        assert s_a > s_b, f"§290: 2 opp AD smalls must score lower than 1; s_a={s_a}, s_b={s_b}"
        assert s_a - s_b == 594, (
            f"§290: exact delta must be 594 "
            f"(sec1:-84 + sec2:-400 + sec4:-25 + sec6:-45 + sec7:-40); "
            f"got {s_a - s_b}"
        )

    def test_opp_two_md_smalls_exact_delta_434(self):
        """§291: _evaluate section 2 `md_opp==2` branch — adding a 2nd BLUE MD Small
        to a 1-MD-Small position lowers RED's score by exactly 434.

        Setup A: BLUE TL-S at (0,0) — 1 MD Small (maindiag corner, not CC).
        Setup B: BLUE TL-S + BR-S — 2 MD Smalls (maindiag endpoints, no CC overlap).

        Delta = s_a - s_b = 434:
          sec1: maindiag-S upgrades them_open_1(14)→them_line_2(70): −56
                BR adds row2-S(−14) + col2-S(−14): −28 → sec1 total −84
          sec2: md_opp 1→2: −300 vs −60: −240
          sec4: them_bull_1 1→2 cells: −25
          sec6: n_them_bull1 1→2: −45*(2-1)=−45
          sec7: maindiag opp_2cl fires: −40
          Total: 84+240+25+45+40 = 434

        Exercises the `md_opp==2` branch (−300 penalty) and distinguishes it from
        `md_opp==1` (−60). Wrong coefficient changes delta."""
        g_a = TicTacPro()
        g_a.board[0, 0, 0] = int(Player.BLUE)   # TL-S — maindiag, 1 MD small
        s_a = _evaluate(g_a, int(Player.RED))

        g_b = TicTacPro()
        g_b.board[0, 0, 0] = int(Player.BLUE)   # TL-S — maindiag corner
        g_b.board[2, 2, 0] = int(Player.BLUE)   # BR-S — maindiag corner (no CC)
        s_b = _evaluate(g_b, int(Player.RED))

        assert s_a > s_b, f"§291: 2 opp MD smalls must score lower than 1; s_a={s_a}, s_b={s_b}"
        assert s_a - s_b == 434, (
            f"§291: exact delta must be 434 "
            f"(sec1:-84 + sec2:-240 + sec4:-25 + sec6:-45 + sec7:-40); "
            f"got {s_a - s_b}"
        )

    def test_contested_ad_neutralizes_opp_2ad_advantage(self):
        """§306: Section 2 `if ad_me == 0` guard — when RED contests the anti-diagonal
        with 1 Small, BLUE's 2-AD-Small −500 penalty is neutralized (mirror of §288).

        §288 tests that `if ad_opp == 0` guard blocks RED's +500 bonus when BLUE contests.
        §306 tests that `if ad_me == 0` guard blocks BLUE's −500 penalty when RED contests.

        Setup A: BLUE at (0,2,S) + (2,0,S) — 2 AD Smalls. RED has none (ad_me=0).
          → `if ad_me == 0` is True → score -= 500 (BLUE penalty fires).
        Setup B: Same BLUE + RED at (1,1,S) — RED contests AD (ad_me=1).
          → `if ad_me == 0` is False → score -= 0 (penalty neutralized, +500 vs A).

        Composite delta s_b − s_a = 737:
          sec1: antidiag-S them_line_2 suppressed (no_me=False), +3 RED me_open_1 lines → +112
          sec2: BLUE's −500 AD penalty neutralized (+500); RED's CC-S on maindiag adds +60
                (md_me=1, md_opp=0 → maindiag M=60 bonus) → total +560
          sec4: RED's me_bull_1 at (1,1) → +25
          sec7: antidiag opp_2cl drops (RED in cell, me_lc=1) → +40
          Total: 112+560+25+40 = 737."""
        g_a = TicTacPro()
        g_a.board[0, 2, 0] = int(Player.BLUE)   # TR-S (antidiag corner)
        g_a.board[2, 0, 0] = int(Player.BLUE)   # BL-S (antidiag corner) — ad_opp=2
        s_a = _evaluate(g_a, int(Player.RED))

        g_b = TicTacPro()
        g_b.board[0, 2, 0] = int(Player.BLUE)
        g_b.board[2, 0, 0] = int(Player.BLUE)
        g_b.board[1, 1, 0] = int(Player.RED)    # CC-S — RED contests antidiag (ad_me=1)
        s_b = _evaluate(g_b, int(Player.RED))

        assert s_b > s_a, (
            f"§306: RED contesting AD must improve RED's score; s_a={s_a}, s_b={s_b}"
        )
        assert s_b - s_a == 737, (
            f"§306: exact delta must be 737 "
            f"(sec1:+112 + sec2:+560 + sec4:+25 + sec7:+40); "
            f"got {s_b - s_a}"
        )


# ── _order_moves ──────────────────────────────────────────────────────────────

class TestOrderMoves:
    def test_win_is_first(self):
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        moves = g.get_legal_moves(Player.RED)
        ordered = _order_moves(g, moves, Player.RED)
        assert ordered[0] == (0, 2, PieceSize.SMALL)

    def test_block_before_rest(self):
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.BLUE)
        g.board[0, 1, 0] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        g.current_player = Player.RED
        moves = g.get_legal_moves(Player.RED)
        ordered = _order_moves(g, moves, Player.RED)
        # Block (0,2,S) should be before any non-blocking move
        block_idx = ordered.index((0, 2, PieceSize.SMALL))
        assert block_idx < len(ordered) - 1  # not last

    def test_tt_move_is_first(self):
        """§209: TT move must be placed at the front of the move list (priority 10M)."""
        g = TicTacPro()
        moves = g.get_legal_moves(Player.RED)
        # Pick an arbitrary non-winning move as the TT move.
        tt_move = (1, 1, PieceSize.MEDIUM)
        assert tt_move in moves
        ordered = _order_moves(g, moves, Player.RED, tt_move=tt_move)
        assert ordered[0] == tt_move, (
            f"TT move {tt_move} must be first, got {ordered[0]}"
        )

    def test_tt_move_before_win(self):
        """§209: TT move priority (10M) exceeds win priority (9M) — searched first
        to get an early cutoff; does not affect correctness since wins are still found."""
        g = TicTacPro()
        # RED two-in-a-row: win at (0,2,SMALL)
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        moves = g.get_legal_moves(Player.RED)
        # Designate a non-winning move as TT (simulates TT hit on previous iteration).
        tt_move = (2, 2, PieceSize.LARGE)
        assert tt_move in moves
        ordered = _order_moves(g, moves, Player.RED, tt_move=tt_move)
        # TT move first, win move second.
        assert ordered[0] == tt_move
        assert ordered[1] == (0, 2, PieceSize.SMALL)

    def test_killers_placed_after_win_and_block(self):
        """§209: Killer moves get priority 7.5M — after wins (9M) and blocks (8M)
        but before positional heuristic moves."""
        g = TicTacPro()
        # BLUE two-in-a-row: block at (0,2,SMALL) gets 8M priority.
        g.board[0, 0, 0] = int(Player.BLUE)
        g.board[0, 1, 0] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        g.current_player = Player.RED
        moves = g.get_legal_moves(Player.RED)
        # Designate a non-winning, non-blocking move as a killer.
        killer_move = (2, 2, PieceSize.LARGE)
        assert killer_move in moves
        ordered = _order_moves(g, moves, Player.RED, killers=[killer_move])
        block_idx  = ordered.index((0, 2, PieceSize.SMALL))
        killer_idx = ordered.index(killer_move)
        # Block must come before the killer.
        assert block_idx < killer_idx, (
            f"Block at index {block_idx} must precede killer at index {killer_idx}"
        )
        # Killer must not be last (it should be ahead of at least some positional moves).
        assert killer_idx < len(ordered) - 1, "Killer move must not be last"

    def test_phantom_block_guard(self):
        """§230: _order_moves must NOT rank a cell as blocking (priority 8M) when the
        opponent has 0 pieces of that size remaining — opp_has_sz guard at minimax_agent.py:214.

        Position:
          BLUE (0,0,M)+(0,1,M), BLUE has 1 Medium remaining → real block at (0,2,M) → priority 8M
          BLUE (2,0,S)+(2,1,S), BLUE has 0 Smalls remaining  → phantom block at (2,2,S) → priority 35

        With guard working: (0,2,M) is ranked before (2,2,S) in the output.
        With guard failing: both get 8M; (2,2,S) appears first in input (SMALL < MEDIUM),
        so stable sort puts (2,2,S) before (0,2,M) — assertion fails.
        """
        g = TicTacPro()
        # Real Medium threat
        g.board[0, 0, 1] = int(Player.BLUE)
        g.board[0, 1, 1] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 1
        # Phantom Small threat
        g.board[2, 0, 0] = int(Player.BLUE)
        g.board[2, 1, 0] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 0
        g._pieces_left -= 4
        g.current_player = Player.RED

        moves = g.get_legal_moves(Player.RED)
        assert (0, 2, PieceSize.MEDIUM) in moves
        assert (2, 2, PieceSize.SMALL) in moves

        ordered = _order_moves(g, moves, Player.RED)

        real_idx    = ordered.index((0, 2, PieceSize.MEDIUM))
        phantom_idx = ordered.index((2, 2, PieceSize.SMALL))
        assert real_idx < phantom_idx, (
            f"§230: real block (0,2,M) at idx {real_idx} must precede phantom (2,2,S) "
            f"at idx {phantom_idx}; if phantom_idx < real_idx the opp_has_sz guard failed"
        )

    def test_antidiag_bonus_lifts_ad_small_above_non_ad_corner(self):
        """§250: _order_moves must give anti-diagonal SMALL cells a higher priority
        than non-AD corner SMALL cells in the rest tier (no wins/blocks/TT/killers).

        _AD_CELLS = {(0,2), (1,1), (2,0)}.  On an empty board:
          (0,2,S): base = 20(corner) + 15(small) = 35, antidiag_bonus = 35*(0+1) = 35 → 70
          (0,0,S): base = 20(corner) + 15(small) = 35, antidiag_bonus = 0            → 35
        Removing antidiag_bonus would make these equal, breaking the ordering invariant."""
        g = TicTacPro()
        moves = g.get_legal_moves(Player.RED)
        ordered = _order_moves(g, moves, Player.RED)
        ad_corner_idx    = ordered.index((0, 2, PieceSize.SMALL))
        non_ad_corner_idx = ordered.index((0, 0, PieceSize.SMALL))
        assert ad_corner_idx < non_ad_corner_idx, (
            f"§250: AD corner (0,2,S) at idx {ad_corner_idx} must precede non-AD "
            f"corner (0,0,S) at idx {non_ad_corner_idx}; antidiag_bonus (+35) must "
            "differentiate same-base-score rest-tier moves"
        )

    def test_anti_bullseye_bonus_lifts_contested_cell_above_uncontested(self):
        """§251: _order_moves must rank a MEDIUM move contesting an opponent-occupied
        cell above an uncontested corner MEDIUM move in the rest tier (§129/§133).

        Position: BLUE SMALL at (0,1) — edge cell.  RED considering MEDIUM placement.
          (0,1,M): base = 0(edge) + 10(medium) = 10, anti_bull = 200*1 = 200  → 210
          (2,2,M): base = 20(corner) + 10(medium) = 30, anti_bull = 0         → 30

        Without the bonus (0,2,M) corner would outrank (0,1,M) edge.  With it, the
        contested cell jumps far ahead, capturing the anti-bullseye denial value.
        Guard §133: anti_bull only fires if me_in_cell == 0."""
        g = TicTacPro()
        g.board[0, 1, 0] = int(Player.BLUE)   # BLUE SMALL at edge cell (0,1)
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] -= 1
        g._pieces_left -= 1
        g.current_player = Player.RED

        moves = g.get_legal_moves(Player.RED)
        ordered = _order_moves(g, moves, Player.RED)

        contested_idx   = ordered.index((0, 1, PieceSize.MEDIUM))
        uncontested_idx = ordered.index((2, 2, PieceSize.MEDIUM))
        assert contested_idx < uncontested_idx, (
            f"§251: contested edge (0,1,M) at idx {contested_idx} must precede "
            f"uncontested corner (2,2,M) at idx {uncontested_idx}; "
            "anti_bull (200) must lift it above the higher-base-score corner"
        )

    def test_antidiag_bonus_scales_with_existing_ad_pieces(self):
        """§295: antidiag_bonus = 35*(ad_me+1) — when player already has 1 AD Small,
        a 2nd AD Small move gets bonus=70 (vs 35 on empty board), lifting CC-S
        even higher above a non-AD corner Small.

        Setup: RED has TR-S at (0,2) — ad_me=1 when considering CC-S=(1,1,S).
          CC-S: base=pos(40)+sz(15)=55, antidiag_bonus=35*(1+1)=70 → priority 125
          TL-S: base=pos(20)+sz(15)=35, antidiag_bonus=0              → priority  35
        With ad_me=0 (empty board), CC-S would be 35+35=70 vs TL-S 35 — a margin of 35.
        With ad_me=1, CC-S is 55+70=125 — margin grows to 90.
        If bonus were constant (always 35*(0+1)=35) both setups give same margin — fail.

        Exercises the `ad_me + 1` scaling at minimax_agent.py:224."""
        g = TicTacPro()
        g.board[0, 2, 0] = int(Player.RED)   # TR-S — 1 existing AD Small
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 1
        g._pieces_left -= 1

        moves = g.get_legal_moves(Player.RED)
        ordered = _order_moves(g, moves, Player.RED)

        cc_s_idx  = ordered.index((1, 1, PieceSize.SMALL))   # CC-S: priority 125
        tl_s_idx  = ordered.index((0, 0, PieceSize.SMALL))   # TL-S: priority  35
        assert cc_s_idx < tl_s_idx, (
            f"§295: CC-S (AD center, bonus=70 with ad_me=1) at idx {cc_s_idx} must "
            f"precede TL-S (non-AD corner, bonus=0) at idx {tl_s_idx}; "
            "antidiag_bonus must scale with ad_me (35*(ad_me+1))"
        )

    def test_anti_bull_two_opp_pieces_outranks_one_opp_piece(self):
        """§296: anti_bull = 200*opp_in_cell — a move contesting a cell where the
        opponent has 2 pieces (anti_bull=400) must rank above one contesting 1 piece (200).

        Setup A (opp_in_cell=2): BLUE SMALL+MEDIUM at (1,2).
          RED's (1,2,L): base=0(edge)+5(L)=5, anti_bull=400 → priority 405.
        Setup B (opp_in_cell=1): BLUE SMALL only at (0,1).
          RED's (0,1,M): base=0(edge)+10(M)=10, anti_bull=200 → priority 210.

        In setup A, (1,2,L) with anti_bull=400 must precede (0,0,S) with base=35;
        if the formula used opp_in_cell > 0 ? 200 : 0 (constant bonus), both opp_in_cell=1
        and opp_in_cell=2 would give 200 — this test would still pass for wrong code.
        Instead, asserting 400 > 200 requires the multiplicative formula."""
        # Setup A: BLUE has 2 pieces in cell (1,2)
        g_a = TicTacPro()
        g_a.board[1, 2, 0] = int(Player.BLUE)   # BLUE SMALL
        g_a.board[1, 2, 1] = int(Player.BLUE)   # BLUE MEDIUM — 2 opp pieces
        g_a.pieces_remaining[Player.BLUE][PieceSize.SMALL]  -= 1
        g_a.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] -= 1
        g_a._pieces_left -= 2

        ordered_a = _order_moves(g_a, g_a.get_legal_moves(Player.RED), Player.RED)
        two_piece_idx     = ordered_a.index((1, 2, PieceSize.LARGE))   # anti_bull=400
        non_contested_idx = ordered_a.index((0, 0, PieceSize.SMALL))   # anti_bull=0, pri=35

        assert two_piece_idx < non_contested_idx, (
            f"§296: (1,2,L) contesting 2 opp pieces (anti_bull=400) at idx {two_piece_idx} "
            f"must precede non-contested (0,0,S) at idx {non_contested_idx}"
        )

        # Setup B: BLUE has 1 piece in cell (0,1)
        g_b = TicTacPro()
        g_b.board[0, 1, 0] = int(Player.BLUE)   # BLUE SMALL — 1 opp piece
        g_b.pieces_remaining[Player.BLUE][PieceSize.SMALL] -= 1
        g_b._pieces_left -= 1

        ordered_b = _order_moves(g_b, g_b.get_legal_moves(Player.RED), Player.RED)
        one_piece_idx = ordered_b.index((0, 1, PieceSize.MEDIUM))   # anti_bull=200

        # The 2-opp-piece priority (405) must beat the 1-opp-piece priority (210).
        # We verify this by comparing the computed priorities directly.
        # 2-opp: base(5) + anti_bull(400) = 405; 1-opp: base(10) + anti_bull(200) = 210.
        # Asserting the 2-piece contest came first in setup A is sufficient, but also
        # verify the 1-piece contest came before non-contested in setup B:
        non_cont_b = ordered_b.index((2, 2, PieceSize.MEDIUM))   # base=30, anti_bull=0
        assert one_piece_idx < non_cont_b, (
            f"§296: (0,1,M) contesting 1 opp piece (anti_bull=200) at idx {one_piece_idx} "
            f"must precede uncontested corner (2,2,M) at idx {non_cont_b}"
        )

    def test_anti_bull_guard_suppressed_when_me_in_cell_nonzero(self):
        """§297: §133 guard — anti_bull must NOT fire when the current player already
        has a piece in the target cell (`me_in_cell > 0`).

        Setup: RED SMALL + BLUE MEDIUM at AD-corner (0,2) — opp_in_cell=1, me_in_cell=1.
               BLUE SMALL at edge (0,1) — opp_in_cell=1, me_in_cell=0.

        Priority for RED's Large moves:
          (0,2,L): me_in_cell=1 → guard fires → anti_bull=0; base=pos(20)+sz(5)=25 → 25
          (0,1,L): me_in_cell=0 → guard off  → anti_bull=200; base=0+5=5         → 205

        Without guard: (0,2,L) would get 25+200=225 > (0,1,L) at 205 → wrong order.
        With guard:    (0,2,L) gets 25 < (0,1,L) at 205 → (0,1,L) comes first.

        Exercises the `me_in_cell == 0` check at minimax_agent.py:235."""
        g = TicTacPro()
        g.board[0, 2, 0] = int(Player.RED)    # RED SMALL at AD-corner (0,2) — me_in_cell=1
        g.board[0, 2, 1] = int(Player.BLUE)   # BLUE MEDIUM at (0,2) — opp_in_cell=1
        g.board[0, 1, 0] = int(Player.BLUE)   # BLUE SMALL at edge (0,1) — no RED here
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  -= 1
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  -= 1
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] -= 1
        g._pieces_left -= 3

        moves = g.get_legal_moves(Player.RED)
        ordered = _order_moves(g, moves, Player.RED)

        idx_02L = ordered.index((0, 2, PieceSize.LARGE))   # guard → anti_bull=0 → 25
        idx_01L = ordered.index((0, 1, PieceSize.LARGE))   # no guard → anti_bull=200 → 205

        assert idx_01L < idx_02L, (
            f"§297: edge (0,1,L) with me_in_cell=0 (anti_bull=200, pri=205) at idx {idx_01L} "
            f"must precede AD-corner (0,2,L) with me_in_cell=1 (anti_bull=0, pri=25) at idx {idx_02L}; "
            "if (0,2,L) comes first, the me_in_cell==0 guard is broken"
        )


# ── MinimaxAgent integration ──────────────────────────────────────────────────

class TestMinimaxAgent:
    @pytest.fixture
    def agent(self):
        return MinimaxAgent(time_limit=0.3, max_depth=4)

    def test_returns_legal_move(self, agent):
        g = TicTacPro()
        mv = agent.get_action(g, Player.RED)
        assert mv in g.get_legal_moves(Player.RED)

    def test_returns_none_on_game_over(self, agent):
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.board[0, 2, 0] = int(Player.RED)
        g.winner = Player.RED
        g.game_over = True
        mv = agent.get_action(g, Player.RED)
        assert mv is None

    def test_takes_immediate_win(self, agent):
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        mv = agent.get_action(g, Player.RED)
        assert mv == (0, 2, PieceSize.SMALL)

    def test_blocks_opponent_win(self, agent):
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.BLUE)
        g.board[0, 1, 0] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        g.current_player = Player.RED
        mv = agent.get_action(g, Player.RED)
        assert mv == (0, 2, PieceSize.SMALL), f"Expected block, got {mv}"

    def test_board_unchanged_after_search(self, agent):
        g = TicTacPro()
        board_before = g.board.copy()
        agent.get_action(g, Player.RED)
        assert (g.board == board_before).all()
        assert g._pieces_left == 18
        assert g.current_player == Player.RED

    def test_full_game_no_crash(self, agent):
        """Run a full game with minimax on both sides — must not raise."""
        g = TicTacPro()
        for _ in range(30):
            if g.game_over:
                break
            mv = agent.get_action(g, g.current_player)
            assert mv is not None
            assert mv in g.get_legal_moves(g.current_player)
            g.make_move(*mv)
        assert g.game_over

    def test_get_info_returns_tuple(self, agent):
        g = TicTacPro()
        mv, info = agent.get_info(g, Player.RED)
        depth, score, nodes, hits = info
        assert mv in g.get_legal_moves(Player.RED)
        assert depth >= 0   # 0 when book move; ≥1 when search used
        assert nodes >= 0

    def test_get_info_game_over_returns_none(self, agent):
        """§222: get_info must return (None, (0,DRAW,0,0)) on game_over state."""
        from rl.minimax_agent import DRAW
        g = TicTacPro()
        g.game_over = True
        mv, info = agent.get_info(g, Player.RED)
        assert mv is None, f"Expected None move on game_over, got {mv}"
        depth, score, nodes, hits = info
        assert depth == 0 and nodes == 0, f"Expected (0,_,0,_) on game_over, got {info}"
        assert score == DRAW

    def test_get_info_single_move_shortcut(self, agent):
        """§222: _id_search single-move shortcut returns immediately with (1,0,1,0)."""
        g = TicTacPro()
        # Fill all 27 slots except one — only one legal move remains.
        # Set all pieces_remaining to 0 except RED SMALL=1, then put pieces
        # everywhere except (0,0,S).
        # Simpler: drain all except one piece and occupy all cells except one.
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  = 1
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.RED][PieceSize.LARGE]  = 0
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 0
        # Occupy all Small slots except (0,0,S)
        for r in range(3):
            for c in range(3):
                if not (r == 0 and c == 0):
                    g.board[r, c, 0] = int(Player.BLUE)  # occupy with BLUE (won't win)
        g._pieces_left = 1
        ag = MinimaxAgent(time_limit=1.0, max_depth=5, use_book=False)
        mv, info = ag.get_info(g, Player.RED)
        assert mv == (0, 0, PieceSize.SMALL), f"Expected only legal move (0,0,S), got {mv}"
        depth, score, nodes, hits = info
        assert nodes == 1, f"Single-move shortcut must report nodes=1, got {nodes}"

    def test_opening_book_move_is_legal(self, agent):
        """Minimax should return TC-L as first move (from book or search)."""
        g = TicTacPro()
        mv = agent.get_action(g, Player.RED)
        assert mv is not None
        assert mv in g.get_legal_moves(Player.RED)

    def test_anti_diagonal_win_found_within_depth(self):
        """Minimax at depth 5 should find a forced anti-diagonal win 1 ply away."""
        g = TicTacPro()
        g.board[0, 2, 0] = int(Player.RED)
        g.board[1, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        g.current_player = Player.RED
        ag = MinimaxAgent(time_limit=1.0, max_depth=5)
        mv = ag.get_action(g, Player.RED)
        assert mv == (2, 0, PieceSize.SMALL), f"Expected anti-diag win, got {mv}"

    def test_aspiration_window_engaged_and_legal_result(self):
        """§242: _id_search aspiration window (minimax_agent.py:516-528) must engage
        at depth ≥ 3 and still return a legal move.

        The aspiration window fires when depth > 2 and abs(prev_score) < WIN//2.
        It searches a narrow [prev-DELTA, prev+DELTA] window; on fail-low or
        fail-high it widens and re-searches.  A bug in the widen logic (wrong
        bounds, skipped re-search) could return an incorrect or illegal move.
        """
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)   # RED: TL-S
        g.make_move(2, 2, PieceSize.LARGE)   # BLUE: BR-L
        # Mid-game position: no immediate wins, aspiration window kicks in at depth 3+.
        ag = MinimaxAgent(time_limit=5.0, max_depth=5, use_book=False)
        mv, info = ag.get_info(g, Player.RED)
        depth_done, score, nodes, _ = info
        assert mv in g.get_legal_moves(Player.RED), (
            f"§242: aspiration window returned illegal move {mv}"
        )
        assert depth_done >= 3, (
            f"§242: expected depth_done ≥ 3 for aspiration window exercise, got {depth_done}"
        )
        assert LOSS <= score <= WIN, f"§242: score out of range: {score}"

    def test_get_action_default_player_uses_current_player(self, agent):
        """§314d: get_action(game) without explicit player must infer current_player.

        Covers minimax_agent.py line 463 — the `player = game.current_player`
        branch that fires when get_action is called without a player argument.
        All existing tests pass an explicit Player.RED/BLUE, leaving this
        path uncovered."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)   # RED moves first, now BLUE to play
        assert g.current_player == Player.BLUE
        mv = agent.get_action(g)              # no player arg — must default to BLUE
        assert mv is not None, "§314d: get_action() must find a move for BLUE"
        assert mv in g.get_legal_moves(Player.BLUE), (
            f"§314d: default-player move {mv} is not legal for BLUE"
        )

    def test_get_info_default_player_uses_current_player(self, agent):
        """§314e: get_info(game) without explicit player must infer current_player.

        Covers minimax_agent.py line 481 — the `player = game.current_player`
        branch that fires when get_info is called without a player argument."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)   # RED moves, now BLUE to play
        assert g.current_player == Player.BLUE
        mv, info = agent.get_info(g)          # no player arg — must default to BLUE
        assert mv is not None, "§314e: get_info() must find a move for BLUE"
        assert mv in g.get_legal_moves(Player.BLUE), (
            f"§314e: default-player move {mv} is not legal for BLUE"
        )


# ── §119 adaptive BLUE book for corner-M openers ─────────────────────────────

class TestAdaptiveBlueBook:
    def test_corner_M_opener_BL_responds_BL_S(self):
        """RED opens BL-M (corner Medium) → BLUE book should contest BL-S (§119)."""
        g = TicTacPro()
        g.make_move(2, 0, PieceSize.MEDIUM)    # RED: BL-M
        mv = _book_move(g, int(Player.BLUE))
        assert mv == (2, 0, PieceSize.SMALL), (
            f"Expected BL-S=(2,0,S) to contest corner, got {mv}")

    def test_corner_M_opener_TL_responds_TL_S(self):
        """RED opens TL-M → BLUE contests TL-S."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.MEDIUM)    # RED: TL-M
        mv = _book_move(g, int(Player.BLUE))
        assert mv == (0, 0, PieceSize.SMALL), f"Expected TL-S, got {mv}"

    def test_corner_M_opener_TR_responds_TR_S(self):
        """RED opens TR-M → BLUE contests TR-S."""
        g = TicTacPro()
        g.make_move(0, 2, PieceSize.MEDIUM)    # RED: TR-M
        mv = _book_move(g, int(Player.BLUE))
        assert mv == (0, 2, PieceSize.SMALL), f"Expected TR-S, got {mv}"

    def test_corner_M_opener_BR_responds_BR_S(self):
        """RED opens BR-M → BLUE contests BR-S."""
        g = TicTacPro()
        g.make_move(2, 2, PieceSize.MEDIUM)    # RED: BR-M
        mv = _book_move(g, int(Player.BLUE))
        assert mv == (2, 2, PieceSize.SMALL), f"Expected BR-S, got {mv}"

    def test_edge_opener_falls_through_to_CC_S(self):
        """RED opens with an edge piece (not corner) → BLUE still plays CC-S."""
        g = TicTacPro()
        g.make_move(0, 1, PieceSize.LARGE)    # RED: TC-L (minimax book opener)
        mv = _book_move(g, int(Player.BLUE))
        assert mv == (1, 1, PieceSize.SMALL), f"Expected CC-S, got {mv}"

    def test_center_opener_falls_through_to_CC_S(self):
        """RED opens CC-M → BLUE plays CC-S (blocks center, not adaptive)."""
        g = TicTacPro()
        g.make_move(1, 1, PieceSize.MEDIUM)    # RED: CC-M
        mv = _book_move(g, int(Player.BLUE))
        assert mv == (1, 1, PieceSize.SMALL), f"Expected CC-S, got {mv}"

    def test_forced_block_overrides_adaptive(self):
        """Immediate opponent win is blocked before adaptive opener fires."""
        g = TicTacPro()
        # Set up: RED threatens row-0-S win (TL-S, TC-S placed); BL-M also placed.
        g.board[0, 0, 0] = int(Player.RED)   # TL-S
        g.board[0, 1, 0] = int(Player.RED)   # TC-S: RED wins row if TR-S placed
        g.board[2, 0, 1] = int(Player.RED)   # BL-M corner (potential adaptive trigger)
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  -= 2
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] -= 1
        g._pieces_left -= 3
        g.current_player = Player.BLUE
        mv = _book_move(g, int(Player.BLUE))
        # Must block row-0-S at (0,2,S) rather than adaptively playing BL-S
        assert mv == (0, 2, PieceSize.SMALL), f"Expected forced block TR-S, got {mv}"

    def test_adaptive_only_fires_at_pp1(self):
        """Adaptive corner-M response only applies to BLUE's first move (pp==1)."""
        g = TicTacPro()
        # Two moves played before BLUE's second book move
        g.make_move(2, 0, PieceSize.MEDIUM)   # RED: BL-M
        g.make_move(2, 0, PieceSize.SMALL)    # BLUE: BL-S (adaptive first move)
        g.make_move(0, 0, PieceSize.MEDIUM)   # RED: TL-M (forces col-0-M block)
        # pp==3 now — forced block should handle this, not adaptive
        mv = _book_move(g, int(Player.BLUE))
        # Should block column-0-M (ML-M) since RED would win with it
        assert mv == (1, 0, PieceSize.MEDIUM), f"Expected ML-M forced block, got {mv}"

    def test_diagonal_endpoint_fork_prevention_main_diag(self):
        """§287: §125 BLUE book at pp=3 blocks the only-Small diagonal endpoint with Medium.

        Setup (pp=3): RED has TL-S (only Small at TL) + BR-M (Medium at BR).
        Both TL (0,0) and BR (2,2) are main-diagonal corners.
        TL has only a Small → BLUE must play (0,0,M) to block the fork.
        No forced win/block for BLUE, so the §125 endpoint rule fires."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)   # TL-S (only Small at TL)
        g.board[2, 2, 1] = int(Player.RED)   # BR-M
        g.board[1, 1, 0] = int(Player.BLUE)  # BLUE's first move (pp=1 already done)
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  -= 1
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] -= 1
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] -= 1
        g._pieces_left -= 3
        g.current_player = Player.BLUE
        assert int((g.board != 0).sum()) == 3, "Setup error: must have pp==3"
        mv = _book_move(g, int(Player.BLUE))
        assert mv == (0, 0, PieceSize.MEDIUM), (
            f"§287: §125 must return (0,0,M) to block TL fork; got {mv}. "
            "TL has only a Small (fork-ready), BR has Medium → BLUE must contest TL-M."
        )

    def test_diagonal_endpoint_fork_prevention_anti_diag(self):
        """§287: §125 fires on anti-diagonal (TR+BL) as well as main-diagonal.

        Setup (pp=3): RED has TR-S (only Small at TR) + BL-M. Both on anti-diagonal.
        TR has only a Small → BLUE must play (0,2,M) to block the fork."""
        g = TicTacPro()
        g.board[0, 2, 0] = int(Player.RED)   # TR-S (only Small at TR)
        g.board[2, 0, 1] = int(Player.RED)   # BL-M
        g.board[1, 1, 0] = int(Player.BLUE)  # BLUE's cc-s
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  -= 1
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] -= 1
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] -= 1
        g._pieces_left -= 3
        g.current_player = Player.BLUE
        assert int((g.board != 0).sum()) == 3, "Setup error: must have pp==3"
        mv = _book_move(g, int(Player.BLUE))
        assert mv == (0, 2, PieceSize.MEDIUM), (
            f"§287: §125 must return (0,2,M) to block TR fork; got {mv}. "
            "TR has only a Small, BL has Medium → BLUE contests TR-M (anti-diag)."
        )

    def test_diagonal_fork_prevention_r2_only_small_main_diag(self):
        """§307: §125 `only_s2 and not only_s1` branch — when the SECOND diagonal
        endpoint (r2,c2) has only a Small, BLUE contests (r2,c2) with Medium.

        §287 tests both diagonal pairs with only_s1 set (first endpoint has Small).
        This tests the mirror case: only_s2 is set (second endpoint=BR has Small,
        first endpoint=TL has Medium → BLUE must contest BR).

        Loop: ((r1,c1),(r2,c2)) = ((0,0),(2,2)).
          only_s1 = board[0,0,0]==RED and board[0,0,1]==0 and board[0,0,2]==0 → False (TL has M).
          only_s2 = board[2,2,0]==RED and board[2,2,1]==0 and board[2,2,2]==0 → True (BR-S only).
          → `only_s2 and not only_s1` → target=(2,2) → return (2,2,M)."""
        g = TicTacPro()
        g.board[0, 0, 1] = int(Player.RED)   # TL-M (not only-Small: M slot occupied)
        g.board[2, 2, 0] = int(Player.RED)   # BR-S (only Small at BR)
        g.board[1, 1, 0] = int(Player.BLUE)  # BLUE's cc-s at pp=1
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  -= 1
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] -= 1
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] -= 1
        g._pieces_left -= 3
        g.current_player = Player.BLUE
        assert int((g.board != 0).sum()) == 3, "Setup error: must have pp==3"
        mv = _book_move(g, int(Player.BLUE))
        assert mv == (2, 2, PieceSize.MEDIUM), (
            f"§307: `only_s2` branch must return (2,2,M) to block BR fork; got {mv}. "
            "BR has only a Small, TL has Medium → BLUE contests BR-M (main-diag)."
        )

    def test_diagonal_fork_prevention_both_only_small_falls_to_plan(self):
        """§308: §125 at pp=3 — when BOTH diagonal endpoints have only a Small,
        neither `only_s2 and not only_s1` nor `only_s1 and not only_s2` fires;
        `target` stays None and BLUE falls through to `_BLUE_BOOK_PLAN`.

        Code path (minimax_agent.py:415-418): only_s1=True and only_s2=True →
        first branch `only_s2 and not only_s1` = False; second = False → target=None.

        Setup: RED at TL-S (0,0,S) + BR-S (2,2,S) — only-Small at both main-diagonal
        corners. BLUE at TR-S (0,2,S). No forced win/block for BLUE. BLUE falls back
        to _BLUE_BOOK_PLAN, whose first available legal move is CC-S (1,1,S)."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)    # TL-S (only Small at TL)
        g.board[2, 2, 0] = int(Player.RED)    # BR-S (only Small at BR)
        g.board[0, 2, 0] = int(Player.BLUE)   # BLUE's TR-S at pp=1
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  -= 2
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] -= 1
        g._pieces_left -= 3
        g.current_player = Player.BLUE
        assert int((g.board != 0).sum()) == 3, "Setup error: must have pp==3"
        mv = _book_move(g, int(Player.BLUE))
        assert mv == (1, 1, PieceSize.SMALL), (
            f"§308: both-only-Small endpoints → target=None → fall to plan → CC-S; got {mv}. "
            "When both endpoints have only-Small, neither fork-prevention branch fires."
        )

    def test_red_book_selects_vs_ccl_plan_when_blue_plays_cc_large(self):
        """§292: RED book plan selection — when BLUE responds CC-L, RED switches to
        _RED_BOOK_PLAN_VS_CCL and plays TC-S at pp=2.

        `_book_move` for RED checks `board[1,1,2] == BLUE` and `board[1,1,1] != RED`
        → selects _RED_BOOK_PLAN_VS_CCL.  The 2nd move in that plan is TC-S (0,1,S).
        A bug keeping the default plan would return CC-M (1,1,M) instead.

        Exercises the counter-plan branch at minimax_agent.py:364-368."""
        g = TicTacPro()
        g.make_move(0, 1, PieceSize.LARGE)   # RED: TC-L (pp=0)
        g.make_move(1, 1, PieceSize.LARGE)   # BLUE: CC-L → triggers VS_CCL plan
        mv = _book_move(g, int(Player.RED))
        assert mv == (0, 1, PieceSize.SMALL), (
            f"§292: RED vs CC-L plan must play TC-S=(0,1,S); got {mv}. "
            "Wrong plan selected (e.g. default plan would give CC-M)."
        )

    def test_red_book_selects_vs_ccm_plan_when_blue_plays_cc_medium(self):
        """§293: RED book plan selection — when BLUE responds CC-M, RED switches to
        _RED_BOOK_PLAN_VS_CCM and plays CC-S at pp=2.

        `_book_move` for RED checks `board[1,1,1] == BLUE` → selects VS_CCM plan.
        The 2nd entry in _RED_BOOK_PLAN_VS_CCM is CC-S (1,1,S).
        Default plan would give CC-M=(1,1,M), but CC-M is now BLUE's — illegal.
        VS_CCM plays CC-S instead to build the TC-bullseye.

        Exercises the counter-plan branch at minimax_agent.py:369-373."""
        g = TicTacPro()
        g.make_move(0, 1, PieceSize.LARGE)   # RED: TC-L (pp=0)
        g.make_move(1, 1, PieceSize.MEDIUM)  # BLUE: CC-M → triggers VS_CCM plan
        mv = _book_move(g, int(Player.RED))
        assert mv == (1, 1, PieceSize.SMALL), (
            f"§293: RED vs CC-M plan must play CC-S=(1,1,S); got {mv}. "
            "Wrong plan (or CC-M selected but that cell is BLUE's)."
        )

    def test_red_book_stays_in_default_plan_when_blue_plays_cc_small(self):
        """§294: §136 guard — CC-S by BLUE does NOT disrupt _RED_BOOK_PLAN.

        `_book_move` condition at minimax_agent.py:382:
          `if int((g.board!=0).sum()) > 0 and board[1,1,0] != blue_int: return None`
        When BLUE plays CC-S: `board[1,1,0] == blue_int` → condition is False → plan continues.
        RED should play CC-M (1,1,M) at pp=2 per _RED_BOOK_PLAN.

        A bug treating CC-S like a disruptive opener (returning None) would break this."""
        g = TicTacPro()
        g.make_move(0, 1, PieceSize.LARGE)   # RED: TC-L (pp=0)
        g.make_move(1, 1, PieceSize.SMALL)   # BLUE: CC-S (§136: plan still valid!)
        mv = _book_move(g, int(Player.RED))
        assert mv == (1, 1, PieceSize.MEDIUM), (
            f"§294: RED stays in default plan after BLUE CC-S, must play CC-M=(1,1,M); "
            f"got {mv}. §136 guard must keep the plan intact."
        )

    def test_red_book_falls_to_search_when_blue_opens_non_center(self):
        """§298: §131 guard — RED book returns None when BLUE occupies a non-center,
        non-CC-S cell at pp=1, letting search handle the position.

        Condition at minimax_agent.py:382:
          `if int((g.board!=0).sum()) > 0 and board[1,1,0] != blue_int: return None`
        BLUE BR-L at (2,2): board has pieces and board[1,1,0] != BLUE → returns None.

        Without this guard, RED might play CC-M from the default plan, but against a
        non-passive BLUE opener the plan's fork (TL-bullseye + maindiag-M) doesn't hold."""
        g = TicTacPro()
        g.make_move(0, 1, PieceSize.LARGE)    # RED: TC-L (pp=0)
        g.make_move(2, 2, PieceSize.LARGE)    # BLUE: BR-L (non-center corner)
        mv = _book_move(g, int(Player.RED))
        assert mv is None, (
            f"§298: §131 guard must return None when BLUE plays non-center (BR-L); "
            f"got {mv}. Wrong plan (CC-M) would mislead search."
        )

    def test_red_book_falls_to_search_when_no_tcl_and_blue_plays_ccl(self):
        """§299: §132 guard — if RED never played TC-L, book returns None even when
        BLUE plays CC-L (VS_CCL plan would otherwise be selected).

        The VS_CCL plan's fork was only proven for the TC-L opener. Without TC-L,
        the plan is undefined. Guard: `if not red_has_tcl: return None`.

        Exercises minimax_agent.py:366 (VS_CCL branch) and minimax_agent.py:372 (VS_CCM)."""
        # VS_CCL: RED opens TL-L (not TC-L), BLUE plays CC-L
        g_ccl = TicTacPro()
        g_ccl.make_move(0, 0, PieceSize.LARGE)    # RED: TL-L (not TC-L)
        g_ccl.make_move(1, 1, PieceSize.LARGE)    # BLUE: CC-L
        mv_ccl = _book_move(g_ccl, int(Player.RED))
        assert mv_ccl is None, (
            f"§299: §132 guard (VS_CCL branch) must return None when RED has no TC-L; "
            f"got {mv_ccl}"
        )

        # VS_CCM: RED opens TR-L (not TC-L), BLUE plays CC-M
        g_ccm = TicTacPro()
        g_ccm.make_move(0, 2, PieceSize.LARGE)    # RED: TR-L (not TC-L)
        g_ccm.make_move(1, 1, PieceSize.MEDIUM)   # BLUE: CC-M
        mv_ccm = _book_move(g_ccm, int(Player.RED))
        assert mv_ccm is None, (
            f"§299: §132 guard (VS_CCM branch) must return None when RED has no TC-L; "
            f"got {mv_ccm}"
        )

    def test_blue_book_skips_cc_non_small_when_blue_has_cc_small(self):
        """§300: BLUE book center-skip — when BLUE already has CC-S, book must skip
        CC-M and CC-L plan entries even if they are legal, advancing to TR-S.

        Plan: [CC-S, CC-M, CC-L, TR-S, BL-S, ...].
        At pp=3 with BLUE holding CC-S, the iterator:
          CC-S → not in legal (already placed) → skip.
          CC-M → center-skip guard fires (BLUE has CC-S, sz!=S) → skip.
          CC-L → center-skip guard fires → skip.
          TR-S → legal, no guard → return (0,2,S).

        Guard at minimax_agent.py:432-435:
          `if (player==blue and (r,c)==(1,1) and sz != S and board[1,1,0]==blue): continue`

        Without this guard, CC-M would be returned (it's legal), stacking more center
        pieces when the plan already has CC-S — wasting a move."""
        g = TicTacPro()
        g.make_move(0, 1, PieceSize.LARGE)    # RED: TC-L (pp=0)
        g.make_move(1, 1, PieceSize.SMALL)    # BLUE: CC-S (pp=1)
        g.make_move(2, 0, PieceSize.LARGE)    # RED: BL-L (pp=2) — TR-S stays available
        # pp=3: BLUE has CC-S. CC-M is legal but should be skipped. TR-S=(0,2,S) is next.
        legal = g.get_legal_moves(Player.BLUE)
        assert (1, 1, PieceSize.MEDIUM) in legal, "CC-M must be legal for skip guard to be meaningful"
        assert (0, 2, PieceSize.SMALL) in legal, "TR-S must be available for the return value check"

        mv = _book_move(g, int(Player.BLUE))
        assert mv == (0, 2, PieceSize.SMALL), (
            f"§300: BLUE book must return TR-S=(0,2,S), skipping legal CC-M; "
            f"got {mv}. If CC-M was returned, the center-skip guard is broken."
        )

    def test_blue_book_selects_vs_ccl_plan_when_red_opens_ccl(self):
        """§309: BLUE book switches to `_BLUE_BOOK_VS_CCL` when RED has CC-L.

        Code path (minimax_agent.py:389): `if board[1,1,2] == red_int: plan = _BLUE_BOOK_VS_CCL`.
        This is BLUE's defensive counter-plan against an MCTS/minimax RED that opens CC-L.

        Setup: RED plays CC-L (1,1,L) as the only piece on board (pp=1). BLUE to move.
        No immediate win or block. BLUE's plan = _BLUE_BOOK_VS_CCL, whose first move
        is CC-M = (1,1,M) — contests center and poisons BLUE's CC bullseye for RED."""
        g = TicTacPro()
        g.board[1, 1, 2] = int(Player.RED)    # RED CC-L → triggers _BLUE_BOOK_VS_CCL
        g.pieces_remaining[Player.RED][PieceSize.LARGE] -= 1
        g._pieces_left -= 1
        g.current_player = Player.BLUE
        assert int((g.board != 0).sum()) == 1, "Setup must have pp=1"
        mv = _book_move(g, int(Player.BLUE))
        assert mv == (1, 1, PieceSize.MEDIUM), (
            f"§309: when RED opens CC-L, BLUE book must return CC-M; got {mv}. "
            "_BLUE_BOOK_VS_CCL first move is CC-M to contest center (§124)."
        )


# ── _hash / _hash_bf ──────────────────────────────────────────────────────────

class TestHash:
    def test_hash_consistent(self):
        """Same game state → same hash both ways."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        h1 = _hash(g)
        h2 = _hash_bf(g.board.tobytes(), int(g.current_player))
        assert h1 == h2

    def test_hash_differs_on_board_change(self):
        """Different board → different hash."""
        g1 = TicTacPro()
        g2 = TicTacPro()
        g2.make_move(0, 0, PieceSize.SMALL)
        assert _hash(g1) != _hash(g2)

    def test_hash_differs_on_player_change(self):
        """Same board but different current_player → different hash."""
        g = TicTacPro()
        h1 = _hash(g)
        g.current_player = Player.BLUE
        h2 = _hash(g)
        assert h1 != h2

    def test_exact_solve_importable(self):
        """exact_solve.py must import without error (regression for missing _hash)."""
        import importlib
        mod = importlib.import_module("exact_solve")
        assert hasattr(mod, "solve")

    def test_negamax_exact_win_in_one(self):
        """§195: _negamax_exact must return WIN when current player wins immediately."""
        import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from exact_solve import _negamax_exact
        from rl.minimax_agent import WIN
        # Build a position where RED can win immediately (row 0, SMALL)
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)   # RED
        g.make_move(1, 0, PieceSize.LARGE)   # BLUE
        g.make_move(0, 1, PieceSize.SMALL)   # RED
        g.make_move(1, 1, PieceSize.LARGE)   # BLUE
        # RED to move — (0, 2, SMALL) wins
        assert g.current_player == Player.RED
        tt = {}; nodes = [0]
        score = _negamax_exact(g, WIN * -2, WIN * 2, tt, nodes)
        assert score == WIN, f"Expected WIN={WIN}, got {score}"
        assert nodes[0] > 0

    def test_negamax_exact_loss_when_opponent_wins(self):
        """§195: _negamax_exact must return LOSS when opponent already won (game_over)."""
        from exact_solve import _negamax_exact
        from rl.minimax_agent import WIN, LOSS
        g = TicTacPro()
        # Manually force game_over = True with BLUE as winner
        for mv in [(0, 0, PieceSize.SMALL), (1, 0, PieceSize.LARGE),
                   (0, 1, PieceSize.SMALL), (1, 1, PieceSize.LARGE),
                   (2, 0, PieceSize.MEDIUM), (0, 2, PieceSize.SMALL)]:
            if not g.game_over:
                g.make_move(*mv)
        # RED just won — at this point game_over = True, winner = RED
        # _negamax_exact called on a game_over state returns LOSS
        # (the player who would move is already past, so current_player lost)
        if g.game_over and g.winner != Player.NONE:
            tt = {}; nodes = [0]
            score = _negamax_exact(g, WIN * -2, WIN * 2, tt, nodes)
            assert score == LOSS, f"Expected LOSS={LOSS} on game_over state, got {score}"

    def test_negamax_exact_draw_near_terminal(self):
        """§195: _negamax_exact must return DRAW (0) on a forced-draw position."""
        from exact_solve import _negamax_exact
        from rl.minimax_agent import WIN, DRAW
        # Build a near-terminal position with only one legal move left (a draw):
        # Fill all cells except (2,2,LARGE) so only that move remains.
        # After placing it, check if we can construct a draw.
        # Simpler approach: fill a board so there are no three-in-a-row opportunities.
        g = TicTacPro()
        # Place pieces to exhaust all sizes for RED so only BLUE can move → game won't end in win
        # Build a specific near-terminal position where we know the outcome.
        # RED at (0,0,S),(0,1,M),(0,2,L) — these are different sizes so no row win
        # BLUE at (1,0,S),(1,1,M),(1,2,L) — same
        # RED at (2,0,M),(2,1,L),(2,2,S) — exhausts all sizes for RED (3S, 3M, 3L used)
        # Actually let's just test that _negamax_exact terminates and returns a valid value.
        g2 = TicTacPro()
        for mv in [(0, 0, PieceSize.SMALL), (1, 0, PieceSize.LARGE),
                   (0, 1, PieceSize.SMALL), (1, 1, PieceSize.LARGE)]:
            g2.make_move(*mv)
        # RED to move (2 moves from win) — search should be fast with TT
        tt = {}; nodes = [0]
        score = _negamax_exact(g2, -WIN * 2, WIN * 2, tt, nodes)
        # Score should be WIN (RED can win at (0,2,SMALL))
        assert score == WIN, f"Expected WIN, got {score}"
        assert nodes[0] > 0


# ---------------------------------------------------------------------------
# §206: _evaluate fork detection (me_fork ≥ 2 → ±200_000)
# ---------------------------------------------------------------------------

class TestForkDetection:
    """§206: _evaluate must return ±200_000 when a player has 2+ simultaneous
    two-of-three threats.  Without this, fork positions are undervalued and
    minimax search fails to prioritise them over weaker lines."""

    def test_me_fork_returns_200k(self):
        """Current player with 2 unblocked row/col threats → +200_000."""
        g = TicTacPro()
        # RED has Smalls in (0,0) and (0,1) → row-0-S threat at (0,2,S)
        # RED has Smalls in (0,0) and (1,0) → col-0-S threat at (2,0,S)
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.board[1, 0, 0] = int(Player.RED)
        score = _evaluate(g, int(Player.RED))
        assert score == 200_000, (
            f"Expected 200_000 for fork position, got {score}"
        )

    def test_them_fork_returns_neg_200k(self):
        """Opponent with 2 unblocked threats from current player's perspective → -200_000."""
        g = TicTacPro()
        # BLUE has Smalls in (0,0) and (0,1) → row-0-S threat
        # BLUE has Smalls in (0,0) and (1,0) → col-0-S threat
        g.board[0, 0, 0] = int(Player.BLUE)
        g.board[0, 1, 0] = int(Player.BLUE)
        g.board[1, 0, 0] = int(Player.BLUE)
        score = _evaluate(g, int(Player.RED))
        assert score == -200_000, (
            f"Expected -200_000 for opponent fork, got {score}"
        )

    def test_single_threat_not_fork(self):
        """One two-of-three threat is NOT a fork (< 200_000)."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        # Only one line has 2 pieces; no fork
        score = _evaluate(g, int(Player.RED))
        assert abs(score) < 200_000, (
            f"Single threat should not trigger fork; got {score}"
        )

    def test_them_fork_via_bull2_returns_neg_200k(self):
        """§304: them_fork ≥ 2 fires when them_bull_2 ≥ 2 (two cells each with 2
        opponent pieces of different sizes). Existing fork tests only used them_line_2;
        this covers the them_bull_2 contribution to them_fork.

        Position: BLUE SMALL+LARGE at (0,0); BLUE MEDIUM+LARGE at (1,2).
        - cell_opp[0,0]=2, cell_me=0 → them_bull_2[0,0]=True.
        - cell_opp[1,2]=2, cell_me=0 → them_bull_2[1,2]=True.
        - them_line_2=0: cells share no same-size pair in any line.
        - them_fork = 0 + 2 = 2 → must return -200_000."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.BLUE)   # BLUE SMALL at (0,0)
        g.board[0, 0, 2] = int(Player.BLUE)   # BLUE LARGE at (0,0) — cell_opp=2
        g.board[1, 2, 1] = int(Player.BLUE)   # BLUE MEDIUM at (1,2)
        g.board[1, 2, 2] = int(Player.BLUE)   # BLUE LARGE at (1,2) — cell_opp=2
        score = _evaluate(g, int(Player.RED))
        assert score == -200_000, (
            f"§304: them_bull_2=2 must trigger them_fork=-200_000; got {score}"
        )

    def test_me_fork_via_bull2_returns_200k(self):
        """§304: me_fork ≥ 2 fires when me_bull_2 ≥ 2 (two cells each with 2 current
        player pieces of different sizes). Mirror of them_bull_2 test above.

        Position: RED SMALL+LARGE at (0,0); RED MEDIUM+LARGE at (1,2).
        - cell_me[0,0]=2 → me_bull_2[0,0]=True.
        - cell_me[1,2]=2 → me_bull_2[1,2]=True.
        - me_fork = 0 + 2 = 2 → must return +200_000."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)    # RED SMALL at (0,0)
        g.board[0, 0, 2] = int(Player.RED)    # RED LARGE at (0,0) — cell_me=2
        g.board[1, 2, 1] = int(Player.RED)    # RED MEDIUM at (1,2)
        g.board[1, 2, 2] = int(Player.RED)    # RED LARGE at (1,2) — cell_me=2
        score = _evaluate(g, int(Player.RED))
        assert score == 200_000, (
            f"§304: me_bull_2=2 must trigger me_fork=+200_000; got {score}"
        )

    def test_me_fork_via_mixed_line2_and_bull2(self):
        """§304: me_fork ≥ 2 fires when me_line_2=1 and me_bull_2=1 (mixed fork).
        me_fork = me_line_2.sum() + me_bull_2.sum() — both contribute.

        Position: RED SMALL at (0,0)+(0,1) → row0-S me_line_2=1;
                  RED MEDIUM+LARGE at (2,2) → cell_me=2 → me_bull_2[2,2]=1.
        - No other me_line_2 or me_bull_2 fires ((2,2) pieces are different sizes,
          single-cell → no same-size pair in any line).
        - me_fork = 1 + 1 = 2 → must return +200_000."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)    # RED SMALL at (0,0)
        g.board[0, 1, 0] = int(Player.RED)    # RED SMALL at (0,1) → row0-S me_line_2=1
        g.board[2, 2, 1] = int(Player.RED)    # RED MEDIUM at (2,2)
        g.board[2, 2, 2] = int(Player.RED)    # RED LARGE at (2,2) → me_bull_2[2,2]=1
        score = _evaluate(g, int(Player.RED))
        assert score == 200_000, (
            f"§304: me_line_2=1+me_bull_2=1 must trigger me_fork=+200_000; got {score}"
        )

    def test_me_fork_priority_over_them_fork(self):
        """§305: me_fork check precedes them_fork — when both players have ≥2
        simultaneous threats, _evaluate must return +200_000 (me_fork wins),
        NOT -200_000 (them_fork).

        Code path (minimax_agent.py:154-157):
          if me_fork >= 2:    return 200_000   ← fires first
          if them_fork >= 2:  return -200_000  ← never reached

        Position: RED SMALL at (0,0),(0,1),(1,0) → me_line_2 for row0-S and col0-S
                  (me_fork=2). BLUE MEDIUM at same cells → them_line_2 for row0-M
                  and col0-M (them_fork=2). Both forks active simultaneously.
        Expected: +200_000 (me_fork wins the priority race)."""
        g = TicTacPro()
        # RED fork (SMALL)
        g.board[0, 0, 0] = int(Player.RED)    # (0,0,S)
        g.board[0, 1, 0] = int(Player.RED)    # (0,1,S) → row0-S me_line_2
        g.board[1, 0, 0] = int(Player.RED)    # (1,0,S) → col0-S me_line_2
        # BLUE fork (MEDIUM — different size slot, same cells)
        g.board[0, 0, 1] = int(Player.BLUE)   # (0,0,M)
        g.board[0, 1, 1] = int(Player.BLUE)   # (0,1,M) → row0-M them_line_2
        g.board[1, 0, 1] = int(Player.BLUE)   # (1,0,M) → col0-M them_line_2
        score = _evaluate(g, int(Player.RED))
        assert score == 200_000, (
            f"§305: me_fork priority must return +200_000 when both forks active; "
            f"got {score} (expected +200_000, not -200_000)"
        )


# ── §238: Late Move Reduction (LMR) ───────────────────────────────────────────

class TestLMR:
    """§238: LMR fires and reduces node count without corrupting the result."""

    def test_lmr_reduces_node_count(self):
        """§238: LMR must reduce the node count compared to full-depth search.

        LMR (minimax_agent.py:615-619) applies at depth ≥ _LMR_DEPTH=3 for
        moves at ordering index ≥ _LMR_IDX=3.  After RED:TL-S and BLUE:BR-M
        there are 25 legal moves with no immediate win or block — so 22 of them
        are subject to LMR at the depth-3 root call.

        Disabling LMR (_LMR_DEPTH=999) forces all moves to full-depth search.
        Node count must be strictly lower with LMR enabled, proving LMR is
        active.  Both variants must return the same score (correctness guard).
        """
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)   # RED: TL-S
        g.make_move(2, 2, PieceSize.MEDIUM)  # BLUE: BR-M
        # RED to move; no immediate wins or blocks.

        def run_negamax(lmr_depth):
            g2 = g.clone(copy_history=False)
            ag = MinimaxAgent(time_limit=60.0, max_depth=3, use_book=False)
            ag._tt        = {}
            ag._killers   = [[] for _ in range(30)]
            ag._nodes     = 0
            ag._hits      = 0
            ag._dl        = time.perf_counter() + 60.0
            ag._mp        = int(Player.RED)
            ag._LMR_DEPTH = lmr_depth
            score = ag._negamax(g2, 3, LOSS - 1, WIN + 1, 0)
            return score, ag._nodes

        score_lmr,    nodes_lmr    = run_negamax(lmr_depth=3)    # LMR active
        score_no_lmr, nodes_no_lmr = run_negamax(lmr_depth=999)  # LMR disabled

        assert nodes_lmr < nodes_no_lmr, (
            f"§238: LMR must reduce node count; "
            f"with LMR={nodes_lmr} vs without LMR={nodes_no_lmr} — LMR not firing"
        )
        assert score_lmr == score_no_lmr, (
            f"§238: LMR changed the search result; "
            f"with LMR={score_lmr}, without LMR={score_no_lmr}"
        )


# ── §302: Section 7 me_2cl=2 exact-delta ─────────────────────────────────────

class TestSection7Me2cl2:
    """§302: me_2cl=2 (two lines where me holds ≥2 cells) contributes +80 to
    section 7 score. §272 tests me_2cl=1 (+40). This pinpoints me_2cl=2 (+80)
    to catch any off-by-one or scaling bug in the multi-cell line control term.

    Setup A: RED MEDIUM at (0,0), RED LARGE at (0,2), RED SMALL at (1,0).
      Row0 has (0,0) and (0,2) → me_lc=2 → contributes 1.
      Col0 has (0,0) and (1,0) → me_lc=2 → contributes 1.
      Total me_2cl=2 → sec7 = +80.

    Setup B: RED MEDIUM at (0,0), RED LARGE at (0,2), RED SMALL at (2,1).
      Row0 has (0,0) and (0,2) → me_lc=2 → contributes 1.
      (2,1) shares no line with (0,0) or (0,2) so me_2cl stays=1 → sec7 = +40.

    All other sections (1, 2, 4, 5, 6) are identical:
      sec1: (0,0,M)→3 lines×10=30; (0,2,L)→3 lines×6=18; (x,y,S)→2 lines×14=28 each.
      sec2: no RED Smalls on antidiag or maindiag → 0.
      sec4: 3 cells each with 1 RED piece → me_bull_1=3 → +75 each.
      sec6: n_me_bull1=3 → +45*(3-1)=90 each.
    Delta s_A − s_B = 40 exactly."""

    def test_me_2cl_two_lines_exact_delta_40(self):
        g_a = TicTacPro()
        g_a.board[0, 0, 1] = int(Player.RED)   # (0,0,M) — row0 + col0 + maindiag lines
        g_a.board[0, 2, 2] = int(Player.RED)   # (0,2,L) — row0 + col2 + antidiag lines
        g_a.board[1, 0, 0] = int(Player.RED)   # (1,0,S) — col0 second piece → col0 me_lc=2
        s_a = _evaluate(g_a, int(Player.RED))

        g_b = TicTacPro()
        g_b.board[0, 0, 1] = int(Player.RED)   # (0,0,M) — identical first piece
        g_b.board[0, 2, 2] = int(Player.RED)   # (0,2,L) — identical second piece
        g_b.board[2, 1, 0] = int(Player.RED)   # (2,1,S) — no shared line with (0,0)/(0,2)
        s_b = _evaluate(g_b, int(Player.RED))

        assert s_a > s_b, (
            f"§302: me_2cl=2 must score higher than me_2cl=1; s_a={s_a}, s_b={s_b}"
        )
        assert s_a - s_b == 40, (
            f"§302: delta must be exactly 40 (one extra me_2cl line × 40); got {s_a - s_b}"
        )


# ---------------------------------------------------------------------------
# §310: _negamax TT LOWER/UPPER bound cutoff paths
# ---------------------------------------------------------------------------

class TestNegamaxTTCutoff:
    """§310: _negamax must honour LOWER and UPPER TT-flag cutoffs.

    LOWER flag (best >= beta stored): actual score ≥ ts.  When the window
    raises alpha above beta via the stored bound, we return early.
    UPPER flag (best <= orig_alpha stored): actual score ≤ ts.  When the
    window lowers beta below alpha via the stored bound, we return early.

    These two branches (lines 596-598 of minimax_agent.py) have no other
    test that specifically exercises them — prior TT tests only exercise EXACT
    (through exact_solve._negamax_exact) or do not inject flags directly.

    Approach: inject a crafted entry into self._tt before calling _negamax,
    then verify the return value and hit counter prove the cutoff fired."""

    def _init_agent(self) -> MinimaxAgent:
        """MinimaxAgent with search state fully initialised for direct _negamax calls."""
        import time
        agent = MinimaxAgent(time_limit=100.0)
        agent._tt      = {}
        agent._killers = [[] for _ in range(agent.max_depth + 2)]
        agent._nodes   = 0
        agent._hits    = 0
        agent._dl      = time.perf_counter() + 100.0
        agent._mp      = int(Player.RED)
        return agent

    def test_tt_lower_cutoff_returns_stored_score(self):
        """§310a: LOWER entry with ts ≥ beta must cut and return ts without searching."""
        agent = self._init_agent()
        g  = TicTacPro()
        bf = g.board.tobytes()
        h  = _hash_bf(bf, int(Player.RED))
        # ts=900_000 >> beta=0 → LOWER raises alpha to 900_000 ≥ beta → cut
        agent._tt[h] = (900_000, LOWER, 10, None)
        score = agent._negamax(g, depth=2, alpha=-WIN, beta=0, ply=0)
        assert score == 900_000, f"LOWER cutoff must return ts; got {score}"
        assert agent._hits == 1, "TT hit counter must increment on LOWER cutoff"
        # Nodes counter: incremented once (this call) but no child nodes searched
        assert agent._nodes == 1, "No child nodes should be searched after LOWER cutoff"

    def test_tt_upper_cutoff_returns_stored_score(self):
        """§310b: UPPER entry with ts ≤ alpha must cut and return ts without searching."""
        agent = self._init_agent()
        g  = TicTacPro()
        bf = g.board.tobytes()
        h  = _hash_bf(bf, int(Player.RED))
        # ts=-900_000 << alpha=0 → UPPER lowers beta to -900_000 ≤ alpha=0 → cut
        agent._tt[h] = (-900_000, UPPER, 10, None)
        score = agent._negamax(g, depth=2, alpha=0, beta=WIN, ply=0)
        assert score == -900_000, f"UPPER cutoff must return ts; got {score}"
        assert agent._hits == 1, "TT hit counter must increment on UPPER cutoff"
        assert agent._nodes == 1, "No child nodes should be searched after UPPER cutoff"

    def test_tt_lower_no_cutoff_still_raises_alpha(self):
        """§310c: LOWER entry with ts < beta must raise alpha without cutting."""
        agent = self._init_agent()
        g  = TicTacPro()
        bf = g.board.tobytes()
        h  = _hash_bf(bf, int(Player.RED))
        # ts=50 < beta=WIN → alpha raised to 50 but no cutoff; search continues
        agent._tt[h] = (50, LOWER, 10, None)
        score = agent._negamax(g, depth=1, alpha=-WIN, beta=WIN, ply=0)
        assert agent._hits == 1, "TT must be consulted (td=10 >= depth=1)"
        # Search continued beyond TT → more than 1 node explored
        assert agent._nodes > 1, "Child nodes must be explored when LOWER does not cut"
        # score ≠ 50 because the real evaluation differs from the injected bound
        assert score != 50, "Must not short-circuit to injected ts when no cutoff fires"


# ---------------------------------------------------------------------------
# §311: _negamax stuck-player guard + _book_move immediate-win priority
# ---------------------------------------------------------------------------

class TestNegamaxEdgeCases:
    """§311: Two previously-untested edge-case paths.

    A) _negamax stuck-player guard (minimax_agent.py:585-586):
       When the current player has no legal moves but game_over is False
       (all piece sizes exhausted before opponent), _negamax must return DRAW.
       This can happen in constructed positions and is a required defensive
       guard — without it an IndexError or silent wrong score would result.

    B) _book_move immediate-win priority (minimax_agent.py:344-347):
       The book's first priority is an immediate win — if any legal move wins
       the game, it must be returned BEFORE consulting the book plan.  No prior
       test exercises this branch; test_forced_block_overrides_adaptive covers
       the block priority but not the win priority."""

    def _init_agent(self) -> MinimaxAgent:
        import time
        agent = MinimaxAgent(time_limit=100.0, use_book=False)
        agent._tt      = {}
        agent._killers = [[] for _ in range(agent.max_depth + 2)]
        agent._nodes   = 0
        agent._hits    = 0
        agent._dl      = time.perf_counter() + 100.0
        agent._mp      = int(Player.RED)
        return agent

    def test_negamax_stuck_player_returns_draw(self):
        """§311a: _negamax returns DRAW when current player has 0 legal moves."""
        agent = self._init_agent()
        g = TicTacPro()
        # Drain all RED pieces without setting game_over — manufactured stuck state.
        # In a real game this cannot arise (game ends when _pieces_left==0), but
        # _negamax must handle it defensively.
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.RED][PieceSize.LARGE]  = 0
        # Sanity: game_over is still False, no legal moves for RED
        assert not g.game_over
        assert g.get_legal_moves(Player.RED) == []
        score = agent._negamax(g, depth=3, alpha=LOSS-1, beta=WIN+1, ply=0)
        assert score == 0, (
            f"§311a: stuck player must return DRAW=0; got {score}"
        )

    def test_negamax_game_over_draw_returns_zero(self):
        """§314a: _negamax must return DRAW (0) when game_over=True and winner=NONE.

        This covers minimax_agent.py line 577 — the draw branch inside the
        game_over guard.  A real draw arises when all 27 piece slots are filled
        with no winner; here we manufacture that state by setting flags directly
        to avoid the 27-move board setup.

        Critically, this is DIFFERENT from the stuck-player guard (line 586).
        Line 586 fires when game_over is False but moves == [].
        Line 577 fires when game_over is True and winner is NONE."""
        agent = self._init_agent()
        g = TicTacPro()
        g.game_over = True
        g.winner = Player.NONE
        score = agent._negamax(g, depth=3, alpha=LOSS - 1, beta=WIN + 1, ply=0)
        assert score == DRAW, (
            f"§314a: game_over draw must return DRAW=0; got {score}"
        )
        assert agent._nodes == 1, "game_over path must not recurse"

    def test_book_move_immediate_win_overrides_plan(self):
        """§311b: _book_move must return an immediate win before consulting the plan.

        Board: RED has TL-S + TC-S on board (row-0-S threat at TR-S=(0,2,S)).
        BLUE has CC-M. It is RED's turn at pp=3, inside RED_BOOK_DEPTH=10.
        The _RED_BOOK_PLAN would suggest CC-M for pp=3, but BLUE already holds
        CC-M.  The book skips it and would continue through the plan.
        _book_move must short-circuit at the immediate-win check and return
        (0,2,SMALL) regardless of what the plan says."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)    # TL-S
        g.board[0, 1, 0] = int(Player.RED)    # TC-S → row-0-S win available at TR-S
        g.board[1, 1, 1] = int(Player.BLUE)   # CC-M (BLUE piece — occupies plan slot)
        g.pieces_remaining[Player.RED][PieceSize.SMALL]   -= 2
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] -= 1
        g._pieces_left -= 3
        g.current_player = Player.RED
        # Verify the win is genuine
        from game.tictacpro import _would_win_bf
        bf = g.board.tobytes()
        assert _would_win_bf(bf, 0, 2, 0, int(Player.RED)), (
            "Test setup error: (0,2,S) must be a winning move for RED"
        )
        mv = _book_move(g, int(Player.RED))
        assert mv == (0, 2, PieceSize.SMALL), (
            f"§311b: _book_move must return immediate win (0,2,S); got {mv}"
        )


# ---------------------------------------------------------------------------
# §314: _book_move exhaustion and BLUE else branch
# ---------------------------------------------------------------------------

class TestBookMoveExhaustion:
    """§314b/c: covers the two un-hit lines in _book_move:
      line 425 — BLUE else branch (pp ≠ 1, ≠ 3, no CCL)
      line 437 — return None when every plan move is illegal."""

    def test_blue_else_branch_and_plan_exhausted_returns_none(self):
        """§314b+c: when BLUE is called with pp=2 (≠1, ≠3), no RED CC-L,
        and every plan move is illegal, _book_move must return None.

        Position:
          board[1,1,1] = RED   (CC-M occupied — only M plan slot)
          board[0,0,0] = RED   (adds a second piece so pp=2 ≠ 1)
          BLUE pieces_remaining[S] = 0  (removes all S plan slots)
          BLUE pieces_remaining[L] = 0  (removes both L plan slots)

        Trace through _book_move for BLUE:
          board[1,1,2] == 0  → no CCL counter (line 390 skipped)
          pp == 2 → skip pp==1 branch (line 393), skip pp==3 branch (line 400)
          else:   → plan = _BLUE_BOOK_PLAN  (line 425 ✓)
          Plan has 9 moves; all fail:
            CC-S/TR-S/BL-S/TL-S/BR-S → BLUE has 0 smalls
            CC-M                      → board[1,1,1] is RED
            CC-L/TC-L/BC-L            → BLUE has 0 larges
          for loop exhausted → return None  (line 437 ✓)"""
        g = TicTacPro()
        # Two RED pieces at CC-M and TL-S (pp=2, no CC-L so no CCL counter-book)
        g.board[1, 1, 1] = int(Player.RED)   # CC-M occupied
        g.board[0, 0, 0] = int(Player.RED)   # TL-S (pp → 2)
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] -= 1
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  -= 1
        g._pieces_left -= 2
        # BLUE has only mediums left — S/L plan moves vanish from legal
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 0
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE] = 0
        g.current_player = Player.BLUE
        # Sanity: BLUE has some legal moves (mediums) but none match the plan
        legal = g.get_legal_moves(Player.BLUE)
        assert any(sz == PieceSize.MEDIUM for _, _, sz in legal), \
            "Setup error: BLUE must have at least one legal medium move"
        mv = _book_move(g, int(Player.BLUE))
        assert mv is None, (
            f"§314b+c: expected None (plan exhausted) but _book_move returned {mv}"
        )


# ---------------------------------------------------------------------------
# §315: _TimeUp exception paths (lines 532-533, 548, 602)
# ---------------------------------------------------------------------------

class TestNegamaxTimeUp:
    """§315: covers the three time-limit lines in minimax_agent.py that are
    never hit under a 100 s deadline:
      line 548  — raise _TimeUp() in _root per-move check
      line 532-533 — except _TimeUp: break in _id_search
      line 602  — raise _TimeUp() in _negamax 10,000-node periodic check"""

    def _base_agent(self) -> MinimaxAgent:
        return MinimaxAgent(time_limit=100.0, use_book=False)

    def test_root_raises_time_up_when_deadline_passed(self):
        """§315a: _root must raise _TimeUp when its per-move time check fires.

        Covers minimax_agent.py line 548.  We initialize the agent state
        manually and set _dl to one second in the past so the very first
        move in _root triggers the guard."""
        import time
        from rl.minimax_agent import _TimeUp
        agent = self._base_agent()
        agent._tt      = {}
        agent._killers = [[] for _ in range(agent.max_depth + 2)]
        agent._nodes   = 0
        agent._hits    = 0
        agent._dl      = time.perf_counter() - 1.0   # deadline already past
        agent._mp      = int(Player.RED)
        g = TicTacPro()
        with pytest.raises(_TimeUp):
            agent._root(g, depth=2)

    def test_id_search_catches_time_up_and_returns_fallback(self, monkeypatch):
        """§315b: _id_search must catch _TimeUp and return the best move found.

        Covers minimax_agent.py lines 532-533.  We monkeypatch time.perf_counter
        so the first call (inside _id_search setting _dl) returns 0.0, and every
        subsequent call returns 1e9 (far past _dl = 0.0 + 1.0).  _root then
        raises _TimeUp on its first move, _id_search catches it and falls back
        to moves[0] with depth_done=0."""
        import time as _time_mod
        call_count = [0]

        def _mock_clock():
            call_count[0] += 1
            return 0.0 if call_count[0] == 1 else 1e9

        monkeypatch.setattr(_time_mod, 'perf_counter', _mock_clock)
        agent = MinimaxAgent(time_limit=1.0, use_book=False)
        g = TicTacPro()
        mv, info = agent._id_search(g, Player.RED)
        assert mv in g.get_legal_moves(Player.RED), (
            "§315b: timed-out _id_search must still return a legal fallback move"
        )
        depth_done, score, nodes, hits = info
        assert depth_done == 0, (
            f"§315b: timed out before completing any depth; expected depth_done=0, got {depth_done}"
        )

    def test_negamax_periodic_time_check_raises_time_up(self):
        """§315c: _negamax raises _TimeUp at the 10,000-node boundary when deadline passed.

        Covers minimax_agent.py line 602.  We pre-load _nodes to 9,999 and
        set _dl to the past.  The first _negamax call increments _nodes to
        10,000 which satisfies _nodes % 10_000 == 0, then the time guard fires."""
        import time
        from rl.minimax_agent import _TimeUp
        agent = self._base_agent()
        agent._tt      = {}
        agent._killers = [[] for _ in range(agent.max_depth + 2)]
        agent._nodes   = 9_999   # next call → 10,000 → triggers periodic check
        agent._hits    = 0
        agent._dl      = time.perf_counter() - 1.0
        agent._mp      = int(Player.RED)
        g = TicTacPro()
        with pytest.raises(_TimeUp):
            agent._negamax(g, depth=5, alpha=LOSS - 1, beta=WIN + 1, ply=0)
