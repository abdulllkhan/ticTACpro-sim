"""Unit and regression tests for MCTSAgent and NeuralMCTSNode helpers."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from game.tictacpro import TicTacPro, Player, PieceSize, _would_win_bf
from rl.mcts_agent import (MCTSAgent, _push_undo, _pop_undo, _push_known_undo,
                            _sort_untried, _apply_move, _LAZY)


# ── push/pop undo correctness ─────────────────────────────────────────────────

class TestPushPopUndo:
    def _game(self):
        return TicTacPro()

    def test_push_undo_restores_board(self):
        g = self._game()
        before = g.board.copy()
        tok = _push_undo(g, 0, 0, PieceSize.SMALL)
        _pop_undo(g, tok)
        assert (g.board == before).all()

    def test_push_undo_restores_pieces_remaining(self):
        g = self._game()
        before = g.pieces_remaining[Player.RED][PieceSize.SMALL]
        tok = _push_undo(g, 0, 0, PieceSize.SMALL)
        _pop_undo(g, tok)
        assert g.pieces_remaining[Player.RED][PieceSize.SMALL] == before

    def test_push_undo_restores_pieces_left(self):
        g = self._game()
        tok = _push_undo(g, 0, 0, PieceSize.SMALL)
        _pop_undo(g, tok)
        assert g._pieces_left == 18

    def test_push_undo_restores_current_player(self):
        g = self._game()
        tok = _push_undo(g, 0, 0, PieceSize.SMALL)
        _pop_undo(g, tok)
        assert g.current_player == Player.RED

    def test_push_known_undo_win_restores_state(self):
        g = self._game()
        # Set up a winning position for RED
        g.make_move(0, 0, PieceSize.SMALL)
        g.make_move(1, 0, PieceSize.SMALL)
        g.make_move(0, 1, PieceSize.SMALL)
        g.make_move(1, 1, PieceSize.SMALL)
        # RED about to win at (0,2,S) — don't use make_move (switches player)
        # Manually set current player back to RED for the test
        g.current_player = Player.RED
        prev_pl = g._pieces_left
        tok = _push_known_undo(g, 0, 2, PieceSize.SMALL, is_win=False)
        _pop_undo(g, tok)
        assert g._pieces_left == prev_pl

    def test_multiple_push_pop_roundtrip(self):
        g = self._game()
        original_board = g.board.copy()
        t1 = _push_undo(g, 0, 0, PieceSize.SMALL)
        t2 = _push_undo(g, 1, 1, PieceSize.SMALL)
        _pop_undo(g, t2)
        _pop_undo(g, t1)
        assert (g.board == original_board).all()
        assert g._pieces_left == 18
        assert g.current_player == Player.RED

    def test_push_undo_win_restores_winner_and_game_over(self):
        """Winning move pushed then undone must fully restore game_over/winner state."""
        g = self._game()
        # Build RED 2-in-a-row: (0,0,S), then BLUE (1,0,S), then RED (0,1,S)
        g.make_move(0, 0, PieceSize.SMALL)  # RED
        g.make_move(1, 0, PieceSize.SMALL)  # BLUE
        g.make_move(0, 1, PieceSize.SMALL)  # RED
        # BLUE move so it's RED's turn again
        g.make_move(2, 2, PieceSize.SMALL)  # BLUE
        # Now (0,2,S) completes RED's row — a winning move
        assert not g.game_over
        prev_winner = g.winner
        prev_game_over = g.game_over
        prev_player = g.current_player
        tok = _push_undo(g, 0, 2, PieceSize.SMALL)
        assert g.game_over, "_push_undo must detect and set game_over on win"
        assert g.winner == Player.RED, "_push_undo must set winner to RED"
        _pop_undo(g, tok)
        assert g.game_over == prev_game_over, "game_over not restored after undo"
        assert g.winner == prev_winner, "winner not restored after undo"
        assert g.current_player == prev_player, "current_player not restored after undo"

    def test_push_known_undo_is_win_true_restores_game_over(self):
        """_push_known_undo(is_win=True) must set game_over; undo must clear it."""
        g = self._game()
        g.make_move(0, 0, PieceSize.SMALL)
        g.make_move(1, 0, PieceSize.SMALL)
        g.make_move(0, 1, PieceSize.SMALL)
        g.make_move(2, 2, PieceSize.SMALL)
        tok = _push_known_undo(g, 0, 2, PieceSize.SMALL, is_win=True)
        assert g.game_over
        assert g.winner == Player.RED
        _pop_undo(g, tok)
        assert not g.game_over
        assert g.winner == Player.NONE

    def test_push_undo_draw_detected_and_restored(self):
        """§240: _push_undo must detect draw (game_over=True, winner=NONE) when
        _pieces_left reaches 0, and _pop_undo must restore game_over=False.

        This exercises the `elif g._pieces_left == 0: g.game_over = True` branch
        at mcts_agent.py:146-147 — the same branch exists in _push_known_undo
        but had no dedicated test in TestPushPopUndo.
        """
        g = self._game()
        g._pieces_left = 1   # force: next push will exhaust all pieces
        assert not g.game_over and g.winner == Player.NONE
        tok = _push_undo(g, 0, 0, PieceSize.LARGE)  # non-winning; no 2 Larges in any line
        assert g.game_over is True, "§240: _push_undo must set game_over on draw"
        assert g.winner == Player.NONE, "§240: draw has no winner"
        _pop_undo(g, tok)
        assert g.game_over is False, "§240: _pop_undo must clear game_over after draw"
        assert g._pieces_left == 1, "§240: _pop_undo must restore _pieces_left"
        assert g.winner == Player.NONE, "§240: winner stays NONE after undo"

    def test_push_known_undo_draw_detected_and_restored(self):
        """§240: _push_known_undo must also detect/restore draw — same branch as
        _push_undo (mcts_agent.py:165-166) but skips would_win check."""
        g = self._game()
        g._pieces_left = 1
        tok = _push_known_undo(g, 0, 0, PieceSize.LARGE, is_win=False)
        assert g.game_over is True, "§240: _push_known_undo must set game_over on draw"
        assert g.winner == Player.NONE
        _pop_undo(g, tok)
        assert g.game_over is False, "§240: _pop_undo must restore game_over=False"
        assert g._pieces_left == 1


# ── _apply_move ───────────────────────────────────────────────────────────────

class TestApplyMove:
    """§241: _apply_move — no-undo move application (used in clone-based rollouts).

    _apply_move (mcts_agent.py:111-124) is the fallback for C-rollout-unavailable
    paths and the rollout_agent branch.  It has no undo capability and had no
    direct unit tests — only implicit coverage via _rollout_undo integration tests.
    """

    def test_places_piece_on_board(self):
        """_apply_move must write the piece into board[r,c,sz_idx]."""
        g = TicTacPro()
        _apply_move(g, 1, 1, PieceSize.MEDIUM, is_win=False)
        assert g.board[1, 1, 1] == int(Player.RED)

    def test_decrements_pieces_remaining(self):
        g = TicTacPro()
        before = g.pieces_remaining[Player.RED][PieceSize.SMALL]
        _apply_move(g, 0, 0, PieceSize.SMALL, is_win=False)
        assert g.pieces_remaining[Player.RED][PieceSize.SMALL] == before - 1

    def test_switches_player_on_normal_move(self):
        g = TicTacPro()
        _apply_move(g, 0, 0, PieceSize.SMALL, is_win=False)
        assert g.current_player == Player.BLUE

    def test_win_sets_game_over_and_winner(self):
        """§241: is_win=True path must set game_over=True and winner=current_player."""
        g = TicTacPro()
        _apply_move(g, 0, 0, PieceSize.SMALL, is_win=True)
        assert g.game_over is True
        assert g.winner == Player.RED

    def test_draw_detected_when_pieces_left_zero(self):
        """§241: _apply_move must detect draw (game_over=True, winner=NONE) when
        _pieces_left reaches 0 — the `elif g._pieces_left == 0` branch at line 121."""
        g = TicTacPro()
        g._pieces_left = 1
        _apply_move(g, 0, 0, PieceSize.LARGE, is_win=False)
        assert g.game_over is True, "§241: draw must set game_over"
        assert g.winner == Player.NONE, "§241: draw has no winner"


# ── _sort_untried ─────────────────────────────────────────────────────────────

class TestSortUntried:
    def test_winning_move_last_for_pop(self):
        """Winning move should be at end of list (pop() returns it first)."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        g.make_move(1, 0, PieceSize.SMALL)
        g.make_move(0, 1, PieceSize.SMALL)
        g.make_move(1, 1, PieceSize.SMALL)
        # RED can win at (0,2,S)
        moves = g.get_legal_moves(Player.RED)
        sorted_moves = _sort_untried(g, moves, guided=False)
        assert sorted_moves[-1] == (0, 2, PieceSize.SMALL)

    def test_guided_returns_unmodified(self):
        """guided=True should return moves without sort (no DQN sort)."""
        g = TicTacPro()
        moves = g.get_legal_moves()
        original_order = list(moves)
        result = _sort_untried(g, list(moves), guided=True)
        assert result == original_order

    def test_no_crash_on_single_move(self):
        """Single-move list shouldn't crash."""
        g = TicTacPro()
        moves = [(0, 0, PieceSize.SMALL)]
        result = _sort_untried(g, moves, guided=False)
        assert len(result) == 1

    def test_blocking_move_before_rest(self):
        """Blocking move should appear after rest but before winning in pop() order."""
        g = TicTacPro()
        # Set up BLUE threat at row 0 (S pieces in col 0 and col 1)
        g.make_move(0, 0, PieceSize.SMALL)   # RED
        g.make_move(1, 0, PieceSize.SMALL)   # BLUE — make RED move first
        # Reset: manually set up a threat
        g2 = TicTacPro()
        g2.board[0, 0, 0] = int(Player.BLUE)
        g2.board[0, 1, 0] = int(Player.BLUE)
        g2.pieces_remaining[Player.BLUE][PieceSize.SMALL] -= 2
        g2._pieces_left -= 2
        g2.current_player = Player.RED
        moves = g2.get_legal_moves(Player.RED)
        sorted_moves = _sort_untried(g2, moves, guided=False)
        # (0,2,S) blocks BLUE's row 0 — should appear before any non-blocking rest move in pop() order
        blocking = (0, 2, PieceSize.SMALL)
        if blocking in sorted_moves:
            rest_idxs = [i for i, m in enumerate(sorted_moves)
                         if m != blocking and sorted_moves[-1] != m]
            block_idx = sorted_moves.index(blocking)
            assert all(block_idx > ri for ri in rest_idxs), (
                f"Blocking move at idx {block_idx} should be after all rest moves, "
                f"rest at {rest_idxs}"
            )

    def test_sort_untried_phantom_block_guard(self):
        """§224: _sort_untried must NOT classify a move as blocking when the
        opponent has 0 pieces of that size (same phantom-block class as §79/§82)."""
        g = TicTacPro()
        # BLUE has 2 Small pieces in row 0 — apparent threat at (0,2,S)
        g.board[0, 0, 0] = int(Player.BLUE)
        g.board[0, 1, 0] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        # Drain all BLUE Smalls — threat is phantom
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 0
        g.current_player = Player.RED
        moves = g.get_legal_moves(Player.RED)
        sorted_moves = _sort_untried(g, moves, guided=False)
        # The apparent block at (0,2,S) must NOT be at end of list (not classified blocking)
        # In a sorted list: [rest..., blocking..., winning...], if it's in 'rest',
        # it will NOT be at the last position(s).
        # Check: (0,2,S) is NOT the last element (which would be the "highest priority")
        # unless it's independently prioritized as a winning move (it's not — RED has no threat)
        apparent_block = (0, 2, PieceSize.SMALL)
        if apparent_block in sorted_moves:
            last_idx = len(sorted_moves) - 1
            idx = sorted_moves.index(apparent_block)
            # The last element(s) would be blocking/winning. If it's in rest, it's NOT last.
            # Since there are no winning moves for RED here, the last element is a blocking
            # move only if phantom-block guard failed. Verify it's NOT last.
            # Actually: with no genuine threat (opp has 0 Smalls), there should be no blocking
            # moves at all. So (0,2,S) must be in rest, not at the very end.
            assert idx < last_idx, (
                "§224: (0,2,S) classified as blocking despite BLUE having 0 Smalls — "
                f"phantom-block guard failed. Index {idx} of {len(sorted_moves)}"
            )

    def test_rest_tier_ad_center_popped_before_non_ad_corner(self):
        """§254: Within rest tier, AD center (1,1,S) priority=74 must appear AFTER
        non-AD corner (0,0,S) priority=14 in the result list so pop() yields it first.
        Layout is [rest_low..., rest_high..., blocking..., winning...].
        """
        g = TicTacPro()
        moves = g.get_legal_moves(Player.RED)
        sorted_moves = _sort_untried(g, moves, guided=False)
        ad_center_idx    = sorted_moves.index((1, 1, PieceSize.SMALL))
        non_ad_corner_idx = sorted_moves.index((0, 0, PieceSize.SMALL))
        assert ad_center_idx > non_ad_corner_idx, (
            f"§254: AD center (1,1,S) at idx {ad_center_idx} must be after "
            f"non-AD corner (0,0,S) at idx {non_ad_corner_idx} so MCTS pops it first"
        )

    def test_rest_tier_ad_corner_popped_before_non_ad_corner(self):
        """§279: Within rest tier, AD corner (0,2,S) priority=54 must appear AFTER
        non-AD corner (0,0,S) priority=14 so MCTS pops the AD corner first.

        _REST_PRIORITY_FLAT for (0,2,S): AD_S_bonus(40) + corner(10) + size(4) = 54.
        _REST_PRIORITY_FLAT for (0,0,S): no AD bonus + corner(10) + size(4) = 14.
        Sorted ascending → (0,0,S) at idx X, (0,2,S) at idx Y, with Y > X.
        pop() returns last element first → (0,2,S) is popped before (0,0,S).
        §254 tested AD center vs non-AD corner; this tests AD corner vs non-AD corner,
        exercising the AD_bonus(+40) for non-center AD cells."""
        g = TicTacPro()
        moves = g.get_legal_moves(Player.RED)
        sorted_moves = _sort_untried(g, moves, guided=False)
        ad_corner_idx     = sorted_moves.index((0, 2, PieceSize.SMALL))   # AD corner, priority 54
        non_ad_corner_idx = sorted_moves.index((0, 0, PieceSize.SMALL))   # non-AD corner, priority 14
        assert ad_corner_idx > non_ad_corner_idx, (
            f"§279: AD corner (0,2,S) priority=54 at idx {ad_corner_idx} must be after "
            f"non-AD corner (0,0,S) priority=14 at idx {non_ad_corner_idx}; "
            "AD_bonus (+40) for non-center AD cells must be non-zero"
        )

    def test_rest_tier_corner_popped_before_edge(self):
        """§280: Within rest tier, non-AD corner (2,2,S) priority=14 must appear AFTER
        non-AD edge (0,1,S) priority=4, so MCTS pops the corner first.

        _REST_PRIORITY_FLAT for (2,2,S): non-AD corner(10) + size(4) = 14.
        _REST_PRIORITY_FLAT for (0,1,S): edge non-AD(0) + size(4) = 4.
        Sorted ascending → edge at idx X, corner at idx Y, with Y > X.
        This tests the positional corner bonus (+10) in _REST_PRIORITY_FLAT."""
        g = TicTacPro()
        moves = g.get_legal_moves(Player.RED)
        sorted_moves = _sort_untried(g, moves, guided=False)
        corner_idx = sorted_moves.index((2, 2, PieceSize.SMALL))  # priority 14
        edge_idx   = sorted_moves.index((0, 1, PieceSize.SMALL))  # priority 4
        assert corner_idx > edge_idx, (
            f"§280: non-AD corner (2,2,S) priority=14 at idx {corner_idx} must be after "
            f"non-AD edge (0,1,S) priority=4 at idx {edge_idx}; "
            "corner positional bonus (+10) must lift it above edge (0)"
        )

    def test_two_winning_moves_both_at_tail(self):
        """§281: When RED has exactly two simultaneous winning moves, both appear at the
        tail of the sorted list (winning tier) so pop() yields them ahead of blocking/rest.

        Position: RED SMALL at (0,0), (0,1), (1,0) — exactly 2 winning moves:
          (0,2,S): row0 pair (0,0,S)+(0,1,S)
          (2,0,S): col0 pair (0,0,S)+(1,0,S)
        No other row/col/diag produces 2-of-3 RED Smalls with this 3-piece setup.
        Sorted list must have (0,2,S) and (2,0,S) both in the last 2 positions."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.board[1, 0, 0] = int(Player.RED)
        # Don't decrement pieces_remaining — RED shows 3 SMALL remaining so
        # get_legal_moves still offers (0,2,S) and (2,0,S) as legal placements.
        g._pieces_left -= 3
        moves = g.get_legal_moves(Player.RED)
        assert (0, 2, PieceSize.SMALL) in moves
        assert (2, 0, PieceSize.SMALL) in moves
        sorted_moves = _sort_untried(g, moves, guided=False)
        tail = set(sorted_moves[-2:])
        assert (0, 2, PieceSize.SMALL) in tail, (
            f"§281: (0,2,S) must be in tail (winning tier); "
            f"tail={tail}, full={sorted_moves[-5:]}"
        )
        assert (2, 0, PieceSize.SMALL) in tail, (
            f"§281: (2,0,S) must be in tail (winning tier); "
            f"tail={tail}, full={sorted_moves[-5:]}"
        )


# ── MCTSAgent integration tests ───────────────────────────────────────────────

class TestMCTSAgent:
    @pytest.fixture
    def agent(self):
        # Short time limit for test speed
        return MCTSAgent(time_limit=0.1, max_simulations=500)

    def test_returns_legal_move(self, agent):
        g = TicTacPro()
        mv = agent.get_action(g, Player.RED)
        assert mv in g.get_legal_moves(Player.RED)

    def test_returns_none_on_game_over(self, agent):
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        g.make_move(1, 0, PieceSize.SMALL)
        g.make_move(0, 1, PieceSize.SMALL)
        g.make_move(1, 1, PieceSize.SMALL)
        g.make_move(0, 2, PieceSize.SMALL)   # RED wins
        assert g.game_over
        mv = agent.get_action(g, Player.RED)
        assert mv is None

    def test_immediate_win_taken(self, agent):
        """MCTS fast path: take an immediate win without building a tree."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        mv = agent.get_action(g, Player.RED)
        assert mv == (0, 2, PieceSize.SMALL)

    def test_blocks_opponent_win(self, agent):
        """MCTS should block an immediate BLUE win when RED moves."""
        agent2 = MCTSAgent(time_limit=0.3, max_simulations=2000)
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.BLUE)
        g.board[0, 1, 0] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        g.current_player = Player.RED
        mv = agent2.get_action(g, Player.RED)
        # RED must block at (0,2,S) or BLUE wins next turn
        assert mv == (0, 2, PieceSize.SMALL), f"Expected block at (0,2,S), got {mv}"

    def test_all_simulations_legal(self, agent):
        """MCTS should never place a piece on an occupied slot."""
        g = TicTacPro()
        for _ in range(12):
            if g.game_over:
                break
            mv = agent.get_action(g, g.current_player)
            assert mv is not None
            assert mv in g.get_legal_moves(g.current_player)
            g.make_move(*mv)

    def test_board_unchanged_after_get_action(self, agent):
        """MCTS must not mutate the game state it was given."""
        g = TicTacPro()
        board_before = g.board.copy()
        pl_before = g._pieces_left
        agent.get_action(g, Player.RED)
        assert (g.board == board_before).all()
        assert g._pieces_left == pl_before
        assert g.current_player == Player.RED

    def test_tree_reuse_does_not_crash(self, agent):
        """Tree reuse across moves should not raise exceptions."""
        g = TicTacPro()
        mv1 = agent.get_action(g, g.current_player)
        g.make_move(*mv1)
        mv2 = agent.get_action(g, g.current_player)
        assert mv2 in g.get_legal_moves(g.current_player) or mv2 is None

    def test_reset_clears_tree(self, agent):
        """reset() should discard the cached subtree without errors."""
        g = TicTacPro()
        agent.get_action(g, Player.RED)
        agent.reset()
        assert agent._reuse_node is None

    def test_get_info_returns_tuple(self, agent):
        g = TicTacPro()
        mv, info = agent.get_info(g, Player.RED)
        sims, wr = info
        assert mv in g.get_legal_moves(Player.RED)
        assert sims >= 0
        assert 0.0 <= wr <= 1.0

    def test_get_info_win_fastpath_returns_zero_sims_and_full_winrate(self, agent):
        """§216: get_info must return (0 sims, 1.0 winrate) via fast-path when
        an immediate win exists — no tree search should run."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        mv, (sims, wr) = agent.get_info(g, Player.RED)
        assert mv == (0, 2, PieceSize.SMALL), f"fast-path must return winning move, got {mv}"
        assert sims == 0, f"win fast-path must skip tree search (sims=0), got {sims}"
        assert wr == 1.0, f"win fast-path must return winrate=1.0, got {wr}"

    def test_randomize_expansion_still_legal(self):
        """§214: randomize_expansion=True (§179) must still return legal moves
        and correctly find immediate wins."""
        from rl.mcts_agent import MCTSAgent
        agent = MCTSAgent(time_limit=0.1, max_simulations=200, randomize_expansion=True)
        g = TicTacPro()
        mv = agent.get_action(g, Player.RED)
        assert mv in g.get_legal_moves(), f"randomized agent returned illegal move {mv}"

    def test_randomize_expansion_finds_win(self):
        """§214: With randomize_expansion=True, immediate wins must still be
        returned (via fast-path before shuffling)."""
        from rl.mcts_agent import MCTSAgent
        agent = MCTSAgent(time_limit=0.1, max_simulations=200, randomize_expansion=True)
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        mv = agent.get_action(g, Player.RED)
        assert mv == (0, 2, PieceSize.SMALL), (
            f"randomize_expansion must not prevent taking immediate win, got {mv}"
        )

    def test_tree_reuse_recycles_existing_visits(self):
        """§227: When the opponent responds with a previously-explored move, the
        reused root node must be detached from the old tree (parent=None), proving
        the existing subtree was recycled rather than discarded.

        Observable signal: after _build_tree reuses a child node as the new root,
        it sets root.parent = None.  If reuse DID NOT happen (new root created from
        scratch), the captured child object still has parent pointing to the old
        best-child node (non-None).
        """
        agent = MCTSAgent(time_limit=0.5, max_simulations=2000)
        g = TicTacPro()

        # Round 1: build a tree deep enough that the best child has explored grandchildren.
        mv1 = agent.get_action(g, Player.RED)
        assert agent._reuse_node is not None, "Sanity: _reuse_node must be set after get_action"

        saved = agent._reuse_node
        if not saved.children:
            pytest.skip("Best child has no explored grandchildren — increase sims for reuse test")

        # Pick the first explored grandchild as the opponent's response.
        opp_child  = saved.children[0]
        opp_move   = opp_child.move
        assert opp_child.visits > 0, "Explored grandchild must have visits > 0"

        # Advance the game: our move then the opponent's explored response.
        g.make_move(*mv1)
        g.make_move(*opp_move)

        # Round 2: _build_tree must find opp_child in saved.children and set its parent=None.
        mv2 = agent.get_action(g, g.current_player)
        assert mv2 in g.get_legal_moves(g.current_player), "Round 2 must return a legal move"

        # Verify reuse: _build_tree sets root.parent = None on the reused node.
        assert opp_child.parent is None, (
            "§227: Reused node must have parent=None after _build_tree detaches it — "
            f"parent is still {opp_child.parent} (tree reuse did not fire)"
        )


# ── _rollout_undo terminal state handling ────────────────────────────────────

class TestRolloutUndo:
    """§219: _rollout_undo must return correct values on already-terminal states."""

    @pytest.fixture
    def agent(self):
        return MCTSAgent(time_limit=0.1, max_simulations=200)

    def _red_wins_game(self):
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)  # RED
        g.make_move(1, 0, PieceSize.SMALL)  # BLUE
        g.make_move(0, 1, PieceSize.SMALL)  # RED
        g.make_move(1, 1, PieceSize.SMALL)  # BLUE
        g.make_move(0, 2, PieceSize.SMALL)  # RED wins
        return g

    def test_rollout_win_returns_one_for_winner(self, agent):
        g = self._red_wins_game()
        assert g.game_over and g.winner == Player.RED
        assert agent._rollout_undo(g, Player.RED) == 1.0

    def test_rollout_win_returns_zero_for_loser(self, agent):
        g = self._red_wins_game()
        assert agent._rollout_undo(g, Player.BLUE) == 0.0

    def test_rollout_draw_returns_half(self, agent):
        g = TicTacPro()
        g._pieces_left = 0
        g.game_over = True
        # winner stays Player.NONE (default)
        assert agent._rollout_undo(g, Player.RED) == 0.5
        assert agent._rollout_undo(g, Player.BLUE) == 0.5

    def test_rollout_live_game_returns_valid(self, agent):
        g = TicTacPro()
        result = agent._rollout_undo(g.clone(), Player.RED)
        assert result in (0.0, 0.5, 1.0)


# ── §303: _sort_untried blocking with LARGE pieces ───────────────────────────

class TestSortUntriedLargeBlock:
    """§303: _sort_untried blocking tier must fire for LARGE pieces, not just SMALL.
    §212/§234 test_blocking_move_before_rest used SMALL-only blocking. This pins
    the Large-size path (_WIN_LINES[2]) to catch any sz_idx miscalculation.

    Position: BLUE LARGE at (0,2) and (1,2) → col-2 Large threat at (2,2,L).
    RED has no winning moves. The blocking move (2,2,L) must appear at the tail
    of the sorted list (blocking tier) so MCTS pops it before rest moves."""

    def test_large_block_is_at_tail(self):
        g = TicTacPro()
        g.board[0, 2, 2] = int(Player.BLUE)   # BLUE LARGE at (0,2)
        g.board[1, 2, 2] = int(Player.BLUE)   # BLUE LARGE at (1,2) — col2 2-of-3 threat
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE] -= 2
        g._pieces_left -= 2
        g.current_player = Player.RED
        moves = g.get_legal_moves(Player.RED)
        sorted_moves = _sort_untried(g, moves, guided=False)
        # (2,2,L) blocks col-2 Large → must be in blocking tier at tail
        assert sorted_moves[-1] == (2, 2, PieceSize.LARGE), (
            f"§303: LARGE blocking move (2,2,L) must be last in sorted list "
            f"(blocking tier); tail={sorted_moves[-5:]}"
        )

    def test_large_block_before_rest_moves(self):
        """§303: Every rest move must have a lower index than the LARGE block."""
        g = TicTacPro()
        g.board[0, 2, 2] = int(Player.BLUE)
        g.board[1, 2, 2] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE] -= 2
        g._pieces_left -= 2
        g.current_player = Player.RED
        moves = g.get_legal_moves(Player.RED)
        sorted_moves = _sort_untried(g, moves, guided=False)
        block_idx = sorted_moves.index((2, 2, PieceSize.LARGE))
        rest_idxs = [i for i, m in enumerate(sorted_moves)
                     if m != (2, 2, PieceSize.LARGE)
                     and not _would_win_bf(g.board.tobytes(), m[0], m[1], m[2]-1, int(Player.RED))]
        assert all(block_idx > ri for ri in rest_idxs), (
            f"§303: LARGE block (idx {block_idx}) must be after all rest moves "
            f"(rest indices: {rest_idxs})"
        )


# ---------------------------------------------------------------------------
# §312: _rollout_undo rollout_agent path
# ---------------------------------------------------------------------------

class TestRolloutUndoWithAgent:
    """§312: _rollout_undo rollout_agent branch (mcts_agent.py:618-629).

    When MCTSAgent is constructed with a rollout_agent, _rollout_undo must
    use that agent to drive the rollout via _push_undo/_pop_undo, then restore
    the game state completely.  This path is never exercised by the existing
    TestRolloutUndo tests, which always use MCTSAgent without a rollout_agent
    (defaulting to fast_rollout_c when C extension is available).

    A MinimaxAgent (depth-1, no book) is used as the rollout_agent: it is
    a valid get_action() provider and terminates quickly."""

    def test_rollout_with_agent_returns_valid_result(self):
        """§312a: With rollout_agent, _rollout_undo returns 0.0, 0.5, or 1.0."""
        from rl.minimax_agent import MinimaxAgent
        ra = MinimaxAgent(time_limit=5.0, max_depth=1, use_book=False)
        agent = MCTSAgent(time_limit=0.1, rollout_agent=ra)
        g = TicTacPro()
        result = agent._rollout_undo(g.clone(), Player.RED)
        assert result in (0.0, 0.5, 1.0), (
            f"§312a: rollout_agent path must return 0.0/0.5/1.0; got {result}"
        )

    def test_rollout_with_agent_restores_game_state(self):
        """§312b: _rollout_undo with rollout_agent must undo all moves after rollout.

        The rollout applies moves via _push_undo and reverts via _pop_undo.
        After _rollout_undo returns, g must be byte-identical to its state
        before the call (board, pieces_remaining, current_player, game_over)."""
        from rl.minimax_agent import MinimaxAgent
        import numpy as np
        ra = MinimaxAgent(time_limit=5.0, max_depth=1, use_book=False)
        agent = MCTSAgent(time_limit=0.1, rollout_agent=ra)
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)   # RED
        g.make_move(2, 2, PieceSize.LARGE)   # BLUE
        board_before  = g.board.copy()
        pr_red_before  = dict(g.pieces_remaining[Player.RED])
        pr_blue_before = dict(g.pieces_remaining[Player.BLUE])
        pl_before      = g._pieces_left
        cp_before      = g.current_player
        result = agent._rollout_undo(g, Player.RED)
        assert result in (0.0, 0.5, 1.0)
        assert np.array_equal(g.board, board_before), (
            "§312b: _rollout_undo (rollout_agent) must restore board"
        )
        assert g.pieces_remaining[Player.RED]  == pr_red_before
        assert g.pieces_remaining[Player.BLUE] == pr_blue_before
        assert g._pieces_left    == pl_before
        assert g.current_player  == cp_before

    def test_rollout_with_agent_terminal_state_skips_agent(self):
        """§312c: On an already-terminal state, _rollout_undo returns immediately
        without calling rollout_agent.get_action() (game_over check at line 614)."""
        from rl.minimax_agent import MinimaxAgent
        calls = []
        class CapturingAgent:
            def get_action(self, game, player, epsilon=0.0):
                calls.append(1)
                return None
        agent = MCTSAgent(time_limit=0.1, rollout_agent=CapturingAgent())
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        g.make_move(1, 0, PieceSize.SMALL)
        g.make_move(0, 1, PieceSize.SMALL)
        g.make_move(1, 1, PieceSize.SMALL)
        g.make_move(0, 2, PieceSize.SMALL)   # RED wins → game_over=True
        assert g.game_over
        result = agent._rollout_undo(g, Player.RED)
        assert result == 1.0
        assert calls == [], (
            "§312c: _rollout_undo must not call rollout_agent when game is already over"
        )


# ---------------------------------------------------------------------------
# §316: Python rollout fallback paths (C extension unavailable)
# ---------------------------------------------------------------------------

class TestPythonRolloutFallback:
    """§316: covers the two Python-clone rollout branches that are dead when the
    C extension is available (C_ROLLOUT_AVAILABLE=True in this environment):

      mcts_agent.py lines 509-520 — else-else rollout in _build_tree's sim loop
      mcts_agent.py lines 634-643 — Python clone fallback in _rollout_undo

    Both paths are tested by monkeypatching C_ROLLOUT_AVAILABLE=False so the
    C extension is bypassed.  Also covers line 444 (g.get_legal_moves() fallback
    during lazy expansion when _get_sorted_ex=None).
    """

    def test_build_tree_python_rollout(self, monkeypatch):
        """§316a: _build_tree must run Python clone rollout when C ext unavailable.

        Covers mcts_agent.py lines 511-520 and line 444.
        With C_ROLLOUT_AVAILABLE=False:
          _get_sorted_ex=None  → expansion uses g.get_legal_moves() (line 444)
          _c_rollout=None      → rollout falls to else-else Python clone (line 511)
        The agent must still return a legal move."""
        import rl.mcts_agent as _ma
        monkeypatch.setattr(_ma, 'C_ROLLOUT_AVAILABLE', False)
        agent = MCTSAgent(time_limit=0.1)
        g = TicTacPro()
        mv = agent.get_action(g, Player.RED)
        assert mv in g.get_legal_moves(Player.RED), (
            "§316a: Python rollout fallback must still produce a legal move"
        )

    def test_rollout_undo_python_fallback(self, monkeypatch):
        """§316b: _rollout_undo must run Python clone rollout when C ext unavailable.

        Covers mcts_agent.py lines 634-643.
        _rollout_undo is called directly (no rollout_agent) with C_ROLLOUT_AVAILABLE=False
        so `if C_ROLLOUT_AVAILABLE:` is False and the Python gc.clone() path runs."""
        import rl.mcts_agent as _ma
        monkeypatch.setattr(_ma, 'C_ROLLOUT_AVAILABLE', False)
        agent = MCTSAgent(time_limit=0.1)   # rollout_agent=None
        g = TicTacPro()
        result = agent._rollout_undo(g, Player.RED)
        assert result in (0.0, 0.5, 1.0), (
            f"§316b: Python _rollout_undo fallback must return valid outcome; got {result}"
        )

    def test_rollout_undo_python_fallback_restores_nothing(self, monkeypatch):
        """§316c: Python fallback clones the game — original state must be unchanged."""
        import numpy as np
        import rl.mcts_agent as _ma
        monkeypatch.setattr(_ma, 'C_ROLLOUT_AVAILABLE', False)
        agent = MCTSAgent(time_limit=0.1)
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        board_before = g.board.copy()
        cp_before    = g.current_player
        pl_before    = g._pieces_left
        agent._rollout_undo(g, Player.RED)
        assert np.array_equal(g.board, board_before), \
            "§316c: original game board must not be mutated by Python clone rollout"
        assert g.current_player == cp_before
        assert g._pieces_left   == pl_before


# ---------------------------------------------------------------------------
# §317: MCTSAgent edge paths — game_over, no-children fallback, guided rollout
# ---------------------------------------------------------------------------

class TestMCTSAgentEdgePaths:
    """§317: covers five uncovered lines in mcts_agent.py:
      line 305        — get_info returns (None,(0,0.0)) on game_over state
      lines 294-296   — get_action returns moves[0] when root has no children
      lines 317-318   — get_info returns moves[0] when root has no children
      line 510        — _rollout_undo called from within _build_tree sim loop
                        when rollout_agent is set and game is not terminal
    """

    def test_get_info_game_over_returns_none_and_zero(self):
        """§317a: get_info must return (None, (0, 0.0)) on game_over state.

        Covers mcts_agent.py line 305 — the early-return guard at the top of
        get_info when game.game_over is True.  The existing TestMCTSAgent only
        covers the game_over guard in get_action (not get_info)."""
        agent = MCTSAgent(time_limit=0.1)
        g = TicTacPro()
        g.game_over = True
        g.winner = Player.RED
        mv, info = agent.get_info(g, Player.RED)
        assert mv is None, f"§317a: get_info game_over must return None move; got {mv}"
        assert info == (0, 0.0), f"§317a: get_info game_over must return (0, 0.0); got {info}"

    def test_get_action_no_root_children_returns_fallback(self):
        """§317b: get_action must return moves[0] when root has no children.

        Covers mcts_agent.py lines 294-296.
        With max_simulations=0 the simulation loop never runs, root.children is
        empty, and get_action falls back to returning the first legal move."""
        agent = MCTSAgent(time_limit=100.0, max_simulations=0)
        g = TicTacPro()
        mv = agent.get_action(g, Player.RED)
        assert mv in g.get_legal_moves(Player.RED), (
            f"§317b: no-children fallback must return a legal move; got {mv}"
        )

    def test_get_info_no_root_children_returns_fallback(self):
        """§317c: get_info must return (moves[0], (0, 0.0)) when root has no children.

        Covers mcts_agent.py lines 317-318.
        Same mechanism as §317b: max_simulations=0 keeps root childless."""
        agent = MCTSAgent(time_limit=100.0, max_simulations=0)
        g = TicTacPro()
        mv, info = agent.get_info(g, Player.RED)
        assert mv in g.get_legal_moves(Player.RED), (
            f"§317c: no-children fallback must return a legal move; got {mv}"
        )
        assert info == (0, 0.0), f"§317c: expected (0, 0.0) with no children; got {info}"

    def test_build_tree_calls_rollout_undo_when_c_ext_off(self, monkeypatch):
        """§317d: _build_tree calls _rollout_undo (line 510) when rollout_agent is set
        and C_ROLLOUT_AVAILABLE=False.

        The `elif _rollout_agent is not None:` branch (line 509-510) is only reached
        when _c_rollout=None.  With the C extension available, _c_rollout=fast_rollout_c
        always and line 510 is skipped.  Monkeypatching C_ROLLOUT_AVAILABLE=False makes
        _c_rollout=None, so the rollout falls through to the guided _rollout_undo call.

        A side-effect spy on _rollout_undo counts invocations — must be > 0."""
        import rl.mcts_agent as _ma
        monkeypatch.setattr(_ma, 'C_ROLLOUT_AVAILABLE', False)
        from rl.minimax_agent import MinimaxAgent
        ra = MinimaxAgent(time_limit=0.5, max_depth=1, use_book=False)
        agent = MCTSAgent(time_limit=0.1, rollout_agent=ra)
        call_count = [0]
        _orig = agent._rollout_undo

        def _spy(g, rp):
            call_count[0] += 1
            return _orig(g, rp)

        agent._rollout_undo = _spy
        g = TicTacPro()
        mv = agent.get_action(g, Player.RED)
        assert mv in g.get_legal_moves(Player.RED), (
            f"§317d: guided rollout must still return a legal move; got {mv}"
        )
        assert call_count[0] > 0, (
            "§317d: _rollout_undo must be called by _build_tree when rollout_agent is set "
            "and C_ROLLOUT_AVAILABLE=False (line 510 never executed)"
        )

    def test_stale_game_over_synced_before_rollout_undo(self, monkeypatch):
        """§345: _build_tree must sync g.game_over before calling _rollout_undo.

        Bug (pre-fix): the undo loop restores local _game_over cache but NOT
        g.game_over.  When _pundo is called during expansion and the move wins,
        g.game_over becomes True.  After the undo loop, _game_over is correctly
        False but g.game_over stays stale-True.  _rollout_undo then exits early
        with the wrong (stale) winner, biasing the search.

        Setup: RED has 2 Small pieces in a row (0,0 and 0,1), BLUE to move.
        RED has no immediate win (not their turn); but at depth-2, RED can
        complete the row → depth-2 expansion via _pundo sets g.game_over=True.
        After undo, _rollout_undo must NOT see g.game_over=True (the stale value).

        The fix (§345) adds g.game_over = _game_over / g.winner = _winner just
        before the _rollout_undo call, ensuring the stale flag is cleared."""
        import rl.mcts_agent as _ma
        monkeypatch.setattr(_ma, 'C_ROLLOUT_AVAILABLE', False)
        from rl.minimax_agent import MinimaxAgent
        ra = MinimaxAgent(time_limit=0.5, max_depth=1, use_book=False)
        agent = MCTSAgent(time_limit=0.1, max_simulations=80, rollout_agent=ra)
        # Position: RED has 2 Small in a row (0,0 and 0,1), BLUE to move.
        # Depth-2 expansion (RED plays 0,2,S to win) will stale g.game_over.
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)   # RED
        g.make_move(2, 2, PieceSize.SMALL)   # BLUE (safe)
        g.make_move(0, 1, PieceSize.SMALL)   # RED — 2-in-a-row, BLUE to move
        seen_game_over = []
        _orig = agent._rollout_undo

        def _spy(game_obj, rp):
            seen_game_over.append(game_obj.game_over)
            return _orig(game_obj, rp)

        agent._rollout_undo = _spy
        mv = agent.get_action(g, Player.BLUE)
        assert mv in g.get_legal_moves(Player.BLUE), (
            f"§345: agent must still return a legal move after fix; got {mv}"
        )
        assert seen_game_over, "§345: _rollout_undo must be called at least once"
        assert all(not go for go in seen_game_over), (
            f"§345: g.game_over must be False in every _rollout_undo call (stale sync bug); "
            f"got True in {sum(go for go in seen_game_over)} of {len(seen_game_over)} calls"
        )


# ---------------------------------------------------------------------------
# §319: Draw transitions in MCTS sim loop (lines 417, 465)
# ---------------------------------------------------------------------------

class TestMCTSDrawTransition:
    """§319: covers the two _pcs_left==0 draw guards inside _run_simulations:
      line 465 — draw detected during EXPANSION (last piece placed, no win)
      line 417 — draw detected during SELECTION (walking to an already-expanded
                  draw-terminal child reduces _pcs_left to 0)

    Setup: 17 of 18 pieces placed (RED all 9, BLUE 8 of 9), no winner, BLUE to
    move.  BLUE has 1 Large piece left and 4 legal moves, ALL of which draw.

    With max_simulations=10:
      Sims 1-4: expand each draw child → _pcs_left drops 1→0 during expansion
                → line 465 fires on each expansion
      Sims 5+:  selection walks root→child → _pcs_left drops 1→0 in selection
                → line 417 fires"""

    def _near_draw_game(self) -> TicTacPro:
        """Return a 17-piece position (BLUE to move, 1 Large left, all moves draw).

        Board layout (no same-size 3-in-a-row, no bullseye):
          RED  S: (0,0) (1,2) (2,1)
          RED  M: (0,1) (1,0) (2,2)
          RED  L: (0,2) (1,1) (2,1)   ← (2,1) stacks RED S and RED L
          BLUE S: (0,1) (1,0) (2,0)
          BLUE M: (0,0) (1,2) (2,1)   ← (2,1) has RED S/L + BLUE M (different slots)
          BLUE L: (0,1) (1,2)

        Empty Large slots → BLUE's 4 legal moves: (0,0,L) (1,0,L) (2,0,L) (2,2,L)

        No cell has 2+ BLUE pieces in the same type combination that would create a
        bullseye upon placing the 3rd:
          (0,0): BLUE M only     → (0,0,L) no bullseye ✓
          (1,0): BLUE S only     → (1,0,L) no bullseye ✓
          (2,0): BLUE S only     → (2,0,L) no bullseye ✓  (BLUE M is at (2,1) not (2,0))
          (2,2): no BLUE pieces  → (2,2,L) no bullseye ✓
        None of these four completes a same-size line either (verified manually)."""
        g = TicTacPro()
        R, B = int(Player.RED), int(Player.BLUE)
        # RED pieces
        g.board[0, 0, 0] = R   # (0,0,S)
        g.board[1, 2, 0] = R   # (1,2,S)
        g.board[2, 1, 0] = R   # (2,1,S)
        g.board[0, 1, 1] = R   # (0,1,M)
        g.board[1, 0, 1] = R   # (1,0,M)
        g.board[2, 2, 1] = R   # (2,2,M)
        g.board[0, 2, 2] = R   # (0,2,L)
        g.board[1, 1, 2] = R   # (1,1,L)
        g.board[2, 1, 2] = R   # (2,1,L)
        # BLUE pieces — BLUE M moved from (2,0) to (2,1) to remove the (2,0) bullseye threat
        g.board[0, 1, 0] = B   # (0,1,S)
        g.board[1, 0, 0] = B   # (1,0,S)
        g.board[2, 0, 0] = B   # (2,0,S)
        g.board[0, 0, 1] = B   # (0,0,M)
        g.board[1, 2, 1] = B   # (1,2,M)
        g.board[2, 1, 1] = B   # (2,1,M)  ← was (2,0,M); moved to avoid bullseye at (2,0)
        g.board[0, 1, 2] = B   # (0,1,L)
        g.board[1, 2, 2] = B   # (1,2,L)
        # Piece counts: RED exhausted; BLUE has 1 Large left
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.RED][PieceSize.LARGE]  = 0
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 1
        g._pieces_left    = 1
        g.current_player  = Player.BLUE
        return g

    def test_draw_during_expansion_and_selection(self):
        """§319: MCTS must handle draw (line 465 + line 417) without crash or
        state corruption, returning a legal move from the draw position."""
        g = self._near_draw_game()
        legal = g.get_legal_moves(Player.BLUE)
        assert len(legal) == 4, f"Setup: expected 4 legal BLUE large moves, got {legal}"
        assert not g.game_over,  "Setup: game must not be over yet"
        assert g.winner == Player.NONE, "Setup: no winner yet"
        # 10 sims: first 4 expand each draw child (line 465); next 6 revisit them (line 417)
        agent = MCTSAgent(time_limit=100.0, max_simulations=10)
        mv = agent.get_action(g, Player.BLUE)
        assert mv in legal, (
            f"§319: MCTS on near-draw board must return a legal move; got {mv}"
        )

    def test_draw_state_not_mutated_after_search(self):
        """§319 sanity: _build_tree must not mutate the original game state."""
        import numpy as np
        g = self._near_draw_game()
        board_before = g.board.copy()
        pl_before    = g._pieces_left
        cp_before    = g.current_player
        agent = MCTSAgent(time_limit=100.0, max_simulations=10)
        agent.get_action(g, Player.BLUE)
        assert np.array_equal(g.board, board_before), \
            "§319: _build_tree must not mutate original board"
        assert g._pieces_left   == pl_before
        assert g.current_player == cp_before


# ---------------------------------------------------------------------------
# §318: _rollout_undo stuck-player mv=None break paths (lines 623, 640)
# ---------------------------------------------------------------------------

class TestRolloutUndoMvNoneBreak:
    """§318: covers the two `if mv is None: break` guards in _rollout_undo
    that fire when the current player has no legal moves but game_over is False
    (manufactured stuck-player state).

      line 623 — rollout_agent path: get_action() returns None for stuck player
      line 640 — Python clone path (C ext off): pick_rollout_move returns (None,False)
    """

    def test_rollout_undo_agent_path_mv_none_breaks_loop(self):
        """§318a: _rollout_undo must break via line 623 when rollout_agent.get_action
        returns None because the current player has no legal moves.

        Setup: RED's all piece counts set to 0 (stuck player), game_over=False.
        _rollout_undo enters the while loop (not game_over), calls
        rollout_agent.get_action(g, RED) → None (no legal moves for RED),
        hits `if mv is None: break` (line 622-623), returns 0.5 (draw)."""
        from rl.minimax_agent import MinimaxAgent
        ra = MinimaxAgent(time_limit=0.5, max_depth=1, use_book=False)
        agent = MCTSAgent(time_limit=0.1, rollout_agent=ra)
        g = TicTacPro()
        # Drain ALL of RED's pieces — stuck player (game_over stays False)
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.RED][PieceSize.LARGE]  = 0
        assert not g.game_over, "Setup error: game must not be over"
        assert g.get_legal_moves(Player.RED) == [], "Setup error: RED must have no moves"
        result = agent._rollout_undo(g, Player.RED)
        # winner = Player.NONE (no one won) → DRAW → 0.5
        assert result == 0.5, (
            f"§318a: stuck-player rollout via agent must return 0.5 (draw); got {result}"
        )

    def test_rollout_undo_python_path_mv_none_breaks_loop(self, monkeypatch):
        """§318b: _rollout_undo Python clone path must break via line 640 when
        pick_rollout_move returns (None, False) for a stuck player.

        With C_ROLLOUT_AVAILABLE=False and rollout_agent=None, the code takes the
        Python clone rollout path (lines 634-643).  Setting RED's pieces to 0 makes
        pick_rollout_move return (None, False), triggering the break at line 639-640."""
        import rl.mcts_agent as _ma
        monkeypatch.setattr(_ma, 'C_ROLLOUT_AVAILABLE', False)
        agent = MCTSAgent(time_limit=0.1)   # rollout_agent=None
        g = TicTacPro()
        # Drain RED's pieces — pick_rollout_move returns (None, False) on RED's turn
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.RED][PieceSize.LARGE]  = 0
        assert not g.game_over, "Setup error: game must not be over"
        result = agent._rollout_undo(g, Player.RED)
        # No winner in the clone → DRAW → 0.5
        assert result == 0.5, (
            f"§318b: stuck-player Python clone rollout must return 0.5 (draw); got {result}"
        )


# ---------------------------------------------------------------------------
# §320: line 465 — draw detected during CHILD expansion in fast path
# ---------------------------------------------------------------------------

class TestMCTSLine465ChildExpansion:
    """§320: cover mcts_agent.py line 465 — the `elif _pcs_left == 0` guard inside
    the `if _nnw >= 0:` fast-path expansion block.

    Why line 465 is hard to hit:
      Root node is initialized with g.get_legal_moves() → _n_non_winning=-1 →
      always uses the _push_undo else-branch.  Line 465 is only reachable when a
      CHILD node (initialized with _LAZY → _get_sorted_ex → _n_non_winning>=0) is
      expanded and its move is the last piece placed with no win.

    Board design — 16 pieces placed (8 RED + 8 BLUE), _pieces_left=2:
      RED  S: (0,0) (1,2)          BLUE S: (0,1) (2,2)
      RED  M: (0,2) (1,0) (2,1)   BLUE M: (0,0) (1,1) (2,0)
      RED  L: (0,1) (1,2) (2,0)   BLUE L: (0,2) (1,0) (2,1)
      pieces_remaining: RED {S:1,M:0,L:0}, BLUE {S:1,M:0,L:0}
      current_player: RED

    Simulation trace (max_simulations=10):
      Sims 1-5:  root untried_moves non-empty → stay at root, expand via _push_undo
                 (_nnw=-1). Each creates a child node with _LAZY. _pcs_left=1 after.
      Sims 6-10: root fully expanded. Selection descends: apply RED S move
                 (_pcs_left: 2→1). Child has _LAZY → _get_sorted_ex called →
                 _nnw=4 (all 4 remaining BLUE S moves are non-winning draws).
                 Fast path: pop BLUE S, _pcs_left: 1→0, is_win=False →
                 `elif _pcs_left == 0: _game_over = True`  ← LINE 465 fires!

    Board verification (via make_move): every combination of RED S + BLUE S yields
    game_over=True, winner=NONE (draw). No same-size line or bullseye in any
    completion. RED S moves: {(0,2),(1,0),(1,1),(2,0),(2,1)} — all non-winning."""

    def _board(self) -> TicTacPro:
        g = TicTacPro()
        R, B = int(Player.RED), int(Player.BLUE)
        # RED pieces (2S + 3M + 3L)
        g.board[0, 0, 0] = R   # (0,0,S)
        g.board[1, 2, 0] = R   # (1,2,S)
        g.board[0, 2, 1] = R   # (0,2,M)
        g.board[1, 0, 1] = R   # (1,0,M)
        g.board[2, 1, 1] = R   # (2,1,M)
        g.board[0, 1, 2] = R   # (0,1,L)
        g.board[1, 2, 2] = R   # (1,2,L)
        g.board[2, 0, 2] = R   # (2,0,L)
        # BLUE pieces (2S + 3M + 3L)
        g.board[0, 1, 0] = B   # (0,1,S)
        g.board[2, 2, 0] = B   # (2,2,S)
        g.board[0, 0, 1] = B   # (0,0,M)
        g.board[1, 1, 1] = B   # (1,1,M)
        g.board[2, 0, 1] = B   # (2,0,M)
        g.board[0, 2, 2] = B   # (0,2,L)
        g.board[1, 0, 2] = B   # (1,0,L)
        g.board[2, 1, 2] = B   # (2,1,L)
        g.pieces_remaining[Player.RED][PieceSize.SMALL]   = 1
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM]  = 0
        g.pieces_remaining[Player.RED][PieceSize.LARGE]   = 0
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  = 1
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 0
        g._pieces_left   = 2
        g.current_player = Player.RED
        return g

    def test_line_465_fires_during_child_expansion(self):
        """§320: MCTS runs without crash on the 16-piece board and returns a legal
        move.  Sims 6-10 hit line 465 (child expansion fast-path, _pcs_left→0,
        non-winning → draw detected inline)."""
        g = self._board()
        legal = g.get_legal_moves(Player.RED)
        assert len(legal) == 5, f"§320 setup: expected 5 RED S moves, got {legal}"
        assert not g.game_over,        "§320 setup: game must not already be over"
        assert g.winner == Player.NONE, "§320 setup: winner must be NONE"
        agent = MCTSAgent(time_limit=100.0, max_simulations=10)
        mv = agent.get_action(g, Player.RED)
        assert mv in legal, f"§320: MCTS must return a legal RED move; got {mv}"

    def test_line_465_state_not_mutated(self):
        """§320: game state is unmodified after search that exercises line 465."""
        import numpy as np
        g = self._board()
        board_before = g.board.copy()
        pl_before    = g._pieces_left
        cp_before    = g.current_player
        agent = MCTSAgent(time_limit=100.0, max_simulations=10)
        agent.get_action(g, Player.RED)
        assert np.array_equal(g.board, board_before), "§320: board mutated by search"
        assert g._pieces_left   == pl_before,          "§320: _pieces_left mutated"
        assert g.current_player == cp_before,          "§320: current_player mutated"


# ---------------------------------------------------------------------------
# §321: _root_all_terminal fast path (lines 571-572, 579-601)
# ---------------------------------------------------------------------------

class TestMCTSRootAllTerminal:
    """§321: cover _build_tree's _root_all_terminal fast path (lines 571-572, 579-601).

    This path fires when EVERY root child has move_is_win=True.  get_action() pre-
    screens wins and never calls _build_tree in such positions, so _build_tree must
    be called directly.

    Board design: manufactured M layer where RED M fills columns 0 and 1 in all 3
    rows (6 RED M pieces — not legal in a real game but TicTacPro does not validate
    board state at construction time).  The 3 remaining empty M cells are
    (0,2,M), (1,2,M), (2,2,M), each completing a same-row win for RED.

    RED has pieces_remaining[M]=1 (1 piece left to place).  All 3 M cells are wins
    so 3 root children will be created, ALL with move_is_win=True.  After the 3rd
    expansion (sim 3) the terminal check triggers (lines 571-572), sets
    _root_all_terminal=True and breaks.  Remaining sims (4-5) run the fast path
    loop (lines 579-601)."""

    def _all_win_board(self) -> TicTacPro:
        """Manufactured board: 6 RED M in cols 0-1 of every row; 3 winning M cells."""
        g = TicTacPro()
        R = int(Player.RED)
        for r in range(3):
            g.board[r, 0, 1] = R   # (r,0,M)
            g.board[r, 1, 1] = R   # (r,1,M)
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] = 1   # the 1 M left to place
        g.pieces_remaining[Player.RED][PieceSize.LARGE]  = 0
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 0
        g._pieces_left   = 1
        g.current_player = Player.RED
        return g

    def test_root_all_terminal_triggers_fast_path(self):
        """§321: _build_tree detects all-winning root, sets _root_all_terminal, and
        runs the fast-path loop without crashing."""
        g = self._all_win_board()
        legal = g.get_legal_moves(Player.RED)
        assert len(legal) == 3, f"§321: expected 3 winning M moves, got {legal}"
        agent = MCTSAgent(time_limit=100.0, max_simulations=5)
        g_clone = g.clone()
        root = agent._build_tree(g_clone, Player.RED)
        assert root is not None
        assert root.children, "§321: root must have children after _build_tree"
        assert all(c.move_is_win for c in root.children), \
            "§321: all root children must be terminal wins"

    def test_root_all_terminal_best_move_is_a_win(self):
        """§321: get_action catches the win in pre-screening (before _build_tree),
        so confirm the manufactured board returns a winning move via _build_tree root."""
        g = self._all_win_board()
        legal = g.get_legal_moves(Player.RED)
        agent = MCTSAgent(time_limit=100.0, max_simulations=5)
        g_clone = g.clone()
        root = agent._build_tree(g_clone, Player.RED)
        best = max(root.children, key=lambda c: c.visits)
        assert best.move in legal, "§321: best move from all-terminal root must be a legal win"


# ---------------------------------------------------------------------------
# §322: MCTSNode.uct_select Python fallback (lines 229-236)
# ---------------------------------------------------------------------------

class TestMCTSNodeUCTSelectPythonFallback:
    """§322: cover MCTSNode.uct_select() lines 229-236 — the Python for-loop fallback
    that fires when ch_n==0 or ch_wr is None (numpy path not taken).

    This method is dead in the _build_tree hot path (UCT is inlined at line 406),
    but it is a public method on MCTSNode and the fallback branch must be tested to
    pin the correctness of UCT tie-breaking and the best-child selection logic."""

    def test_uct_select_python_fallback_picks_highest_uct(self):
        """§322: with ch_n=0/ch_wr=None, uct_select uses the Python loop and picks
        the child with the highest winrate + explore_base * uct_scale."""
        from rl.mcts_agent import MCTSNode, _LAZY

        node = MCTSNode(None, None, [], None)
        node.visits = 4   # explore_base = UCT_C * sqrt(log(4))

        child_a = MCTSNode((0, 0, 1), node, _LAZY, 1, False, -1, 0, 1)
        child_b = MCTSNode((0, 1, 1), node, _LAZY, 1, False, -1, 1, 1)
        # High winrate but low exploration bonus
        child_a.visits = 3; child_a.wins = 2.7
        child_a.winrate   = 0.9
        child_a.uct_scale = 0.01
        # Lower winrate but large exploration bonus
        child_b.visits = 1; child_b.wins = 0.0
        child_b.winrate   = 0.0
        child_b.uct_scale = 1.0

        node.children = [child_a, child_b]
        # Leave ch_n=0 and ch_wr=None → forces Python fallback (lines 229-236)

        result = node.uct_select()
        # child_a score = 0.9 + explore_base*0.01; child_b = 0 + explore_base*1.0
        # explore_base = UCT_C * sqrt(log(4)) ≈ 1.414 * 1.177 ≈ 1.664
        # child_a ≈ 0.9 + 0.017 ≈ 0.917; child_b ≈ 1.664
        # child_b wins
        assert result is child_b, (
            "§322: Python fallback UCT must pick the child with higher UCT score"
        )

    def test_uct_select_python_fallback_single_child(self):
        """§322: single child — Python loop returns it unconditionally."""
        from rl.mcts_agent import MCTSNode, _LAZY

        node = MCTSNode(None, None, [], None)
        node.visits = 1
        child = MCTSNode((1, 1, 2), node, _LAZY, 1, False, -1, 0, 2)
        child.visits = 1; child.wins = 0.5
        child.winrate = 0.5; child.uct_scale = 0.7
        node.children = [child]

        result = node.uct_select()
        assert result is child, "§322: single child must always be selected"

    def test_uct_select_numpy_path_picks_highest_uct(self):
        """§322: with ch_n>0 and ch_wr set, uct_select uses numpy vectorised UCT
        (line 228) and picks the child with the highest winrate+exploration score."""
        import numpy as np
        from rl.mcts_agent import MCTSNode, _LAZY

        node = MCTSNode(None, None, [], None)
        node.visits = 4   # explore_base = UCT_C * sqrt(log(4))

        child_a = MCTSNode((0, 0, 1), node, _LAZY, 1, False, -1, 0, 1)
        child_b = MCTSNode((0, 1, 1), node, _LAZY, 1, False, -1, 1, 1)
        child_a.winrate   = 0.9; child_a.uct_scale = 0.01   # low exploration bonus
        child_b.winrate   = 0.0; child_b.uct_scale = 1.0    # large exploration bonus

        node.children = [child_a, child_b]
        # Set ch_n>0 and ch_wr ≠ None → numpy path (line 228)
        node.ch_n  = 2
        node.ch_wr = np.array([0.9, 0.0])
        node.ch_us = np.array([0.01, 1.0])

        result = node.uct_select()
        # child_b wins: explore_base ≈ 1.66; child_a≈0.917, child_b≈1.66
        assert result is child_b, \
            "§322: numpy UCT must pick child_b (higher exploration term wins)"


# ---------------------------------------------------------------------------
# §323: _root_all_terminal fast-path timeout break (line 586)
# ---------------------------------------------------------------------------

class TestMCTSRootAllTerminalTimeout:
    """§323: cover line 586 — the `break` inside the _root_all_terminal fast-path
    loop when the time budget is exhausted.

    Strategy: monkeypatch time.perf_counter so the first 2 calls return 0.0
    (main-loop setup + sims=0 check) and all subsequent calls return 1e9 (expired).
    The main loop expands 3 winning moves then breaks; the fast path starts at sims=3.
    The time check fires at sims=64 (64 % 64 == 0) and 1e9 > deadline → break."""

    def _all_win_board(self) -> TicTacPro:
        g = TicTacPro()
        R = int(Player.RED)
        for r in range(3):
            g.board[r, 0, 1] = R
            g.board[r, 1, 1] = R
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] = 1
        g.pieces_remaining[Player.RED][PieceSize.LARGE]  = 0
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 0
        g._pieces_left   = 1
        g.current_player = Player.RED
        return g

    def test_fast_path_breaks_on_deadline(self, monkeypatch):
        """§323: deadline expiry during fast-path UCT loop triggers line 586 break."""
        import time
        call_count = [0]
        def _mock_clock():
            call_count[0] += 1
            # Calls 1-2: deadline setup + sims=0 main check — stay in main loop
            return 0.0 if call_count[0] <= 2 else 1e9

        monkeypatch.setattr(time, 'perf_counter', _mock_clock)

        g = self._all_win_board()
        agent = MCTSAgent(time_limit=100.0, max_simulations=128)  # > 64 = _check
        g_clone = g.clone()
        root = agent._build_tree(g_clone, Player.RED)
        assert root is not None, "§323: _build_tree must return a root even after timeout"
        # Deadline fired in fast path: sims stopped well before max_simulations
        assert agent._last_sims < 128, \
            f"§323: timeout should stop sims early; got {agent._last_sims}"
