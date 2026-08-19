"""Unit tests for the TicTacPro game engine."""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from game.tictacpro import TicTacPro, Player, PieceSize, pick_rollout_move, _would_win_bf, would_win


class TestGameInit:
    def test_board_is_empty(self):
        g = TicTacPro()
        assert g.board.sum() == 0

    def test_red_starts(self):
        g = TicTacPro()
        assert g.current_player == Player.RED

    def test_pieces_initialised(self):
        g = TicTacPro()
        for player in [Player.RED, Player.BLUE]:
            for size in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]:
                assert g.pieces_remaining[player][size] == 3

    def test_game_not_over(self):
        g = TicTacPro()
        assert not g.game_over
        assert g.winner == Player.NONE

    def test_pieces_left_is_18(self):
        """§265: _pieces_left must be 18 on init (2 players × 3 sizes × 3 pieces each).
        draw detection relies on this counter being correct from the very first move."""
        g = TicTacPro()
        assert g._pieces_left == 18, f"§265: _pieces_left should be 18, got {g._pieces_left}"

    def test_move_history_empty(self):
        """§266: move_history must be [] on init so replay and history consumers see
        a clean slate — a leftover list would corrupt history-based features."""
        g = TicTacPro()
        assert g.move_history == [], f"§266: move_history must be [], got {g.move_history}"


class TestLegalMoves:
    def test_initial_legal_moves_count(self):
        g = TicTacPro()
        moves = g.get_legal_moves()
        # 9 positions × 3 sizes = 27
        assert len(moves) == 27

    def test_legal_moves_after_small_placed(self):
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        # (0,0,SMALL) slot is now occupied — no one can reuse that slot.
        # BLUE can still place MEDIUM/LARGE at (0,0), so 26 legal moves remain.
        moves_blue = g.get_legal_moves(Player.BLUE)
        assert len(moves_blue) == 26

    def test_no_moves_for_exhausted_size(self):
        """§234: get_legal_moves must exclude moves of sizes the player has run out of.

        Previous version placed 3 Smalls in row 0 which won the game — then
        get_legal_moves returned [] (game over) and the assertion was vacuously True.
        Fixed: place 3 Smalls in non-winning positions via board manipulation, then
        verify the move list genuinely excludes Smalls.
        """
        g = TicTacPro()
        # Place 3 RED Smalls in non-winning positions: (0,0), (0,1), (1,0)
        # Not a row (missing (0,2)), not a col (missing (2,0)), not a diag.
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.board[1, 0, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] = 0
        g._pieces_left -= 3
        assert not g.game_over, "Setup error: game must not be over"
        moves = g.get_legal_moves(Player.RED)
        assert len(moves) > 0, "RED still has Medium/Large — must have legal moves"
        for r, c, sz in moves:
            assert sz != PieceSize.SMALL, (
                f"§234: RED has 0 Smalls but get_legal_moves returned {(r,c,sz)}"
            )

    def test_game_over_returns_empty(self):
        g = TicTacPro()
        g.game_over = True
        assert g.get_legal_moves() == []

    def test_explicit_non_current_player_uses_that_players_pieces(self):
        """§246: get_legal_moves(player) must use piece counts for *player*, not
        current_player.  Callers (minimax move-ordering, threat detection) query the
        opponent's legal moves while it is their own turn.

        Setup: RED plays, then BLUE's Medium count is zeroed out.  current_player is
        BLUE.  Querying RED (non-current) must still return Medium moves; querying
        BLUE (current) must exclude Mediums because BLUE has none left."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)       # RED plays — now BLUE's turn
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 0  # exhaust BLUE's Mediums
        assert g.current_player == Player.BLUE

        red_moves  = g.get_legal_moves(Player.RED)   # non-current player
        blue_moves = g.get_legal_moves(Player.BLUE)  # current player

        red_has_medium  = any(sz == PieceSize.MEDIUM for _, _, sz in red_moves)
        blue_has_medium = any(sz == PieceSize.MEDIUM for _, _, sz in blue_moves)

        assert red_has_medium, (
            "§246: get_legal_moves(RED) while BLUE's turn must use RED's piece counts; "
            "RED still has Mediums but none appeared in the move list"
        )
        assert not blue_has_medium, (
            "§246: get_legal_moves(BLUE) must respect BLUE's 0 Mediums; "
            "a Medium move appeared despite BLUE having none"
        )


class TestMakeMove:
    def test_basic_move(self):
        g = TicTacPro()
        ok = g.make_move(1, 1, PieceSize.MEDIUM)
        assert ok
        assert g.board[1, 1, PieceSize.MEDIUM - 1] == Player.RED

    def test_turn_alternates(self):
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        assert g.current_player == Player.BLUE

    def test_pieces_decremented(self):
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        assert g.pieces_remaining[Player.RED][PieceSize.SMALL] == 2

    def test_invalid_same_slot(self):
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL, player=Player.RED)
        # Can't place another SMALL at the same slot (even by BLUE)
        ok = g.make_move(0, 0, PieceSize.SMALL, player=Player.BLUE)
        assert not ok

    def test_stacking_allowed(self):
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL, player=Player.RED)
        ok = g.make_move(0, 0, PieceSize.MEDIUM, player=Player.BLUE)
        assert ok

    def test_out_of_bounds_row_returns_false(self):
        """§244: make_move must return False for row outside [0, BOARD_SIZE)."""
        g = TicTacPro()
        ok = g.make_move(3, 0, PieceSize.SMALL)  # row 3 is out of bounds
        assert not ok, "§244: out-of-bounds row must return False"
        assert (g.board == 0).all(), "Board must not be modified on invalid move"

    def test_out_of_bounds_col_returns_false(self):
        """§244: make_move must return False for col outside [0, BOARD_SIZE)."""
        g = TicTacPro()
        ok = g.make_move(0, -1, PieceSize.SMALL)  # col -1 is out of bounds
        assert not ok, "§244: out-of-bounds col must return False"
        assert (g.board == 0).all(), "Board must not be modified on invalid move"

    def test_exhausted_pieces_returns_false(self):
        """§244: make_move must return False when the player has 0 pieces of that size."""
        g = TicTacPro()
        g.pieces_remaining[Player.RED][PieceSize.SMALL] = 0
        ok = g.make_move(0, 0, PieceSize.SMALL)
        assert not ok, "§244: placing an exhausted size must return False"
        assert (g.board == 0).all(), "Board must not be modified on invalid move"

    def test_make_move_returns_false_when_game_over(self):
        """§245: make_move must return False and not modify state when game is already over."""
        g = TicTacPro()
        g.game_over = True
        g.winner = Player.RED
        board_before = g.board.copy()
        ok = g.make_move(0, 0, PieceSize.SMALL)
        assert not ok, "§245: make_move must return False when game_over=True"
        assert (g.board == board_before).all(), "Board must not change when game is over"

    def test_make_move_appends_to_move_history(self):
        """§248: make_move must append (row, col, size, player) to move_history on success.

        The clone() and reset() tests implicitly check move_history length, but no test
        verifies the actual tuple content.  A refactor changing the append format (e.g.
        dropping player, swapping order) would be invisible to existing tests."""
        g = TicTacPro()
        g.make_move(1, 2, PieceSize.MEDIUM)   # RED plays
        assert len(g.move_history) == 1, "One move must produce one history entry"
        assert g.move_history[0] == (1, 2, PieceSize.MEDIUM, Player.RED), (
            f"§248: move_history[0] must be (1,2,MEDIUM,RED); got {g.move_history[0]}"
        )
        g.make_move(0, 0, PieceSize.SMALL)    # BLUE plays
        assert g.move_history[1] == (0, 0, PieceSize.SMALL, Player.BLUE), (
            f"§248: move_history[1] must be (0,0,SMALL,BLUE); got {g.move_history[1]}"
        )

    def test_stacking_both_pieces_on_board(self):
        """§284: After stacking SMALL then MEDIUM at the same cell, both pieces must
        remain on the board independently (board[r,c,0]=RED, board[r,c,1]=BLUE).

        test_stacking_allowed only verifies make_move returns True. This test verifies
        the board state after stacking: both size slots are set to the correct player.
        A bug overwriting the SMALL slot when MEDIUM is placed would be caught here."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)    # RED places SMALL at (0,0)
        g.make_move(0, 0, PieceSize.MEDIUM)   # BLUE places MEDIUM at (0,0)
        assert g.board[0, 0, 0] == int(Player.RED), (
            "§284: SMALL slot (0,0,0) must remain RED after MEDIUM is stacked above"
        )
        assert g.board[0, 0, 1] == int(Player.BLUE), (
            "§284: MEDIUM slot (0,0,1) must be BLUE after stacking"
        )
        assert g.board[0, 0, 2] == 0, (
            "§284: LARGE slot (0,0,2) must remain 0 (no LARGE placed)"
        )

    def test_make_move_does_not_switch_player_on_win(self):
        """§249: make_move must NOT switch current_player when the move ends the game.

        push() has an explicit test for this invariant (test_push_does_not_switch_player_on_win)
        because negamax uses current_player to identify the winner.  make_move must
        uphold the same invariant — the winning player stays as current_player.

        Uses natural alternating turns (no explicit player arg) so current_player==RED
        when RED makes the winning move at (0,2,S)."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)   # RED (current_player RED → BLUE)
        g.make_move(2, 2, PieceSize.LARGE)   # BLUE (current_player BLUE → RED)
        g.make_move(0, 1, PieceSize.SMALL)   # RED (current_player RED → BLUE)
        g.make_move(2, 1, PieceSize.LARGE)   # BLUE (current_player BLUE → RED)
        g.make_move(0, 2, PieceSize.SMALL)   # RED wins (current_player stays RED)
        assert g.game_over is True
        assert g.winner == Player.RED
        assert g.current_player == Player.RED, (
            "§249: make_move must not switch current_player after a win; "
            f"current_player={g.current_player} but winner=RED"
        )


class TestWinConditions:
    def _setup_row_win(self, player=Player.RED, size=PieceSize.SMALL):
        g = TicTacPro()
        opponent = Player.BLUE if player == Player.RED else Player.RED
        for col in range(3):
            g.board[0, col, size - 1] = player
        g.winner = player
        g.game_over = True
        return g

    def test_row_win_detected(self):
        g = TicTacPro()
        # Drive a row win step by step
        g.make_move(0, 0, PieceSize.SMALL, player=Player.RED)
        g.make_move(0, 1, PieceSize.SMALL, player=Player.RED)
        g.make_move(0, 2, PieceSize.SMALL, player=Player.RED)
        assert g.winner == Player.RED
        assert g.game_over

    def test_col_win_detected(self):
        g = TicTacPro()
        for row in range(3):
            g.make_move(row, 0, PieceSize.MEDIUM, player=Player.RED)
        assert g.winner == Player.RED

    def test_diagonal_win(self):
        g = TicTacPro()
        for i in range(3):
            g.make_move(i, i, PieceSize.LARGE, player=Player.RED)
        assert g.winner == Player.RED

    def test_anti_diagonal_win(self):
        g = TicTacPro()
        for i in range(3):
            g.make_move(i, 2 - i, PieceSize.SMALL, player=Player.RED)
        assert g.winner == Player.RED

    def test_bullseye_win(self):
        g = TicTacPro()
        g.make_move(1, 1, PieceSize.SMALL, player=Player.RED)
        g.make_move(1, 1, PieceSize.MEDIUM, player=Player.RED)
        g.make_move(1, 1, PieceSize.LARGE, player=Player.RED)
        assert g.winner == Player.RED

    def test_no_win_mid_game(self):
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL, player=Player.RED)
        g.make_move(0, 1, PieceSize.SMALL, player=Player.RED)
        assert g.winner == Player.NONE
        assert not g.game_over

    # §212: BLUE win conditions — win detection must work for both players.
    def test_blue_row_win_detected(self):
        g = TicTacPro()
        for col in range(3):
            g.make_move(0, col, PieceSize.SMALL, player=Player.BLUE)
        assert g.winner == Player.BLUE
        assert g.game_over

    def test_blue_col_win_detected(self):
        g = TicTacPro()
        for row in range(3):
            g.make_move(row, 2, PieceSize.MEDIUM, player=Player.BLUE)
        assert g.winner == Player.BLUE

    def test_blue_diagonal_win(self):
        g = TicTacPro()
        for i in range(3):
            g.make_move(i, i, PieceSize.LARGE, player=Player.BLUE)
        assert g.winner == Player.BLUE

    def test_blue_anti_diagonal_win(self):
        g = TicTacPro()
        for i in range(3):
            g.make_move(i, 2 - i, PieceSize.SMALL, player=Player.BLUE)
        assert g.winner == Player.BLUE

    def test_blue_bullseye_win(self):
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL,  player=Player.BLUE)
        g.make_move(0, 0, PieceSize.MEDIUM, player=Player.BLUE)
        g.make_move(0, 0, PieceSize.LARGE,  player=Player.BLUE)
        assert g.winner == Player.BLUE
        assert g.game_over

    def test_red_wins_not_blue(self):
        """When RED wins, winner must be RED, not BLUE."""
        g = TicTacPro()
        for col in range(3):
            g.make_move(0, col, PieceSize.SMALL, player=Player.RED)
        assert g.winner == Player.RED
        assert g.winner != Player.BLUE


class TestStateTensor:
    def test_shape(self):
        g = TicTacPro()
        st = g.get_state_tensor()
        assert st.shape == (61,)

    def test_dtype(self):
        g = TicTacPro()
        st = g.get_state_tensor()
        assert st.dtype == np.float32

    def test_initial_board_zeros(self):
        g = TicTacPro()
        st = g.get_state_tensor()
        # First 54 values (board) should be 0 on empty board
        assert np.all(st[:54] == 0)

    def test_current_player_feature(self):
        g = TicTacPro()
        st = g.get_state_tensor()
        # Last feature: 1.0 if RED (first player)
        assert st[-1] == 1.0

    def test_piece_features_normalised(self):
        g = TicTacPro()
        st = g.get_state_tensor()
        # Piece features (indices 54-59) should be 1.0 at start
        assert np.allclose(st[54:60], 1.0)

    def test_state_changes_after_move(self):
        g = TicTacPro()
        st_before = g.get_state_tensor().copy()
        g.make_move(0, 0, PieceSize.SMALL)
        st_after = g.get_state_tensor()
        assert not np.array_equal(st_before, st_after)

    def test_current_player_feature_blue(self):
        """§243: get_state_tensor buf[60] must be 0.0 when current player is BLUE.
        The existing test only covers RED (buf[60]=1.0); this covers the else branch."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)  # RED moves → BLUE's turn
        assert g.current_player == Player.BLUE
        assert g.get_state_tensor()[60] == 0.0, (
            "§243: buf[60] must be 0.0 when BLUE is current player"
        )

    def test_red_piece_in_slots_0_26(self):
        """§243: A RED piece at (r,c,sz_idx) must appear as 1.0 in buf[r*9+c*3+sz_idx]."""
        g = TicTacPro()
        g.make_move(1, 2, PieceSize.MEDIUM)   # RED places Medium at (1,2)
        st = g.get_state_tensor()
        flat_idx = 1 * 9 + 2 * 3 + 1   # r*9 + c*3 + sz_idx (Medium=sz 2, idx=1)
        assert st[flat_idx] == 1.0, (
            f"§243: RED piece at (1,2,M) must be 1.0 at buf[{flat_idx}], got {st[flat_idx]}"
        )
        assert st[27 + flat_idx] == 0.0, (
            f"§243: BLUE slot at buf[{27+flat_idx}] must be 0.0 (no BLUE piece here)"
        )

    def test_blue_piece_in_slots_27_53(self):
        """§243: A BLUE piece at (r,c,sz_idx) must appear as 1.0 in buf[27+r*9+c*3+sz_idx]."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)    # RED moves
        g.make_move(2, 1, PieceSize.LARGE)    # BLUE places Large at (2,1)
        st = g.get_state_tensor()
        flat_idx = 2 * 9 + 1 * 3 + 2   # r*9 + c*3 + sz_idx (Large=sz 3, idx=2)
        assert st[27 + flat_idx] == 1.0, (
            f"§243: BLUE piece at (2,1,L) must be 1.0 at buf[{27+flat_idx}], got {st[27+flat_idx]}"
        )
        assert st[flat_idx] == 0.0, (
            f"§243: RED slot at buf[{flat_idx}] must be 0.0 (no RED piece here)"
        )


class TestClone:
    def test_clone_independence(self):
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        clone = g.clone()
        clone.make_move(1, 1, PieceSize.SMALL)
        # Original should not be affected
        assert g.board[1, 1, 0] == Player.NONE

    def test_clone_equals_original(self):
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        clone = g.clone()
        assert np.array_equal(g.board, clone.board)
        assert g.current_player == clone.current_player

    def test_clone_copies_move_history_by_default(self):
        """§220: clone() must copy move_history when copy_history=True (default)."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        clone = g.clone()
        assert clone.move_history == g.move_history
        assert clone.move_history is not g.move_history  # independent copy

    def test_clone_no_history_returns_empty_list(self):
        """§220: clone(copy_history=False) must return empty move_history."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        g.make_move(1, 1, PieceSize.MEDIUM)
        clone = g.clone(copy_history=False)
        assert clone.move_history == []

    def test_clone_no_history_preserves_board_state(self):
        """§220: clone(copy_history=False) must still copy board, pieces, etc."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        clone = g.clone(copy_history=False)
        assert np.array_equal(g.board, clone.board)
        assert g.current_player == clone.current_player
        assert g._pieces_left == clone._pieces_left

    def test_clone_preserves_game_over_and_winner(self):
        """§261: clone() must copy game_over and winner — callers use clone() on finished
        positions (e.g. MCTS rollout clone at terminal node) and must see the result."""
        g = TicTacPro()
        # Drive RED to a row win so game_over=True and winner=RED
        for col in range(3):
            g.make_move(0, col, PieceSize.SMALL, player=Player.RED)
        assert g.game_over is True
        assert g.winner == Player.RED
        clone = g.clone()
        assert clone.game_over is True, "§261: clone must preserve game_over=True"
        assert clone.winner == Player.RED, "§261: clone must preserve winner=RED"

    def test_clone_pieces_remaining_independence(self):
        """§262: clone's pieces_remaining dict must be independent — mutating it must
        not affect the original (clone() creates new dicts, not shallow copies)."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)   # RED uses 1 Small → RED has 2 left
        clone = g.clone()
        original_count = g.pieces_remaining[Player.RED][PieceSize.SMALL]
        clone.pieces_remaining[Player.RED][PieceSize.SMALL] = 0  # exhaust clone's Smalls
        assert g.pieces_remaining[Player.RED][PieceSize.SMALL] == original_count, (
            "§262: mutating clone pieces_remaining must not affect original"
        )


class TestDraw:
    def test_draw_when_all_pieces_placed(self):
        """When all 18 pieces are placed without a win, game ends as draw."""
        g = TicTacPro()
        # Layout verified move-by-move to produce no 3-in-a-row or bullseye:
        # RED S: (0,0)(1,2)(2,1); BLUE S: (0,1)(1,0)(2,2) — no line for either
        # RED M: (0,1)(1,0)(2,2); BLUE M: (0,0)(1,2)(2,1) — same cells swapped, no line
        # RED L: (0,0)(0,2)(2,0); BLUE L: (1,1)(0,1)(2,2) — no line for either
        # No cell holds all 3 sizes for one player → no bullseye.
        moves = [
            (0, 0, PieceSize.SMALL,  Player.RED),
            (0, 1, PieceSize.SMALL,  Player.BLUE),
            (1, 2, PieceSize.SMALL,  Player.RED),
            (1, 0, PieceSize.SMALL,  Player.BLUE),
            (2, 1, PieceSize.SMALL,  Player.RED),
            (2, 2, PieceSize.SMALL,  Player.BLUE),
            (0, 1, PieceSize.MEDIUM, Player.RED),
            (0, 0, PieceSize.MEDIUM, Player.BLUE),
            (1, 0, PieceSize.MEDIUM, Player.RED),
            (1, 2, PieceSize.MEDIUM, Player.BLUE),
            (2, 2, PieceSize.MEDIUM, Player.RED),
            (2, 1, PieceSize.MEDIUM, Player.BLUE),
            (0, 0, PieceSize.LARGE,  Player.RED),
            (1, 1, PieceSize.LARGE,  Player.BLUE),
            (0, 2, PieceSize.LARGE,  Player.RED),
            (0, 1, PieceSize.LARGE,  Player.BLUE),
            (2, 0, PieceSize.LARGE,  Player.RED),
            (2, 2, PieceSize.LARGE,  Player.BLUE),
        ]
        for r, c, sz, pl in moves:
            if g.game_over:
                break
            g.make_move(r, c, sz, player=pl)
        assert g.game_over
        assert g.winner == Player.NONE

    def test_pieces_left_decrements(self):
        g = TicTacPro()
        assert g._pieces_left == 18
        g.make_move(0, 0, PieceSize.SMALL)
        assert g._pieces_left == 17

    def test_win_on_last_piece_beats_draw(self):
        """§285: When the last piece placed wins the game, winner must be set (not draw).

        Production code at tictacpro.py:169 checks `wins` before `_pieces_left==0`.
        If the order were reversed, winner would be NONE (draw) even when the final move
        wins the game. This test verifies the correct priority.

        Setup: _pieces_left=1, RED has 1 SMALL remaining, 2 RED Smalls already in row0.
        Placing (0,2,S) wins row0 AND empties _pieces_left. Winner must be RED not NONE."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] = 1
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.RED][PieceSize.LARGE] = 0
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 0
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE] = 0
        g._pieces_left = 1
        g.current_player = Player.RED
        ok = g.make_move(0, 2, PieceSize.SMALL)
        assert ok, "§285: make_move must succeed"
        assert g.game_over is True, "§285: game must be over after last piece"
        assert g.winner == Player.RED, (
            f"§285: winner must be RED (win on last piece), not {g.winner}; "
            "make_move checks 'wins' before '_pieces_left==0'"
        )


class TestStateTensorNormalized:
    def test_shape_and_dtype(self):
        g = TicTacPro()
        st = g.get_state_tensor_normalized()
        assert st.shape == (61,)
        assert st.dtype == np.float32

    def test_red_perspective_initial(self):
        """On empty board with RED to move: first 54 features zero."""
        g = TicTacPro()
        st = g.get_state_tensor_normalized()
        assert np.all(st[:54] == 0)

    def test_first_mover_flag_red(self):
        """§180: buf[60] = 1.0 when current player is RED."""
        g = TicTacPro()
        assert g.get_state_tensor_normalized()[60] == 1.0

    def test_first_mover_flag_blue(self):
        """§180: buf[60] = 0.0 when current player is BLUE."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)   # RED moves, now BLUE's turn
        assert g.current_player == Player.BLUE
        assert g.get_state_tensor_normalized()[60] == 0.0

    def test_perspective_flip_blue(self):
        """When BLUE moves, BLUE's pieces appear in slots 0-26, RED's in 27-53."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)   # RED places at (0,0,S)
        # Now it's BLUE's turn. RED's piece is at board[0,0,0].
        # Normalized: current player=BLUE → slots 0-26 are BLUE's, 27-53 are RED's.
        st = g.get_state_tensor_normalized()
        # board[0,0,0] is RED; RED is the opponent from BLUE's perspective → in 27-53
        flat_idx = 0 * 9 + 0 * 3 + 0   # r*9 + c*3 + sz_idx = 0
        assert st[flat_idx] == 0.0            # no BLUE piece here
        assert st[27 + flat_idx] == 1.0       # RED piece shows in opponent slots

    def test_perspective_flip_symmetry(self):
        """Normalized tensor for RED and BLUE on symmetric boards should be symmetric."""
        g_red = TicTacPro()
        g_red.board[0, 0, 0] = int(Player.RED)
        g_red.pieces_remaining[Player.RED][PieceSize.SMALL] = 2
        g_red._pieces_left = 17

        g_blue = TicTacPro()
        g_blue.board[0, 0, 0] = int(Player.BLUE)
        g_blue.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 2
        g_blue._pieces_left = 17
        g_blue.current_player = Player.BLUE

        st_red = g_red.get_state_tensor_normalized()
        st_blue = g_blue.get_state_tensor_normalized()
        # Both boards: current player has 1 piece at (0,0,S) → slot 0 == 1.0
        assert st_red[0] == 1.0
        assert st_blue[0] == 1.0

    def test_piece_count_features_initial(self):
        """§233: buf[54:60] must be 1.0 on a fresh board (all pieces remaining)."""
        g = TicTacPro()
        st = g.get_state_tensor_normalized()
        assert np.allclose(st[54:60], 1.0), (
            f"§233: expected piece counts 1.0 on fresh board, got {st[54:60]}"
        )

    def test_piece_count_features_perspective_aware(self):
        """§233: buf[54:56] are CURRENT player's counts; buf[57:59] are opponent's.
        After RED uses 1 Small (now BLUE's turn): buf[54] = BLUE's S count = 1.0,
        buf[57] = RED's S count = 2/3 (RED has 2 Smalls left)."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)   # RED uses 1 Small → BLUE's turn
        assert g.current_player == Player.BLUE
        st = g.get_state_tensor_normalized()
        # Current player (BLUE) has all pieces remaining: buf[54:57] = 1.0
        assert np.allclose(st[54], 1.0), f"BLUE SMALL should be 1.0, got {st[54]}"
        assert np.allclose(st[55], 1.0), f"BLUE MEDIUM should be 1.0, got {st[55]}"
        assert np.allclose(st[56], 1.0), f"BLUE LARGE should be 1.0, got {st[56]}"
        # Opponent (RED) used 1 Small: buf[57] = 2/3, buf[58:60] = 1.0
        assert np.isclose(st[57], 2.0 / 3.0, atol=1e-6), (
            f"§233: RED SMALL (opp) should be 2/3 after 1 use, got {st[57]}"
        )
        assert np.allclose(st[58], 1.0), f"RED MEDIUM (opp) should be 1.0, got {st[58]}"
        assert np.allclose(st[59], 1.0), f"RED LARGE (opp) should be 1.0, got {st[59]}"

    def test_red_perspective_opponent_in_slots_27_53(self):
        """§282: From RED's perspective, BLUE's pieces must appear in buf[27:54].

        test_perspective_flip_blue covers BLUE's perspective (RED piece → opp slots 27-53).
        This test covers the RED perspective (BLUE piece → opp slots 27-53):
          RED places (0,0,S), BLUE places (2,1,L), then RED to move.
          buf[27 + flat_idx] must be 1.0 for BLUE's piece at (2,1,L).
          buf[flat_idx] must be 0.0 (no RED piece at that location)."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)    # RED
        g.make_move(2, 1, PieceSize.LARGE)    # BLUE at (2,1,L)
        assert g.current_player == Player.RED
        st = g.get_state_tensor_normalized()
        flat_idx = 2 * 9 + 1 * 3 + 2          # (2,1,LARGE) → sz_idx=2
        assert st[27 + flat_idx] == 1.0, (
            f"§282: BLUE piece at (2,1,L) must be 1.0 in buf[{27+flat_idx}] "
            f"(opponent slots from RED's perspective); got {st[27+flat_idx]}"
        )
        assert st[flat_idx] == 0.0, (
            f"§282: buf[{flat_idx}] (RED's own slot at (2,1,L)) must be 0.0; "
            f"got {st[flat_idx]}"
        )

    def test_first_mover_flag_mid_game_red(self):
        """§283: buf[60]=1.0 persists for RED's turns throughout the game, not just initially.

        After RED+BLUE alternate moves, when it's RED's turn again buf[60] must still be 1.0.
        The flag always reflects current player's color (RED=1.0 always for RED turns)."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)   # RED
        g.make_move(1, 1, PieceSize.MEDIUM)  # BLUE
        assert g.current_player == Player.RED  # RED's 2nd turn
        assert g.get_state_tensor_normalized()[60] == 1.0, (
            "§283: buf[60] must be 1.0 for RED's turn mid-game, not just on initial board"
        )


class TestPickRolloutMove:
    def test_returns_winning_move(self):
        """pick_rollout_move must return the immediate win and is_win=True."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        mv, is_win = pick_rollout_move(g)
        assert mv == (0, 2, PieceSize.SMALL)
        assert is_win is True

    def test_returns_blocking_move(self):
        """When the opponent has 2-in-a-row, pick_rollout_move must block."""
        g = TicTacPro()
        g.board[1, 0, 1] = int(Player.BLUE)  # BLUE MEDIUM at (1,0)
        g.board[1, 1, 1] = int(Player.BLUE)  # BLUE MEDIUM at (1,1)
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] -= 2
        g._pieces_left -= 2
        g.current_player = Player.RED
        mv, is_win = pick_rollout_move(g)
        assert mv == (1, 2, PieceSize.MEDIUM)
        assert is_win is False

    def test_win_takes_priority_over_block(self):
        """A winning move must be returned even if a block is also available."""
        g = TicTacPro()
        # RED has 2-in-a-row SMALL at row 2
        g.board[2, 0, 0] = int(Player.RED)
        g.board[2, 1, 0] = int(Player.RED)
        # BLUE also has 2-in-a-row SMALL at row 0 (threat)
        g.board[0, 0, 0] = int(Player.BLUE)
        g.board[0, 1, 0] = int(Player.BLUE)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] -= 2
        g._pieces_left -= 4
        g.current_player = Player.RED
        mv, is_win = pick_rollout_move(g)
        assert mv == (2, 2, PieceSize.SMALL)
        assert is_win is True

    def test_returns_none_when_no_pieces(self):
        """pick_rollout_move returns (None, False) when the current player has no pieces."""
        g = TicTacPro()
        for sz in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]:
            g.pieces_remaining[Player.RED][sz] = 0
        mv, is_win = pick_rollout_move(g)
        assert mv is None
        assert is_win is False

    def test_rest_pick_legal(self):
        """Without wins or blocks, pick_rollout_move must return a legal move."""
        g = TicTacPro()
        mv, is_win = pick_rollout_move(g)
        assert mv in g.get_legal_moves()
        assert is_win is False

    def test_is_win_false_on_non_winning(self):
        """Non-winning move must return is_win=False."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)   # RED
        _, is_win = pick_rollout_move(g)
        assert is_win is False

    def test_bullseye_win_detected(self):
        """pick_rollout_move detects a bullseye win (all 3 sizes in one cell)."""
        g = TicTacPro()
        g.board[1, 1, 0] = int(Player.RED)   # SMALL at center
        g.board[1, 1, 1] = int(Player.RED)   # MEDIUM at center
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 1
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] -= 1
        g._pieces_left -= 2
        g.current_player = Player.RED
        mv, is_win = pick_rollout_move(g)
        assert mv == (1, 1, PieceSize.LARGE)
        assert is_win is True

    def test_board_unchanged_after_call(self):
        """pick_rollout_move must not mutate the game state."""
        g = TicTacPro()
        board_before = g.board.copy()
        pl_before = g._pieces_left
        pick_rollout_move(g)
        assert (g.board == board_before).all()
        assert g._pieces_left == pl_before

    def test_threat_tier_preferred_over_rest(self):
        """§231/§196/§175: pick_rollout_move must prefer a threat (2-in-a-row setup) over a rest pick.

        Position: RED at (0,0,S). In row-major iteration (0,1,S) is the first empty
        Small cell. Placing there gives RED two Smalls in row 0 — a threat. No block or
        win exists. So pick_rollout_move must return (0,1,S) as threat_mv, NOT a rest pick.

        §231: strengthened assertion from 'mv in legal_moves' (always trivially True)
        to the specific threat cell (0,1,S) which is the unique first threat.
        """
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)    # RED at (0,0,S)
        g.make_move(2, 2, PieceSize.LARGE)    # BLUE at (2,2,L) — neutral
        # RED to move: (0,1,S) creates row-0 2-in-a-row (threat), (0,2,S) is rest.
        assert g.current_player == Player.RED
        mv, is_win = pick_rollout_move(g)
        assert is_win is False
        # (0,1,S) must be returned — first threat in iteration order.
        # A rest pick (0,2,S) would indicate the threat tier was ignored.
        assert mv == (0, 1, PieceSize.SMALL), (
            f"§231: expected threat (0,1,S) but got {mv}; "
            "threat tier must outrank rest picks (§175)"
        )

    def test_threat_yields_to_block(self):
        """§175: block must take priority over threat — block > threat in ordering."""
        g = TicTacPro()
        # Build: BLUE has 2 LARGE in col 2 → threat at (2,2,L)
        # RED has 1 SMALL in row 1 → threat at row 1 for RED SMALL
        # But RED cannot win anywhere (no 2-in-a-row for RED that can complete)
        g.make_move(0, 1, PieceSize.SMALL)    # RED at (0,1,S)
        g.make_move(0, 2, PieceSize.LARGE)    # BLUE at (0,2,L)
        g.make_move(0, 0, PieceSize.MEDIUM)   # RED at (0,0,M) — no row-win (diff sizes)
        g.make_move(1, 2, PieceSize.LARGE)    # BLUE at (1,2,L) — BLUE 2-in-a-row col 2 LARGE
        # RED to move — BLUE about to win col 2 LARGE at (2,2,L).
        # RED has 1 SMALL in row 0 which is a weak threat (only 1 piece).
        # Block must take priority over the threat.
        assert g.current_player == Player.RED
        mv, is_win = pick_rollout_move(g)
        assert is_win is False
        # RED must block BLUE's column 2 Large win
        assert mv == (2, 2, PieceSize.LARGE), (
            f"Expected block at (2,2,L), got {mv}"
        )

    def test_phantom_block_guard_zero_pieces(self):
        """§228: pick_rollout_move must NOT return a phantom block when the opponent
        has 0 pieces of that size remaining — same class as §79/§82/§224.

        Position: BLUE occupies (2,0,S) and (2,1,S) [row-2, ci=6/7], apparent threat
        at (2,2,S) [ci=8, last in iteration order].  BLUE has 0 Smalls remaining so
        cannot complete the line.  The opp_has_size guard must suppress block
        classification so the function returns (0,0,S) [ci=0, first empty cell] as a
        rest-pick rather than (2,2,S) as a phantom block.

        This position reliably separates the two outcomes:
          guard works  → rest-pick (0,0,S) is returned first
          guard fails  → block (2,2,S) is returned instead
        """
        g = TicTacPro()
        # BLUE has 2 Smalls in row 2 at (2,0) and (2,1) — apparent threat at (2,2)
        g.board[2, 0, 0] = int(Player.BLUE)
        g.board[2, 1, 0] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 0  # BLUE exhausted all Smalls
        g._pieces_left -= 2
        g.current_player = Player.RED
        mv, is_win = pick_rollout_move(g)
        assert is_win is False, "No win should be possible in this position"
        # Guard working → first empty small cell (0,0,S) [ci=0] returned as rest-pick.
        # Guard failing → phantom block (2,2,S) [ci=8] returned instead.
        assert mv == (0, 0, PieceSize.SMALL), (
            f"§228: pick_rollout_move returned {mv} instead of rest-pick (0,0,S); "
            "if (2,2,S) was returned, the opp_has_size guard failed on phantom block"
        )
        assert mv in g.get_legal_moves(Player.RED), f"Returned move {mv} is not legal"

    def test_bullseye_threat_detected_in_threat_tier(self):
        """§259: pick_rollout_move must detect a bullseye threat (2-of-3 sizes owned in
        one cell) as a threat_mv, preferring it over a rest pick.

        With RED LARGE at (1,2), the function scans SMALL slots first. At (1,2,S) the
        bullseye pair is (M_slot, L_slot). L_slot has RED's piece → bf[L_slot]=cur,
        bf[M_slot]==0 → is_threat=True. The first rest pick is (0,0,S) which occurs
        BEFORE (1,2,S) in row-major scan, so the returned move must be the threat
        (1,2,S), NOT the rest pick (0,0,S).
        """
        g = TicTacPro()
        g.board[1, 2, 2] = int(Player.RED)    # RED LARGE at (1,2) — edge, no diagonals
        g.pieces_remaining[Player.RED][PieceSize.LARGE] -= 1
        g._pieces_left -= 1
        g.current_player = Player.RED
        mv, is_win = pick_rollout_move(g)
        assert is_win is False, "§259: bullseye threat must NOT be flagged as win"
        assert mv == (1, 2, PieceSize.SMALL), (
            f"§259: expected bullseye threat (1,2,S) but got {mv}. "
            "If rest pick (0,0,S) was returned, bullseye threat detection failed."
        )

    def test_bullseye_block_priority_over_line_block(self):
        """§343: pick_rollout_move must prioritize bullseye block over line block.

        Position: BLUE has 2/3 bullseye at CC (CC-L, CC-M) and a separate line
        2/3 threat (TR-S, BR-S in col-2-S). RED's CC-S blocks the bullseye;
        MR-S blocks the line. pick_rollout_move must return CC-S (bullseye block),
        not MR-S (line block).

        Scan order: SMALL cells iterate row-major; CC-S=(1,1) is ci=4, MR-S=(1,2) is ci=5.
        So MR-S (line block) is found AFTER CC-S (bullseye block) in the scan — but this
        tests that even if order were reversed, the bullseye block always wins.
        """
        g = TicTacPro()
        g.board[1, 1, 2] = int(Player.BLUE)   # CC-L
        g.board[1, 1, 1] = int(Player.BLUE)   # CC-M → bullseye threat at CC
        g.board[0, 2, 0] = int(Player.BLUE)   # TR-S
        g.board[2, 2, 0] = int(Player.BLUE)   # BR-S → col-2-S line threat
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 2
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 2
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  = 1
        g._pieces_left -= 4
        g.current_player = Player.RED
        mv, is_win = pick_rollout_move(g)
        assert is_win is False
        assert mv == (1, 1, PieceSize.SMALL), (
            f"§343: pick_rollout_move must return bullseye block (1,1,S), got {mv}. "
            "If (1,2,S) was returned, bullseye block priority was not applied."
        )


# ---------------------------------------------------------------------------
# §203: push() and push_known() direct unit tests
# ---------------------------------------------------------------------------

class TestPushAndPushKnown:
    """§203: push() and push_known() are used by all MCTS search agents but
    had no direct unit tests — a silent bug here would corrupt every search."""

    def test_push_places_piece(self):
        g = TicTacPro()
        g.push(1, 1, PieceSize.MEDIUM)
        assert g.board[1, 1, PieceSize.MEDIUM - 1] == Player.RED

    def test_push_decrements_pieces(self):
        g = TicTacPro()
        g.push(0, 0, PieceSize.SMALL)
        assert g.pieces_remaining[Player.RED][PieceSize.SMALL] == 2

    def test_push_switches_player_on_non_terminal(self):
        g = TicTacPro()
        assert g.current_player == Player.RED
        g.push(0, 0, PieceSize.SMALL)
        assert g.current_player == Player.BLUE

    def test_push_does_not_switch_player_on_win(self):
        """After a winning push, current_player stays as the winner (game_over=True)."""
        g = TicTacPro()
        # Set up RED with two Smalls in row 0; third push wins.
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] = 1  # one left
        g._pieces_left = sum(
            g.pieces_remaining[p][s]
            for p in [Player.RED, Player.BLUE]
            for s in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]
        )
        g.push(0, 2, PieceSize.SMALL)
        assert g.game_over is True
        assert g.winner == Player.RED
        # Player must NOT have switched — negamax uses this to identify the winner.
        assert g.current_player == Player.RED

    def test_push_draw_detected(self):
        """push() must detect a draw when pieces_left reaches 0 with no winner."""
        g = TicTacPro()
        # Force _pieces_left to 1 so the next push triggers draw check.
        g._pieces_left = 1
        g.push(0, 0, PieceSize.LARGE)  # non-winning (no two others in line)
        assert g.game_over is True
        assert g.winner == Player.NONE

    def test_push_known_non_winning_matches_push(self):
        """push_known(is_win=False) must produce identical board/state to push()."""
        g1 = TicTacPro(); g2 = TicTacPro()
        g1.push(1, 1, PieceSize.MEDIUM)
        g2.push_known(1, 1, PieceSize.MEDIUM, is_win=False)
        assert np.array_equal(g1.board, g2.board)
        assert g1.current_player == g2.current_player
        assert g1.game_over == g2.game_over

    def test_push_known_winning_matches_push(self):
        """push_known(is_win=True) must produce identical terminal state to push()."""
        g1 = TicTacPro(); g2 = TicTacPro()
        g1.board[0, 0, 0] = int(Player.RED)
        g1.board[0, 1, 0] = int(Player.RED)
        g2.board[0, 0, 0] = int(Player.RED)
        g2.board[0, 1, 0] = int(Player.RED)
        g1.push(0, 2, PieceSize.SMALL)
        g2.push_known(0, 2, PieceSize.SMALL, is_win=True)
        assert g1.game_over == g2.game_over == True
        assert g1.winner == g2.winner == Player.RED
        assert g1.current_player == g2.current_player

    def test_push_known_draw_detected(self):
        """§247: push_known(is_win=False) with _pieces_left==1 must detect draw.

        push() has a matching test (test_push_draw_detected) but push_known() was
        missing its draw-path coverage.  The draw branch is:
            elif self._pieces_left == 0: self.game_over = True
        If this were silently dropped, rollout termination would hang or skip draws."""
        g = TicTacPro()
        g._pieces_left = 1
        g.push_known(0, 0, PieceSize.LARGE, is_win=False)
        assert g.game_over is True, "§247: push_known must set game_over=True on draw"
        assert g.winner == Player.NONE, (
            "§247: push_known draw must leave winner=NONE; "
            f"got winner={g.winner}"
        )


# ---------------------------------------------------------------------------
# §204: _pieces_left counter invariant + reset() correctness
# ---------------------------------------------------------------------------

class TestPiecesLeftInvariant:
    """§204: _pieces_left must always equal sum(pieces_remaining) throughout a game.
    A desync causes wrong draw detection (fires too early or never)."""

    def _check_invariant(self, g):
        expected = sum(
            g.pieces_remaining[p][s]
            for p in [Player.RED, Player.BLUE]
            for s in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]
        )
        assert g._pieces_left == expected, (
            f"_pieces_left={g._pieces_left} != sum(pieces_remaining)={expected}"
        )

    def test_invariant_holds_at_init(self):
        g = TicTacPro()
        self._check_invariant(g)

    def test_invariant_holds_after_make_move(self):
        g = TicTacPro()
        for mv in [(0, 0, PieceSize.SMALL), (1, 1, PieceSize.MEDIUM),
                   (2, 2, PieceSize.LARGE), (0, 2, PieceSize.SMALL)]:
            g.make_move(*mv)
            self._check_invariant(g)

    def test_invariant_holds_after_push(self):
        g = TicTacPro()
        g.push(0, 0, PieceSize.SMALL)
        self._check_invariant(g)
        g.push(1, 1, PieceSize.MEDIUM)
        self._check_invariant(g)

    def test_invariant_holds_after_push_known(self):
        g = TicTacPro()
        g.push_known(0, 0, PieceSize.SMALL, is_win=False)
        self._check_invariant(g)

    def test_invariant_holds_after_clone(self):
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        g2 = g.clone()
        self._check_invariant(g2)


class TestReset:
    """§204: reset() must restore the game to a clean initial state."""

    def test_reset_clears_board(self):
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        g.reset()
        assert g.board.sum() == 0

    def test_reset_restores_pieces(self):
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        g.reset()
        for player in [Player.RED, Player.BLUE]:
            for size in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]:
                assert g.pieces_remaining[player][size] == 3

    def test_reset_restores_turn(self):
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)  # RED played
        assert g.current_player == Player.BLUE
        g.reset()
        assert g.current_player == Player.RED

    def test_reset_clears_game_over(self):
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.board[0, 2, 0] = int(Player.RED)
        g.game_over = True
        g.winner = Player.RED
        g.reset()
        assert g.game_over is False
        assert g.winner == Player.NONE

    def test_reset_pieces_left_is_18(self):
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        g.reset()
        assert g._pieces_left == 18  # 2 players × 3 sizes × 3 pieces each

    def test_reset_clears_move_history(self):
        """§237: reset() must clear move_history — guards against refactors that manually
        restore each field but forget to reset move_history to []."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        g.make_move(1, 1, PieceSize.MEDIUM)
        assert len(g.move_history) == 2, "Setup error: expected 2 moves in history"
        g.reset()
        assert g.move_history == [], (
            "§237: reset() must clear move_history; "
            f"got {g.move_history} after reset"
        )


# ---------------------------------------------------------------------------
# §205: get_visible_board() tests
# ---------------------------------------------------------------------------

class TestGetVisibleBoard:
    """§205: get_visible_board() is used for display and game history; correctness
    matters for debugging and any UI that uses it."""

    def test_empty_board_all_none(self):
        g = TicTacPro()
        vb = g.get_visible_board()
        assert vb.shape == (3, 3, 2)
        assert (vb[:, :, 0] == int(Player.NONE)).all(), "All cells must show NONE on empty board"

    def test_single_piece_visible(self):
        g = TicTacPro()
        g.board[1, 2, 0] = int(Player.RED)   # (1,2,SMALL)
        vb = g.get_visible_board()
        assert vb[1, 2, 0] == int(Player.RED)
        assert vb[1, 2, 1] == 1  # size 1 = SMALL

    def test_large_visible_over_small(self):
        """Larger piece at same cell must be shown over smaller one."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)    # SMALL
        g.board[0, 0, 2] = int(Player.BLUE)   # LARGE — different player on top
        vb = g.get_visible_board()
        assert vb[0, 0, 0] == int(Player.BLUE), "LARGE (top) must be shown"
        assert vb[0, 0, 1] == 3, "Size must be 3 (LARGE)"

    def test_medium_visible_over_small_not_large(self):
        """MEDIUM is shown when MEDIUM is top; SMALL is below."""
        g = TicTacPro()
        g.board[2, 1, 0] = int(Player.RED)    # SMALL
        g.board[2, 1, 1] = int(Player.BLUE)   # MEDIUM on top, no LARGE
        vb = g.get_visible_board()
        assert vb[2, 1, 0] == int(Player.BLUE)
        assert vb[2, 1, 1] == 2  # MEDIUM

    def test_all_three_sizes_large_shown(self):
        """§255: A cell with all 3 sizes (bullseye-completed) must show LARGE — the
        topmost piece — since get_visible_board iterates sizes in reverse."""
        g = TicTacPro()
        g.board[1, 1, 0] = int(Player.RED)    # SMALL
        g.board[1, 1, 1] = int(Player.BLUE)   # MEDIUM
        g.board[1, 1, 2] = int(Player.RED)    # LARGE — topmost
        vb = g.get_visible_board()
        assert vb[1, 1, 0] == int(Player.RED), "LARGE owner (RED) must be shown"
        assert vb[1, 1, 1] == 3, "Size index must be 3 (LARGE)"


# ── _would_win_bf / would_win ─────────────────────────────────────────────────

class TestWouldWin:
    """§301: Direct unit tests for _would_win_bf and would_win."""

    def test_same_size_row_win(self):
        """Placing the 3rd same-size Small in a row returns True."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        bf = g.board.tobytes()
        assert _would_win_bf(bf, 0, 2, 0, int(Player.RED)) is True

    def test_no_win_with_only_one_in_line(self):
        """Only 1 Red piece in a line — placing 2nd does not win."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        bf = g.board.tobytes()
        assert _would_win_bf(bf, 0, 1, 0, int(Player.RED)) is False

    def test_bullseye_win_via_would_win_bf(self):
        """CC-S + CC-L placed; placing CC-M completes bullseye → True."""
        g = TicTacPro()
        g.board[1, 1, 0] = int(Player.RED)   # CC-S
        g.board[1, 1, 2] = int(Player.RED)   # CC-L (skip MEDIUM)
        bf = g.board.tobytes()
        assert _would_win_bf(bf, 1, 1, 1, int(Player.RED)) is True

    def test_no_bullseye_with_opponent_piece(self):
        """If opponent occupies one slot in the cell, bullseye win is impossible."""
        g = TicTacPro()
        g.board[1, 1, 0] = int(Player.RED)    # CC-S (RED)
        g.board[1, 1, 2] = int(Player.BLUE)   # CC-L (BLUE) — mixed
        bf = g.board.tobytes()
        assert _would_win_bf(bf, 1, 1, 1, int(Player.RED)) is False

    def test_antidiag_win(self):
        """TR-S + BL-S: placing CC-S completes antidiagonal → True."""
        g = TicTacPro()
        g.board[0, 2, 0] = int(Player.RED)   # TR-S
        g.board[2, 0, 0] = int(Player.RED)   # BL-S
        bf = g.board.tobytes()
        assert _would_win_bf(bf, 1, 1, 0, int(Player.RED)) is True

    def test_would_win_matches_would_win_bf(self):
        """would_win(board, ...) must equal _would_win_bf(board.tobytes(), ...) always."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        bf = g.board.tobytes()
        assert would_win(g.board, 0, 2, 0, int(Player.RED)) == _would_win_bf(bf, 0, 2, 0, int(Player.RED))
        assert would_win(g.board, 1, 0, 0, int(Player.RED)) == _would_win_bf(bf, 1, 0, 0, int(Player.RED))

    def test_wrong_player_not_win(self):
        """Two RED pieces in a line — BLUE placing the 3rd is not a win for BLUE."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        bf = g.board.tobytes()
        assert _would_win_bf(bf, 0, 2, 0, int(Player.BLUE)) is False


class TestCoverageGaps:
    """§331: fill remaining game/* coverage gaps."""

    def test_make_move_invalid_size_returns_false(self):
        """Line 153: make_move returns False when size is not a valid PieceSize enum."""
        g = TicTacPro()
        result = g.make_move(0, 0, 99)  # 99 is not a valid PieceSize
        assert result is False

    def test_str_empty_board(self):
        """Lines 302-337: __str__ on a fresh game produces readable output."""
        g = TicTacPro()
        s = str(g)
        assert "0   1   2" in s
        assert "Red pieces" in s
        assert "Blue pieces" in s
        assert "Current player" in s

    def test_str_with_pieces_and_game_over_winner(self):
        """Lines 331-335: __str__ shows winner when game_over is True."""
        g = TicTacPro()
        # Win RED via anti-diagonal: (0,2,S), (1,1,S), (2,0,S)
        g.make_move(0, 2, PieceSize.SMALL)  # RED
        g.make_move(0, 0, PieceSize.LARGE)  # BLUE
        g.make_move(1, 1, PieceSize.SMALL)  # RED
        g.make_move(0, 1, PieceSize.LARGE)  # BLUE
        g.make_move(2, 0, PieceSize.SMALL)  # RED wins
        assert g.game_over and g.winner == Player.RED
        s = str(g)
        assert "RED WINS" in s

    def test_str_draw(self):
        """Line 335: __str__ shows DRAW when winner is NONE and game_over."""
        g = TicTacPro()
        g.game_over = True
        g.winner = Player.NONE
        s = str(g)
        assert "DRAW" in s
