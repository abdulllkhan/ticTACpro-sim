"""Unit tests for the TicTacPro game engine."""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from game.tictacpro import TicTacPro, Player, PieceSize


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
        g = TicTacPro()
        # Use all 3 RED smalls
        for col in range(3):
            g.make_move(0, col, PieceSize.SMALL, player=Player.RED)
        moves = g.get_legal_moves(Player.RED)
        for r, c, sz in moves:
            assert sz != PieceSize.SMALL

    def test_game_over_returns_empty(self):
        g = TicTacPro()
        g.game_over = True
        assert g.get_legal_moves() == []


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
