"""§194: Tests for rollout_c_wrapper — C-accelerated rollout and legal-move helpers."""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import random
import numpy as np
from game.tictacpro import TicTacPro, Player, PieceSize

from game.rollout_c_wrapper import (
    C_ROLLOUT_AVAILABLE,
    fast_rollout_c,
    get_legal_moves_fast,
    get_sorted_legal_c,
    get_sorted_legal_ex_c,
)

pytestmark = pytest.mark.skipif(
    not C_ROLLOUT_AVAILABLE,
    reason="rollout_c.so not available",
)

_VALID_PLAYERS = {Player.RED, Player.BLUE, Player.NONE}


def _mid_game():
    """Build a non-trivial mid-game state with 4 moves played."""
    g = TicTacPro()
    for mv in [(0, 0, PieceSize.SMALL), (1, 1, PieceSize.MEDIUM),
               (2, 2, PieceSize.LARGE), (0, 2, PieceSize.SMALL)]:
        g.make_move(*mv)
    return g


class TestFastRolloutC:
    def test_returns_valid_player(self):
        g = TicTacPro()
        result = fast_rollout_c(g)
        assert result in _VALID_PLAYERS

    def test_does_not_modify_game_state(self):
        g = TicTacPro()
        board_before = bytes(g.board)
        pr_before = {p: dict(g.pieces_remaining[p]) for p in (Player.RED, Player.BLUE)}
        fast_rollout_c(g)
        assert bytes(g.board) == board_before
        for p in (Player.RED, Player.BLUE):
            assert g.pieces_remaining[p] == pr_before[p]

    def test_does_not_modify_mid_game_state(self):
        g = _mid_game()
        board_before = bytes(g.board)
        pr_before = {p: dict(g.pieces_remaining[p]) for p in (Player.RED, Player.BLUE)}
        fast_rollout_c(g)
        assert bytes(g.board) == board_before
        for p in (Player.RED, Player.BLUE):
            assert g.pieces_remaining[p] == pr_before[p]

    def test_result_consistent_across_calls_on_same_state(self):
        """Pure function — same board state returns deterministic result (greedy)."""
        g = _mid_game()
        r1 = fast_rollout_c(g)
        r2 = fast_rollout_c(g)
        assert r1 == r2

    def test_rollout_many_random_games(self):
        """fast_rollout_c must return a valid player for 50 random mid-game states."""
        rng = random.Random(0)
        for _ in range(50):
            g = TicTacPro()
            n = rng.randint(0, 8)
            for _ in range(n):
                moves = g.get_legal_moves(g.current_player)
                if not moves or g.game_over:
                    break
                g.make_move(*rng.choice(moves))
            if g.game_over:
                continue
            result = fast_rollout_c(g)
            assert result in _VALID_PLAYERS, f"Invalid result {result} for game state"

    def test_completed_game_not_called(self):
        """Should not be called on a completed game, but just verify it doesn't crash."""
        g = TicTacPro()
        for mv in [(0, 0, PieceSize.SMALL), (1, 0, PieceSize.SMALL),
                   (0, 1, PieceSize.SMALL), (1, 1, PieceSize.SMALL),
                   (0, 2, PieceSize.SMALL)]:
            if not g.game_over:
                g.make_move(*mv)
        # Whether game_over or not, must not crash
        fast_rollout_c(g)

    def test_red_wins_when_immediate_win_available(self):
        """§236: fast_rollout_c must return Player.RED when RED can win immediately.

        RED has two-in-a-row Small at (0,0) and (0,1); (0,2,S) wins the game.
        The greedy C rollout must detect and play the winning move, returning RED.
        """
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] = 1  # used 2, 1 remaining
        g._pieces_left -= 2
        g.current_player = Player.RED
        assert not g.game_over, "Setup error: game must not already be over"
        result = fast_rollout_c(g)
        assert result == Player.RED, (
            f"§236: fast_rollout_c returned {result!r} instead of Player.RED — "
            "C rollout failed to detect/play the immediate winning move (0,2,S)."
        )

    def test_blue_wins_when_immediate_win_available(self):
        """§236: fast_rollout_c must return Player.BLUE when BLUE can win immediately.

        BLUE has two-in-a-row Small at (0,0) and (0,1); (0,2,S) wins the game.
        """
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.BLUE)
        g.board[0, 1, 0] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 1
        g._pieces_left -= 2
        g.current_player = Player.BLUE
        assert not g.game_over, "Setup error: game must not already be over"
        result = fast_rollout_c(g)
        assert result == Player.BLUE, (
            f"§236: fast_rollout_c returned {result!r} instead of Player.BLUE — "
            "C rollout failed to detect/play the immediate winning move (0,2,S)."
        )


class TestGetLegalMovesFast:
    def test_matches_get_legal_moves_empty_board(self):
        g = TicTacPro()
        fast = sorted(get_legal_moves_fast(g))
        slow = sorted(g.get_legal_moves(Player.RED))
        assert fast == slow

    def test_matches_get_legal_moves_mid_game(self):
        g = _mid_game()
        fast = sorted(get_legal_moves_fast(g))
        slow = sorted(g.get_legal_moves(g.current_player))
        assert fast == slow

    def test_matches_many_random_states(self):
        rng = random.Random(7)
        for _ in range(100):
            g = TicTacPro()
            n = rng.randint(0, 10)
            for _ in range(n):
                moves = g.get_legal_moves(g.current_player)
                if not moves or g.game_over:
                    break
                g.make_move(*rng.choice(moves))
            if g.game_over:
                continue
            fast = sorted(get_legal_moves_fast(g))
            slow = sorted(g.get_legal_moves(g.current_player))
            assert fast == slow, (
                f"Mismatch at move {n}: fast={fast} slow={slow}\n"
                f"board={list(g.board)}"
            )

    def test_does_not_modify_game(self):
        g = _mid_game()
        board_before = bytes(g.board)
        get_legal_moves_fast(g)
        assert bytes(g.board) == board_before


class TestGetSortedLegalC:
    def test_same_set_as_legal_moves_empty(self):
        g = TicTacPro()
        sorted_moves = get_sorted_legal_c(g)
        slow = set(g.get_legal_moves(Player.RED))
        assert set(sorted_moves) == slow

    def test_same_set_as_legal_moves_mid_game(self):
        g = _mid_game()
        sorted_moves = get_sorted_legal_c(g)
        slow = set(g.get_legal_moves(g.current_player))
        assert set(sorted_moves) == slow

    def test_winning_move_at_end(self):
        """get_sorted_legal_c sorts winning moves to the tail (for pop()-first access)."""
        g = TicTacPro()
        # RED is about to win: two SMALL in a row
        g.make_move(0, 0, PieceSize.SMALL)
        g.make_move(1, 0, PieceSize.LARGE)
        g.make_move(0, 1, PieceSize.SMALL)
        g.make_move(1, 1, PieceSize.LARGE)
        # RED's turn: (0,2,SMALL) wins
        assert g.current_player == Player.RED
        sorted_moves = get_sorted_legal_c(g)
        assert sorted_moves[-1] == (0, 2, PieceSize.SMALL), (
            f"Winning move should be last, got {sorted_moves[-1]}"
        )

    def test_does_not_modify_game(self):
        g = _mid_game()
        board_before = bytes(g.board)
        get_sorted_legal_c(g)
        assert bytes(g.board) == board_before

    def test_same_set_many_random_states(self):
        rng = random.Random(13)
        for _ in range(100):
            g = TicTacPro()
            n = rng.randint(0, 10)
            for _ in range(n):
                moves = g.get_legal_moves(g.current_player)
                if not moves or g.game_over:
                    break
                g.make_move(*rng.choice(moves))
            if g.game_over:
                continue
            c_moves = set(get_sorted_legal_c(g))
            py_moves = set(g.get_legal_moves(g.current_player))
            assert c_moves == py_moves

    def test_phantom_block_guard(self):
        """§232: sort_legal_moves_c must NOT classify a cell as blocking when the
        opponent has 0 pieces of that size remaining (opp_has_sz guard in rollout_c.c:177).

        Position:
          BLUE (0,0,M)+(0,1,M), BLUE has 1 Medium remaining → real block at (0,2,M) → at tail
          BLUE (2,0,S)+(2,1,S), BLUE has 0 Smalls remaining  → phantom block at (2,2,S) → rest

        With guard working: last element is (0,2,M) (the only real blocking move).
        With guard failing: (2,2,S) also gets blocking priority and appears after (0,2,M),
        making it the last element instead.
        """
        g = TicTacPro()
        g.board[0, 0, 1] = int(Player.BLUE)
        g.board[0, 1, 1] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 1
        g.board[2, 0, 0] = int(Player.BLUE)
        g.board[2, 1, 0] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 0
        g._pieces_left -= 4
        g.current_player = Player.RED

        sorted_moves = get_sorted_legal_c(g)

        assert (0, 2, PieceSize.MEDIUM) in sorted_moves
        assert (2, 2, PieceSize.SMALL) in sorted_moves
        # Real block must be the last element; phantom must not be last.
        assert sorted_moves[-1] == (0, 2, PieceSize.MEDIUM), (
            f"§232: C phantom-block guard failed — last move is {sorted_moves[-1]}, "
            "expected real block (0,2,M). If (2,2,S) is last, opp_has_sz guard failed."
        )


class TestGetSortedLegalExC:
    def test_same_moves_as_sorted(self):
        g = _mid_game()
        moves_ex, _ = get_sorted_legal_ex_c(g)
        moves_base = get_sorted_legal_c(g)
        assert set(moves_ex) == set(moves_base)

    def test_n_winning_zero_on_empty_board(self):
        g = TicTacPro()
        _, n_winning = get_sorted_legal_ex_c(g)
        assert n_winning == 0

    def test_n_winning_positive_when_win_available(self):
        """n_winning must be ≥1 when current player has an immediate win."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        g.make_move(1, 0, PieceSize.LARGE)
        g.make_move(0, 1, PieceSize.SMALL)
        g.make_move(1, 1, PieceSize.LARGE)
        # RED can win at (0,2,SMALL)
        assert g.current_player == Player.RED
        moves_ex, n_winning = get_sorted_legal_ex_c(g)
        assert n_winning >= 1, f"Expected n_winning≥1, got {n_winning}"
        winning_moves = moves_ex[-n_winning:]
        assert (0, 2, PieceSize.SMALL) in winning_moves

    def test_winning_moves_actually_win(self):
        """Every move reported as winning must actually win the game."""
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)
        g.make_move(1, 0, PieceSize.LARGE)
        g.make_move(0, 1, PieceSize.SMALL)
        g.make_move(1, 1, PieceSize.LARGE)
        moves_ex, n_winning = get_sorted_legal_ex_c(g)
        for mv in moves_ex[-n_winning:]:
            import copy
            g2 = copy.deepcopy(g)
            g2.make_move(*mv)
            assert g2.game_over and g2.winner == Player.RED, (
                f"Reported winning move {mv} did not actually win"
            )

    def test_does_not_modify_game(self):
        g = _mid_game()
        board_before = bytes(g.board)
        get_sorted_legal_ex_c(g)
        assert bytes(g.board) == board_before

    def test_phantom_block_guard(self):
        """§232: sort_legal_moves_ex_c must NOT classify a cell as blocking when the
        opponent has 0 pieces of that size remaining (opp_has_sz guard in rollout_c.c:234).

        Same position as TestGetSortedLegalC::test_phantom_block_guard.
        The n_winning must be 0 and the tail (blocking section) must be [(0,2,M)] only.
        """
        g = TicTacPro()
        g.board[0, 0, 1] = int(Player.BLUE)
        g.board[0, 1, 1] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 1
        g.board[2, 0, 0] = int(Player.BLUE)
        g.board[2, 1, 0] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 0
        g._pieces_left -= 4
        g.current_player = Player.RED

        moves, n_winning = get_sorted_legal_ex_c(g)

        assert n_winning == 0, f"No wins for RED; got n_winning={n_winning}"
        assert (0, 2, PieceSize.MEDIUM) in moves
        assert (2, 2, PieceSize.SMALL) in moves
        # Layout: [rest..., blocking..., winning]. With n_winning=0 and one real block,
        # the last element must be (0,2,M). A phantom-block failure would put (2,2,S) last.
        assert moves[-1] == (0, 2, PieceSize.MEDIUM), (
            f"§232: C ex phantom-block guard failed — last move is {moves[-1]}, "
            "expected real block (0,2,M). If (2,2,S) is last, opp_has_sz guard failed."
        )
