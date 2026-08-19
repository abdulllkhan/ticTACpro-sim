"""Tests for OptimalAgent, BullseyeAgent, MCTSAgent, MinimaxAgent, and NeuralMCTSAgent."""

import pytest
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from game.tictacpro import TicTacPro, Player, PieceSize
from game.optimal_agent import OptimalAgent
from game.bullseye_agent import BullseyeAgent
from rl.mcts_agent import MCTSAgent, _sort_untried
from rl.minimax_agent import MinimaxAgent
from rl.neural_mcts_agent import NeuralMCTSAgent
from rl.network import GPUDQNNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play_full_game(red_agent, blue_agent, max_moves=40):
    game = TicTacPro()
    agents = {Player.RED: red_agent, Player.BLUE: blue_agent}
    for _ in range(max_moves):
        if game.game_over:
            break
        agent = agents[game.current_player]
        mv = agent.get_action(game, game.current_player)
        if mv is None:
            break
        game.make_move(*mv)
    return game.winner


# ── OptimalAgent ──────────────────────────────────────────────────────────────

class TestOptimalAgent:
    def setup_method(self):
        self.agent = OptimalAgent()

    def test_returns_legal_move_on_empty_board(self):
        g = TicTacPro()
        mv = self.agent.get_action(g)
        assert mv in g.get_legal_moves()

    def test_takes_immediate_win(self):
        """Agent must take a winning move when available."""
        g = TicTacPro()
        # Give RED two anti-diagonal S pieces; third should win
        g.make_move(0, 2, PieceSize.SMALL)  # RED: TR-S
        g.make_move(0, 0, PieceSize.LARGE)  # BLUE irrelevant
        g.make_move(1, 1, PieceSize.SMALL)  # RED: CC-S
        g.make_move(0, 1, PieceSize.LARGE)  # BLUE irrelevant
        # RED now has 2/3 anti-diagonal; BL-S wins
        assert g.current_player == Player.RED
        mv = self.agent.get_action(g)
        assert mv == (2, 0, PieceSize.SMALL)

    def test_blocks_opponent_win(self):
        """Agent must block when opponent is about to win."""
        g = TicTacPro()
        # Give BLUE two anti-diagonal S pieces while it's RED's turn
        # Manually set board to simulate BLUE having CC-S and TR-S
        g.board[1, 1, 0] = int(Player.BLUE)   # CC-S for BLUE
        g.board[0, 2, 0] = int(Player.BLUE)   # TR-S for BLUE
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 1  # used 2 already
        assert g.current_player == Player.RED
        mv = self.agent.get_action(g)
        # Must block at BL-S (2, 0, SMALL)
        assert mv == (2, 0, PieceSize.SMALL)

    def test_bullseye_block_priority_over_line_block(self):
        """§333: Block scan must prefer bullseye blocks over line blocks.

        BLUE has CC-L and CC-M (bullseye 2/3 at CC, needs CC-S) and
        TR-S + BR-S (col-2-Small 2/3, needs MR-S).  RED must block one;
        the bullseye threat is more critical and should be blocked first.
        """
        g = TicTacPro()
        g.board[1, 1, 2] = int(Player.BLUE)   # CC-L
        g.board[1, 1, 1] = int(Player.BLUE)   # CC-M  → bullseye threat at CC
        g.board[0, 2, 0] = int(Player.BLUE)   # TR-S
        g.board[2, 2, 0] = int(Player.BLUE)   # BR-S  → col-2-S line threat
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 2
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 2
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  = 1
        g._pieces_left = sum(
            g.pieces_remaining[p][s]
            for p in [Player.RED, Player.BLUE]
            for s in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]
        )
        assert g.current_player == Player.RED
        mv = self.agent.get_action(g)
        assert mv == (1, 1, PieceSize.SMALL), (
            f"§333: expected bullseye block CC-S=(1,1,S), got {mv}. "
            "OptimalAgent must prefer bullseye_block over line_block."
        )

    def test_wins_against_random_as_red(self):
        """OptimalAgent as RED wins at least 9/10 vs random."""
        import random
        wins = 0
        for _ in range(10):
            g = TicTacPro()
            while not g.game_over:
                if g.current_player == Player.RED:
                    mv = self.agent.get_action(g)
                else:
                    moves = g.get_legal_moves()
                    mv = random.choice(moves) if moves else None
                if mv is None:
                    break
                g.make_move(*mv)
            if g.winner == Player.RED:
                wins += 1
        assert wins >= 9, f"Expected ≥9/10 wins as RED, got {wins}"

    def test_wins_against_random_as_blue(self):
        """OptimalAgent as BLUE wins at least 8/10 vs random."""
        import random
        wins = 0
        for _ in range(10):
            g = TicTacPro()
            while not g.game_over:
                if g.current_player == Player.BLUE:
                    mv = self.agent.get_action(g)
                else:
                    moves = g.get_legal_moves()
                    mv = random.choice(moves) if moves else None
                if mv is None:
                    break
                g.make_move(*mv)
            if g.winner == Player.BLUE:
                wins += 1
        assert wins >= 8, f"Expected ≥8/10 wins as BLUE, got {wins}"

    def test_works_as_player_arg(self):
        """get_action with explicit player arg returns a legal move."""
        g = TicTacPro()
        mv = self.agent.get_action(g, player=Player.RED)
        assert mv in g.get_legal_moves()

    def test_no_phantom_block_when_opp_out_of_small(self):
        """Step-3 piece-availability guard: with BLUE having 2/3 anti-diagonal S but 0 Small
        pieces left, the agent must still return a legal move (BL-S is fine here — it advances
        RED's own anti-diagonal plan; the key is that the guard prevents a wasted block when
        RED has NO other anti-diagonal cells available and a better plan move exists elsewhere).
        """
        g = TicTacPro()
        # BLUE has 2/3 anti-diagonal Small pieces but exhausted all 3 Small pieces
        # (third Small placed elsewhere: BL-S slot is empty).
        g.board[0, 2, 0] = int(Player.BLUE)   # TR-S (BLUE)
        g.board[1, 1, 0] = int(Player.BLUE)   # CC-S (BLUE)
        g.board[2, 0, 0] = int(Player.BLUE)   # BL-S (BLUE) — BLUE's 3rd Small elsewhere
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 0  # BLUE has no Small left
        g._pieces_left = sum(g.pieces_remaining[p][s] for p in [Player.RED, Player.BLUE] for s in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE])
        mv = self.agent.get_action(g)
        # With all three anti-diagonal S slots taken (by BLUE), there is no anti-diagonal
        # S cell for either the phantom-block or the plan. Agent must return a legal move
        # (step-5 _RED_PLAN or step-6 fallback).
        assert mv in g.get_legal_moves()
        # The phantom-block step-3 guard must NOT cause a KeyError or infinite loop.
        # Specifically, it should NOT try to block at a BLUE-occupied cell.
        assert mv not in [(0, 2, PieceSize.SMALL), (1, 1, PieceSize.SMALL), (2, 0, PieceSize.SMALL)]


# ── BullseyeAgent ─────────────────────────────────────────────────────────────

class TestBullseyeAgent:
    """§334: BullseyeAgent — sequential bullseye fork strategy."""

    def setup_method(self):
        self.agent = BullseyeAgent()

    def test_returns_legal_move_from_start(self):
        g = TicTacPro()
        mv = self.agent.get_action(g)
        assert mv in g.get_legal_moves()

    def test_wins_immediately_when_possible(self):
        """Takes an immediate win over any other move."""
        g = TicTacPro()
        # RED (mover) has CC-L and CC-M; CC-S completes bullseye → win.
        g.board[1, 1, 2] = int(Player.RED)
        g.board[1, 1, 1] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.LARGE]  = 2
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] = 2
        g._pieces_left = sum(g.pieces_remaining[p][s]
                             for p in [Player.RED, Player.BLUE]
                             for s in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE])
        mv = self.agent.get_action(g, player=Player.RED)
        assert mv == (1, 1, PieceSize.SMALL), f"Expected bullseye win CC-S, got {mv}"

    def test_blocks_opponent_bullseye(self):
        """Blocks opponent's immediate bullseye threat."""
        g = TicTacPro()
        # BLUE has CC-L and CC-M → bullseye threat at CC-S. RED moves next.
        g.board[1, 1, 2] = int(Player.BLUE)
        g.board[1, 1, 1] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 2
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 2
        g._pieces_left = sum(g.pieces_remaining[p][s]
                             for p in [Player.RED, Player.BLUE]
                             for s in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE])
        mv = self.agent.get_action(g, player=Player.RED)
        assert mv == (1, 1, PieceSize.SMALL), f"Expected bullseye block CC-S, got {mv}"

    def test_bullseye_block_priority_over_line_block(self):
        """§333: Bullseye block must be preferred over line block (same as OptimalAgent §333)."""
        g = TicTacPro()
        g.board[1, 1, 2] = int(Player.BLUE)   # CC-L
        g.board[1, 1, 1] = int(Player.BLUE)   # CC-M → bullseye threat at CC
        g.board[0, 2, 0] = int(Player.BLUE)   # TR-S
        g.board[2, 2, 0] = int(Player.BLUE)   # BR-S → col-2-S line threat
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 2
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 2
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  = 1
        g._pieces_left = sum(g.pieces_remaining[p][s]
                             for p in [Player.RED, Player.BLUE]
                             for s in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE])
        mv = self.agent.get_action(g, player=Player.RED)
        assert mv == (1, 1, PieceSize.SMALL), (
            f"§333: expected bullseye block CC-S=(1,1,S), got {mv}"
        )

    def test_completes_own_bullseye_step4(self):
        """Step 4: completes own 2/3 bullseye when opponent hasn't blocked."""
        g = TicTacPro()
        # RED has CC-L and CC-M (2/3 bullseye); no BLUE piece at CC. RED moves next.
        g.board[1, 1, 2] = int(Player.RED)
        g.board[1, 1, 1] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.LARGE]  = 2
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] = 2
        # Give BLUE some pieces elsewhere so it doesn't look like a win yet
        g.board[0, 0, 2] = int(Player.BLUE)  # TL-L  (no threat)
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE] = 2
        g._pieces_left = sum(g.pieces_remaining[p][s]
                             for p in [Player.RED, Player.BLUE]
                             for s in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE])
        mv = self.agent.get_action(g, player=Player.RED)
        # RED should complete the bullseye at CC-S.
        assert mv == (1, 1, PieceSize.SMALL), f"Step4: expected CC-S bullseye completion, got {mv}"

    def test_beats_random_as_blue(self):
        """BullseyeAgent as BLUE wins at least 9/10 vs random."""
        import random
        wins = 0
        for _ in range(10):
            g = TicTacPro()
            while not g.game_over:
                if g.current_player == Player.BLUE:
                    mv = self.agent.get_action(g)
                else:
                    moves = g.get_legal_moves()
                    mv = random.choice(moves) if moves else None
                if mv is None:
                    break
                g.make_move(*mv)
            if g.winner == Player.BLUE:
                wins += 1
        assert wins >= 9, f"BullseyeAgent-BLUE vs random: expected ≥9/10 wins, got {wins}"

    def test_returns_none_when_no_legal_moves(self):
        """Returns None when the board is full and no moves are possible."""
        g = TicTacPro()
        # Fill entire board — 9 cells × 3 sizes = 27 slots.
        for r in range(3):
            for c in range(3):
                for sz in range(3):
                    g.board[r, c, sz] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.RED][PieceSize.LARGE]  = 0
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 0
        g._pieces_left = 0
        mv = self.agent.get_action(g)
        assert mv is None, f"Expected None when no legal moves, got {mv}"

    def test_accepts_player_kwarg(self):
        """player= kwarg is accepted (interface compatibility with other agents)."""
        g = TicTacPro()
        mv = self.agent.get_action(g, player=Player.RED)
        assert mv in g.get_legal_moves()

    def test_accepts_epsilon_kwarg(self):
        """epsilon= kwarg is accepted (interface compatibility)."""
        g = TicTacPro()
        mv = self.agent.get_action(g, epsilon=0.0)
        assert mv in g.get_legal_moves()


# ── MCTSAgent ─────────────────────────────────────────────────────────────────

class TestMCTSAgent:
    def setup_method(self):
        self.agent = MCTSAgent(time_limit=0.2, max_simulations=500)

    def test_returns_legal_move(self):
        g = TicTacPro()
        mv = self.agent.get_action(g)
        assert mv in g.get_legal_moves()

    def test_returns_none_on_game_over(self):
        g = TicTacPro()
        # Force game over
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.board[0, 2, 0] = int(Player.RED)
        g.winner = Player.RED
        g.game_over = True
        mv = self.agent.get_action(g)
        assert mv is None

    def test_wins_against_random(self):
        """MCTS should win most games vs random."""
        import random
        wins = 0
        for _ in range(5):
            g = TicTacPro()
            while not g.game_over:
                if g.current_player == Player.RED:
                    mv = self.agent.get_action(g, g.current_player)
                else:
                    moves = g.get_legal_moves()
                    mv = random.choice(moves) if moves else None
                if mv is None:
                    break
                g.make_move(*mv)
            if g.winner == Player.RED:
                wins += 1
        assert wins >= 3, f"MCTS expected ≥3/5 wins, got {wins}"

    def test_tree_reuse_clears_on_new_game(self):
        """After a full game, _reuse_node is reset when a new game starts."""
        g = TicTacPro()
        import random
        # Play a short game to populate reuse node
        for _ in range(4):
            if g.game_over:
                break
            moves = g.get_legal_moves()
            g.make_move(*random.choice(moves))
        self.agent.get_action(g, g.current_player)
        # _reuse_node may be None if an immediate win was found (early-win fast path)
        # or non-None if tree search was needed. Both are valid.

        # Start a new game — agent should always return a legal move
        g2 = TicTacPro()
        mv = self.agent.get_action(g2, g2.current_player)
        assert mv in g2.get_legal_moves()

    def test_reset_clears_reuse_node(self):
        g = TicTacPro()
        self.agent.get_action(g, g.current_player)
        assert self.agent._reuse_node is not None
        self.agent.reset()
        assert self.agent._reuse_node is None

    def test_get_info_returns_sims_and_winrate(self):
        g = TicTacPro()
        mv, (sims, wr) = self.agent.get_info(g)
        assert mv in g.get_legal_moves()
        assert sims > 0
        assert 0.0 <= wr <= 1.0

    def test_sort_untried_phantom_block_guard(self):
        """§235: _sort_untried must NOT classify a cell as blocking when the
        opponent has 0 pieces of that size remaining (opp_has_sz guard at mcts_agent.py:87).

        Position:
          BLUE (0,0,S)+(0,1,S), BLUE has 1 Small remaining  → real block at (0,2,S) → blocking
          BLUE (2,0,M)+(2,1,M), BLUE has 0 Mediums remaining → phantom block at (2,2,M) → rest

        get_legal_moves iterates sizes first (S→M→L), so (0,2,S) appears before (2,2,M)
        in the moves list. When both are classified as blocking, (2,2,M) appears last in
        the blocking sublist — making it the last element overall. Guard working: (0,2,S)
        is last (only real block); guard failing: (2,2,M) is last (phantom misclassified).
        """
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.BLUE)
        g.board[0, 1, 0] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 1
        g.board[2, 0, 1] = int(Player.BLUE)
        g.board[2, 1, 1] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 0
        g._pieces_left -= 4
        g.current_player = Player.RED

        moves = g.get_legal_moves(Player.RED)
        sorted_moves = _sort_untried(g, moves, guided=False)

        assert (0, 2, PieceSize.SMALL)  in sorted_moves
        assert (2, 2, PieceSize.MEDIUM) in sorted_moves
        assert sorted_moves[-1] == (0, 2, PieceSize.SMALL), (
            f"§235: _sort_untried phantom-block guard failed — last move is {sorted_moves[-1]}, "
            "expected real block (0,2,S). If (2,2,M) is last, opp_has_sz guard failed."
        )

    def test_sort_untried_bullseye_block_before_line_block(self):
        """§341c: _sort_untried must place bullseye block after line block so pop()
        returns the bullseye block first (higher urgency than a line block).

        Position: BLUE has 2/3 bullseye at CC (CC-L, CC-M) and line 2/3 (TR-S, BR-S).
        RED's CC-S blocks the bullseye; MR-S blocks the line.
        Expected: bull_blocking after line_blocking → CC-S is closer to end.
        """
        g = TicTacPro()
        g.board[1, 1, 2] = int(Player.BLUE)   # CC-L
        g.board[1, 1, 1] = int(Player.BLUE)   # CC-M → bullseye threat at CC
        g.board[0, 2, 0] = int(Player.BLUE)   # TR-S
        g.board[2, 2, 0] = int(Player.BLUE)   # BR-S → line threat col-2-S
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 2
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 2
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  = 1
        g._pieces_left -= 4
        g.current_player = Player.RED

        moves = g.get_legal_moves(Player.RED)
        sorted_moves = _sort_untried(g, moves, guided=False)

        bull_idx = sorted_moves.index((1, 1, PieceSize.SMALL))
        line_idx = sorted_moves.index((1, 2, PieceSize.SMALL))
        assert bull_idx > line_idx, (
            f"§341c: _sort_untried must place bullseye block (1,1,S) after line block (1,2,S) "
            f"so pop() returns bullseye first; got bull_idx={bull_idx} line_idx={line_idx}"
        )


# ── MinimaxAgent ──────────────────────────────────────────────────────────────

class TestMinimaxAgent:
    def setup_method(self):
        self.agent = MinimaxAgent(time_limit=0.5, use_book=True)

    def test_returns_legal_move(self):
        g = TicTacPro()
        mv = self.agent.get_action(g)
        assert mv in g.get_legal_moves()

    def test_opening_book_used(self):
        """First move should come from the opening book."""
        g = TicTacPro()
        mv = self.agent.get_action(g)
        # Book's first preferred move is TC-L = (0, 1, LARGE)
        assert mv == (0, 1, PieceSize.LARGE), f"Expected book move (0,1,LARGE), got {mv}"

    def test_takes_immediate_win(self):
        """Minimax must take a winning move even during book phase."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] = 1
        mv = self.agent.get_action(g, Player.RED)
        # Should play TR-S = (0, 2, SMALL) to complete a row win
        assert mv == (0, 2, PieceSize.SMALL)

    def test_wins_against_random(self):
        """Minimax with book should win most games vs random."""
        import random
        wins = 0
        for _ in range(5):
            g = TicTacPro()
            while not g.game_over:
                if g.current_player == Player.RED:
                    mv = self.agent.get_action(g, g.current_player)
                else:
                    moves = g.get_legal_moves()
                    mv = random.choice(moves) if moves else None
                if mv is None:
                    break
                g.make_move(*mv)
            if g.winner == Player.RED:
                wins += 1
        assert wins >= 4, f"Minimax expected ≥4/5 wins, got {wins}"

    def test_no_book_falls_back_to_search(self):
        """Without book, agent still returns a legal move via search."""
        agent_no_book = MinimaxAgent(time_limit=0.5, use_book=False)
        g = TicTacPro()
        mv = agent_no_book.get_action(g)
        assert mv in g.get_legal_moves()

    def test_book_vs_ccl_uses_tc_s(self):
        """When BLUE opens with CC-L, RED book must switch to TC-S plan (not CC-M)."""
        from rl.minimax_agent import _book_move
        g = TicTacPro()
        g.make_move(0, 1, PieceSize.LARGE)   # RED TC-L
        g.make_move(1, 1, PieceSize.LARGE)   # BLUE CC-L
        mv = _book_move(g, int(Player.RED))
        assert mv == (0, 1, PieceSize.SMALL), f"Expected TC-S vs CC-L, got {mv}"

    def test_book_vs_ccs_uses_cc_m(self):
        """When BLUE opens with CC-S, RED book should continue with CC-M."""
        from rl.minimax_agent import _book_move
        g = TicTacPro()
        g.make_move(0, 1, PieceSize.LARGE)   # RED TC-L
        g.make_move(1, 1, PieceSize.SMALL)   # BLUE CC-S
        mv = _book_move(g, int(Player.RED))
        assert mv == (1, 1, PieceSize.MEDIUM), f"Expected CC-M vs CC-S, got {mv}"

    def test_book_vs_ccm_uses_cc_s(self):
        """When BLUE opens with CC-M, RED book must switch to TC-bullseye plan (CC-S next)."""
        from rl.minimax_agent import _book_move
        g = TicTacPro()
        g.make_move(0, 1, PieceSize.LARGE)   # RED TC-L
        g.make_move(1, 1, PieceSize.MEDIUM)  # BLUE CC-M
        mv = _book_move(g, int(Player.RED))
        assert mv == (1, 1, PieceSize.SMALL), f"Expected CC-S vs CC-M, got {mv}"

    def test_blue_book_uses_cc_s(self):
        """BLUE book must open with CC-S on the first BLUE move."""
        from rl.minimax_agent import _book_move
        g = TicTacPro()
        g.make_move(0, 1, PieceSize.LARGE)   # RED TC-L (any opening)
        mv = _book_move(g, int(Player.BLUE))
        assert mv == (1, 1, PieceSize.SMALL), f"Expected BLUE CC-S book opening, got {mv}"

    def test_phantom_block_not_classified_when_opp_out_of_pieces(self):
        """_book_move must not classify a position as forced block when opponent has 0 pieces of that size."""
        from rl.minimax_agent import _book_move
        import numpy as np
        # Simulate a position where BLUE has 2/3 Small pieces in anti-diag but 0 S remaining
        g = TicTacPro()
        g.board[0, 2, 0] = int(Player.BLUE)   # TR-S (BLUE)
        g.board[1, 1, 0] = int(Player.BLUE)   # CC-S (BLUE)
        # Manually set BLUE to have 0 Small pieces left (exhausted)
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 0
        g._pieces_left = sum(g.pieces_remaining[p][s] for p in [Player.RED, Player.BLUE] for s in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE])
        # RED's turn: BLUE threatens BL-S to complete anti-diag-S only if BLUE has S pieces.
        # With BLUE at 0 S, this should NOT be classified as a forced block.
        mv = _book_move(g, int(Player.RED))
        # Should NOT be (2, 0, PieceSize.SMALL) as a phantom block
        # (exact move depends on plan; just verify it's a legal move)
        assert mv is None or mv in g.get_legal_moves()
        # If BL-S is not a phantom block, the plan move (TC-L at pp=0) should be returned
        assert mv == (0, 1, PieceSize.LARGE), f"Expected plan move TC-L, got {mv} (phantom block bug?)"


# ── NeuralMCTSAgent ───────────────────────────────────────────────────────────

class TestNeuralMCTSAgent:
    """Tests for NeuralMCTSAgent v3 (DQN-guided expansion + rollout evaluation).
    Uses a randomly-initialised GPUDQNNetwork to avoid loading a checkpoint."""

    def setup_method(self):
        net = GPUDQNNetwork().to(DEVICE)
        net.eval()
        self.agent = NeuralMCTSAgent(
            policy_net      = net,
            device          = DEVICE,
            time_limit      = 0.2,
            max_simulations = 200,
        )

    def test_returns_legal_move_initial(self):
        g = TicTacPro()
        mv = self.agent.get_action(g)
        assert mv in g.get_legal_moves()

    def test_returns_legal_move_mid_game(self):
        import random
        g = TicTacPro()
        for _ in range(6):
            moves = g.get_legal_moves()
            if not moves or g.game_over:
                break
            g.make_move(*random.choice(moves))
        if not g.game_over:
            mv = self.agent.get_action(g, g.current_player)
            assert mv in g.get_legal_moves()

    def test_no_move_when_game_over(self):
        g = TicTacPro()
        # Set up a finished game
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.board[0, 2, 0] = int(Player.RED)
        g.game_over = True
        g.winner = Player.RED
        mv = self.agent.get_action(g)
        assert mv is None

    def test_reset_clears_reuse_node(self):
        g = TicTacPro()
        self.agent.get_action(g)
        assert self.agent._reuse_node is not None
        self.agent.reset()
        assert self.agent._reuse_node is None

    def test_get_info_returns_sims_and_winrate(self):
        g = TicTacPro()
        mv, (sims, wr) = self.agent.get_info(g)
        assert mv in g.get_legal_moves()
        assert sims > 0
        assert 0.0 <= wr <= 1.0

    def test_get_info_win_fastpath(self):
        """§217: get_info must return (0, 1.0) via win fast-path when immediate
        win exists — no tree search should run."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        mv, (sims, wr) = self.agent.get_info(g, Player.RED)
        assert mv == (0, 2, PieceSize.SMALL), f"win fast-path must return winning move, got {mv}"
        assert sims == 0, f"win fast-path skips tree search (sims=0), got {sims}"
        assert wr == 1.0, f"win fast-path winrate must be 1.0, got {wr}"

    def test_get_info_block_fastpath(self):
        """§217: get_info must return (0, 0.5) via block fast-path when only an
        opponent win is imminent — no tree search should run."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.BLUE)
        g.board[0, 1, 0] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        g.current_player = Player.RED
        mv, (sims, wr) = self.agent.get_info(g, Player.RED)
        assert mv == (0, 2, PieceSize.SMALL), f"block fast-path must return blocking move, got {mv}"
        assert sims == 0, f"block fast-path skips tree search (sims=0), got {sims}"
        assert wr == 0.5, f"block fast-path winrate must be 0.5, got {wr}"

    def test_get_info_bullseye_block_fastpath(self):
        """§341b: get_info must prioritize bullseye block over line block in fast-path.

        Position: BLUE has 2/3 bullseye at CC (CC-L, CC-M), and a line 2/3 (TR-S, BR-S).
        RED plays next — CC-S is the bullseye block, MR-S is the line block.
        get_info fast-path must return CC-S (bullseye block) and sims=0.
        Exercises lines 227-228 (bullseye_block assignment in get_info).
        """
        g = TicTacPro()
        g.board[1, 1, 2] = int(Player.BLUE)   # CC-L
        g.board[1, 1, 1] = int(Player.BLUE)   # CC-M → bullseye threat at CC
        g.board[0, 2, 0] = int(Player.BLUE)   # TR-S
        g.board[2, 2, 0] = int(Player.BLUE)   # BR-S → line threat on col-2-S
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 2
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 2
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  = 1
        g._pieces_left -= 4
        g.current_player = Player.RED
        mv, (sims, wr) = self.agent.get_info(g, Player.RED)
        assert mv == (1, 1, PieceSize.SMALL), (
            f"get_info fast-path must return bullseye block (1,1,S), got {mv}"
        )
        assert sims == 0
        assert wr == 0.5

    def test_sort_moves_returns_subset(self):
        """_sort_moves returns the same moves in a (possibly reordered) list."""
        g = TicTacPro()
        moves = g.get_legal_moves()
        sorted_moves, _n_winning = self.agent._sort_moves(g, moves)
        assert set(sorted_moves) == set(moves)

    def test_sort_moves_winning_at_end_and_n_winning_correct(self):
        """§202: _sort_moves must place winning moves at the END of the returned
        list (so pop() yields them first) and n_winning must equal the number of
        winning moves in the position."""
        g = TicTacPro()
        # Give RED two Smalls in row 0 — (0,2,SMALL) completes the row win.
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        moves = g.get_legal_moves()
        win_mv = (0, 2, PieceSize.SMALL)
        assert win_mv in moves, "Win move must be legal"

        sorted_moves, n_winning = self.agent._sort_moves(g, moves)

        # The winning move(s) must be the last n_winning entries.
        assert n_winning >= 1, "n_winning must be ≥1 when a win exists"
        tail = sorted_moves[-n_winning:]
        assert win_mv in tail, (
            f"Winning move {win_mv} must be in the last {n_winning} entries; "
            f"tail={tail}"
        )

    def test_sort_moves_phantom_block_guard(self):
        """§229: _sort_moves must NOT classify a cell as blocking when the opponent
        has 0 pieces of that size remaining (opp_has_sz guard at neural_mcts_agent.py:503).

        Position:
          BLUE (0,0,M)+(0,1,M), BLUE has 1 Medium remaining → real block at (0,2,M)
          BLUE (2,0,S)+(2,1,S), BLUE has 0 Smalls remaining  → phantom block at (2,2,S)

        Expected layout after _sort_moves:
          rest = [all moves except (0,2,M)]   ← (2,2,S) must land here
          blocking = [(0,2,M)]                 ← only the real block
          winning  = []

        The list is returned as rest+blocking+winning, so the last element must be
        (0,2,M). (2,2,S) must not appear last (it would if mis-classified as blocking).
        """
        g = TicTacPro()
        # Real Medium threat: BLUE occupies (0,0,M) and (0,1,M)
        g.board[0, 0, 1] = int(Player.BLUE)
        g.board[0, 1, 1] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 1  # BLUE has 1 Medium left
        # Phantom Small threat: BLUE occupies (2,0,S) and (2,1,S), 0 Smalls left
        g.board[2, 0, 0] = int(Player.BLUE)
        g.board[2, 1, 0] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 0
        g._pieces_left -= 4
        g.current_player = Player.RED

        moves = g.get_legal_moves(Player.RED)
        assert (0, 2, PieceSize.MEDIUM) in moves, "Real block must be legal"
        assert (2, 2, PieceSize.SMALL) in moves, "Phantom block cell must be legal"

        sorted_moves, n_winning = self.agent._sort_moves(g, moves)

        assert n_winning == 0, "No wins for RED in this position"
        assert len(sorted_moves) == len(moves), "All moves must be returned"
        # Real block must be the last element (only blocking move → layout: rest+[(0,2,M)])
        assert sorted_moves[-1] == (0, 2, PieceSize.MEDIUM), (
            f"§229: last element must be real block (0,2,M); got {sorted_moves[-1]}. "
            "If (2,2,S) appears last, phantom-block guard opp_has_sz failed."
        )

    def test_rollout_returns_valid_result(self):
        """_rollout returns 0.0, 0.5, or 1.0."""
        import random
        g = TicTacPro()
        for _ in range(4):
            mv = random.choice(g.get_legal_moves())
            g.make_move(*mv)
            if g.game_over:
                break
        result = self.agent._rollout(g.clone(), Player.RED)
        assert result in (0.0, 0.5, 1.0)

    def test_eval_leaf_dqn_in_range(self):
        """§154: _eval_leaf_dqn returns value in [0, 1]."""
        g = TicTacPro()
        result = self.agent._eval_leaf_dqn(g, Player.RED)
        assert 0.0 <= result <= 1.0

    def test_eval_leaf_dqn_flips_for_opponent(self):
        """§154: result for BLUE = 1 - result for RED on same position."""
        g = TicTacPro()
        r_red  = self.agent._eval_leaf_dqn(g, Player.RED)
        r_blue = self.agent._eval_leaf_dqn(g, Player.BLUE)
        assert abs(r_red + r_blue - 1.0) < 1e-6

    def test_use_dqn_eval_flag_produces_legal_move(self):
        """§154: NeuralMCTSAgent with use_dqn_eval=True still picks legal moves."""
        net = self.agent.net
        agent_dqn = NeuralMCTSAgent(
            policy_net      = net,
            device          = DEVICE,
            time_limit      = 0.1,
            max_simulations = 100,
            use_dqn_eval    = True,
        )
        g = TicTacPro()
        mv = agent_dqn.get_action(g)
        assert mv in g.get_legal_moves()

    def test_block_fast_path_fires(self):
        """§199: NeuralMCTSAgent must block an opponent's immediate win without
        entering tree search (fast path matches §143 parallel_mcts_agent guard)."""
        g = TicTacPro()
        # BLUE has two Smalls in column 2 — any SMALL at (2,2) wins for BLUE.
        g.board[0, 2, 0] = int(Player.BLUE)
        g.board[1, 2, 0] = int(Player.BLUE)
        # RED moves next; must block at (2,2,SMALL).
        mv = self.agent.get_action(g, player=Player.RED)
        assert mv == (2, 2, PieceSize.SMALL), f"Expected block at (2,2,S), got {mv}"

    def test_block_fast_path_phantom_guard(self):
        """§199/§227: Block fast-path must NOT fire when opponent has 0 pieces of
        that size — phantom guard suppresses fast-path and falls through to tree search.

        The observable signal: when the fast-path returns early it sets
        _reuse_node=None; when the tree search runs it sets _reuse_node=best_child
        (non-None).  Checking _reuse_node is not None after the call confirms the
        phantom guard worked and the tree path executed instead of the fast path.

        Note: MCTS may legitimately choose (2,2,S) as a strong positional move even
        without the phantom block — so asserting mv != (2,2,S) is not reliable.
        """
        g = TicTacPro()
        g.board[0, 2, 0] = int(Player.BLUE)
        g.board[1, 2, 0] = int(Player.BLUE)
        # Drain all BLUE Smalls so the apparent threat is phantom.
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 0
        g._pieces_left = sum(
            g.pieces_remaining[p][s]
            for p in [Player.RED, Player.BLUE]
            for s in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]
        )
        self.agent._reuse_node = None  # reset so we can detect whether tree ran
        mv = self.agent.get_action(g, player=Player.RED)
        assert mv in g.get_legal_moves(), f"Expected legal move, got {mv}"
        # Fast-path clears _reuse_node and returns early; tree search sets it.
        # If _reuse_node is not None, the tree ran → fast-path phantom guard worked.
        assert self.agent._reuse_node is not None, (
            "§199: phantom-block guard must suppress fast-path when BLUE has 0 Smalls; "
            "tree search must run (signaled by _reuse_node being set to best child)"
        )

    def test_bullseye_block_priority_over_line_block(self):
        """§337: Fast-path must prefer bullseye blocks over line blocks.

        Set up: BLUE has CC-L and CC-M (bullseye 2/3 at CC, needs CC-S to win)
        and TR-S + BR-S (column-2-Small 2/3, needs MR-S to win).
        RED moves next.  Bullseye threat (CC-S) and line threat (MR-S) are both
        immediate blocks.  §337 fix ensures bullseye block is returned first.
        """
        g = TicTacPro()
        # BLUE bullseye threat at CC: CC-L and CC-M placed.
        g.board[1, 1, 2] = int(Player.BLUE)   # CC-L
        g.board[1, 1, 1] = int(Player.BLUE)   # CC-M  → BLUE needs CC-S to bullseye
        # BLUE line threat on col-2-Small: TR-S and BR-S placed.
        g.board[0, 2, 0] = int(Player.BLUE)   # TR-S
        g.board[2, 2, 0] = int(Player.BLUE)   # BR-S  → BLUE needs MR-S for col-2-S win
        # Update piece counts to reflect 4 BLUE pieces on board.
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 2  # used 1 CC-L
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 2  # used 1 CC-M
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  = 1  # used 2 TR-S, BR-S
        g._pieces_left = sum(
            g.pieces_remaining[p][s]
            for p in [Player.RED, Player.BLUE]
            for s in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]
        )
        # RED moves next (default).
        mv = self.agent.get_action(g, player=Player.RED)
        # §337: bullseye block (CC-S) must be preferred over line block (MR-S).
        assert mv == (1, 1, PieceSize.SMALL), (
            f"§337: expected bullseye block CC-S=(1,1,S), got {mv}. "
            "Fast-path must prefer bullseye_block over line_block."
        )


# ---------------------------------------------------------------------------
# ParallelMCTSAgent — phantom block regression
# ---------------------------------------------------------------------------

class TestParallelMCTSPhantomBlock:
    """§143: ParallelMCTSAgent fast path must guard opponent block check with
    piece availability (same phantom-block class as §79/§82)."""

    @pytest.fixture(scope="class")
    def agent(self):
        from rl.parallel_mcts_agent import ParallelMCTSAgent
        return ParallelMCTSAgent(n_workers=1, time_limit=0.1, max_simulations=200)

    def test_piece_guard_prevents_phantom_block_logic(self):
        """Direct logic test: piece-guard condition must be False when opponent
        has 0 pieces, even when _would_win_bf returns True for that cell."""
        from game.tictacpro import _would_win_bf
        g = TicTacPro()
        # BLUE has 2/3 anti-diagonal Smalls but 0 Smalls left.
        g.board[0, 2, 0] = int(Player.BLUE)  # TR-S
        g.board[1, 1, 0] = int(Player.BLUE)  # CC-S
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 0
        g._pieces_left = sum(
            g.pieces_remaining[p][s]
            for p in [Player.RED, Player.BLUE]
            for s in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]
        )
        opp_pr = g.pieces_remaining[Player.BLUE]
        bf = g.board.tobytes()
        # _would_win_bf alone returns True (the bug: doesn't check piece count)
        assert _would_win_bf(bf, 2, 0, 0, int(Player.BLUE)), "Line-of-two must exist"
        # With the §143 piece guard, the block should NOT be triggered
        block_triggered = opp_pr[PieceSize.SMALL] > 0 and _would_win_bf(bf, 2, 0, 0, int(Player.BLUE))
        assert not block_triggered, "§143 piece guard must prevent phantom block when opp has 0 Smalls"

    def test_returns_legal_move_when_phantom_would_have_fired(self, agent):
        """Full agent call: returns a legal move even when the unguarded code
        would have returned a phantom block."""
        g = TicTacPro()
        g.board[0, 2, 0] = int(Player.BLUE)  # TR-S
        g.board[1, 1, 0] = int(Player.BLUE)  # CC-S
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] = 0
        g._pieces_left = sum(
            g.pieces_remaining[p][s]
            for p in [Player.RED, Player.BLUE]
            for s in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]
        )
        mv = agent.get_action(g, player=Player.RED)
        assert mv in g.get_legal_moves(), f"Expected legal move, got {mv}"


# ── §211 ParallelMCTSAgent basic functional tests ─────────────────────────────

class TestParallelMCTSBasic:
    """§211: ParallelMCTSAgent basic functionality — returns legal moves, takes
    immediate wins, blocks opponent wins. Uses n_workers=1 to avoid test flakiness
    from spawning many processes."""

    @pytest.fixture(scope="class")
    def agent(self):
        from rl.parallel_mcts_agent import ParallelMCTSAgent
        return ParallelMCTSAgent(n_workers=1, time_limit=0.3, max_simulations=300)

    def test_returns_legal_move_on_empty_board(self, agent):
        g = TicTacPro()
        mv = agent.get_action(g, player=Player.RED)
        assert mv in g.get_legal_moves(), f"Expected legal move on empty board, got {mv}"

    def test_returns_none_on_game_over(self, agent):
        g = TicTacPro()
        g.game_over = True
        assert agent.get_action(g) is None

    def test_takes_immediate_win(self, agent):
        """With two Smalls in a row, must take the winning third."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        mv = agent.get_action(g, player=Player.RED)
        assert mv == (0, 2, PieceSize.SMALL), f"Expected win at (0,2,S), got {mv}"

    def test_blocks_opponent_win(self, agent):
        """With two opponent Smalls in a row, must block the third."""
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.BLUE)
        g.board[0, 1, 0] = int(Player.BLUE)
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        g.current_player = Player.RED
        mv = agent.get_action(g, player=Player.RED)
        assert mv == (0, 2, PieceSize.SMALL), f"Expected block at (0,2,S), got {mv}"

    def test_get_info_returns_tuple(self, agent):
        g = TicTacPro()
        mv, info = agent.get_info(g, player=Player.RED)
        assert mv in g.get_legal_moves()
        sims, wr = info
        assert sims >= 0
        assert 0.0 <= wr <= 1.0

    def test_get_info_win_fastpath_resets_sims(self, agent):
        """§218: get_info must return sims=0 when win fast-path fires, even when
        a prior normal call populated _last_sims with a non-zero value."""
        # First call on empty board populates _last_sims
        g_normal = TicTacPro()
        agent.get_info(g_normal, player=Player.RED)
        assert agent._last_sims > 0, "prerequisite: normal call must set _last_sims>0"
        # Now fast-path position
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        mv, (sims, _wr) = agent.get_info(g, player=Player.RED)
        assert mv == (0, 2, PieceSize.SMALL), f"Expected winning move, got {mv}"
        assert sims == 0, (
            f"§218: _last_sims must be reset to 0 on win fast-path, got {sims}"
        )


# ── §325 ParallelMCTSAgent coverage: uncovered lines 37-47, 80-81, 85, 94, 137-138 ─────

class TestParallelMCTSCoverage:
    """§325: Cover remaining uncovered lines in parallel_mcts_agent.py."""

    def test_worker_run_directly_with_seed(self):
        """Lines 37-47: _worker_run called directly in-process with seed set."""
        from rl.parallel_mcts_agent import _worker_run
        g = TicTacPro()
        result = _worker_run((g, int(Player.RED), 0.1, 50, 42))
        assert isinstance(result, dict)
        for move, count in result.items():
            assert count >= 0

    def test_worker_run_directly_no_seed(self):
        """Lines 37-47: _worker_run with seed=None skips random.seed call."""
        from rl.parallel_mcts_agent import _worker_run
        g = TicTacPro()
        result = _worker_run((g, int(Player.BLUE), 0.1, 30, None))
        assert isinstance(result, dict)

    def test_del_terminates_pool(self):
        """Lines 80-81: __del__ exception handler fires when _pool is missing."""
        from rl.parallel_mcts_agent import ParallelMCTSAgent
        agent = ParallelMCTSAgent(n_workers=1, time_limit=0.1, max_simulations=10)
        pool = agent._pool       # keep pool alive so it gets cleaned up
        del agent._pool          # remove attr → __del__ raises AttributeError
        agent.__del__()          # except branch (lines 80-81) fires, swallowed
        pool.terminate()         # clean up pool separately

    def test_reset_is_noop(self):
        """Line 85: reset() is a no-op for API compatibility."""
        from rl.parallel_mcts_agent import ParallelMCTSAgent
        agent = ParallelMCTSAgent(n_workers=1, time_limit=0.1, max_simulations=10)
        agent.reset()
        assert agent._last_sims == 0
        agent._pool.terminate()

    def test_get_action_default_player(self):
        """Line 94: get_action without player arg reads current_player."""
        from rl.parallel_mcts_agent import ParallelMCTSAgent
        agent = ParallelMCTSAgent(n_workers=1, time_limit=0.1, max_simulations=50)
        g = TicTacPro()
        g.current_player = Player.BLUE
        mv = agent.get_action(g)  # no player arg → uses current_player (BLUE)
        assert mv in g.get_legal_moves(Player.BLUE)
        agent._pool.terminate()

    def test_empty_total_falls_back_to_legal0(self):
        """Lines 137-138: workers return empty dicts when max_simulations=0;
        fall back to legal[0]."""
        from rl.parallel_mcts_agent import ParallelMCTSAgent
        agent = ParallelMCTSAgent(n_workers=1, time_limit=100.0, max_simulations=0)
        g = TicTacPro()
        mv = agent.get_action(g, Player.RED)
        assert mv in g.get_legal_moves()
        agent._pool.terminate()


# ── §346 ParallelMCTSAgent bullseye block priority ───────────────────────────

class TestParallelMCTSBullseyeBlockPriority:
    """§346: ParallelMCTSAgent fast-path must return bullseye block over line block.

    Pre-fix (single block_mv): get_legal_moves() returns Small moves before Large,
    so a line block at (0,2,S) is saved as block_mv before the bullseye block at
    (1,1,L) is reached — bullseye block ignored.

    Fix (§333/§346): dual bull_block_mv / line_block_mv tracking; bullseye wins.
    Board: BLUE S+M at (1,1) [bullseye threat with L] + BLUE S at (0,0)+(0,1)
    [line threat at (0,2,S)]. RED to move. Must return (1,1,L) not (0,2,S)."""

    @pytest.fixture(scope="class")
    def agent(self):
        from rl.parallel_mcts_agent import ParallelMCTSAgent
        return ParallelMCTSAgent(n_workers=1, time_limit=0.1, max_simulations=50)

    def _make_bullseye_plus_line_threat_board(self) -> "TicTacPro":
        """BLUE has M+L at (1,1) [needs S there → bullseye] and S at (0,0)+(0,1)
        [needs S at (0,2) → line]. BLUE has 1 Small remaining — can play either threat.
        RED must block the bullseye (higher priority) at (1,1,S), NOT the line at (0,2,S).
        Old single-block_mv code returns (0,2,S) first (Small scan before Large/Medium,
        ci=2 < ci=4). New §346 dual-track returns (1,1,S) — correct bullseye priority."""
        g = TicTacPro()
        B = int(Player.BLUE)
        g.board[1, 1, 1] = B   # CC-M  (bullseye cell, Medium)
        g.board[1, 1, 2] = B   # CC-L  (bullseye cell, Large) → BLUE needs (1,1,S) to win bullseye
        g.board[0, 0, 0] = B   # TL-S  (line setup)
        g.board[0, 1, 0] = B   # TM-S  → BLUE needs (0,2,S) to win top-row Small line
        # BLUE pieces used: 2 Small, 1 Medium, 1 Large → 1 Small remains → can threaten line
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  -= 2
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] -= 1
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  -= 1
        g._pieces_left = sum(
            g.pieces_remaining[p][s]
            for p in [Player.RED, Player.BLUE]
            for s in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]
        )
        g.current_player = Player.RED
        return g

    def test_bullseye_block_returned_over_line_block(self, agent):
        """§346: fast-path must return (1,1,S) bullseye block, not (0,2,S) line block.

        Both (0,2,S) and (1,1,S) block an opponent win. Small scan comes first and
        (0,2,S) has ci=2 < ci=4 for (1,1,S), so the old single-variable code would
        set block_mv=(0,2,S) and never check (1,1,S).  The §346 dual-track fix must
        detect (1,1,S) as a bullseye block and return it with higher priority."""
        g = self._make_bullseye_plus_line_threat_board()
        mv = agent.get_action(g, Player.RED)
        assert mv == (1, 1, PieceSize.SMALL), (
            f"§346: bullseye block (1,1,S) must take priority over line block (0,2,S); got {mv}"
        )


# ── §181 NeuralMCTS illegal-action masking ────────────────────────────────────

class TestNeuralMCTSLegalMasking:
    """§181: _eval_leaf_dqn must take max Q over legal actions only.

    Illegal slots carry untrained Q-values that can exceed legal-move Q-values.
    The fix computes the action index for each legal move (r*9+c*3+sz-1) and
    takes max over only those indices.
    """

    def test_eval_leaf_dqn_ignores_illegal_high_q(self):
        """Patch Q-values so an illegal slot has a huge value; verify the
        returned win probability reflects the best LEGAL Q, not the illegal one."""
        import numpy as np

        net = GPUDQNNetwork(state_size=61, action_size=27, init_weights=False)
        agent = NeuralMCTSAgent(
            policy_net=net, device=DEVICE, time_limit=0.1, use_dqn_eval=True
        )

        # Make a partial game: three moves placed
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)   # RED occupies (0,0,S)
        g.make_move(1, 1, PieceSize.SMALL)   # BLUE occupies (1,1,S)
        g.make_move(0, 1, PieceSize.MEDIUM)  # RED occupies (0,1,M)

        legal = g.get_legal_moves()
        all_idx = set(range(27))
        legal_idx = {r * 9 + c * 3 + (sz - 1) for r, c, sz in legal}
        illegal_idx = all_idx - legal_idx
        assert illegal_idx, "Need at least one illegal action for this test"

        # Manually override net output so one illegal slot has a very large Q
        # and all legal slots have small Q-values near 0.
        illegal_slot = next(iter(illegal_idx))

        def patched_forward(x):
            out = torch.zeros(x.shape[0], 27)
            out[:, illegal_slot] = 1000.0   # illegal slot — huge value
            out[:, list(legal_idx)[0]] = 0.1  # best legal slot — small value
            return out

        agent.net.forward = patched_forward  # type: ignore[assignment]

        # Call _eval_leaf_dqn directly
        result = agent._eval_leaf_dqn(g, Player.RED)

        # With the §181 fix, max_q should be 0.1 (best legal), not 1000.0 (illegal).
        # raw_p = (0.1 + 1) / 2 = 0.55. Without masking it would be (1000+1)/2 → 1.0.
        assert result < 0.9, (
            f"§181: illegal Q=1000 must be masked; expected result < 0.9, got {result:.4f}. "
            "This indicates the pre-§181 bug (unmasked max) is still present."
        )

    def test_eval_leaf_dqn_returns_legal_best(self):
        """§207: _eval_leaf_dqn uses softmax-weighted average Q over legal moves.

        With one legal move at Q=0.8 and all others at Q=0.0:
        - Old §154 (max Q): raw_p = (0.8+1)/2 = 0.9  → overconfident win signal
        - New §207 (softmax avg): result is in (0.5, 0.9) — less overconfident
        The key invariant is that the result is > 0.5 (positive Q signal reflected)
        but < 0.9 (not as overconfident as max Q).
        """
        net = GPUDQNNetwork(state_size=61, action_size=27, init_weights=False)
        agent = NeuralMCTSAgent(
            policy_net=net, device=DEVICE, time_limit=0.1, use_dqn_eval=True
        )
        # Fresh board: current_player is RED — matches root_player so no flip.
        g = TicTacPro()

        legal = g.get_legal_moves()
        assert legal, "Need legal moves for this test"

        best_legal = legal[-1]
        best_idx = best_legal[0] * 9 + best_legal[1] * 3 + (best_legal[2] - 1)

        def patched_forward(x):
            out = torch.zeros(x.shape[0], 27)
            out[:, best_idx] = 0.8
            return out

        agent.net.forward = patched_forward  # type: ignore[assignment]

        result = agent._eval_leaf_dqn(g, Player.RED)
        # §207: softmax-weighted avg dilutes the 0.8 signal across all legal moves,
        # giving a result between 0.5 (neutral) and 0.9 (old max-Q overconfident).
        assert 0.5 < result < 0.9, (
            f"§207 softmax avg: expected 0.5 < result < 0.9 (got {result:.4f}). "
            "Result > 0.9 indicates old max-Q overconfidence bug; <= 0.5 means Q signal lost."
        )


# ── §324: NeuralMCTSAgent coverage gaps ──────────────────────────────────────

def _make_neural_agent(**kwargs) -> NeuralMCTSAgent:
    net = GPUDQNNetwork().to(DEVICE)
    net.eval()
    defaults = dict(policy_net=net, device=DEVICE, time_limit=0.2, max_simulations=200)
    defaults.update(kwargs)
    return NeuralMCTSAgent(**defaults)


class TestNeuralMCTSNodeUCTSelect:
    """§324a: NeuralMCTSNode.uct_select() — Python fallback (lines 93-100) and
    numpy path (line 92)."""

    def test_uct_select_python_fallback(self):
        from rl.neural_mcts_agent import NeuralMCTSNode, _LAZY
        node = NeuralMCTSNode(None, None, [], None)
        node.visits = 4
        child_a = NeuralMCTSNode((0, 0, 1), node, _LAZY, 1, False, False, -1, 0, 1)
        child_b = NeuralMCTSNode((0, 1, 1), node, _LAZY, 1, False, False, -1, 1, 1)
        child_a.winrate = 0.9;  child_a.uct_scale = 0.01
        child_b.winrate = 0.0;  child_b.uct_scale = 1.0
        node.children = [child_a, child_b]
        result = node.uct_select()
        assert result is child_b, "Python UCT fallback must pick child_b (higher exploration)"

    def test_uct_select_numpy_path(self):
        import numpy as np
        from rl.neural_mcts_agent import NeuralMCTSNode, _LAZY
        node = NeuralMCTSNode(None, None, [], None)
        node.visits = 4
        child_a = NeuralMCTSNode((0, 0, 1), node, _LAZY, 1, False, False, -1, 0, 1)
        child_b = NeuralMCTSNode((0, 1, 1), node, _LAZY, 1, False, False, -1, 1, 1)
        child_a.winrate = 0.9;  child_a.uct_scale = 0.01
        child_b.winrate = 0.0;  child_b.uct_scale = 1.0
        node.children = [child_a, child_b]
        node.ch_n  = 2
        node.ch_wr = np.array([0.9, 0.0])
        node.ch_us = np.array([0.01, 1.0])
        result = node.uct_select()
        assert result is child_b, "Numpy UCT must pick child_b (higher exploration)"


class TestNeuralMCTSAPIEdges:
    """§324b: get_action / get_info edge paths."""

    def test_get_info_game_over_early_return(self):
        """Line 195: get_info must return (None,(0,0.0)) when game is already over."""
        agent = _make_neural_agent()
        g = TicTacPro()
        g.game_over = True; g.winner = Player.RED
        mv, info = agent.get_info(g, Player.RED)
        assert mv is None
        assert info == (0, 0.0)

    def test_get_action_no_children_fallback(self):
        """Lines 186-188: max_simulations=0 → root has no children → legal[0] returned."""
        agent = _make_neural_agent(max_simulations=0)
        g = TicTacPro()
        mv = agent.get_action(g, Player.RED)
        assert mv in g.get_legal_moves(Player.RED)

    def test_get_info_no_children_fallback(self):
        """Lines 220-221: max_simulations=0 → root has no children → legal[0] returned."""
        agent = _make_neural_agent(max_simulations=0)
        g = TicTacPro()
        mv, (sims, wr) = agent.get_info(g, Player.RED)
        assert mv in g.get_legal_moves(Player.RED)
        assert (sims, wr) == (0, 0.0)

    def test_get_action_default_player(self):
        """Line 162: calling get_action without explicit player uses current_player."""
        agent = _make_neural_agent()
        g = TicTacPro()
        g.make_move(0, 0, PieceSize.SMALL)   # RED moves; now BLUE's turn
        mv = agent.get_action(g)              # no player arg
        assert mv in g.get_legal_moves(Player.BLUE)

    def test_get_action_immediate_win_pre_screen(self):
        """Lines 174-175: get_action returns immediately when a winning move is found."""
        agent = _make_neural_agent()
        g = TicTacPro()
        # Place 2 RED Smalls in a row → third Small completes row 0 win.
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        g.current_player = Player.RED
        mv = agent.get_action(g, Player.RED)
        # The winning move must complete the row.
        assert mv is not None
        r, c, sz = mv
        assert sz == PieceSize.SMALL

    def test_timeout_break_in_build_tree(self, monkeypatch):
        """Line 277: deadline expiry during main sim loop triggers break."""
        import time
        call_count = [0]
        def _mock_clock():
            call_count[0] += 1
            return 0.0 if call_count[0] == 1 else 1e9
        monkeypatch.setattr(time, 'perf_counter', _mock_clock)
        agent = _make_neural_agent(time_limit=100.0, max_simulations=1000)
        g = TicTacPro()
        mv = agent.get_action(g, Player.RED)
        assert mv in g.get_legal_moves(Player.RED)
        assert agent._last_sims < 1000, "Timeout must stop sims early"


class TestNeuralMCTSTreeReuse:
    """§324c: tree reuse path in _build_tree (lines 231-236)."""

    def test_tree_reuse_executes_loop(self):
        """Lines 231-236: _build_tree loops over _reuse_node.children when
        _reuse_node is set and g has move_history."""
        agent = _make_neural_agent(time_limit=0.3, max_simulations=300)
        g = TicTacPro()

        # Call 1: agent builds tree for RED; sets _reuse_node = best RED child
        red_move = agent.get_action(g, Player.RED)
        reuse_node = agent._reuse_node
        g.make_move(*red_move)   # RED plays; now BLUE's turn

        # Pick a BLUE move that was explored in the first call (reuse_node.children)
        blue_move = None
        if reuse_node is not None and reuse_node.children:
            blue_move = reuse_node.children[0].move
        if blue_move is None:
            blue_move = g.get_legal_moves()[0]

        g.make_move(*blue_move)  # BLUE plays; now RED's turn again

        # Call 2: _reuse_node.children should contain a child matching blue_move
        mv = agent.get_action(g, Player.RED)
        assert mv in g.get_legal_moves(Player.RED)


class TestNeuralMCTSSimLoopPaths:
    """§324d: simulation loop edge paths — win/draw in selection and expansion,
    deep-node C sort, lazy expansion game_over short-circuit."""

    def test_win_in_fast_path_expansion_via_build_tree(self):
        """Line 345: _build_tree called directly (bypassing get_action pre-screening).
        _sort_moves puts winning move at end → pop() returns it → is_win=True → line 345."""
        agent = _make_neural_agent(max_simulations=5)
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        g.current_player = Player.RED
        g_clone = g.clone()
        root = agent._build_tree(g_clone, Player.RED)
        assert root is not None
        assert any(c.move_is_win for c in root.children), "Winning child must be created"

    def test_win_in_selection_via_build_tree(self):
        """Line 298: after root is fully expanded, selection walks to winning child.
        Must call _build_tree directly; get_action pre-screens wins before tree build."""
        agent = _make_neural_agent(max_simulations=50)
        g = TicTacPro()
        # RED has 2S in row 0 → (0,2,S) win; all other moves non-winning.
        # With 25 total legal RED moves at this position: 50 sims fully expands root (25)
        # then 25 more sims select children; the winning child is selected at some point.
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        g.current_player = Player.RED
        g_clone = g.clone()
        root = agent._build_tree(g_clone, Player.RED)
        assert root is not None

    def test_draw_in_selection(self):
        """Line 300: selection walks to a draw-terminal child (_pcs_left→0, no win)."""
        # Same 17-piece near-draw board as §319 but for NeuralMCTSAgent.
        g = TicTacPro()
        R, B = int(Player.RED), int(Player.BLUE)
        g.board[0, 0, 0] = R; g.board[1, 2, 0] = R; g.board[2, 1, 0] = R
        g.board[0, 1, 1] = R; g.board[1, 0, 1] = R; g.board[2, 2, 1] = R
        g.board[0, 2, 2] = R; g.board[1, 1, 2] = R; g.board[2, 1, 2] = R
        g.board[0, 1, 0] = B; g.board[1, 0, 0] = B; g.board[2, 0, 0] = B
        g.board[0, 0, 1] = B; g.board[1, 2, 1] = B; g.board[2, 1, 1] = B
        g.board[0, 1, 2] = B; g.board[1, 2, 2] = B
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.RED][PieceSize.LARGE]  = 0
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 1
        g._pieces_left = 1; g.current_player = Player.BLUE
        agent = _make_neural_agent(max_simulations=10)
        mv = agent.get_action(g, Player.BLUE)
        assert mv in g.get_legal_moves(Player.BLUE)

    def test_draw_in_fast_path_expansion(self):
        """Line 347: _pcs_left→0 with no win in _nnw>=0 expansion sets _game_over."""
        # Same 16-piece board as §320 but for NeuralMCTSAgent.
        g = TicTacPro()
        R, B = int(Player.RED), int(Player.BLUE)
        g.board[0, 0, 0] = R; g.board[1, 2, 0] = R
        g.board[0, 2, 1] = R; g.board[1, 0, 1] = R; g.board[2, 1, 1] = R
        g.board[0, 1, 2] = R; g.board[1, 2, 2] = R; g.board[2, 0, 2] = R
        g.board[0, 1, 0] = B; g.board[2, 2, 0] = B
        g.board[0, 0, 1] = B; g.board[1, 1, 1] = B; g.board[2, 0, 1] = B
        g.board[0, 2, 2] = B; g.board[1, 0, 2] = B; g.board[2, 1, 2] = B
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  = 1
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.RED][PieceSize.LARGE]  = 0
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  = 1
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 0
        g._pieces_left = 2; g.current_player = Player.RED
        agent = _make_neural_agent(max_simulations=10)
        mv = agent.get_action(g, Player.RED)
        assert mv in g.get_legal_moves(Player.RED)

    def test_game_over_clears_lazy_untried_via_build_tree(self):
        """Line 316: _game_over=True during lazy expansion → node.untried_moves=[].
        Called via _build_tree directly so win children ARE created; selection
        revisits them in later sims → _game_over=True → line 316 fires."""
        agent = _make_neural_agent(max_simulations=50)
        g = TicTacPro()
        g.board[0, 0, 0] = int(Player.RED)
        g.board[0, 1, 0] = int(Player.RED)
        g.pieces_remaining[Player.RED][PieceSize.SMALL] -= 2
        g._pieces_left -= 2
        g.current_player = Player.RED
        g_clone = g.clone()
        root = agent._build_tree(g_clone, Player.RED)
        assert root is not None

    def test_deep_node_c_sort(self):
        """Lines 319-322: deep nodes (_toks_d > max_dqn_depth) use C sort instead
        of DQN. With max_dqn_depth=0, ALL children use C sort (depth 1 > 0)."""
        agent = _make_neural_agent(max_simulations=50, max_dqn_depth=0)
        g = TicTacPro()
        mv = agent.get_action(g, Player.RED)
        assert mv in g.get_legal_moves(Player.RED)


class TestNeuralMCTSEvalPaths:
    """§324e: leaf evaluation paths — DQN eval, Python rollout, exceptions."""

    def test_dqn_eval_path(self):
        """Lines 380-381: use_dqn_eval=True routes evaluation through _eval_leaf_dqn."""
        agent = _make_neural_agent(use_dqn_eval=True, max_simulations=10)
        g = TicTacPro()
        mv = agent.get_action(g, Player.RED)
        assert mv in g.get_legal_moves(Player.RED)

    def test_python_rollout_fallback(self, monkeypatch):
        """Line 390: C_ROLLOUT_AVAILABLE=False and use_dqn_eval=False → _rollout called."""
        import rl.neural_mcts_agent as _nma
        monkeypatch.setattr(_nma, 'C_ROLLOUT_AVAILABLE', False)
        agent = _make_neural_agent(max_simulations=5)
        g = TicTacPro()
        mv = agent.get_action(g, Player.RED)
        assert mv in g.get_legal_moves(Player.RED)

    def test_eval_leaf_dqn_exception_returns_05(self):
        """Lines 474-475: exception inside _eval_leaf_dqn is caught and returns 0.5."""
        agent = _make_neural_agent(use_dqn_eval=True)
        g = TicTacPro()
        # Make net.forward raise unconditionally
        def _boom(x): raise RuntimeError("injected test error")
        agent.net.forward = _boom
        result = agent._eval_leaf_dqn(g, Player.RED)
        assert result == 0.5, f"Exception path must return 0.5; got {result}"

    def test_eval_leaf_dqn_no_legal_moves_uses_mean(self):
        """Line 471: empty legal moves → uses q_vals.mean() instead of masked softmax."""
        agent = _make_neural_agent(use_dqn_eval=True)
        g = TicTacPro()
        # Manufacture a stuck player: no legal moves but game not over
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.RED][PieceSize.LARGE]  = 0
        g.current_player = Player.RED
        result = agent._eval_leaf_dqn(g, Player.RED)
        assert 0.0 <= result <= 1.0, f"No-legal-moves path must return valid prob; got {result}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP requires CUDA")
    def test_eval_leaf_dqn_amp_path(self):
        """Line 459: use_amp=True on CUDA triggers torch.amp.autocast path."""
        agent = _make_neural_agent(use_dqn_eval=True, use_amp=True)
        g = TicTacPro()
        result = agent._eval_leaf_dqn(g, Player.RED)
        assert 0.0 <= result <= 1.0

    def test_eval_leaf_dqn_non_amp_path(self):
        """Line 459 (else branch): use_amp=False bypasses torch.amp.autocast.
        On CUDA the default is use_amp=True, so must explicitly set False."""
        agent = _make_neural_agent(use_dqn_eval=True, use_amp=False)
        g = TicTacPro()
        result = agent._eval_leaf_dqn(g, Player.RED)
        assert 0.0 <= result <= 1.0


class TestNeuralMCTSSortMoves:
    """§324f: _sort_moves edge paths — single move, exception, AMP."""

    def test_sort_moves_single_move_early_return(self):
        """Line 486: _sort_moves with len(moves)<=1 returns (list(moves), 0) directly."""
        agent = _make_neural_agent()
        g = TicTacPro()
        single_move = [(0, 0, PieceSize.SMALL)]
        result, n_win = agent._sort_moves(g, single_move)
        assert result == single_move
        assert n_win == 0

    def test_sort_moves_exception_keeps_original_order(self):
        """Lines 538-539: exception in DQN inference → rest kept in original order.
        _sort_moves calls self.net.features(sv). Replace features with a raising
        nn.Module (direct function assignment is blocked by torch.nn.Module.__setattr__)."""
        agent = _make_neural_agent()
        g = TicTacPro()
        moves = g.get_legal_moves()

        class _RaiserFeatures(torch.nn.Module):
            def forward(self, x): raise RuntimeError("injected test error")

        agent.net.features = _RaiserFeatures()  # valid: it IS an nn.Module

        sorted_moves, n_win = agent._sort_moves(g, moves)
        assert set(sorted_moves) == set(moves), "All moves must be returned even after exception"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP requires CUDA")
    def test_sort_moves_amp_path(self):
        """AMP path: use_amp=True on CUDA triggers torch.amp.autocast in _sort_moves."""
        agent = _make_neural_agent(use_amp=True)
        g = TicTacPro()
        moves = g.get_legal_moves()
        sorted_moves, n_win = agent._sort_moves(g, moves)
        assert set(sorted_moves) == set(moves)

    def test_sort_moves_non_amp_path(self):
        """Lines 532-534 (else branch): use_amp=False bypasses torch.amp.autocast in _sort_moves."""
        agent = _make_neural_agent(use_amp=False)
        g = TicTacPro()
        moves = g.get_legal_moves()
        sorted_moves, n_win = agent._sort_moves(g, moves)
        assert set(sorted_moves) == set(moves)

    def test_sort_moves_bullseye_block_before_line_block(self):
        """§341: _sort_moves places bullseye block after line block (pop() → bullseye first)."""
        agent = _make_neural_agent(use_amp=False)
        g = TicTacPro()
        # BLUE has bullseye 2/3 at CC (CC-L, CC-M) and line 2/3 (TR-S, BR-S col-2-S).
        g.board[1, 1, 2] = int(Player.BLUE)   # CC-L
        g.board[1, 1, 1] = int(Player.BLUE)   # CC-M
        g.board[0, 2, 0] = int(Player.BLUE)   # TR-S
        g.board[2, 2, 0] = int(Player.BLUE)   # BR-S
        g.pieces_remaining[Player.BLUE][PieceSize.LARGE]  = 2
        g.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] = 2
        g.pieces_remaining[Player.BLUE][PieceSize.SMALL]  = 1
        g._pieces_left = sum(g.pieces_remaining[p][s]
                             for p in [Player.RED, Player.BLUE]
                             for s in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE])
        # RED moves; all legal RED moves form the candidate set.
        moves = g.get_legal_moves(Player.RED)
        sorted_moves, n_win = agent._sort_moves(g, moves)
        # Highest-priority block is at the end (pop() target).
        # Bullseye block CC-S = (1,1,S) must come AFTER line block MR-S = (1,2,S).
        bull_idx = sorted_moves.index((1, 1, PieceSize.SMALL))
        line_idx = sorted_moves.index((1, 2, PieceSize.SMALL))
        assert bull_idx > line_idx, (
            f"§341: bullseye block CC-S must be at higher index than line block MR-S "
            f"(got bull_idx={bull_idx}, line_idx={line_idx})"
        )


class TestNeuralMCTSRollout:
    """§324g: _rollout edge paths — stuck player (mv=None break, line 565)."""

    def test_rollout_stuck_player_mv_none_break(self):
        """Line 565: _rollout breaks when mv=None (player has no legal moves)."""
        agent = _make_neural_agent()
        g = TicTacPro()
        # Manufacture stuck state: all RED pieces exhausted
        g.pieces_remaining[Player.RED][PieceSize.SMALL]  = 0
        g.pieces_remaining[Player.RED][PieceSize.MEDIUM] = 0
        g.pieces_remaining[Player.RED][PieceSize.LARGE]  = 0
        g.current_player = Player.RED
        # _rollout clones g, sets game_over=False, runs loop → mv=None → break → draw
        result = agent._rollout(g.clone(), Player.RED)
        assert result in (0.0, 0.5, 1.0)
