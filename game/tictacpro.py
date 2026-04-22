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
from copy import deepcopy


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
        Get all legal moves for the current player
        Returns: List of (row, col, piece_size) tuples
        """
        if player is None:
            player = self.current_player

        if self.game_over:
            return []

        legal_moves = []

        for size in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]:
            # Check if player has pieces of this size remaining
            if self.pieces_remaining[player][size] == 0:
                continue

            for row in range(self.BOARD_SIZE):
                for col in range(self.BOARD_SIZE):
                    if self._is_valid_move(row, col, size):
                        legal_moves.append((row, col, size))

        return legal_moves

    def _is_valid_move(self, row: int, col: int, size: PieceSize) -> bool:
        """Check if placing a piece at (row, col) of given size is valid"""
        size_idx = size - 1  # Convert to 0-indexed

        # Check if this size slot is empty
        if self.board[row, col, size_idx] != Player.NONE:
            return False

        # Small pieces can only be placed in empty cells
        if size == PieceSize.SMALL:
            return True

        # Medium/Large pieces require smaller pieces below (or can go on empty)
        # Actually, in Tic Tac Pro, any size can go on any empty cell
        # But larger pieces cover smaller ones
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
        self.board[row, col, size_idx] = player
        self.pieces_remaining[player][size] -= 1
        self.move_history.append((row, col, size, player))

        # Check for win
        if self._check_win(player):
            self.winner = player
            self.game_over = True
        # Check for draw (no legal moves for either player)
        elif not self.get_legal_moves(Player.RED) and not self.get_legal_moves(Player.BLUE):
            self.game_over = True
        else:
            # Switch players
            self.current_player = Player.BLUE if player == Player.RED else Player.RED

        return True

    def _check_win(self, player: Player) -> bool:
        """Check if the player has won"""
        # Check all three win conditions for each size
        for size in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]:
            if self._check_size_win(player, size):
                return True

        # Check bullseye wins (all three sizes in one cell)
        if self._check_bullseye_win(player):
            return True

        return False

    def _check_size_win(self, player: Player, size: PieceSize) -> bool:
        """Check if player has three in a row of the given size"""
        size_idx = size - 1

        # Check rows
        for row in range(self.BOARD_SIZE):
            if all(self.board[row, col, size_idx] == player for col in range(self.BOARD_SIZE)):
                return True

        # Check columns
        for col in range(self.BOARD_SIZE):
            if all(self.board[row, col, size_idx] == player for row in range(self.BOARD_SIZE)):
                return True

        # Check diagonals
        if all(self.board[i, i, size_idx] == player for i in range(self.BOARD_SIZE)):
            return True

        if all(self.board[i, self.BOARD_SIZE - 1 - i, size_idx] == player for i in range(self.BOARD_SIZE)):
            return True

        return False

    def _check_bullseye_win(self, player: Player) -> bool:
        """Check if player has a bullseye (all three sizes in one cell)"""
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                # Check if all three sizes in this cell belong to the player
                if all(self.board[row, col, size] == player for size in range(self.NUM_PIECE_SIZES)):
                    return True
        return False

    def get_state_tensor(self) -> np.ndarray:
        """
        Get the game state as a tensor for neural network input
        Returns: Flattened state representation
        Shape: (3*3*3*2 + 6 + 1) = 61 features
        - 54 features for board (3x3x3 for each player)
        - 6 features for remaining pieces (3 sizes * 2 players)
        - 1 feature for current player
        """
        # Board state for each player
        red_board = (self.board == Player.RED).astype(np.float32)
        blue_board = (self.board == Player.BLUE).astype(np.float32)

        # Flatten boards
        board_features = np.concatenate([red_board.flatten(), blue_board.flatten()])

        # Remaining pieces
        piece_features = np.array([
            self.pieces_remaining[Player.RED][PieceSize.SMALL] / self.PIECES_PER_SIZE,
            self.pieces_remaining[Player.RED][PieceSize.MEDIUM] / self.PIECES_PER_SIZE,
            self.pieces_remaining[Player.RED][PieceSize.LARGE] / self.PIECES_PER_SIZE,
            self.pieces_remaining[Player.BLUE][PieceSize.SMALL] / self.PIECES_PER_SIZE,
            self.pieces_remaining[Player.BLUE][PieceSize.MEDIUM] / self.PIECES_PER_SIZE,
            self.pieces_remaining[Player.BLUE][PieceSize.LARGE] / self.PIECES_PER_SIZE,
        ], dtype=np.float32)

        # Current player
        current_player_feature = np.array([1.0 if self.current_player == Player.RED else 0.0], dtype=np.float32)

        return np.concatenate([board_features, piece_features, current_player_feature])

    def clone(self):
        """Create a deep copy of the game state"""
        new_game = TicTacPro()
        new_game.board = self.board.copy()
        new_game.pieces_remaining = deepcopy(self.pieces_remaining)
        new_game.current_player = self.current_player
        new_game.winner = self.winner
        new_game.game_over = self.game_over
        new_game.move_history = self.move_history.copy()
        return new_game

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


if __name__ == "__main__":
    # Quick test
    game = TicTacPro()
    print(game)
    print(f"\nLegal moves: {len(game.get_legal_moves())}")

    # Test some moves
    game.make_move(0, 0, PieceSize.SMALL)
    game.make_move(1, 1, PieceSize.SMALL)
    game.make_move(0, 1, PieceSize.SMALL)
    print("\n" + str(game))
