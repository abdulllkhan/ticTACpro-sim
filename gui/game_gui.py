"""
Pygame GUI for Tic Tac Pro

Interactive visual interface to play against the trained AI agent
with move suggestions and analysis.
"""

import pygame
import sys
import numpy as np
from typing import Optional, Tuple, List
import os

from game.tictacpro import TicTacPro, Player, PieceSize
from rl.agent import DQNAgent


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
RED = (220, 50, 50)
DARK_RED = (180, 30, 30)
BLUE = (50, 100, 220)
DARK_BLUE = (30, 70, 180)
GREEN = (50, 200, 50)
YELLOW = (255, 215, 0)
GOLD = (255, 200, 0)

# Board settings
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
BOARD_SIZE = 500
BOARD_MARGIN = 50
CELL_SIZE = BOARD_SIZE // 3

# Piece sizes (radius)
PIECE_SMALL = 20
PIECE_MEDIUM = 35
PIECE_LARGE = 50


class TicTacProGUI:
    """
    Visual interface for Tic Tac Pro with AI opponent

    Features:
    - Beautiful board visualization
    - Drag-and-drop or click to play
    - AI opponent with adjustable difficulty
    - Move suggestions overlay
    - Position evaluation display
    - Game history and replay
    """

    def __init__(self, agent: Optional[DQNAgent] = None, ai_player: Player = Player.BLUE):
        """
        Initialize the GUI

        Args:
            agent: Trained DQN agent (if None, plays human vs human)
            ai_player: Which player the AI controls
        """
        pygame.init()

        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tic Tac Pro - AI Training")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.tiny_font = pygame.font.Font(None, 18)

        # Game state
        self.game = TicTacPro()
        self.agent = agent
        self.ai_player = ai_player

        # UI state
        self.selected_size = PieceSize.SMALL
        self.hover_cell = None
        self.show_suggestions = True
        self.show_q_values = True
        self.game_history = []

        # AI settings
        self.ai_enabled = agent is not None
        self.ai_thinking = False
        self.ai_delay = 500  # ms delay before AI moves

        self.running = True

    def draw_board(self):
        """Draw the game board"""
        # Board background
        board_rect = pygame.Rect(
            BOARD_MARGIN,
            BOARD_MARGIN,
            BOARD_SIZE,
            BOARD_SIZE
        )
        pygame.draw.rect(self.window, LIGHT_GRAY, board_rect)

        # Grid lines
        for i in range(1, 3):
            # Vertical lines
            x = BOARD_MARGIN + i * CELL_SIZE
            pygame.draw.line(
                self.window,
                BLACK,
                (x, BOARD_MARGIN),
                (x, BOARD_MARGIN + BOARD_SIZE),
                3
            )

            # Horizontal lines
            y = BOARD_MARGIN + i * CELL_SIZE
            pygame.draw.line(
                self.window,
                BLACK,
                (BOARD_MARGIN, y),
                (BOARD_MARGIN + BOARD_SIZE, y),
                3
            )

        # Border
        pygame.draw.rect(self.window, BLACK, board_rect, 4)

        # Draw pieces
        self.draw_pieces()

        # Draw hover highlight
        if self.hover_cell is not None:
            row, col = self.hover_cell
            cell_x = BOARD_MARGIN + col * CELL_SIZE
            cell_y = BOARD_MARGIN + row * CELL_SIZE
            highlight_rect = pygame.Rect(cell_x, cell_y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.window, YELLOW, highlight_rect, 3)

    def draw_pieces(self):
        """Draw all pieces on the board"""
        visible_board = self.game.get_visible_board()

        for row in range(3):
            for col in range(3):
                player = Player(visible_board[row, col, 0])
                size_val = visible_board[row, col, 1]

                if player == Player.NONE:
                    continue

                # Get piece position
                cell_x = BOARD_MARGIN + col * CELL_SIZE + CELL_SIZE // 2
                cell_y = BOARD_MARGIN + row * CELL_SIZE + CELL_SIZE // 2

                # Determine color and radius
                color = RED if player == Player.RED else BLUE
                dark_color = DARK_RED if player == Player.RED else DARK_BLUE

                if size_val == 1:
                    radius = PIECE_SMALL
                elif size_val == 2:
                    radius = PIECE_MEDIUM
                else:
                    radius = PIECE_LARGE

                # Draw piece with 3D effect
                # Shadow
                pygame.draw.circle(self.window, dark_color, (cell_x + 2, cell_y + 2), radius)
                # Main piece
                pygame.draw.circle(self.window, color, (cell_x, cell_y), radius)
                # Highlight
                pygame.draw.circle(self.window, WHITE, (cell_x - 5, cell_y - 5), radius // 4)

                # Draw size indicator
                size_text = ['S', 'M', 'L'][size_val - 1]
                text_surface = self.tiny_font.render(size_text, True, WHITE)
                text_rect = text_surface.get_rect(center=(cell_x, cell_y))
                self.window.blit(text_surface, text_rect)

    def draw_piece_selector(self):
        """Draw piece size selector panel"""
        panel_x = BOARD_MARGIN + BOARD_SIZE + 50
        panel_y = BOARD_MARGIN

        # Panel title
        title = self.font.render("Select Size", True, BLACK)
        self.window.blit(title, (panel_x, panel_y))

        # Current player indicator
        current_player = self.game.current_player
        player_color = RED if current_player == Player.RED else BLUE
        player_name = "RED" if current_player == Player.RED else "BLUE"

        player_text = self.small_font.render(f"Current: {player_name}", True, player_color)
        self.window.blit(player_text, (panel_x, panel_y + 40))

        # Piece counts
        y_offset = 80
        for size in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]:
            remaining = self.game.pieces_remaining[current_player][size]
            size_name = ['Small', 'Medium', 'Large'][size - 1]
            radius = [PIECE_SMALL, PIECE_MEDIUM, PIECE_LARGE][size - 1]

            # Button background
            button_rect = pygame.Rect(panel_x, panel_y + y_offset, 200, 60)

            # Highlight selected size
            if size == self.selected_size:
                pygame.draw.rect(self.window, GOLD, button_rect, 0, 5)
            else:
                pygame.draw.rect(self.window, LIGHT_GRAY, button_rect, 0, 5)

            pygame.draw.rect(self.window, BLACK, button_rect, 2, 5)

            # Draw preview piece
            piece_x = panel_x + 40
            piece_y = panel_y + y_offset + 30
            pygame.draw.circle(self.window, player_color, (piece_x, piece_y), radius)

            # Draw text
            text = self.small_font.render(f"{size_name}: {remaining}", True, BLACK)
            self.window.blit(text, (panel_x + 80, panel_y + y_offset + 15))

            y_offset += 70

    def draw_suggestions(self):
        """Draw AI move suggestions overlay"""
        if not self.show_suggestions or self.agent is None or self.game.game_over:
            return

        suggestions = self.agent.get_move_suggestions(
            self.game,
            self.game.current_player,
            top_k=3
        )

        if not suggestions:
            return

        # Draw suggestion markers
        for i, ((row, col, size), q_value) in enumerate(suggestions):
            cell_x = BOARD_MARGIN + col * CELL_SIZE + CELL_SIZE // 2
            cell_y = BOARD_MARGIN + row * CELL_SIZE + CELL_SIZE // 2

            # Draw suggestion circle
            color = GREEN if i == 0 else YELLOW
            pygame.draw.circle(self.window, color, (cell_x, cell_y), 10, 3)

            # Draw rank number
            rank_text = self.tiny_font.render(str(i + 1), True, color)
            self.window.blit(rank_text, (cell_x - 5, cell_y - 8))

    def draw_game_info(self):
        """Draw game information panel"""
        panel_x = BOARD_MARGIN + BOARD_SIZE + 50
        panel_y = BOARD_MARGIN + 300

        # Move count
        move_text = self.small_font.render(
            f"Moves: {len(self.game.move_history)}",
            True,
            BLACK
        )
        self.window.blit(move_text, (panel_x, panel_y))

        # Win/loss status
        if self.game.game_over:
            if self.game.winner == Player.NONE:
                result_text = "DRAW!"
                color = GRAY
            else:
                winner_name = "RED" if self.game.winner == Player.RED else "BLUE"
                result_text = f"{winner_name} WINS!"
                color = RED if self.game.winner == Player.RED else BLUE

            result_surface = self.font.render(result_text, True, color)
            self.window.blit(result_surface, (panel_x, panel_y + 40))

            # New game button
            button_rect = pygame.Rect(panel_x, panel_y + 100, 200, 50)
            pygame.draw.rect(self.window, GREEN, button_rect, 0, 5)
            pygame.draw.rect(self.window, BLACK, button_rect, 2, 5)

            button_text = self.font.render("New Game", True, WHITE)
            text_rect = button_text.get_rect(center=button_rect.center)
            self.window.blit(button_text, text_rect)

        # Controls help
        help_y = WINDOW_HEIGHT - 150
        help_texts = [
            "Controls:",
            "1/2/3: Select piece size",
            "Click: Place piece",
            "S: Toggle suggestions",
            "R: Reset game"
        ]

        for i, text in enumerate(help_texts):
            help_surface = self.tiny_font.render(text, True, BLACK)
            self.window.blit(help_surface, (panel_x, help_y + i * 20))

    def get_cell_from_pos(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Convert mouse position to board cell"""
        x, y = pos

        if (x < BOARD_MARGIN or x > BOARD_MARGIN + BOARD_SIZE or
            y < BOARD_MARGIN or y > BOARD_MARGIN + BOARD_SIZE):
            return None

        col = (x - BOARD_MARGIN) // CELL_SIZE
        row = (y - BOARD_MARGIN) // CELL_SIZE

        return (row, col)

    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click"""
        # Check if new game button clicked
        if self.game.game_over:
            panel_x = BOARD_MARGIN + BOARD_SIZE + 50
            panel_y = BOARD_MARGIN + 300
            button_rect = pygame.Rect(panel_x, panel_y + 100, 200, 50)
            if button_rect.collidepoint(pos):
                self.reset_game()
                return

        # Check piece selector
        panel_x = BOARD_MARGIN + BOARD_SIZE + 50
        panel_y = BOARD_MARGIN + 80

        for size in [PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE]:
            y_offset = (size - 1) * 70
            button_rect = pygame.Rect(panel_x, panel_y + y_offset, 200, 60)
            if button_rect.collidepoint(pos):
                self.selected_size = size
                return

        # Check board click
        cell = self.get_cell_from_pos(pos)
        if cell is not None and not self.game.game_over:
            row, col = cell

            # Try to make move
            if self.game.current_player == self.ai_player and self.ai_enabled:
                return  # Don't allow human to play during AI's turn

            if self.game.make_move(row, col, self.selected_size):
                # Move successful
                pass

    def ai_move(self):
        """Execute AI move"""
        if self.agent is None or self.game.game_over:
            return

        if self.game.current_player != self.ai_player:
            return

        # Get AI action
        action = self.agent.get_action(self.game, self.ai_player, epsilon=0.0)

        if action is not None:
            row, col, size = action
            self.game.make_move(row, col, size)
            self.selected_size = size  # Update selector to match AI choice

    def reset_game(self):
        """Reset the game"""
        self.game.reset()
        self.selected_size = PieceSize.SMALL

    def run(self):
        """Main game loop"""
        ai_move_timer = 0

        while self.running:
            dt = self.clock.tick(60)  # 60 FPS

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)

                elif event.type == pygame.MOUSEMOTION:
                    self.hover_cell = self.get_cell_from_pos(event.pos)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self.selected_size = PieceSize.SMALL
                    elif event.key == pygame.K_2:
                        self.selected_size = PieceSize.MEDIUM
                    elif event.key == pygame.K_3:
                        self.selected_size = PieceSize.LARGE
                    elif event.key == pygame.K_s:
                        self.show_suggestions = not self.show_suggestions
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False

            # AI move logic
            if (self.ai_enabled and
                not self.game.game_over and
                self.game.current_player == self.ai_player):

                ai_move_timer += dt
                if ai_move_timer >= self.ai_delay:
                    self.ai_move()
                    ai_move_timer = 0
            else:
                ai_move_timer = 0

            # Draw everything
            self.window.fill(WHITE)
            self.draw_board()
            self.draw_piece_selector()
            self.draw_suggestions()
            self.draw_game_info()

            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    # Test GUI without AI
    print("Starting Tic Tac Pro GUI (Human vs Human)...")
    gui = TicTacProGUI()
    gui.run()
