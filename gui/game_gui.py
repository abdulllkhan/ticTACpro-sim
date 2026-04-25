"""
TicTacPro GUI — redesigned with dark theme, Q-value heatmap,
win-condition highlight, score tracker, and move history.
"""

import pygame
import sys
import math
import time
import numpy as np
from typing import Optional, Tuple, List

from game.tictacpro import TicTacPro, Player, PieceSize

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG          = (15,  17,  26)   # near-black navy
PANEL_BG    = (22,  27,  42)   # slightly lighter panel
BOARD_BG    = (28,  35,  55)   # board surface
CELL_NORMAL = (35,  44,  68)   # default cell
CELL_HOVER  = (50,  65,  100)  # mouse-over cell
GRID_LINE   = (55,  70,  110)  # grid lines

RED_HUE     = (220, 75,  75)
RED_DARK    = (160, 40,  40)
RED_LIGHT   = (255, 140, 140)
BLUE_HUE    = (70,  130, 230)
BLUE_DARK   = (40,  80,  170)
BLUE_LIGHT  = (140, 190, 255)

WHITE       = (255, 255, 255)
OFF_WHITE   = (210, 215, 230)
MUTED       = (120, 130, 155)
ACCENT      = (100, 200, 255)   # cyan accent
GREEN       = (60,  200, 100)
GOLD        = (255, 200, 50)
WIN_GLOW    = (255, 220, 60)

# Q-value heatmap colours (cool → hot)
HEAT_LOW    = (30,  50,  120)   # low Q — cool blue
HEAT_HIGH   = (200, 60,  60)    # high Q — warm red

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
W, H        = 1180, 780
BOARD_X     = 60
BOARD_Y     = 80
BOARD_PX    = 540           # board pixel size
CELL_PX     = BOARD_PX // 3
PANEL_X     = BOARD_X + BOARD_PX + 40
PANEL_W     = W - PANEL_X - 20

PIECE_R     = {1: 18, 2: 30, 3: 44}   # radius per size


class TicTacProGUI:
    def __init__(self, agent=None, ai_player: Player = Player.BLUE):
        pygame.init()
        pygame.display.set_caption("TicTacPro — AI Challenge")
        self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        self.clock  = pygame.time.Clock()

        # Fonts
        self._init_fonts()

        # Game state
        self.game       = TicTacPro()
        self.agent      = agent
        self.ai_player  = ai_player
        self.human      = Player.RED if ai_player == Player.BLUE else Player.BLUE

        # UI state
        self.selected_size  = PieceSize.MEDIUM
        self.hover_cell     = None          # (row, col) or None
        self.show_hints     = True
        self.show_heatmap   = True
        self.last_ai_move   = None          # (row, col, size) for highlight
        self.win_cells: List[Tuple[int,int]] = []

        # Score
        self.score = {Player.RED: 0, Player.BLUE: 0, "draws": 0}

        # AI timing
        self.ai_delay_ms    = 600
        self.ai_timer       = 0
        self.ai_thinking    = False

        # Move history
        self.move_log: List[str] = []

        # Q-values cache
        self._qv: Optional[np.ndarray] = None
        self._qv_dirty = True

        self.running = True

    # ------------------------------------------------------------------
    # Font helpers
    # ------------------------------------------------------------------

    def _init_fonts(self):
        candidates = ["DejaVu Sans", "Liberation Sans", "FreeSans", "Arial", None]
        def load(name, size):
            for c in candidates:
                try:
                    return pygame.font.SysFont(c, size, bold=(name == "bold"))
                except Exception:
                    pass
            return pygame.font.Font(None, size)

        self.f_title  = load("bold",    28)
        self.f_large  = load("normal",  22)
        self.f_mid    = load("normal",  18)
        self.f_small  = load("normal",  15)
        self.f_tiny   = load("normal",  13)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def cell_center(self, row: int, col: int) -> Tuple[int, int]:
        x = BOARD_X + col * CELL_PX + CELL_PX // 2
        y = BOARD_Y + row * CELL_PX + CELL_PX // 2
        return x, y

    def pos_to_cell(self, pos) -> Optional[Tuple[int, int]]:
        x, y = pos
        if not (BOARD_X <= x < BOARD_X + BOARD_PX and
                BOARD_Y <= y < BOARD_Y + BOARD_PX):
            return None
        return (y - BOARD_Y) // CELL_PX, (x - BOARD_X) // CELL_PX

    # ------------------------------------------------------------------
    # Q-value helpers
    # ------------------------------------------------------------------

    def _refresh_qv(self):
        if self.agent is None or self.game.game_over:
            self._qv = None
            return
        try:
            self._qv = self.agent.get_q_values(self.game)
        except Exception:
            self._qv = None
        self._qv_dirty = False

    def _cell_max_q(self, row: int, col: int) -> Optional[float]:
        if self._qv is None:
            return None
        vals = [self._qv[row * 9 + col * 3 + (s - 1)] for s in [1, 2, 3]]
        return float(max(vals))

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _txt(self, surf, text, font, color, pos, anchor="topleft"):
        s = font.render(str(text), True, color)
        r = s.get_rect(**{anchor: pos})
        surf.blit(s, r)

    def draw_background(self):
        self.screen.fill(BG)

        # Title bar area
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, W, 60))
        self._txt(self.screen, "TicTacPro", self.f_title, ACCENT,
                  (W // 2, 30), anchor="center")

        # Score strip
        r_score = f"RED  {self.score[Player.RED]}"
        b_score = f"BLUE  {self.score[Player.BLUE]}"
        d_score = f"Draws  {self.score['draws']}"
        self._txt(self.screen, r_score, self.f_mid, RED_HUE,  (180, 30), anchor="center")
        self._txt(self.screen, d_score, self.f_mid, MUTED,    (W // 2, 30), anchor="center")
        self._txt(self.screen, b_score, self.f_mid, BLUE_HUE, (W - 180, 30), anchor="center")

    def draw_board(self):
        board_rect = pygame.Rect(BOARD_X, BOARD_Y, BOARD_PX, BOARD_PX)

        # Shadow
        shadow = pygame.Surface((BOARD_PX + 8, BOARD_PX + 8), pygame.SRCALPHA)
        shadow.fill((0, 0, 0, 80))
        self.screen.blit(shadow, (BOARD_X + 4, BOARD_Y + 4))

        pygame.draw.rect(self.screen, BOARD_BG, board_rect, border_radius=8)

        # Q-value heatmap overlay
        if self.show_heatmap and self._qv is not None:
            qs = [self._cell_max_q(r, c) for r in range(3) for c in range(3)]
            q_min, q_max = min(qs), max(qs)
            q_range = max(q_max - q_min, 1e-6)
            for r in range(3):
                for c in range(3):
                    q = self._cell_max_q(r, c)
                    if q is None:
                        continue
                    t = (q - q_min) / q_range
                    col_heat = tuple(int(HEAT_LOW[i] + t * (HEAT_HIGH[i] - HEAT_LOW[i]))
                                     for i in range(3))
                    cell_surf = pygame.Surface((CELL_PX - 4, CELL_PX - 4), pygame.SRCALPHA)
                    cell_surf.fill((*col_heat, 80))
                    cx = BOARD_X + c * CELL_PX + 2
                    cy = BOARD_Y + r * CELL_PX + 2
                    self.screen.blit(cell_surf, (cx, cy))

        # Hover highlight
        if self.hover_cell and not self.game.game_over:
            r, c = self.hover_cell
            pygame.draw.rect(self.screen, CELL_HOVER,
                             (BOARD_X + c * CELL_PX + 2,
                              BOARD_Y + r * CELL_PX + 2,
                              CELL_PX - 4, CELL_PX - 4),
                             border_radius=4)

        # Win cells glow
        for (r, c) in self.win_cells:
            glow = pygame.Surface((CELL_PX - 2, CELL_PX - 2), pygame.SRCALPHA)
            glow.fill((*WIN_GLOW, 60))
            self.screen.blit(glow, (BOARD_X + c * CELL_PX + 1,
                                    BOARD_Y + r * CELL_PX + 1))

        # Grid lines
        for i in range(1, 3):
            x = BOARD_X + i * CELL_PX
            y = BOARD_Y + i * CELL_PX
            pygame.draw.line(self.screen, GRID_LINE,
                             (x, BOARD_Y), (x, BOARD_Y + BOARD_PX), 2)
            pygame.draw.line(self.screen, GRID_LINE,
                             (BOARD_X, y), (BOARD_X + BOARD_PX, y), 2)

        # Border
        pygame.draw.rect(self.screen, GRID_LINE, board_rect, 2, border_radius=8)

        # Pieces
        self._draw_pieces()

        # Hint overlay (top-3 suggestions)
        if self.show_hints and not self.game.game_over:
            self._draw_hints()

        # AI last-move indicator
        if self.last_ai_move:
            r, c, _ = self.last_ai_move
            cx, cy = self.cell_center(r, c)
            pygame.draw.rect(self.screen, ACCENT,
                             (BOARD_X + c * CELL_PX + 1, BOARD_Y + r * CELL_PX + 1,
                              CELL_PX - 2, CELL_PX - 2),
                             2, border_radius=4)

        # Q-value numbers (small, per-cell best)
        if self.show_heatmap and self._qv is not None:
            for r in range(3):
                for c in range(3):
                    q = self._cell_max_q(r, c)
                    if q is not None:
                        cx, cy = self.cell_center(r, c)
                        self._txt(self.screen, f"{q:+.1f}", self.f_tiny,
                                  (*OFF_WHITE, 180), (cx, cy + CELL_PX // 2 - 14),
                                  anchor="center")

    def _draw_pieces(self):
        visible = self.game.get_visible_board()
        for r in range(3):
            for c in range(3):
                player_id = int(visible[r, c, 0])
                size_val  = int(visible[r, c, 1])
                if player_id == Player.NONE.value:
                    continue

                player = Player(player_id)
                cx, cy = self.cell_center(r, c)
                radius = PIECE_R[size_val]
                color  = RED_HUE  if player == Player.RED  else BLUE_HUE
                dark   = RED_DARK if player == Player.RED  else BLUE_DARK
                light  = RED_LIGHT if player == Player.RED else BLUE_LIGHT

                # Shadow
                pygame.draw.circle(self.screen, dark, (cx + 3, cy + 3), radius)
                # Body
                pygame.draw.circle(self.screen, color, (cx, cy), radius)
                # Inner ring for medium/large
                if size_val >= 2:
                    pygame.draw.circle(self.screen, dark, (cx, cy), radius - 8, 2)
                # Highlight
                hlx = cx - radius // 3
                hly = cy - radius // 3
                hl_r = max(3, radius // 4)
                hl_surf = pygame.Surface((hl_r * 2, hl_r * 2), pygame.SRCALPHA)
                pygame.draw.circle(hl_surf, (*light, 160), (hl_r, hl_r), hl_r)
                self.screen.blit(hl_surf, (hlx - hl_r, hly - hl_r))

                # Size label
                label = ["S", "M", "L"][size_val - 1]
                self._txt(self.screen, label, self.f_small, WHITE,
                          (cx, cy), anchor="center")

                # Win-cell sparkle
                if (r, c) in self.win_cells:
                    t = time.time()
                    pulse = int(80 + 80 * math.sin(t * 6))
                    sparkle = pygame.Surface((radius * 2 + 8, radius * 2 + 8),
                                            pygame.SRCALPHA)
                    pygame.draw.circle(sparkle, (*WIN_GLOW, pulse),
                                       (radius + 4, radius + 4), radius + 4, 3)
                    self.screen.blit(sparkle, (cx - radius - 4, cy - radius - 4))

    def _draw_hints(self):
        if self.agent is None:
            return
        try:
            suggestions = self.agent.get_move_suggestions(
                self.game, self.game.current_player, top_k=3)
        except Exception:
            return

        colors = [GREEN, GOLD, MUTED]
        for i, ((r, c, sz), q) in enumerate(suggestions):
            cx, cy = self.cell_center(r, c)
            col = colors[i]
            # Dashed ring
            pygame.draw.circle(self.screen, col, (cx, cy), PIECE_R[sz] + 6, 2)
            # Rank badge
            badge_r = pygame.Rect(cx + PIECE_R[sz] - 2, cy - PIECE_R[sz] - 2, 16, 16)
            pygame.draw.rect(self.screen, col, badge_r, border_radius=3)
            self._txt(self.screen, str(i + 1), self.f_tiny, BG,
                      badge_r.center, anchor="center")

    def draw_panel(self):
        """Right-side panel: turn indicator, piece inventory, controls, history."""
        px = PANEL_X
        pygame.draw.rect(self.screen, PANEL_BG,
                         (px - 10, 65, PANEL_W + 10, H - 70), border_radius=8)

        y = 80

        # ---------- Turn indicator ----------
        if self.game.game_over:
            if self.game.winner == Player.NONE:
                msg, col = "DRAW", MUTED
            else:
                name = "RED" if self.game.winner == Player.RED else "BLUE"
                col  = RED_HUE if self.game.winner == Player.RED else BLUE_HUE
                msg  = f"{name} WINS!"
            self._txt(self.screen, msg, self.f_title, col, (px + PANEL_W // 2, y),
                      anchor="center")
        else:
            cur = self.game.current_player
            col = RED_HUE if cur == Player.RED else BLUE_HUE
            who = "RED" if cur == Player.RED else "BLUE"
            tag = " (you)" if cur == self.human else " (AI)"
            self._txt(self.screen, f"{who}{tag}", self.f_large, col,
                      (px + PANEL_W // 2, y), anchor="center")

        y += 36

        # Thinking indicator
        if self.ai_thinking:
            dots = "." * (int(time.time() * 2) % 4)
            self._txt(self.screen, f"AI thinking{dots}", self.f_small, ACCENT,
                      (px + PANEL_W // 2, y), anchor="center")
        y += 28

        # ---------- Piece inventory ----------
        self._draw_inventory(px, y)
        y += 170

        # ---------- Size selector ----------
        self._draw_size_selector(px, y)
        y += 90

        # ---------- Toggle buttons ----------
        self._draw_toggles(px, y)
        y += 46

        # ---------- Move history ----------
        self._draw_history(px, y)
        y += len(self.move_log) * 18 + 10

        # ---------- Controls ----------
        self._draw_controls(px)

        # ---------- New Game button when game over ----------
        if self.game.game_over:
            btn = pygame.Rect(px, H - 100, PANEL_W - 10, 44)
            pygame.draw.rect(self.screen, GREEN, btn, border_radius=8)
            self._txt(self.screen, "New Game  [R]", self.f_large, BG,
                      btn.center, anchor="center")

    def _draw_inventory(self, px: int, y: int):
        self._txt(self.screen, "PIECES REMAINING", self.f_small, MUTED, (px, y))
        y += 22
        for player, col in [(Player.RED, RED_HUE), (Player.BLUE, BLUE_HUE)]:
            name = "RED" if player == Player.RED else "BLUE"
            self._txt(self.screen, name, self.f_small, col, (px, y))
            rx = px + 70
            for sz in [1, 2, 3]:
                count = self.game.pieces_remaining[player][sz]
                r = PIECE_R[sz] - 4
                cx = rx + r + 2
                cy = y + 10
                # draw small piece icons
                dark = RED_DARK if player == Player.RED else BLUE_DARK
                pygame.draw.circle(self.screen, dark, (cx + 2, cy + 2), r)
                pygame.draw.circle(self.screen, col,  (cx, cy), r)
                self._txt(self.screen, str(count), self.f_tiny, WHITE,
                          (cx, cy), anchor="center")
                rx += r * 2 + 16
            # label S M L under icons
            self._txt(self.screen, "S", self.f_tiny, MUTED, (px + 77,  y + 22))
            self._txt(self.screen, "M", self.f_tiny, MUTED, (px + 104, y + 22))
            self._txt(self.screen, "L", self.f_tiny, MUTED, (px + 133, y + 22))
            y += 44

    def _draw_size_selector(self, px: int, y: int):
        self._txt(self.screen, "PLACE SIZE", self.f_small, MUTED, (px, y))
        y += 22
        cur = self.game.current_player
        col = RED_HUE if cur == Player.RED else BLUE_HUE
        dark = RED_DARK if cur == Player.RED else BLUE_DARK
        names = ["Small", "Medium", "Large"]
        bw = (PANEL_W - 10 - 12) // 3
        for i, sz in enumerate([1, 2, 3]):
            bx = px + i * (bw + 6)
            selected = (self.selected_size == sz)
            bg = col if selected else PANEL_BG
            border = col if selected else GRID_LINE
            pygame.draw.rect(self.screen, bg,
                             (bx, y, bw, 52), border_radius=6)
            pygame.draw.rect(self.screen, border,
                             (bx, y, bw, 52), 2, border_radius=6)
            # mini piece
            mx = bx + bw // 2
            my = y + 22
            r  = PIECE_R[sz] - 6
            pygame.draw.circle(self.screen, dark if not selected else (0,0,0,0),
                               (mx + 1, my + 1), r)
            pygame.draw.circle(self.screen,
                               WHITE if selected else col, (mx, my), r)
            self._txt(self.screen, names[i][0], self.f_tiny,
                      BG if selected else WHITE, (mx, my), anchor="center")
            self._txt(self.screen, str(sz), self.f_tiny, MUTED,
                      (bx + bw // 2, y + 40), anchor="center")

    def _draw_toggles(self, px: int, y: int):
        for i, (label, attr, key) in enumerate([
            ("Hints [S]",   "show_hints",   None),
            ("Heatmap [H]", "show_heatmap", None),
        ]):
            bx = px + i * ((PANEL_W - 10) // 2 + 4)
            on = getattr(self, attr)
            bg     = (30, 80, 50) if on else PANEL_BG
            border = GREEN if on else GRID_LINE
            pygame.draw.rect(self.screen, bg,
                             (bx, y, (PANEL_W - 10) // 2 - 4, 34), border_radius=6)
            pygame.draw.rect(self.screen, border,
                             (bx, y, (PANEL_W - 10) // 2 - 4, 34), 2, border_radius=6)
            self._txt(self.screen, label, self.f_tiny,
                      GREEN if on else MUTED,
                      (bx + (PANEL_W - 10) // 4 - 2, y + 17), anchor="center")

    def _draw_history(self, px: int, y: int):
        self._txt(self.screen, "MOVE HISTORY", self.f_small, MUTED, (px, y))
        y += 20
        visible = self.move_log[-12:]  # last 12 moves
        for i, entry in enumerate(visible):
            shade = OFF_WHITE if i == len(visible) - 1 else MUTED
            self._txt(self.screen, entry, self.f_tiny, shade, (px, y + i * 18))

    def _draw_controls(self, px: int):
        lines = [
            ("1/2/3", "pick size"),
            ("S",     "hints on/off"),
            ("H",     "heatmap on/off"),
            ("R",     "new game"),
            ("ESC",   "quit"),
        ]
        y = H - 20 - len(lines) * 18
        for key, desc in lines:
            self._txt(self.screen, key, self.f_tiny, ACCENT, (px, y))
            self._txt(self.screen, desc, self.f_tiny, MUTED,  (px + 32, y))
            y += 18

    # ------------------------------------------------------------------
    # Logic
    # ------------------------------------------------------------------

    def _detect_win_cells(self):
        """Populate self.win_cells from game state after a win."""
        self.win_cells = []
        if not self.game.game_over or self.game.winner == Player.NONE:
            return
        visible = self.game.get_visible_board()
        w = self.game.winner.value

        def check_line(cells):
            for r, c in cells:
                if visible[r, c, 0] != w:
                    return False
            return True

        def same_size(cells):
            sizes = [visible[r, c, 1] for r, c in cells]
            return len(set(sizes)) == 1 and sizes[0] != 0

        lines = [
            [(r, c) for c in range(3)] for r in range(3)
        ] + [
            [(r, c) for r in range(3)] for c in range(3)
        ] + [
            [(i, i) for i in range(3)],
            [(i, 2 - i) for i in range(3)],
        ]

        for line in lines:
            if check_line(line) and same_size(line):
                self.win_cells = line
                return

        # Bullseye check
        for r in range(3):
            for c in range(3):
                stack = self.game.board[r, c]
                layers = [stack[l] for l in range(3)
                          if stack[l][0] == w and stack[l][1] == l + 1]
                if len(layers) == 3:
                    self.win_cells = [(r, c)]
                    return

    def _log_move(self, player: Player, r: int, c: int, sz: int):
        sym = "R" if player == Player.RED else "B"
        pos = f"({r},{c})"
        size_name = ["S", "M", "L"][sz - 1]
        n = len(self.move_log) + 1
        self.move_log.append(f"{n:>2}. {sym} {pos} {size_name}")

    def handle_click(self, pos):
        # New-game button
        if self.game.game_over:
            btn = pygame.Rect(PANEL_X, H - 100, PANEL_W - 10, 44)
            if btn.collidepoint(pos):
                self.reset_game()
                return

        # Size selector buttons
        y_sel = 80 + 36 + 28 + 170 + 22
        bw = (PANEL_W - 10 - 12) // 3
        for i, sz in enumerate([1, 2, 3]):
            bx = PANEL_X + i * (bw + 6)
            if pygame.Rect(bx, y_sel, bw, 52).collidepoint(pos):
                self.selected_size = sz
                return

        # Toggle buttons
        y_tog = y_sel + 90
        hw = (PANEL_W - 10) // 2 - 4
        for i, attr in enumerate(["show_hints", "show_heatmap"]):
            bx = PANEL_X + i * (hw + 8)
            if pygame.Rect(bx, y_tog, hw, 34).collidepoint(pos):
                setattr(self, attr, not getattr(self, attr))
                return

        # Board click
        cell = self.pos_to_cell(pos)
        if cell is None or self.game.game_over:
            return
        if self.game.current_player == self.ai_player:
            return  # AI's turn
        r, c = cell
        if self.game.make_move(r, c, self.selected_size):
            self._log_move(self.game.winner if self.game.game_over
                           else Player(3 - self.game.current_player.value),
                           r, c, self.selected_size)
            self._qv_dirty = True
            if self.game.game_over:
                self._on_game_over()

    def _ai_move(self):
        if self.agent is None or self.game.game_over:
            return
        if self.game.current_player != self.ai_player:
            return
        action = self.agent.get_action(self.game, self.ai_player, epsilon=0.0)
        if action is not None:
            r, c, sz = action
            self.game.make_move(r, c, sz)
            self.last_ai_move = (r, c, sz)
            self._log_move(self.ai_player, r, c, sz)
            self._qv_dirty = True
            if self.game.game_over:
                self._on_game_over()

    def _on_game_over(self):
        self._detect_win_cells()
        if self.game.winner == Player.RED:
            self.score[Player.RED] += 1
        elif self.game.winner == Player.BLUE:
            self.score[Player.BLUE] += 1
        else:
            self.score["draws"] += 1

    def reset_game(self):
        self.game.reset()
        self.selected_size  = PieceSize.MEDIUM
        self.last_ai_move   = None
        self.win_cells      = []
        self.move_log       = []
        self._qv            = None
        self._qv_dirty      = True
        self.ai_thinking    = False
        self.ai_timer       = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        while self.running:
            dt = self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self.hover_cell = self.pos_to_cell(event.pos)
                elif event.type == pygame.KEYDOWN:
                    k = event.key
                    if k == pygame.K_1:
                        self.selected_size = 1
                    elif k == pygame.K_2:
                        self.selected_size = 2
                    elif k == pygame.K_3:
                        self.selected_size = 3
                    elif k == pygame.K_s:
                        self.show_hints = not self.show_hints
                    elif k == pygame.K_h:
                        self.show_heatmap = not self.show_heatmap
                        self._qv_dirty = True
                    elif k == pygame.K_r:
                        self.reset_game()
                    elif k in (pygame.K_ESCAPE, pygame.K_q):
                        self.running = False

            # AI turn logic
            is_ai_turn = (
                self.agent is not None
                and not self.game.game_over
                and self.game.current_player == self.ai_player
            )
            if is_ai_turn:
                self.ai_thinking = True
                self.ai_timer += dt
                if self.ai_timer >= self.ai_delay_ms:
                    self._ai_move()
                    self.ai_timer  = 0
                    self.ai_thinking = False
            else:
                self.ai_timer    = 0
                self.ai_thinking = False

            # Refresh Q-values when dirty and it's the human's turn
            if self._qv_dirty and self.show_heatmap and not is_ai_turn:
                self._refresh_qv()

            # Draw
            self.draw_background()
            self.draw_board()
            self.draw_panel()
            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    gui = TicTacProGUI()
    gui.run()
