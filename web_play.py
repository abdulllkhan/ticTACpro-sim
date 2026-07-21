#!/usr/bin/env python3
"""
Web interface for TicTacPro.
Run: python3 web_play.py --checkpoint checkpoints_spark/final.pt
Then open http://localhost:5000
"""

import argparse
import threading
import os
import torch
from flask import Flask, jsonify, request, render_template_string
from game.tictacpro import TicTacPro, Player, PieceSize
from game.optimal_agent import OptimalAgent
from rl.agent import GPUDQNAgent, DQNAgent
from rl.minimax_agent import MinimaxAgent
from rl.mcts_agent import MCTSAgent
from rl.neural_mcts_agent import NeuralMCTSAgent
from rl.parallel_mcts_agent import ParallelMCTSAgent

app = Flask(__name__)

_game        = TicTacPro()
_agent       = None
_optimal     = OptimalAgent()
_minimax     = MinimaxAgent(time_limit=2.0, max_depth=18)
_mcts        = MCTSAgent(time_limit=2.0, max_simulations=10_000_000)
_mcts_g      = MCTSAgent(time_limit=2.0, max_simulations=10_000_000,
                         rollout_agent=OptimalAgent())
_pmcts        = None          # initialised in main() after worker pool spawned
_neural       = None          # initialised after DQN loads in main()
_ckpt_episodes = 0            # loaded from checkpoint metadata
_ckpt_epsilon  = 0.0          # loaded from checkpoint metadata
_ai_player   = Player.BLUE
_human_player= Player.RED
_mode        = "ai"          # "ai" | "optimal" | "minimax" | "mcts" | "mcts_g" | "pmcts" | "neural" | "human"
_series_n    = 3             # best-of N (1 = no series)
_lock        = threading.Lock()
_scores      = {Player.RED: 0, Player.BLUE: 0}
_series_wins = {Player.RED: 0, Player.BLUE: 0}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _state_dict():
    board = []
    for r in range(3):
        row = []
        for c in range(3):
            row.append([int(_game.board[r, c, sz]) for sz in range(3)])
        board.append(row)

    pieces = {}
    for color, player in [("red", Player.RED), ("blue", Player.BLUE)]:
        pieces[color] = {
            "S": int(_game.pieces_remaining[player][PieceSize.SMALL]),
            "M": int(_game.pieces_remaining[player][PieceSize.MEDIUM]),
            "L": int(_game.pieces_remaining[player][PieceSize.LARGE]),
        }

    legal = []
    if not _game.game_over:
        for (r, c, s) in _game.get_legal_moves():
            legal.append(f"{r},{c},{int(s)}")

    history = []
    for (r, c, s, p) in _game.move_history[-20:]:
        history.append({"row": r, "col": c, "size": int(s), "player": int(p)})

    series_target = (_series_n + 1) // 2
    return {
        "board":        board,
        "pieces":       pieces,
        "current_player": int(_game.current_player),
        "winner":       int(_game.winner),
        "game_over":    _game.game_over,
        "legal":        legal,
        "history":      history,
        "ai_player":    int(_ai_player),
        "human_player": int(_human_player),
        "mode":         _mode,
        "series_n":     _series_n,
        "series_target": series_target,
        "scores": {
            str(int(Player.RED)):  _scores[Player.RED],
            str(int(Player.BLUE)): _scores[Player.BLUE],
        },
        "series_wins": {
            str(int(Player.RED)):  _series_wins[Player.RED],
            str(int(Player.BLUE)): _series_wins[Player.BLUE],
        },
    }


def _q_values():
    if _agent is None or _game.game_over:
        return None
    q = _agent.get_q_values(_game)           # uses correct API
    grid = []
    for r in range(3):
        row = []
        for c in range(3):
            row.append([float(q[r * 9 + c * 3 + s]) for s in range(3)])
        grid.append(row)
    return grid


def _ai_move():
    if _mode not in ("ai", "optimal", "minimax", "mcts", "mcts_g", "pmcts", "neural"):
        return
    if _game.game_over or _game.current_player != _ai_player:
        return
    if _mode == "optimal":
        action = _optimal.get_action(_game)
    elif _mode == "minimax":
        action = _minimax.get_action(_game, _ai_player)
    elif _mode == "mcts":
        action = _mcts.get_action(_game, _ai_player)
    elif _mode == "mcts_g":
        action = _mcts_g.get_action(_game, _ai_player)
    elif _mode == "pmcts":
        if _pmcts is None:
            return
        action = _pmcts.get_action(_game, _ai_player)
    elif _mode == "neural":
        if _neural is None:
            return
        action = _neural.get_action(_game, _ai_player)
    else:
        if _agent is None:
            return
        action = _agent.get_action(_game, _ai_player, epsilon=0.0)
    if action is None:
        return
    r, c, s = action
    _game.make_move(r, c, s)
    if _game.winner == _ai_player:
        _scores[_ai_player] += 1
        _series_wins[_ai_player] += 1


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML,
        ckpt_episodes=_ckpt_episodes,
        ckpt_epsilon=_ckpt_epsilon)


@app.route("/api/state")
def api_state():
    with _lock:
        s = _state_dict()
        s["q_values"] = _q_values()
    return jsonify(s)


@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.json
    with _lock:
        if _game.game_over:
            return jsonify({"error": "Game over", **_state_dict()})
        r  = int(data["row"])
        c  = int(data["col"])
        sz = PieceSize(int(data["size"]))

        # In human-vs-human any current player can move
        if _mode in ("ai", "optimal", "minimax", "mcts", "mcts_g", "pmcts", "neural") and _game.current_player != _human_player:
            return jsonify({"error": "Not your turn", **_state_dict()})

        if not _game.make_move(r, c, sz):
            return jsonify({"error": "Invalid move", **_state_dict()})

        if _game.winner == _human_player:
            _scores[_human_player] += 1
            _series_wins[_human_player] += 1
        elif _mode == "human" and _game.winner != Player.NONE:
            winner = _game.winner
            _scores[winner] += 1
            _series_wins[winner] += 1

        if not _game.game_over and _mode in ("ai", "optimal", "minimax", "mcts", "mcts_g", "pmcts", "neural"):
            _ai_move()

        s = _state_dict()
        s["q_values"] = _q_values()
    return jsonify(s)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global _ai_player, _human_player, _mode, _series_n
    with _lock:
        data = request.json or {}
        if "ai_player" in data:
            _ai_player   = Player.RED if int(data["ai_player"]) == 1 else Player.BLUE
            _human_player= Player.BLUE if _ai_player == Player.RED else Player.RED
        if "mode" in data:
            _mode = data["mode"]
        if "series_n" in data:
            _series_n = int(data["series_n"])
        if data.get("reset_series"):
            _series_wins[Player.RED]  = 0
            _series_wins[Player.BLUE] = 0

        _game.reset()
        # Clear MCTS subtree caches so stale nodes from the previous game are
        # not mistakenly reused in the new game.
        _mcts.reset()
        _mcts_g.reset()
        if _pmcts is not None:
            _pmcts.reset()
        if _neural is not None:
            _neural.reset()

        if _mode in ("ai", "optimal", "minimax", "mcts", "mcts_g", "pmcts", "neural") and _ai_player == Player.RED:
            _ai_move()

        s = _state_dict()
        s["q_values"] = _q_values()
    return jsonify(s)


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>TicTacPro</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;600;700;800;900&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
<style>
body { font-family: 'Lexend', sans-serif; background: #f8f9fa; }
.material-symbols-outlined { font-variation-settings: 'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 24; }

.board-cell {
  border-radius: 20px;
  box-shadow: inset 0 2px 6px rgba(0,0,0,0.07);
  transition: background 0.18s, box-shadow 0.15s;
  cursor: not-allowed; position: relative;
}
.board-cell.can-place {
  cursor: pointer;
  box-shadow: inset 0 2px 6px rgba(0,0,0,0.07), 0 0 0 2.5px rgba(182,23,34,0.35);
  background: rgba(182,23,34,0.04) !important;
}
.board-cell.can-place:hover {
  box-shadow: inset 0 2px 6px rgba(0,0,0,0.07), 0 0 0 3px #b61722;
  background: rgba(182,23,34,0.09) !important;
}
.board-cell.win-cell { animation: win-pulse 0.65s infinite alternate; }
@keyframes win-pulse {
  from { background: #f3f4f5; }
  to   { background: #d1fae5; box-shadow: 0 0 0 3px #006b2d; }
}

.piece-s { width:28px;  height:28px;  border-radius:9999px; }
.piece-m { width:64px;  height:64px;  border-radius:9999px; background:transparent !important; border:8px solid; }
.piece-l { width:96px;  height:96px;  border-radius:9999px; background:transparent !important; border:12px solid; }
.piece-red.piece-s  { background: radial-gradient(circle at 30% 30%,#ff6b6b,#c92a2a); box-shadow:0 3px 8px rgba(182,23,34,.35); }
.piece-blue.piece-s { background: radial-gradient(circle at 30% 30%,#4dabf7,#1864ab); box-shadow:0 3px 8px rgba(24,100,171,.35); }
.piece-red.piece-m  { border-color:#c92a2a; box-shadow:0 2px 8px rgba(182,23,34,.2); }
.piece-blue.piece-m { border-color:#1864ab; box-shadow:0 2px 8px rgba(24,100,171,.2); }
.piece-red.piece-l  { border-color:#b61722; box-shadow:0 3px 12px rgba(182,23,34,.25); }
.piece-blue.piece-l { border-color:#0058be; box-shadow:0 3px 12px rgba(24,100,171,.25); }

/* Ghost on hover */
.board-cell.can-place .ghost { opacity:0.10; }
.board-cell.can-place:hover .ghost { opacity:0.40; }
.ghost { opacity:0; position:absolute; border-radius:9999px; pointer-events:none; transition:opacity .15s;
         top:50%; left:50%; transform:translate(-50%,-50%); z-index:20; }
/* Absolute-center helper for nested piece circles */
.piece-abs { position:absolute; border-radius:9999px; top:50%; left:50%; transform:translate(-50%,-50%); }
.ghost.rs { width:28px; height:28px; background:#c92a2a; }
.ghost.rm { width:64px; height:64px; border:8px solid #c92a2a; background:transparent; }
.ghost.rl { width:96px; height:96px; border:12px solid #b61722; background:transparent; }
.ghost.bs { width:28px; height:28px; background:#1864ab; }
.ghost.bm { width:64px; height:64px; border:8px solid #1864ab; background:transparent; }
.ghost.bl { width:96px; height:96px; border:12px solid #0058be; background:transparent; }

/* Inventory dots */
.inv-dot { display:inline-block; border-radius:9999px; vertical-align:middle; transition:opacity .2s; }
.inv-dot.spent { opacity:0.15; }

/* Toggles */
.tog { width:40px; height:22px; border-radius:11px; position:relative; cursor:pointer; transition:background .2s; background:#e1e3e4; flex-shrink:0; }
.tog.on { background:#b61722; }
.tog::after { content:''; width:16px; height:16px; background:#fff; border-radius:50%; position:absolute; top:3px; left:3px; transition:left .2s; box-shadow:0 1px 3px rgba(0,0,0,.2); }
.tog.on::after { left:21px; }

/* Overlay */
.overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.55); align-items:center; justify-content:center; z-index:200; }
.overlay.show { display:flex; }
@keyframes bounce-in { 0%{transform:scale(.8);opacity:0} 70%{transform:scale(1.04)} 100%{transform:scale(1);opacity:1} }
.ov-card { animation:bounce-in .3s ease; }
@keyframes spin { to { transform:rotate(360deg); } }
.spinner { animation:spin 1s linear infinite; }
@keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.7)} }
.pdot { animation:pulse-dot 1.1s infinite; }

/* Segmented control */
.seg-btn { flex:1; padding:6px 4px; border-radius:8px; font-size:12px; font-weight:600; cursor:pointer; transition:all .15s; color:#5b403e; background:transparent; border:none; }
.seg-btn.active { background:white; color:#b61722; box-shadow:0 1px 4px rgba(0,0,0,.12); }
</style>
</head>
<body class="h-screen overflow-hidden flex flex-col text-[#191c1d]">

<!-- Header -->
<header class="bg-white border-b-2 border-[#edeeef] shadow-sm flex justify-between items-center px-6 py-3 z-50 flex-shrink-0">
  <h1 class="text-2xl font-black text-[#b61722]">TicTac<span class="text-[#0058be]">Pro</span></h1>

  <!-- Score + Series -->
  <div class="flex items-center gap-4">
    <!-- Game score -->
    <div class="bg-[#edeeef] px-4 py-1.5 rounded-full text-sm font-bold flex items-center gap-2">
      <span class="text-[#b61722]">Red <span id="sc-red">0</span></span>
      <span class="text-[#8f6f6d]">·</span>
      <span class="text-[#0058be]"><span id="sc-blue">0</span> Blue</span>
    </div>
    <!-- Series score (hidden when series_n=1) -->
    <div id="series-bar" class="hidden items-center gap-2 bg-[#f3f4f5] px-4 py-1.5 rounded-full text-xs font-bold border border-[#edeeef]">
      <span class="text-[#8f6f6d]">Series</span>
      <span class="text-[#b61722]" id="sw-red">0</span>
      <span class="text-[#8f6f6d]">–</span>
      <span class="text-[#0058be]" id="sw-blue">0</span>
      <span class="text-[#8f6f6d]" id="sw-label"></span>
    </div>
  </div>

  <div class="w-24"></div>
</header>

<div class="flex flex-1 overflow-hidden">

  <!-- Board area -->
  <main class="flex-1 flex flex-col items-center justify-center gap-5 p-8">

    <!-- Status bar -->
    <div class="flex items-center gap-3 bg-white px-6 py-2.5 rounded-full shadow-sm border border-[#edeeef] min-w-[260px] justify-center">
      <div id="sdot" class="w-3.5 h-3.5 rounded-full bg-[#b61722] pdot flex-shrink-0"></div>
      <span id="stxt" class="text-sm font-semibold">Loading…</span>
      <svg id="spinner" class="spinner w-4 h-4 text-[#b61722] hidden" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
      </svg>
    </div>

    <!-- 3×3 Board -->
    <div class="bg-white p-4 rounded-[28px] shadow-md border-2 border-[#edeeef]">
      <div id="board" class="grid grid-cols-3 gap-3"></div>
    </div>

    <p class="text-[11px] text-[#8f6f6d] tracking-wide text-center">
      Click a cell to place · 1/2/3 to change size · R to reset · No gobbling: each size slot is independent
    </p>
  </main>

  <!-- Sidebar -->
  <aside class="w-[300px] bg-white border-l-2 border-[#edeeef] flex flex-col p-5 gap-5 overflow-y-auto flex-shrink-0">

    <!-- Mode selector -->
    <div>
      <p class="text-[11px] font-bold uppercase tracking-widest text-[#8f6f6d] mb-2">Game Mode</p>
      <div class="bg-[#f3f4f5] p-1 rounded-xl flex gap-1 flex-wrap">
        <button id="mode-ai"      class="seg-btn active" onclick="setMode('ai')">vs AI</button>
        <button id="mode-optimal" class="seg-btn"        onclick="setMode('optimal')">vs Optimal</button>
        <button id="mode-minimax" class="seg-btn"        onclick="setMode('minimax')">vs Minimax</button>
        <button id="mode-mcts"    class="seg-btn"        onclick="setMode('mcts')">vs MCTS</button>
        <button id="mode-mcts_g"  class="seg-btn"        onclick="setMode('mcts_g')">vs Guided MCTS</button>
        <button id="mode-pmcts"   class="seg-btn"        onclick="setMode('pmcts')">vs Parallel MCTS</button>
        <button id="mode-neural"  class="seg-btn"        onclick="setMode('neural')">vs Neural MCTS</button>
        <button id="mode-human"   class="seg-btn"        onclick="setMode('human')">vs Human</button>
      </div>
    </div>

    <!-- Side + Series (only in AI mode) -->
    <div id="ai-options">
      <p class="text-[11px] font-bold uppercase tracking-widest text-[#8f6f6d] mb-2">AI Options</p>
      <div class="bg-[#f3f4f5] rounded-2xl p-3 flex flex-col gap-3">
        <div class="flex items-center justify-between text-sm">
          <span class="font-medium">Play as Blue (2nd)</span>
          <div id="tog-side" class="tog" onclick="toggleSide()"></div>
        </div>
        <div class="flex items-center justify-between text-sm">
          <span class="font-medium">Best of</span>
          <div class="flex gap-1">
            <button onclick="setSeries(1)" data-sn="1" class="sn-btn px-2 py-0.5 rounded text-xs font-bold bg-[#b61722] text-white">1</button>
            <button onclick="setSeries(3)" data-sn="3" class="sn-btn px-2 py-0.5 rounded text-xs font-bold bg-[#edeeef] text-[#5b403e]">3</button>
            <button onclick="setSeries(5)" data-sn="5" class="sn-btn px-2 py-0.5 rounded text-xs font-bold bg-[#edeeef] text-[#5b403e]">5</button>
            <button onclick="setSeries(7)" data-sn="7" class="sn-btn px-2 py-0.5 rounded text-xs font-bold bg-[#edeeef] text-[#5b403e]">7</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Pieces remaining -->
    <div>
      <p class="text-[11px] font-bold uppercase tracking-widest text-[#8f6f6d] mb-2 flex items-center gap-1">
        <span class="material-symbols-outlined text-[16px]">category</span> Pieces Remaining
      </p>
      <div class="bg-[#f3f4f5] rounded-2xl p-3 flex flex-col gap-3" id="inv"></div>
    </div>

    <!-- Size selector -->
    <div>
      <p class="text-[11px] font-bold uppercase tracking-widest text-[#8f6f6d] mb-2 flex items-center gap-1">
        <span class="material-symbols-outlined text-[16px]">straighten</span> Place Size
      </p>
      <div class="flex gap-2" id="sz-btns">
        <button onclick="setSize(1)" data-sz="1"
          class="sz-btn flex-1 flex flex-col items-center py-3 rounded-2xl border-2 border-transparent bg-[#f3f4f5] gap-1.5">
          <div class="w-5 h-5 rounded-full bg-[#8f6f6d] sz-circle"></div>
          <span class="text-xs font-bold">Small</span>
          <span class="text-[10px] bg-[#edeeef] px-2 rounded font-mono">1</span>
        </button>
        <button onclick="setSize(2)" data-sz="2"
          class="sz-btn flex-1 flex flex-col items-center py-3 rounded-2xl border-2 border-[#b61722] bg-red-50 gap-1.5">
          <div class="w-7 h-7 rounded-full border-4 border-[#b61722] sz-circle"></div>
          <span class="text-xs font-bold text-[#b61722]">Med</span>
          <span class="text-[10px] bg-red-100 text-[#b61722] px-2 rounded font-mono">2</span>
        </button>
        <button onclick="setSize(3)" data-sz="3"
          class="sz-btn flex-1 flex flex-col items-center py-3 rounded-2xl border-2 border-transparent bg-[#f3f4f5] gap-1.5">
          <div class="w-9 h-9 rounded-full border-4 border-[#8f6f6d] sz-circle"></div>
          <span class="text-xs font-bold">Large</span>
          <span class="text-[10px] bg-[#edeeef] px-2 rounded font-mono">3</span>
        </button>
      </div>
    </div>

    <!-- Settings -->
    <div>
      <p class="text-[11px] font-bold uppercase tracking-widest text-[#8f6f6d] mb-2 flex items-center gap-1">
        <span class="material-symbols-outlined text-[16px]">tune</span> Display
      </p>
      <div class="bg-[#f3f4f5] rounded-2xl p-3 flex flex-col gap-3">
        <div class="flex items-center justify-between text-sm">
          <span class="font-medium">Q-value heatmap</span>
          <div id="tog-heat" class="tog on" onclick="toggleHeat()"></div>
        </div>
        <div class="flex items-center justify-between text-sm">
          <span class="font-medium">Best move hint</span>
          <div id="tog-hint" class="tog on" onclick="toggleHint()"></div>
        </div>
      </div>
    </div>

    <!-- Buttons -->
    <div class="flex gap-2">
      <button onclick="resetGame(false)"
        class="flex-1 bg-[#006b2d] hover:bg-[#00873b] text-white font-bold py-3 rounded-xl shadow-[0_3px_0_#005321] active:translate-y-px active:shadow-none transition-all text-sm">
        New Game (R)
      </button>
      <button id="reset-series-btn" onclick="resetGame(true)"
        class="hidden px-3 bg-[#edeeef] hover:bg-[#e1e3e4] text-[#5b403e] font-bold py-3 rounded-xl text-xs transition-colors">
        Reset Series
      </button>
    </div>

    <!-- Move history -->
    <div class="flex-1 min-h-0">
      <p class="text-[11px] font-bold uppercase tracking-widest text-[#8f6f6d] mb-2 flex items-center gap-1">
        <span class="material-symbols-outlined text-[16px]">history</span> Move History
      </p>
      <div id="hist" class="flex flex-col gap-1 overflow-y-auto max-h-40"></div>
    </div>

  </aside>
</div>

<!-- Footer -->
<footer class="bg-white border-t border-[#edeeef] px-6 py-1.5 flex justify-between items-center flex-shrink-0">
  <span class="text-[10px] text-[#8f6f6d] tracking-wide uppercase">Same-size row or bullseye stack wins · No gobbling — each size has its own slot</span>
  <span class="text-[10px] text-[#8f6f6d]">Trained {{ ckpt_episodes // 1000 }}k episodes · ε = {{ "%.2f" | format(ckpt_epsilon) }}</span>
</footer>

<!-- Game-over overlay -->
<div class="overlay" id="overlay">
  <div class="ov-card bg-white rounded-3xl p-10 text-center shadow-2xl max-w-sm w-full mx-4">
    <div id="ov-emoji" class="text-5xl mb-3">🎉</div>
    <h2 id="ov-title" class="text-3xl font-black mb-1">You Win!</h2>
    <p id="ov-sub" class="text-sm text-[#8f6f6d] mb-1">The AI has been defeated.</p>
    <p id="ov-series" class="text-sm font-bold text-[#006b2d] mb-6"></p>
    <div class="flex gap-3">
      <button onclick="resetGame(false)"
        class="flex-1 bg-[#006b2d] hover:bg-[#00873b] text-white font-bold py-3 rounded-xl shadow-md transition-colors">
        Next Game
      </button>
      <button id="ov-series-reset" onclick="resetGame(true)"
        class="hidden flex-1 bg-[#edeeef] hover:bg-[#e1e3e4] text-[#5b403e] font-bold py-3 rounded-xl transition-colors text-sm">
        New Series
      </button>
    </div>
  </div>
</div>

<script>
const SZ     = ['','s','m','l'];
const SZNAME = ['','Small','Med','Large'];
const POS    = [['top-left','top-mid','top-right'],['mid-left','center','mid-right'],['bot-left','bot-mid','bot-right']];
const RED    = '#b61722', BLUE = '#0058be';

let state = null, selSize = 2, showHeat = true, showHint = true, asBlue = false;
let currentMode = 'ai', currentSeries = 1;
const BOT_MODES = new Set(['ai', 'optimal', 'minimax', 'mcts', 'mcts_g', 'pmcts', 'neural']);
const isVsBot   = () => BOT_MODES.has(currentMode);
const botLabel  = () => ({ai:'AI', optimal:'Optimal', minimax:'Minimax',
                          mcts:'MCTS', mcts_g:'Guided MCTS',
                          pmcts:'Parallel MCTS', neural:'Neural MCTS'}[currentMode] || 'Bot');

// ── Build 9 board cells ───────────────────────────────────────────
const boardEl = document.getElementById('board');
for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
  const cell = document.createElement('div');
  cell.id = `cell-${r}-${c}`;
  cell.className = 'board-cell bg-[#f3f4f5] w-[120px] h-[120px] flex items-center justify-center';
  cell.innerHTML =
    `<div id="heat-${r}-${c}" class="absolute inset-0 rounded-[20px] pointer-events-none transition-all"></div>` +
    `<div id="piece-${r}-${c}" class="z-10 absolute inset-0"></div>` +
    `<div id="ghost-${r}-${c}" class="ghost"></div>` +
    `<div id="hint-${r}-${c}" class="hidden absolute -top-2 -right-2 bg-yellow-400 text-yellow-900 text-[10px] font-bold px-2 py-0.5 rounded-full shadow z-30 flex items-center gap-0.5">` +
      `<span class="material-symbols-outlined text-[11px]">star</span>best` +
    `</div>`;
  cell.addEventListener('click', () => onCellClick(r, c));
  boardEl.appendChild(cell);
}

// ── Size selector ─────────────────────────────────────────────────
function setSize(s) {
  selSize = s;
  document.querySelectorAll('.sz-btn').forEach(b => {
    const a = +b.dataset.sz === s;
    b.className = `sz-btn flex-1 flex flex-col items-center py-3 rounded-2xl border-2 gap-1.5 transition-colors ${
      a ? 'border-[#b61722] bg-red-50' : 'border-transparent bg-[#f3f4f5] hover:border-[#b61722]'}`;
    const circle = b.querySelector('.sz-circle');
    if (circle) {
      if (+b.dataset.sz === 1) { circle.style.background = a ? '#b61722' : ''; circle.style.borderColor=''; }
      else { circle.style.borderColor = a ? '#b61722' : ''; }
    }
    const lbl = b.querySelector('span');
    if (lbl) lbl.style.color = a ? '#b61722' : '';
  });
  updateGhosts();
}

// ── Mode selector ────────────────────────────────────────────────
function setMode(m) {
  currentMode = m;
  ['ai','optimal','minimax','mcts','mcts_g','pmcts','neural','human'].forEach(id =>
    document.getElementById(`mode-${id}`).classList.toggle('active', m === id)
  );
  const vsBot = BOT_MODES.has(m);
  document.getElementById('ai-options').style.display = vsBot ? '' : 'none';
  document.getElementById('tog-hint').closest('div.flex').style.display = vsBot ? '' : 'none';
  resetGame(false);
}

// ── Series selector ──────────────────────────────────────────────
function setSeries(n) {
  currentSeries = n;
  document.querySelectorAll('.sn-btn').forEach(b => {
    const a = +b.dataset.sn === n;
    b.className = `sn-btn px-2 py-0.5 rounded text-xs font-bold ${a ? 'bg-[#b61722] text-white' : 'bg-[#edeeef] text-[#5b403e]'}`;
  });
  resetGame(true);
}

// ── Ghost preview ────────────────────────────────────────────────
function updateGhosts() {
  if (!state) return;
  const legal   = new Set(state.legal);
  const active  = isVsBot() ? state.human_player : state.current_player;
  const pc      = active === 1 ? 'r' : 'b';
  const sz      = SZ[selSize];
  const canMove = !state.game_over;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    const ghost  = document.getElementById(`ghost-${r}-${c}`);
    const cell   = document.getElementById(`cell-${r}-${c}`);
    const isLegal = canMove && legal.has(`${r},${c},${selSize}`) &&
                    (!isVsBot() || state.current_player === state.human_player);
    ghost.className = `ghost ${pc}${sz}`;
    cell.classList.toggle('can-place', isLegal);
  }
}

// ── Board render ─────────────────────────────────────────────────
function renderBoard() {
  if (!state) return;
  const qv = state.q_values;
  let minQ = Infinity, maxQ = -Infinity;
  if (qv) for (let r=0;r<3;r++) for (let c=0;c<3;c++) for (let s=0;s<3;s++) {
    minQ = Math.min(minQ, qv[r][c][s]);
    maxQ = Math.max(maxQ, qv[r][c][s]);
  }

  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    const layers = state.board[r][c]; // [player_small, player_medium, player_large]
    const piece  = document.getElementById(`piece-${r}-${c}`);
    const heat   = document.getElementById(`heat-${r}-${c}`);
    const hint   = document.getElementById(`hint-${r}-${c}`);
    const cell   = document.getElementById(`cell-${r}-${c}`);

    // Pieces — all 3 layers rendered as concentric circles (large ring → medium ring → small dot)
    piece.innerHTML = layers.map((p, sz) =>
      p ? `<div class="piece-${p===1?'red':'blue'} piece-${SZ[sz+1]} piece-abs"></div>` : ''
    ).join('');

    // Heatmap
    if (qv && showHeat && isVsBot()) {
      const t = (Math.max(...qv[r][c]) - minQ) / (maxQ - minQ || 1);
      heat.style.background = t > .7 ? 'rgba(182,23,34,.13)' : t > .4 ? 'rgba(210,153,34,.10)' : 'transparent';
    } else { heat.style.background = 'transparent'; }

    // Hint badge — reset
    hint.classList.add('hidden'); hint.classList.remove('flex');

    // Win glow — highlight cells where winner has any piece
    cell.classList.toggle('win-cell', state.game_over && state.winner && layers.some(p => p === state.winner));
  }

  // Best move badge
  const myTurn = !state.game_over &&
    (!isVsBot() || state.current_player === state.human_player);
  if (showHint && qv && myTurn && isVsBot()) {
    let best = null, bestQ = -Infinity;
    for (const key of state.legal) {
      const [lr, lc, ls] = key.split(',').map(Number);
      const q = qv[lr][lc][ls - 1];
      if (q > bestQ) { bestQ = q; best = {r: lr, c: lc}; }
    }
    if (best) {
      const h = document.getElementById(`hint-${best.r}-${best.c}`);
      h.classList.remove('hidden'); h.classList.add('flex');
    }
  }
  updateGhosts();
}

// ── Inventory ────────────────────────────────────────────────────
function renderInv() {
  if (!state) return;
  document.getElementById('inv').innerHTML = ['red','blue'].map(col => {
    const p  = state.pieces[col];
    const bc = col === 'red' ? '#c92a2a' : '#1864ab';
    const tc = col === 'red' ? RED : BLUE;
    const dot = (w, h, extra, cnt, i) =>
      `<span class="inv-dot${i >= cnt ? ' spent' : ''}" style="width:${w};height:${h};${extra}"></span>`;
    return `<div class="flex items-center gap-2">
      <span class="text-xs font-bold w-8" style="color:${tc}">${col[0].toUpperCase()+col.slice(1)}</span>
      <div class="flex items-center gap-1 flex-wrap">
        ${[0,1,2].map(i=>dot('10px','10px',`background:${bc};`        ,p.S,i)).join('')}
        <span class="w-1"></span>
        ${[0,1,2].map(i=>dot('16px','16px',`border:3px solid ${bc};background:transparent;`,p.M,i)).join('')}
        <span class="w-1"></span>
        ${[0,1,2].map(i=>dot('22px','22px',`border:4px solid ${bc};background:transparent;`,p.L,i)).join('')}
      </div>
      <span class="ml-auto text-[10px] text-[#8f6f6d]">${p.S}·${p.M}·${p.L}</span>
    </div>`;
  }).join('');
}

// ── Status bar ───────────────────────────────────────────────────
function renderStatus() {
  if (!state) return;
  const dot = document.getElementById('sdot');
  const txt = document.getElementById('stxt');
  const spn = document.getElementById('spinner');
  dot.classList.add('pdot'); spn.classList.add('hidden');

  if (state.game_over) {
    dot.classList.remove('pdot');
    dot.style.background = state.winner === 0 ? '#8f6f6d' : state.winner === 1 ? RED : BLUE;
    txt.textContent = state.winner === 0 ? 'Draw!' :
      (state.winner === state.human_player && isVsBot()) ? '🎉 You win!' :
      isVsBot() ? `${botLabel()} wins!` :
      `${state.winner === 1 ? 'Red' : 'Blue'} wins!`;
  } else if (isVsBot() && state.current_player === state.ai_player) {
    dot.style.background = state.ai_player === 1 ? RED : BLUE;
    spn.classList.remove('hidden'); dot.classList.remove('pdot');
    txt.textContent = `${botLabel()} thinking…`;
  } else {
    const whose = !isVsBot()
      ? (state.current_player === 1 ? 'Red' : 'Blue')
      : (state.human_player === 1 ? 'Red' : 'Blue');
    dot.style.background = state.current_player === 1 ? RED : BLUE;
    txt.textContent = `${whose}'s turn`;
  }
}

// ── History ──────────────────────────────────────────────────────
function renderHistory() {
  if (!state) return;
  document.getElementById('hist').innerHTML = [...state.history].reverse().map(m => {
    const col = m.player === 1 ? RED : BLUE;
    const who = !isVsBot() ? (m.player === 1 ? 'Red' : 'Blue') :
                m.player === state.human_player ? 'You' :
                botLabel();
    return `<div class="flex items-center gap-2 text-xs bg-[#f3f4f5] rounded-lg px-3 py-1.5">
      <span class="w-2 h-2 rounded-full flex-shrink-0" style="background:${col}"></span>
      <span class="font-medium">${who}</span>
      <span class="text-[#8f6f6d]">${SZNAME[m.size]} → ${POS[m.row][m.col]}</span>
    </div>`;
  }).join('');
}

// ── Scores + Series ──────────────────────────────────────────────
function renderScores() {
  if (!state) return;
  document.getElementById('sc-red').textContent  = state.scores['1']  ?? 0;
  document.getElementById('sc-blue').textContent = state.scores['2'] ?? 0;

  const seriesBar = document.getElementById('series-bar');
  const rstBtn    = document.getElementById('reset-series-btn');
  const ovRst     = document.getElementById('ov-series-reset');
  const seriesLbl = document.getElementById('sw-label');

  if (state.series_n > 1 && isVsBot()) {
    seriesBar.classList.remove('hidden'); seriesBar.classList.add('flex');
    rstBtn.classList.remove('hidden');
    ovRst.classList.remove('hidden'); ovRst.classList.add('flex');
    document.getElementById('sw-red').textContent  = state.series_wins['1']  ?? 0;
    document.getElementById('sw-blue').textContent = state.series_wins['2'] ?? 0;
    seriesLbl.textContent = `(Bo${state.series_n})`;
  } else {
    seriesBar.classList.add('hidden');
    rstBtn.classList.add('hidden');
    ovRst.classList.add('hidden');
  }
}

// ── Overlay ──────────────────────────────────────────────────────
function showOverlay() {
  if (!state?.game_over) return;
  const emoji = document.getElementById('ov-emoji');
  const title = document.getElementById('ov-title');
  const sub   = document.getElementById('ov-sub');
  const ser   = document.getElementById('ov-series');

  if (state.winner === 0) {
    emoji.textContent='🤝'; title.textContent='Draw!'; title.style.color='#5b403e';
    sub.textContent='No legal moves remaining.';
  } else if (isVsBot()) {
    if (state.winner === state.human_player) {
      emoji.textContent='🎉'; title.textContent='You Win!'; title.style.color='#006b2d';
      sub.textContent = `You beat the ${botLabel()} agent!`;
    } else {
      emoji.textContent='🤖'; title.textContent = `${botLabel()} Wins`;
      title.style.color='#b61722';
      const subs = {
        optimal: 'The hardcoded agent follows the perfect anti-diagonal strategy.',
        minimax: 'Minimax + opening book: anti-diagonal plan then alpha-beta search.',
        mcts:    'MCTS ran 1000+ simulations to find the best move.',
        mcts_g:  'Guided MCTS used OptimalAgent rollouts for domain-aware search.',
        pmcts:   'Parallel MCTS ran 500K+ simulations across 16 CPU cores.',
        neural:  'Neural MCTS combined tree search with GPU Q-value evaluation.',
        ai:      'The DQN agent has been training hard. Try again!',
      };
      sub.textContent = subs[currentMode] || `${botLabel()} wins!`;
    }
  } else {
    const col = state.winner === 1 ? 'Red' : 'Blue';
    emoji.textContent = state.winner === 1 ? '🔴' : '🔵';
    title.textContent = `${col} Wins!`;
    title.style.color = state.winner === 1 ? '#b61722' : '#0058be';
    sub.textContent = 'Great game!';
  }

  // Series progress
  if (state.series_n > 1 && isVsBot()) {
    const rw = state.series_wins['1'] ?? 0, bw = state.series_wins['2'] ?? 0;
    const t  = state.series_target;
    const seriesWinner = rw >= t ? 'Red' : bw >= t ? 'Blue' : null;
    ser.textContent = seriesWinner
      ? `${seriesWinner} wins the Best of ${state.series_n} series!`
      : `Series: Red ${rw} – ${bw} Blue (first to ${t})`;
    ser.style.color = seriesWinner === 'Red' ? '#b61722' : seriesWinner === 'Blue' ? '#0058be' : '#006b2d';
  } else { ser.textContent = ''; }

  document.getElementById('overlay').classList.add('show');
}

// ── Full render ──────────────────────────────────────────────────
function render() {
  renderBoard();
  renderInv();
  renderStatus();
  renderHistory();
  renderScores();
  if (state?.game_over) setTimeout(showOverlay, 600);
}

// ── Cell click ───────────────────────────────────────────────────
async function onCellClick(r, c) {
  if (!state || state.game_over) return;
  if (isVsBot() && state.current_player !== state.human_player) return;
  if (!state.legal.includes(`${r},${c},${selSize}`)) {
    // Cell has no room for the selected size — show which sizes ARE legal
    const available = [1,2,3].filter(s => state.legal.includes(`${r},${c},${s}`));
    if (available.length > 0) {
      const names = available.map(s => ['S','M','L'][s-1]).join('/');
      document.getElementById('stxt').textContent = `Select ${names} for that cell`;
    }
    return;
  }

  document.getElementById('overlay').classList.remove('show');

  // Show bot-thinking immediately (before fetch returns)
  if (isVsBot()) {
    document.getElementById('stxt').textContent = `${botLabel()} thinking…`;
    document.getElementById('spinner').classList.remove('hidden');
    document.getElementById('sdot').classList.remove('pdot');
  }

  try {
    const res = await fetch('/api/move', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({row: r, col: c, size: selSize})
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state = await res.json();
    if (state.error) { console.warn('Move error:', state.error); }
  } catch(e) {
    console.error('Move failed:', e);
    document.getElementById('stxt').textContent = 'Error — try again';
    document.getElementById('spinner').classList.add('hidden');
    return;
  }
  render();
}

// ── Reset ────────────────────────────────────────────────────────
async function resetGame(resetSeries) {
  document.getElementById('overlay').classList.remove('show');
  const res = await fetch('/api/reset', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      ai_player:    asBlue ? 1 : 2,
      mode:         currentMode,
      series_n:     currentSeries,
      reset_series: !!resetSeries,
    })
  });
  state = await res.json();
  render();
}

// ── Toggle helpers ───────────────────────────────────────────────
function toggleHeat() {
  showHeat = !showHeat;
  document.getElementById('tog-heat').classList.toggle('on', showHeat);
  renderBoard();
}
function toggleHint() {
  showHint = !showHint;
  document.getElementById('tog-hint').classList.toggle('on', showHint);
  renderBoard();
}
function toggleSide() {
  asBlue = !asBlue;
  document.getElementById('tog-side').classList.toggle('on', asBlue);
  resetGame(false);
}

// ── Keyboard ─────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if      (e.key==='1') setSize(1);
  else if (e.key==='2') setSize(2);
  else if (e.key==='3') setSize(3);
  else if (e.key==='r'||e.key==='R') resetGame(false);
  else if (e.key==='Escape') document.getElementById('overlay').classList.remove('show');
});

// ── Boot ─────────────────────────────────────────────────────────
fetch('/api/state').then(r=>r.json()).then(s => { state = s; render(); });
</script>
</body>
</html>"""


def main():
    global _agent, _neural, _pmcts, _ckpt_episodes, _ckpt_epsilon

    parser = argparse.ArgumentParser(description="TicTacPro Web Interface")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to DQN checkpoint (optional — other modes work without one)")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint: try explicit path, then common defaults
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        for candidate in ("checkpoints_spark/final.pt",
                          "checkpoints_normalized/latest.pt",
                          "checkpoints/final.pt"):
            if os.path.exists(candidate):
                ckpt_path = candidate
                break

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading agent from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        _ckpt_episodes = int(ckpt.get("episode_count", 0))
        _ckpt_epsilon  = float(ckpt.get("epsilon", 0.0))
        if "policy_net" in ckpt:
            # buffer_size=1 avoids allocating a 4.88GB GPU buffer (10M default) that
            # would compete with any concurrent training run for VRAM and cause a 30s
            # startup delay.  Inference-only use needs zero buffer entries.
            _agent = GPUDQNAgent(device=device, compile_model=False, buffer_size=1)
        else:
            _agent = DQNAgent(device=device)
        _agent.load(ckpt_path)
        _agent.epsilon = 0.0

        # Warm up torch.compile — first inference is slow (JIT compilation)
        print("Warming up model (first inference)...")
        _warmup_game = TicTacPro()
        _agent.get_action(_warmup_game, Player.RED, epsilon=0.0)

        # NeuralMCTS backed by the loaded DQN network
        _major  = torch.cuda.get_device_properties(0).major if device.type == "cuda" else 0
        amp_dt  = torch.bfloat16 if _major >= 9 else torch.float16
        raw_net = getattr(_agent.policy_net, "_orig_mod", _agent.policy_net)
        _neural = NeuralMCTSAgent(
            policy_net      = raw_net,
            device          = device,
            time_limit      = 2.0,
            max_simulations = 10_000_000,
            use_amp         = True,
            amp_dtype       = amp_dt,
        )
        print(f"DQN agent ready on {device}  (episodes={_ckpt_episodes:,}, ε={_ckpt_epsilon:.3f})")
    else:
        if ckpt_path:
            print(f"WARNING: checkpoint not found: {ckpt_path}")
        print("No DQN checkpoint loaded — 'vs AI' and 'vs Neural MCTS' modes disabled.")
        print("Other modes (Minimax, MCTS, Parallel MCTS, Optimal, Human) are fully available.")

    import multiprocessing as _mp
    n_workers = min(16, max(1, _mp.cpu_count() - 2))
    _pmcts = ParallelMCTSAgent(n_workers=n_workers, time_limit=2.0, max_simulations=10_000_000)
    print(f"Parallel MCTS ready ({n_workers} workers)")
    print(f"Open http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
