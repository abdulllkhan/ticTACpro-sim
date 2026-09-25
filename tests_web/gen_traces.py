"""Generate Python-engine game traces for the JS parity harness.

Run:  PYTHONPATH=<repo root> python3 tests_web/gen_traces.py

Simulates 500 uniform-random games with random.Random(42) and writes
tests_web/traces.json (compact). Per ply: the sorted legal-move list as
"r,c,s" strings (s converted to 0-INDEXED to match JS SMALL=0..LARGE=2),
the chosen move, and the post-move state: game_over, winner,
current_player (mover stays current on game end), and the 27 board bytes
in board.tobytes() order (r*9 + c*3 + s, s already 0-indexed).
"""
import json
import os
import random

from game.tictacpro import TicTacPro, PieceSize, Player

N_GAMES = 500
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces.json")


def play_game(rng):
    game = TicTacPro()
    plies = []
    while not game.game_over:
        # (r, c, s) with s 0-indexed; sorted for order-independent comparison.
        legal = sorted((r, c, int(s) - 1) for r, c, s in game.get_legal_moves())
        r, c, s = legal[rng.randrange(len(legal))]
        assert game.make_move(r, c, PieceSize(s + 1))
        plies.append({
            "legal": [f"{lr},{lc},{ls}" for lr, lc, ls in legal],
            "move": [r, c, s],
            "game_over": game.game_over,
            "winner": int(game.winner),
            "current_player": int(game.current_player),
            "board": list(game.board.tobytes()),
        })
    return {"plies": plies, "final_move_count": len(plies)}


def main():
    rng = random.Random(42)
    games = [play_game(rng) for _ in range(N_GAMES)]
    total = sum(g["final_move_count"] for g in games)
    draws = sum(1 for g in games if g["plies"][-1]["winner"] == 0)
    with open(OUT, "w") as f:
        json.dump({"games": games}, f, separators=(",", ":"))
    print(f"Wrote {len(games)} games ({total} plies, {draws} draws) to {OUT}")


if __name__ == "__main__":
    main()
