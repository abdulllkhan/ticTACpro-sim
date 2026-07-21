"""
Offline game solver for TicTacPro.

Uses parallel alpha-beta search (20 CPUs) to determine the game-theoretic
value from the starting position: WIN for RED, WIN for BLUE, or DRAW.

Run: python3 solve_game.py [--time 300] [--workers 20]

Results are saved to solve_results.json for use in FINDINGS.md.
"""

import sys, os, time, json, argparse, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.tictacpro import TicTacPro, Player, PieceSize
from rl.minimax_agent import MinimaxAgent, _push, _pop, \
                             _order_moves, _hash, _evaluate, WIN, LOSS, DRAW


# ── Per-move deep search (runs in worker process) ─────────────────────────────
def _search_root_move(args):
    """Worker: search one root move to max depth within time_limit seconds."""
    r, c, sz_int, time_limit = args
    g = TicTacPro()
    sz = PieceSize(sz_int)
    # Make the root move (RED's first move)
    g.make_move(r, c, sz)

    # Run minimax for BLUE (minimiser)
    agent = MinimaxAgent(time_limit=time_limit, max_depth=18, use_book=False)
    # We want the value FROM BLUE's perspective after RED played (r,c,sz)
    # i.e., we maximise for BLUE
    action, (depth, score, nodes, hits) = agent.get_info(g, player=Player.BLUE)

    return {
        "move":  (r, c, sz_int),
        "depth": depth,
        "score": score,          # from BLUE's perspective → negate for RED
        "nodes": nodes,
        "hits":  hits,
    }


def solve(time_per_move: float = 60.0, workers: int = 20):
    """
    Evaluate every RED first move in parallel.
    Returns the best first move and the game-theoretic assessment.
    """
    g = TicTacPro()
    first_moves = g.get_legal_moves()
    print(f"Solving {len(first_moves)} root moves with {workers} workers, "
          f"{time_per_move:.0f}s/move budget")

    args = [(r, c, int(sz), time_per_move) for r, c, sz in first_moves]

    t0 = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        results = pool.map(_search_root_move, args)
    elapsed = time.perf_counter() - t0

    POS = ['TL','TC','TR','ML','CC','MR','BL','BC','BR']
    SZN = {1:'S', 2:'M', 3:'L'}

    print(f"\nResults ({elapsed:.1f}s total):")
    print(f"{'Move':8s} {'Depth':>6s} {'Score(RED)':>12s} {'Nodes':>10s}")
    print("-" * 50)

    best_move   = None
    best_score  = LOSS - 1

    for res in sorted(results, key=lambda x: -(-x['score'])):  # sort by RED score desc
        r, c, sz_int = res['move']
        red_score = -res['score']   # negate BLUE's score → RED's score
        pos = POS[r*3+c]
        label = "WIN " if red_score >= WIN//2 else \
                "LOSS" if red_score <= LOSS//2 else \
                f"{red_score:+5d}"
        print(f"{pos}-{SZN[sz_int]:4s}  depth={res['depth']:3d}  "
              f"score={label}  nodes={res['nodes']:,}")
        if red_score > best_score:
            best_score = red_score
            best_move  = (r, c, sz_int)

    print(f"\nBest first move: {POS[best_move[0]*3+best_move[1]]}-{SZN[best_move[2]]}  "
          f"score={best_score:+d}")

    outcome = "RED_WIN" if best_score >= WIN//2 else \
              "BLUE_WIN" if best_score <= LOSS//2 else "UNCERTAIN"
    print(f"Game-theoretic assessment: {outcome}")

    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "time_per_move_s": time_per_move,
        "workers": workers,
        "total_time_s": elapsed,
        "results": results,
        "best_first_move": best_move,
        "best_score_red": best_score,
        "outcome": outcome,
    }
    with open("solve_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → solve_results.json")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time",    type=float, default=60.0,
                        help="Seconds per root move (default 60)")
    parser.add_argument("--workers", type=int,   default=min(20, mp.cpu_count()),
                        help="Parallel workers")
    args = parser.parse_args()
    solve(args.time, args.workers)
