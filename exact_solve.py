"""
Exact game-tree solver for TicTacPro.

Uses pure negamax with alpha-beta and TT but NO heuristic — only terminal
detection (WIN/LOSS/DRAW from game_over + winner). Searches to game completion.

This gives the true game-theoretic value: RED_WIN, BLUE_WIN, or DRAW.

Run: python3 exact_solve.py [--workers N] [--time T]
"""
import sys, os, time, json, argparse, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.tictacpro import TicTacPro, Player, PieceSize, _would_win_bf, _SWITCH_PLAYER, _SIZES
from rl.minimax_agent import _push, _pop, _order_moves, _hash, WIN, LOSS, DRAW

EXACT = 0; LOWER = 1; UPPER = 2


def _negamax_exact(g, alpha, beta, tt, nodes):
    """Exact negamax — no heuristic, searches to game completion."""
    nodes[0] += 1

    if g.game_over:
        if g.winner != Player.NONE:
            return LOSS   # current player is the loser (_push always switches player)
        return DRAW

    moves = g.get_legal_moves()
    if not moves:
        return DRAW

    h = _hash(g)
    if h in tt:
        ts, tf, td = tt[h]
        if tf == EXACT: return ts
        if tf == LOWER: alpha = max(alpha, ts)
        if tf == UPPER: beta  = min(beta,  ts)
        if alpha >= beta: return ts

    moves_ord = _order_moves(g, moves, g.current_player, None, None)
    orig_alpha = alpha
    best = LOSS - 1

    for mv in moves_ord:
        tok   = _push(g, *mv)
        score = -_negamax_exact(g, -beta, -alpha, tt, nodes)
        _pop(g, tok)
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    flag = EXACT if best > orig_alpha and best < beta else \
           LOWER if best >= beta else UPPER
    tt[h] = (best, flag, 0)
    return best


def _solve_root_move(args):
    r, c, sz_int = args
    g = TicTacPro()
    sz = PieceSize(sz_int)
    g.make_move(r, c, sz)

    tt    = {}
    nodes = [0]
    t0    = time.perf_counter()
    score = -_negamax_exact(g, LOSS - 1, WIN + 1, tt, nodes)
    elapsed = time.perf_counter() - t0

    # score is from RED's perspective (RED made first move)
    return {
        "move": (r, c, sz_int),
        "score_red": score,
        "nodes": nodes[0],
        "elapsed_s": elapsed,
    }


def solve(workers: int = 20):
    g = TicTacPro()
    first_moves = g.get_legal_moves()
    print(f"Exact solve: {len(first_moves)} root moves, {workers} workers")

    args = [(r, c, int(sz)) for r, c, sz in first_moves]

    POS = ['TL','TC','TR','ML','CC','MR','BL','BC','BR']
    SZN = {1:'S', 2:'M', 3:'L'}

    t0 = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        results = pool.map(_solve_root_move, args)
    elapsed = time.perf_counter() - t0

    print(f"\nResults ({elapsed:.1f}s total):")
    print(f"{'Move':8s} {'Score(RED)':>12s} {'Nodes':>12s} {'Time':>8s}")
    print("-" * 50)

    best_score = LOSS - 1
    best_move  = None

    for res in sorted(results, key=lambda x: -x['score_red']):
        r, c, sz_int = res['move']
        sc = res['score_red']
        label = "WIN " if sc >= WIN//2 else "LOSS" if sc <= LOSS//2 else "DRAW"
        pos = POS[r*3+c]
        print(f"{pos}-{SZN[sz_int]:4s}  {label:>12s}  {res['nodes']:>12,}  {res['elapsed_s']:>7.1f}s")
        if sc > best_score:
            best_score = sc
            best_move  = (r, c, sz_int)

    outcome = "RED_WIN" if best_score >= WIN//2 else \
              "BLUE_WIN" if best_score <= LOSS//2 else "DRAW"

    print(f"\nBest first move: {POS[best_move[0]*3+best_move[1]]}-{SZN[best_move[2]]}")
    print(f"Game-theoretic value: {outcome}")

    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "total_time_s": elapsed,
        "workers": workers,
        "results": results,
        "best_first_move": best_move,
        "best_score_red": best_score,
        "outcome": outcome,
    }
    with open("exact_solve_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved → exact_solve_results.json")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(20, mp.cpu_count()))
    args = parser.parse_args()
    solve(args.workers)
