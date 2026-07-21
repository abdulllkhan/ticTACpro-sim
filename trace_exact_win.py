"""
Trace the principal variation of TicTacPro under exact minimax.
Both players play optimally (no heuristic, pure terminal detection).
Extracts RED's winning sequence from the TC-L opening.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.tictacpro import TicTacPro, Player, PieceSize
from rl.minimax_agent import _push, _pop, _order_moves, _hash, WIN, LOSS, DRAW

EXACT = 0; LOWER = 1; UPPER = 2

POS = {(0,0):'TL',(0,1):'TC',(0,2):'TR',
       (1,0):'ML',(1,1):'CC',(1,2):'MR',
       (2,0):'BL',(2,1):'BC',(2,2):'BR'}
SZN = {PieceSize.SMALL:'S', PieceSize.MEDIUM:'M', PieceSize.LARGE:'L'}

def fmt(mv):
    r, c, sz = mv
    return f"{POS[(r,c)]}-{SZN[sz]}"


def negamax_pv(g, alpha, beta, tt, nodes):
    """Returns (score, best_move)."""
    nodes[0] += 1

    if g.game_over:
        if g.winner != Player.NONE:
            return LOSS, None
        return DRAW, None

    moves = g.get_legal_moves()
    if not moves:
        return DRAW, None

    h = _hash(g)
    tt_move = None
    if h in tt:
        ts, tf, tt_move = tt[h]
        if tf == EXACT:
            return ts, tt_move
        if tf == LOWER: alpha = max(alpha, ts)
        if tf == UPPER: beta  = min(beta,  ts)
        if alpha >= beta: return ts, tt_move

    moves_ord = _order_moves(g, moves, g.current_player, tt_move, None)
    orig_alpha = alpha
    best       = LOSS - 1
    best_mv    = None

    for mv in moves_ord:
        tok   = _push(g, *mv)
        score, _ = negamax_pv(g, -beta, -alpha, tt, nodes)
        score = -score
        _pop(g, tok)
        if score > best:
            best   = score
            best_mv = mv
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    flag = EXACT if best > orig_alpha and best < beta else \
           LOWER if best >= beta else UPPER
    tt[h] = (best, flag, best_mv)
    return best, best_mv


def trace(first_move=(0, 1, PieceSize.LARGE)):
    g = TicTacPro()
    tt = {}
    nodes = [0]
    red_moves = []
    blue_moves = []

    # Force first move
    r, c, sz = first_move
    g.make_move(r, c, sz)
    red_moves.append(first_move)
    print(f"[pp=0] RED: {fmt(first_move)} (forced)")

    pp = 1
    t0 = time.perf_counter()
    while not g.game_over:
        player = g.current_player
        t1 = time.perf_counter()
        score, mv = negamax_pv(g, LOSS - 1, WIN + 1, tt, nodes)
        elapsed = time.perf_counter() - t1

        label = "WIN" if score >= WIN//2 else "LOSS" if score <= LOSS//2 else "DRAW"
        side = "RED" if player == Player.RED else "BLUE"
        print(f"[pp={pp}] {side}: {fmt(mv)}  score={label}  nodes={nodes[0]:,}  t={elapsed:.1f}s")

        if player == Player.RED:
            red_moves.append(mv)
        else:
            blue_moves.append(mv)

        g.make_move(*mv)
        pp += 1

    total = time.perf_counter() - t0
    result = 'RED wins' if g.winner == Player.RED else \
             'BLUE wins' if g.winner == Player.BLUE else 'Draw'
    print(f"\nResult: {result}  ({total:.1f}s total, {nodes[0]:,} nodes)")

    print(f"\nRED moves: {[fmt(m) for m in red_moves]}")
    print(f"BLUE moves: {[fmt(m) for m in blue_moves]}")

    # Print as book plan
    print("\n_RED_BOOK_PLAN (exact optimal):")
    for i, mv in enumerate(red_moves):
        r, c, sz = mv
        sz_name = f"PieceSize.{sz.name}"
        print(f"    ({r}, {c}, {sz_name}),  # {fmt(mv)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "mcts_line":
        # Trace after MCTS deviation: TC-L + CC-S + CC-M + BL-S(MCTS) + TR-S(forced)
        # i.e., RED gets TC-L and TR-S on board; MCTS has CC-S and BL-S
        g2 = TicTacPro()
        setup = [
            (0, 1, PieceSize.LARGE),   # RED TC-L
            (1, 1, PieceSize.SMALL),   # BLUE CC-S
            (1, 1, PieceSize.MEDIUM),  # RED CC-M
            (2, 0, PieceSize.SMALL),   # BLUE BL-S (MCTS deviation)
            (0, 2, PieceSize.SMALL),   # RED TR-S (forced block)
        ]
        print("Setup: TC-L(RED) + CC-S(BLUE) + CC-M(RED) + BL-S(BLUE) + TR-S(RED forced)")
        for mv in setup:
            g2.make_move(*mv)
        # Now trace from BLUE's pp=5 turn
        tt2 = {}
        nodes2 = [0]
        pp = 5
        t0 = time.perf_counter()
        while not g2.game_over:
            player = g2.current_player
            t1 = time.perf_counter()
            score, mv = negamax_pv(g2, LOSS - 1, WIN + 1, tt2, nodes2)
            elapsed = time.perf_counter() - t1
            label = "WIN" if score >= WIN//2 else "LOSS" if score <= LOSS//2 else "DRAW"
            side = "RED" if player == Player.RED else "BLUE"
            print(f"[pp={pp}] {side}: {fmt(mv)}  score={label}  nodes={nodes2[0]:,}  t={elapsed:.1f}s")
            g2.make_move(*mv)
            pp += 1
        total = time.perf_counter() - t0
        result = 'RED wins' if g2.winner == Player.RED else 'BLUE wins' if g2.winner == Player.BLUE else 'Draw'
        print(f"\nResult: {result}  ({total:.1f}s total, {nodes2[0]:,} nodes)")
    elif len(sys.argv) > 1 and sys.argv[1] == "actual_state":
        # Trace from actual MCTS game state at pp=8 (after observed forced blocks)
        # Board: TC-L(R), CC-S(B), CC-M(R), BL-S(B), TR-S(R), BL-M(B), BL-L(R), CC-L(B)
        g3 = TicTacPro()
        setup = [
            (0, 1, PieceSize.LARGE),   # RED TC-L
            (1, 1, PieceSize.SMALL),   # BLUE CC-S
            (1, 1, PieceSize.MEDIUM),  # RED CC-M
            (2, 0, PieceSize.SMALL),   # BLUE BL-S
            (0, 2, PieceSize.SMALL),   # RED TR-S (forced block)
            (2, 0, PieceSize.MEDIUM),  # BLUE BL-M (builds BL bullseye)
            (2, 0, PieceSize.LARGE),   # RED BL-L (forced block)
            (1, 1, PieceSize.LARGE),   # BLUE CC-L
        ]
        print("Setup: TC-L(R)+CC-S(B)+CC-M(R)+BL-S(B)+TR-S(R)+BL-M(B)+BL-L(R)+CC-L(B)")
        for mv in setup:
            g3.make_move(*mv)
        tt3 = {}
        nodes3 = [0]
        pp = 8
        t0 = time.perf_counter()
        while not g3.game_over:
            player = g3.current_player
            t1 = time.perf_counter()
            score, mv = negamax_pv(g3, LOSS - 1, WIN + 1, tt3, nodes3)
            elapsed = time.perf_counter() - t1
            label = "WIN" if score >= WIN//2 else "LOSS" if score <= LOSS//2 else "DRAW"
            side = "RED" if player == Player.RED else "BLUE"
            print(f"[pp={pp}] {side}: {fmt(mv)}  score={label}  nodes={nodes3[0]:,}  t={elapsed:.1f}s")
            g3.make_move(*mv)
            pp += 1
        total = time.perf_counter() - t0
        result = 'RED wins' if g3.winner == Player.RED else 'BLUE wins' if g3.winner == Player.BLUE else 'Draw'
        print(f"\nResult: {result}  ({total:.1f}s, {nodes3[0]:,} nodes)")
    else:
        trace()
