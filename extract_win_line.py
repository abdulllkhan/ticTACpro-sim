"""
Extract the forced win sequence for RED starting with TC-L.
Uses deep minimax (no book, no time limit) to find the explicit win path.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.tictacpro import TicTacPro, Player, PieceSize
from rl.minimax_agent import MinimaxAgent, _push, _pop, _order_moves, _hash, WIN, LOSS, DRAW

POS = {(0,0):'TL',(0,1):'TC',(0,2):'TR',
       (1,0):'ML',(1,1):'CC',(1,2):'MR',
       (2,0):'BL',(2,1):'BC',(2,2):'BR'}
SZN = {PieceSize.SMALL:'S', PieceSize.MEDIUM:'M', PieceSize.LARGE:'L'}

def fmt(mv):
    r, c, sz = mv
    return f"{POS[(r,c)]}-{SZN[sz]}"

def trace_win(time_limit=60.0, max_depth=18):
    g = TicTacPro()
    agent = MinimaxAgent(time_limit=time_limit, max_depth=max_depth, use_book=False)

    # Force TC-L as first move
    g.make_move(0, 1, PieceSize.LARGE)
    print(f"[pp=0] RED: TC-L (book)")

    pp = 1
    while not g.game_over:
        player = g.current_player
        action, (depth, score, nodes, hits) = agent.get_info(g, player)
        label = "WIN" if score >= WIN//2 else "LOSS" if score <= LOSS//2 else f"{score:+d}"
        side = "RED" if player == Player.RED else "BLUE"
        print(f"[pp={pp}] {side}: {fmt(action)}  score={label}  depth={depth}  nodes={nodes:,}")
        g.make_move(*action)
        pp += 1

    print(f"\nResult: {'RED wins' if g.winner == Player.RED else 'BLUE wins' if g.winner == Player.BLUE else 'Draw'}")

if __name__ == "__main__":
    trace_win()
