import sys; sys.path.insert(0,'.')
from game.tictacpro import TicTacPro as TicTacProGame, Player
from rl.minimax_agent import MinimaxAgent
from rl.mcts_agent import MCTSAgent
import time

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40

mm = MinimaxAgent(max_depth=6, time_limit=2.0)
mcts = MCTSAgent(time_limit=2.0)

wins_mm = wins_mcts = draws = 0
t0 = time.time()
for i in range(N):
    g = TicTacProGame()
    red = mm if i < N // 2 else mcts
    blue = mcts if i < N // 2 else mm
    while not g.game_over:
        agent = red if g.current_player == Player.RED else blue
        mv = agent.get_action(g, g.current_player)
        g.make_move(*mv)
    w = g.winner
    if w == Player.NONE or w is None:
        draws += 1
    elif (i < N // 2 and w == Player.RED) or (i >= N // 2 and w == Player.BLUE):
        wins_mm += 1
    else:
        wins_mcts += 1
    if (i + 1) % 10 == 0:
        total = wins_mm + draws + wins_mcts
        wr = 100 * (wins_mm + 0.5 * draws) / total
        print(f'[{i+1}/{N}] {wins_mm}W {draws}D {wins_mcts}L  WR={wr:.1f}%', flush=True)

total = wins_mm + draws + wins_mcts
wr = 100 * (wins_mm + 0.5 * draws) / total
print(f'\nFinal: {wins_mm}W {draws}D {wins_mcts}L  WR={wr:.1f}%')
print(f'Time: {time.time()-t0:.1f}s')
