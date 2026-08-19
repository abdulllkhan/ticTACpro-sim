"""
Round-robin tournament for TicTacPro agents.

Agents:
  random    — uniform random legal moves
  optimal   — hardcoded rule-based (win→block→anti-diagonal S)
  dqn       — trained GPUDQNAgent (greedy, ε=0)
  minimax   — alpha-beta negamax (iterative deepening, 1s/move)
  mcts      — pure MCTS UCT (1s/move)
  neural    — MCTS + DQN leaf evaluation (1s/move, GPU)

Every ordered pair plays N_GAMES games (agent_A plays RED, agent_B plays BLUE),
then swaps. Results saved to tournament_results.json.

Usage:
  python3 tournament.py [--games 5] [--time 1.0] [--no-slow]
"""

import sys, os, time, json, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from game.tictacpro import TicTacPro, Player, PieceSize
from game.optimal_agent import OptimalAgent
from game.bullseye_agent import BullseyeAgent
from rl.agent import GPUDQNAgent
from rl.minimax_agent import MinimaxAgent
from rl.mcts_agent import MCTSAgent
from rl.neural_mcts_agent import NeuralMCTSAgent
from rl.parallel_mcts_agent import ParallelMCTSAgent


# ── Random agent ──────────────────────────────────────────────────────────────
class RandomAgent:
    def get_action(self, game, player=None, epsilon=0.0):
        moves = game.get_legal_moves()
        return random.choice(moves) if moves else None


# ── Play one game ─────────────────────────────────────────────────────────────
def play_game(red_agent, blue_agent, max_moves: int = 100) -> Player:
    """Play one game; return winner (Player.RED / Player.BLUE / Player.NONE)."""
    game = TicTacPro()
    # Reset tree-reuse caches so stale nodes from the previous game are not
    # incorrectly matched against this game's moves.
    for agent in (red_agent, blue_agent):
        if hasattr(agent, "reset"):
            agent.reset()
    agents = {Player.RED: red_agent, Player.BLUE: blue_agent}
    for _ in range(max_moves):
        if game.game_over:
            break
        agent  = agents[game.current_player]
        action = agent.get_action(game, game.current_player, epsilon=0.0)
        if action is None:
            break
        game.make_move(*action)
    return game.winner


# ── Match: N games each direction ─────────────────────────────────────────────
def run_match(name_a, agent_a, name_b, agent_b, n: int, verbose: bool = True):
    wins = {name_a: 0, name_b: 0, "draw": 0}
    for i in range(n):
        w = play_game(agent_a, agent_b)
        if w == Player.RED:   wins[name_a] += 1
        elif w == Player.BLUE: wins[name_b] += 1
        else:                  wins["draw"]  += 1
        if verbose:
            label = name_a if w == Player.RED else (name_b if w == Player.BLUE else "draw")
            print(f"  [{i+1}/{n}] RED={name_a:<12s} BLUE={name_b:<12s}  →  {label}")

    for i in range(n):
        w = play_game(agent_b, agent_a)
        if w == Player.RED:   wins[name_b] += 1
        elif w == Player.BLUE: wins[name_a] += 1
        else:                  wins["draw"]  += 1
        if verbose:
            label = name_b if w == Player.RED else (name_a if w == Player.BLUE else "draw")
            print(f"  [{n+i+1}/{2*n}] RED={name_b:<12s} BLUE={name_a:<12s}  →  {label}")

    return wins


# ── Build agents ──────────────────────────────────────────────────────────────
def _load_net(checkpoint: str, device):
    """Load a DQN policy_net from a checkpoint file."""
    # buffer_size=1: tournament only needs inference, not a 4.88GB replay buffer
    agent = GPUDQNAgent(compile_model=False, device=device, buffer_size=1)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    net_state = ck.get("policy_net") or ck.get("model_state_dict") or ck
    raw_net = getattr(agent.policy_net, "_orig_mod", agent.policy_net)
    raw_net.load_state_dict(net_state)
    agent.policy_net.eval()
    agent.epsilon = 0.0
    return agent, raw_net


def build_agents(time_limit: float, checkpoint: str, skip_slow: bool,
                 checkpoint2: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agents = {}

    agents["random"]   = RandomAgent()
    agents["optimal"]  = OptimalAgent()
    agents["bullseye"] = BullseyeAgent()  # §334: sequential bullseye fork strategy

    # DQN (original biased checkpoint)
    print("Loading DQN checkpoint…")
    dqn, raw_net = _load_net(checkpoint, device)
    agents["dqn"] = dqn
    print(f"  DQN ready  ({checkpoint})")

    if not skip_slow:
        agents["minimax"] = MinimaxAgent(time_limit=time_limit, max_depth=18)

        agents["mcts"]    = MCTSAgent(time_limit=time_limit, max_simulations=10_000_000)

        # Guided MCTS: use OptimalAgent as rollout policy instead of random
        agents["mcts_g"]  = MCTSAgent(
            time_limit      = time_limit,
            max_simulations = 10_000_000,
            rollout_agent   = OptimalAgent(),
        )

        # §334: BullseyeAgent-guided MCTS (uses sequential bullseye fork for rollouts)
        agents["mcts_b"] = MCTSAgent(
            time_limit      = time_limit,
            max_simulations = 10_000_000,
            rollout_agent   = BullseyeAgent(),
        )

        _major = torch.cuda.get_device_properties(0).major if device.type == "cuda" else 0
        amp_dt = torch.bfloat16 if _major >= 9 else torch.float16

        # NeuralMCTS with original biased DQN
        agents["neural"] = NeuralMCTSAgent(
            policy_net      = raw_net,
            device          = device,
            time_limit      = time_limit,
            max_simulations = 10_000_000,
            use_amp         = True,
            amp_dtype       = amp_dt,
        )

        # §154: NeuralMCTS with DQN leaf evaluation instead of rollout
        agents["neural_dqn_eval"] = NeuralMCTSAgent(
            policy_net      = raw_net,
            device          = device,
            time_limit      = time_limit,
            max_simulations = 10_000_000,
            use_amp         = True,
            amp_dtype       = amp_dt,
            use_dqn_eval    = True,
        )

        # NeuralMCTS with normalized perspective DQN (if provided)
        if checkpoint2 and os.path.exists(checkpoint2):
            print(f"Loading normalized DQN checkpoint…")
            _, raw_net2 = _load_net(checkpoint2, device)
            agents["neural2"] = NeuralMCTSAgent(
                policy_net      = raw_net2,
                device          = device,
                time_limit      = time_limit,
                max_simulations = 10_000_000,
                use_amp         = True,
                amp_dtype       = amp_dt,
            )
            print(f"  NeuralMCTS (normalized) ready  ({checkpoint2})")
        else:
            if checkpoint2:
                print(f"  WARNING: checkpoint2 not found: {checkpoint2} — skipping neural2")

        # Parallel MCTS: 16 workers, ~500K sims per 2-second move
        import multiprocessing as _mp
        n_workers = min(16, max(1, _mp.cpu_count() - 2))
        agents["pmcts"] = ParallelMCTSAgent(
            n_workers       = n_workers,
            time_limit      = time_limit,
            max_simulations = 10_000_000,
        )
        print(f"  Minimax / MCTS / GuidedMCTS / NeuralMCTS / ParallelMCTS ({n_workers} workers) ready")

    return agents, device


# ── Tournament ────────────────────────────────────────────────────────────────
def run_tournament(agents: dict, n_games: int):
    names  = list(agents.keys())
    pairs  = [(a, b) for i, a in enumerate(names) for b in names[i+1:]]
    totals = {n: {"wins": 0, "losses": 0, "draws": 0, "games": 0} for n in names}
    records = []

    print(f"\nTournament: {len(pairs)} match-ups, {n_games*2} games each\n")
    t0 = time.perf_counter()

    for a, b in pairs:
        print(f"── {a}  vs  {b} ──────────")
        t1 = time.perf_counter()
        wins = run_match(a, agents[a], b, agents[b], n_games)
        elapsed = time.perf_counter() - t1
        record  = {"a": a, "b": b, "wins": wins, "elapsed_s": round(elapsed, 1)}
        records.append(record)

        for name in (a, b):
            totals[name]["wins"]   += wins.get(name, 0)
            totals[name]["draws"]  += wins["draw"]
            totals[name]["games"]  += n_games * 2
            totals[name]["losses"] += n_games * 2 - wins.get(name, 0) - wins["draw"]

        print(f"  → {a}: {wins[a]}W {wins['draw']}D {wins[b]}L  "
              f"| {b}: {wins[b]}W {wins['draw']}D {wins[a]}L  "
              f"({elapsed:.1f}s)\n")

    total_elapsed = time.perf_counter() - t0

    # Rankings
    print("\n════ RANKINGS ═══════════════════════════════")
    print(f"{'Agent':<12s} {'W':>5s} {'D':>5s} {'L':>5s} {'G':>5s} {'WR%':>7s}")
    print("─" * 45)
    ranked = sorted(totals.items(),
                    key=lambda x: x[1]["wins"] / max(x[1]["games"], 1), reverse=True)
    for name, t in ranked:
        wr = 100 * t["wins"] / max(t["games"], 1)
        print(f"{name:<12s} {t['wins']:>5d} {t['draws']:>5d} {t['losses']:>5d} "
              f"{t['games']:>5d} {wr:>6.1f}%")

    print(f"\nTotal time: {total_elapsed:.1f}s")

    return {
        "timestamp":    time.strftime("%Y-%m-%d %H:%M"),
        "n_games_each": n_games,
        "matches":      records,
        "totals":       totals,
        "total_time_s": round(total_elapsed, 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games",    type=int,   default=5,
                        help="Games per direction per pair (default 5)")
    parser.add_argument("--time",     type=float, default=1.0,
                        help="Time limit per move for tree-search agents (default 1.0s)")
    parser.add_argument("--checkpoint", default="checkpoints_spark/final.pt",
                        help="DQN checkpoint path (original biased model)")
    parser.add_argument("--checkpoint2", default=None,
                        help="Optional 2nd DQN checkpoint (normalized perspective model → neural2 agent)")
    parser.add_argument("--no-slow",  action="store_true",
                        help="Skip Minimax/MCTS/NeuralMCTS (fast agents only)")
    args = parser.parse_args()

    agents, device = build_agents(args.time, args.checkpoint, args.no_slow,
                                  checkpoint2=args.checkpoint2)

    results = run_tournament(agents, args.games)

    out_path = "tournament_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")
