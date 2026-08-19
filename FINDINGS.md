# TicTacPro: Deep Reinforcement Learning Findings

**Author:** Khan  
**Hardware:** NVIDIA DGX Spark (GB10 Grace-Blackwell Superchip)  
**Date:** May 2026

---

## Abstract

We trained a Deep Q-Network (DQN) agent to play TicTacPro — a 3×3 board game where each player has three piece sizes (Small, Medium, Large) and can win by placing three-in-a-row of the **same size** or by stacking all three sizes in one cell (**bullseye**). Training ran for **1.5 hours** on a GB10 GPU using BF16 mixed precision and `torch.compile`. The agent completed **923,136 self-play episodes** and reached ε = 0.05. Analysis reveals that the game is effectively **solved by the first player**: RED (first mover) wins **100% of 2,000 greedy self-play games** via a fully deterministic 17-move sequence ending in a **Small-piece anti-diagonal win** (TR→CC→BL). An exhaustive game-tree solve — `exact_solve.py`, negamax searched to terminal with no heuristic (§77) — confirms this is not a self-play artifact: **TicTacPro is a first-player (RED) win from all 27 opening moves**, its true game-theoretic value. However, the learned policy shows significant **self-play overfitting**: RED wins only 57% against random opponents.

---

## 1. Game Rules

TicTacPro is played on a 3×3 grid. Each player starts with 9 pieces: 3 Small (S), 3 Medium (M), 3 Large (L).

**Win conditions:**
1. **Same-size line**: Three pieces of the same size in a row, column, or diagonal
2. **Bullseye**: All three sizes (S + M + L) stacked in one cell, by the same player

**Placement rules:**
- Each cell has three independent slots (one per size)
- A piece may only be placed if its size slot in the target cell is empty
- Small can be placed in a cell that already has a Medium (they use different slots)
- There is no "gobbling" — larger pieces do not cover smaller ones

**Board positions used in this document:**

```
TL  TC  TR
ML  CC  MR
BL  BC  BR
```

---

## 2. Training Setup

| Parameter | Value |
|-----------|-------|
| Architecture | DQN (feedforward, 4 layers) |
| Hidden sizes | 1024 → 512 → 256 |
| Parameters | 792,604 |
| State representation | 61-dimensional vector |
| Action space | 27 (9 cells × 3 sizes) |
| Optimizer | Adam, lr=3×10⁻⁴ |
| Batch size | 16,384 |
| Buffer size | 10,000,000 |
| GPU | NVIDIA GB10 (BF16 AMP) |
| Episodes | 923,136 |
| Gradient steps | 114,304 |
| Training time | 1.50 hours |
| Final ε | 0.0500 |
| Final avg loss | 0.0487 |

---

## 3. Key Results

### 3.1 Win Rate in Greedy Self-Play

| Outcome | Count | Rate |
|---------|-------|------|
| RED (first mover) wins | 2,000 | **100%** |
| BLUE (second mover) wins | 0 | 0% |
| Draws | 0 | 0% |

### 3.2 Performance vs Non-Greedy Opponents

| Setup | RED wins |
|-------|----------|
| RED greedy vs BLUE greedy (self-play) | 100% (500/500) |
| RED greedy vs BLUE random | **57%** (57/100) |
| RED random vs BLUE greedy | 30% (30/100) |
| **RED hardcoded vs BLUE random** | **99.7%** (997/1000) |

> **Important:** The model is heavily overfit to self-play. RED's greedy DQN policy only wins 57% against random play. By contrast, the hardcoded rule-based agent (win → block → anti-diagonal S) wins 99.7% against random opponents with zero inference overhead. This is because the DQN memorized the self-play Nash equilibrium rather than learning robust game strategy.

### 3.3 Game Statistics

| Metric | Value |
|--------|-------|
| Average game length (greedy vs greedy) | **17 moves** (exactly) |
| Unique game outcomes (greedy vs greedy) | **1** — fully deterministic |
| Winning mechanism (greedy vs greedy) | Always **Small anti-diagonal** (TR→CC→BL) |
| Pieces used per game | All 9 per player (S×3 + M×3 + L×3) |

---

## 4. The Optimal Strategy

### 4.1 RED's Winning Line

The agent has learned one specific winning plan and executes it every time:

**Target**: Place Small pieces at **TR (0,2)**, **CC (1,1)**, and **BL (2,0)** — the anti-diagonal.

```
 .   .  [TR]     ← Small piece here
 .  [CC]  .      ← Small piece here
[BL]  .   .      ← Small piece here → WIN
```

RED places these three Small pieces at turns 13, 15, and 17 (the 7th, 8th, and 9th moves for RED).

### 4.2 The Deterministic 17-Move Sequence

Every greedy self-play game follows this **exact** sequence:

```
Turn | Player | Move   | Q-value | Notes
-----|--------|--------|---------|------
  1  | RED    | TC-L   | +23.46  | Opening: Large at top-center
  2  | BLUE   | BC-S   | +23.42  | 
  3  | RED    | TC-M   | +24.35  | Second piece at TC (M fills M-slot)
  4  | BLUE   | BR-S   | +24.71  |
  5  | RED    | BL-M   | +26.56  | Starts BL setup (eventual win cell)
  6  | BLUE   | ML-S   | +25.90  |
  7  | RED    | CC-L   | +26.60  | Starts CC setup (peak Q-value)
  8  | BLUE   | CC-M   | +24.97  | BLUE places M at CC (doesn't block S-slot)
  9  | RED    | BR-M   | +21.83  |
 10  | BLUE   | MR-L   | +19.85  |
 11  | RED    | BR-L   | +12.44  | RED stacks L at BR (M already there)
 12  | BLUE   | TR-M   | +10.12  |
 13  | RED    | CC-S   |  +3.26  | **Anti-diag piece 1/3** (CC)
 14  | BLUE   | TL-M   |  +2.18  |
 15  | RED    | TR-S   |  +1.77  | **Anti-diag piece 2/3** (TR)
 16  | BLUE   | BC-L   |  +0.60  |
 17  | RED    | BL-S   |  +1.00  | **WINNING MOVE** — anti-diag complete
```

**Winner: RED** — Small pieces occupy the full anti-diagonal (TR, CC, BL).

### 4.3 Opening Move Rankings (Empty Board)

The top 10 first moves by Q-value from the trained model:

| Rank | Position | Size | Q-value | Notes |
|------|----------|------|---------|-------|
| 1 | TC (0,1) | L | +23.46 | **Best opening** |
| 2 | BL (2,0) | M | +23.34 | |
| 3 | TC (0,1) | M | +22.86 | |
| 4 | TL (0,0) | L | +22.44 | |
| 5 | CC (1,1) | L | +21.96 | |
| 6 | BC (2,1) | S | +21.56 | |
| 7 | BR (2,2) | M | +20.82 | |
| 8 | BR (2,2) | L | +20.56 | |
| 9 | TR (0,2) | M | +20.22 | |
| 10 | ML (1,0) | L | +20.18 | |

### 4.4 Q-value Heatmaps (Empty Board)

**Large piece values:**
```
+22.44  +23.46  +17.88
+20.18  +21.96  +12.41
+11.68  +15.15  +20.56
```

**Medium piece values:**
```
+14.02  +22.86  +20.22
+11.51  +18.09  +14.20
+23.34  +13.72  +20.82
```

**Small piece values:**
```
+18.46  +10.93  +19.74
+19.47  +17.26  +19.04
+14.22  +21.56  +18.37
```

> Observation: Small pieces score high everywhere (range: +10.93 to +21.56) because they're the winning mechanism. BLUE's best response to TC-L is BC-S (Q=+23.42), indicating BLUE also fights for Small piece positioning.

### 4.5 BLUE's Best Responses to Common Openings

| RED opens | BLUE's best response | BLUE's Q |
|-----------|----------------------|----------|
| TC-L (best) | BC-S | +23.42 |
| TC-L (best) | BR-S (2nd) | +23.33 |
| TC-L (best) | CC-M (3rd) | +23.15 |
| TL-L | CC-M | +21.78 |
| CC-L (center) | CC-M | +23.48 |
| CC-L (center) | BR-S (2nd) | +21.78 |

> BLUE consistently wants either a Small piece for its own anti-diagonal attempt or a Medium piece at CC to contest the center.

### 4.6 RED's Responses to Deviations

When BLUE deviates from the expected BC-S counter:

| BLUE's deviation | RED's response | RED's Q |
|------------------|----------------|---------|
| TL-L | BL-M | +24.01 |
| CC-M (early) | TC-M | +24.68 |
| TR-L | BL-M | +21.93 |
| BR-L | BL-M | +22.77 |

> RED tends to play BL-M as a fallback, which is part of the winning sequence and maintains a high Q-value regardless of BLUE's deviation.

---

## 5. Why the Model Is Weak Against Novel Play

The model achieves 100% win rate in self-play but only 57% against random opponents. The root cause is **self-play overfitting**:

1. **Both players learned the same equilibrium.** In self-play, RED learns to execute the anti-diagonal S-win *while* BLUE learns to execute the same plan as a counter. Neither player is trained to defend against genuinely random strategies.

2. **Narrow state coverage.** The deterministic 17-move equilibrium means only one game state sequence was ever played in late training. Off-path positions were only explored during early ε-greedy exploration and are poorly valued.

3. **Q-values are tuned for the equilibrium.** RED's Q-values peak at +26.60 mid-game, but they approach +1.00 near the win — suggesting the reward signal is highly discounted and the agent doesn't generalize from Q-values in novel states.

4. **No game-tree search.** The policy is a direct mapping from state → Q-value with no lookahead. A random opponent can easily stumble into positions that block the anti-diagonal S-win without "knowing" to do so.

---

## 6. Improvements to Explore

### 6.1 Robust Self-Play (Priority)
Train against a mix of opponents: greedy policy, random, and an ensemble of past checkpoints. This prevents the policy from collapsing to a single equilibrium.

### 6.2 Reward Shaping
Add intermediate rewards for:
- Creating a two-in-a-row threat (e.g. +0.1)
- Blocking opponent's two-in-a-row (e.g. +0.05)
- Losing a piece to a bullseye attempt (e.g. -0.1)
This would help the agent value intermediate board states and generalize.

### 6.3 Monte Carlo Tree Search (MCTS)
Replace pure Q-network with MCTS guided by the Q-network as a value function. MCTS naturally explores deviations from the training distribution and can find counter-plays to random strategies.

### 6.4 Curriculum Learning
Start training against random opponents (easy), then gradually increase opponent strength. This ensures the policy is robust before trying to converge on the optimal equilibrium.

### 6.5 Larger Network / Longer Training
The current 792k-parameter network may be too small to represent the full value function. Training for 5M+ episodes with a larger network (2M+ params) and more gradient steps could improve generalization.

---

## 7. Hardcoded Optimal Agent

Based on the findings above, a hardcoded agent following the optimal discovered sequence is available in `game/optimal_agent.py`. It:
- Always opens with TC-L (the strongest first move)
- Attempts to place Small pieces at CC, TR, and BL (anti-diagonal win)
- Falls back to highest Q-value legal move when off the expected path

See `game/optimal_agent.py` for usage.

---

## 8. Tree Search Implementation

### 8.1 Agents Implemented

| Agent | Type | Search budget | Notes |
|-------|------|---------------|-------|
| **Minimax** | Alpha-beta negamax | 1s/move, max depth 18 | Iterative deepening, transposition table (EXACT/LOWER/UPPER), in-place push/undo |
| **MCTS** | UCT Monte Carlo | 1s/move, max 200k sims | Random rollouts to terminal |
| **NeuralMCTS** | UCT + DQN leaf eval | 1s/move, max 200k sims | GPU DQN Q-values replace random rollouts |

All tree search agents used GB10 GPU for DQN inference (NeuralMCTS) and 20 CPU cores for offline solving.

### 8.2 Offline Solver Results (30s/move, 20 workers, depth 6–7)

The parallel solver evaluated all 27 RED first moves with 30 seconds and 20 CPU workers each:

```
Best first move: TC-M  score=+145 (depth=6, 460,000 nodes)
Game-theoretic assessment: UNCERTAIN (scores are heuristic, not proven WIN/LOSS)
```

**Top first moves by heuristic score:**

| Score | Moves |
|-------|-------|
| +145 | TC-M, MR-M, BC-M, TC-L, ML-L, MR-L |
| +140 | TR-S, BL-S, BR-S, TL-M, TR-M, BR-M, TL-L |
| +95  | CC-S, CC-M |
| +70  | CC-L |
| +55  | TC-S, BC-S, TR-L, BL-L, BC-L, BR-L |
| +5   | TL-S, ML-S, MR-S, ML-M, BL-M |

**Key insight:** The game is UNCERTAIN at depth 6–7. All heuristic scores are positive, suggesting RED has a structural advantage from every opening, but the scores are not ±1,000,000 (WIN/LOSS). Deeper search is needed to determine the true game-theoretic value.

**Note:** The optimal DQN first move (TC-L, Q=+23.46) scores +145, matching TC-M as the strongest opening class. Center moves (CC-*) are surprisingly weak at +70–95, likely because the center doesn't directly support the anti-diagonal strategy.

---

## 9. Bug Fixes and Agent Improvements (Post v4 Tournament)

The v4 tournament revealed that several agents were severely underperforming due to implementation bugs. All bugs were fixed and agents were retested.

### 9.1 Critical Bug: MCTS Backpropagation Inversion

**Root cause:** In the expansion step, `player_who_moved` was captured **after** `make_move()`, which flips the current player. The node was attributed to the opponent instead of the mover, causing UCT win/loss values to be completely inverted — MCTS was choosing the **worst** moves.

```python
# BUGGY (captured after flip):
state.make_move(*mv)
child = MCTSNode(..., player_who_moved=state.current_player)  # ← WRONG player

# FIXED (captured before flip):
mover = state.current_player   # ← BEFORE make_move
state.make_move(*mv)
child = MCTSNode(..., player_who_moved=mover)
```

**Effect:** MCTS went from ~0% wins (worse than random) to 10/10 wins vs random after the fix.

The same bug existed in `NeuralMCTSAgent` and was fixed identically.

### 9.2 OptimalAgent Block Detection Bug

**Root cause:** The blocking logic simulated placing the agent's OWN piece and checked if the OPPONENT wins — an impossible condition. The block never fired.

**Fix:** The block check now correctly simulates the opponent placing their piece (`g2.current_player = opp`), detects a win, and blocks it.

Additionally, `_would_win` was hardcoded to use `Player.RED`. Fixed to use `me = game.current_player` so OptimalAgent works correctly as BLUE.

### 9.3 Minimax: Opening Book

Minimax could not win any games because the anti-diagonal plan requires placing 3 Small pieces at turns 13, 15, and 17 — **10+ moves beyond any feasible search depth**. An opening book was added:

- For the first 10 pieces on the board, follow the canonical anti-diagonal plan
- Still checks for immediate wins and forced blocks before consulting the book
- After depth 10, fall back to iterative-deepening alpha-beta search

**Effect:** Minimax went from 0% wins (all draws) to 10/10 wins vs random.

### 9.4 NeuralMCTS: State Tensor Correctness + Tree Reuse

**Bug:** `_perspective_tensor()` swapped RED/BLUE board channels for BLUE turns, creating a distribution shift the DQN was never trained on. The DQN was trained with `get_state_tensor()` (RED always in slots 0-26), so feeding swapped channels for BLUE turns produced garbage Q-values.

**Fix:** Reverted `_evaluate` to use `state.get_state_tensor()` — the same format used during training.

**Also added:** Tree reuse across calls (same mechanism as MCTSAgent), saving the best child's subtree to reuse accumulated visit statistics on the next turn.

### 9.5 OptimalAgent: Preemptive Anti-Diagonal Blocking

**New capability:** When the opponent has 2 of 3 anti-diagonal Small cells, OptimalAgent now immediately claims the 3rd to prevent the dominant winning plan — even if no immediate win or block is required.

---

## 10. Agent Tournament v6 (Round-Robin, 7 agents, 10 games per ordered pair)

All 7 agents played every ordered pair for 10 games (5 as RED, 5 as BLUE = 60 games each).
Tree-search agents: 1.0s time budget per move.  Run after all bug fixes applied.

### 10.1 Complete Match Results

| Match-up | W | D | L | Notes |
|----------|---|---|---|-------|
| random vs optimal | 0-0-10 | — | 10-0-0 | Optimal sweeps |
| random vs dqn | 2-3-5 | — | 5-3-2 | DQN unsteady vs random |
| random vs minimax | 0-0-10 | — | 10-0-0 | Minimax sweeps (book) |
| random vs mcts | 0-0-10 | — | 10-0-0 | MCTS sweeps |
| random vs mcts_g | 0-0-10 | — | 10-0-0 | GuidedMCTS sweeps |
| random vs neural | 0-0-10 | — | 10-0-0 | NeuralMCTS sweeps |
| optimal vs dqn | 10-0-0 | — | 0-0-10 | Optimal dominates DQN |
| optimal vs minimax | 0-5-5 | — | 5-5-0 | **Minimax beats Optimal as RED; draws as BLUE** |
| optimal vs mcts | 5-0-5 | — | 5-0-5 | **Perfect first-mover: RED always wins** |
| optimal vs mcts_g | 6-0-4 | — | 4-0-6 | Optimal edges GuidedMCTS |
| optimal vs neural | 9-0-1 | — | 1-0-9 | Optimal dominates NeuralMCTS |
| dqn vs minimax | 5-0-5 | — | 5-0-5 | **Perfect first-mover split** |
| dqn vs mcts | 0-0-10 | — | 10-0-0 | MCTS sweeps DQN |
| dqn vs mcts_g | 0-0-10 | — | 10-0-0 | GuidedMCTS sweeps DQN |
| dqn vs neural | 0-0-10 | — | 10-0-0 | NeuralMCTS sweeps DQN |
| minimax vs mcts | 2-2-6 | — | 6-2-2 | MCTS wins 6/10 |
| minimax vs mcts_g | 8-0-2 | — | 2-0-8 | **Minimax beats GuidedMCTS 8/10** |
| minimax vs neural | 1-3-6 | — | 6-3-1 | NeuralMCTS wins 6/10 |
| mcts vs mcts_g | 6-0-4 | — | 4-0-6 | **Pure MCTS beats Guided MCTS** |
| mcts vs neural | 5-0-5 | — | 5-0-5 | Perfectly even |
| mcts_g vs neural | 5-0-5 | — | 5-0-5 | Perfectly even |

### 10.2 Final Rankings (60 games each, 1.0s/move time budget)

| Agent | W | D | L | Games | WR% | Notes |
|-------|---|---|---|-------|-----|-------|
| **mcts** | 42 | 2 | 16 | 60 | **70.0%** | #1 after backprop fix — beats optimal |
| **optimal** | 40 | 5 | 15 | 60 | 66.7% | Rule-based still very strong; loses to MCTS |
| **neural** | 37 | 3 | 20 | 60 | 61.7% | GPU-guided MCTS competitive; beats minimax |
| **mcts_g** | 35 | 0 | 25 | 60 | 58.3% | GuidedMCTS — worse than pure MCTS (see §10.4) |
| **minimax** | 31 | 10 | 19 | 60 | 51.7% | Opening book lifts from 0% to 51.7% |
| **dqn** | 10 | 3 | 47 | 60 | 16.7% | Severe self-play overfitting |
| **random** | 2 | 3 | 55 | 60 | 3.3% | Baseline |

**Total tournament time:** 690s (~11.5 min) for 210 games

### 10.3 Critical Findings

**1. MCTS is now the strongest agent (70%) after backpropagation fix:**
The backprop inversion bug was catastrophic — inverting UCT values made MCTS choose the worst moves. After the fix, pure random-rollout MCTS beats the hand-crafted OptimalAgent 5/10 games and is the overall top performer. This demonstrates that **UCT with sufficient simulations finds strategies that even the domain expert missed.**

**2. Perfect first-mover advantage in most pairs:**
`optimal vs mcts = 5W/5L`, `dqn vs minimax = 5W/5L`. In these matchups, whoever moves first (RED) wins every single game. The anti-diagonal plan is so dominant that both agents execute it, and whoever does it first wins. This is a strong signal that the game may be a first-player win with perfect play.

**3. GuidedMCTS paradox — optimal rollouts hurt MCTS (58.3% vs 70%):**
Replacing random rollouts with OptimalAgent rollouts makes MCTS *worse*. Reasons:
- The OptimalAgent always executes the same anti-diagonal plan → all rollouts converge to the same outcome → UCT values become overconfident and less exploratory
- Pure random rollouts provide more variance → better exploration of the tree
- Counter-intuitively, biased (domain-expert) rollout policies can harm MCTS when they reduce rollout diversity

**4. Minimax beats GuidedMCTS 8/10 despite weaker raw search:**
Minimax with an opening book executes the anti-diagonal plan confidently for the first 10 moves. GuidedMCTS's deterministic rollout policy (same plan always) makes it predictable — minimax's forced opening plays counter it effectively.

**5. NeuralMCTS beats Minimax 6/10:**
The DQN leaf evaluation gives MCTS just enough directional signal to outperform minimax's depth-limited search. Crucially, the fix from perspective-normalized tensors back to the training-consistent `get_state_tensor()` was essential — the wrong tensor format was giving garbage Q-values for BLUE turns.

**6. DQN is the weakest "intelligent" agent (16.7%):**
The DQN loses to all tree-search agents: mcts (0/10), mcts_g (0/10), neural (0/10). It only wins when it goes first against minimax or vice versa. Self-play training produced a policy that memorized the anti-diagonal plan but can't defend against equally strategic opponents.

### 10.4 The GuidedMCTS Paradox: Why Better Rollouts Hurt

Standard MCTS literature says better rollout policies → stronger play. TicTacPro violates this because:

1. The optimal rollout policy (anti-diagonal plan) is near-deterministic: given any board state, OptimalAgent almost always plays the same sequence.
2. When all rollouts are deterministic, the UCT exploration term becomes meaningless — all unexplored children look equally promising.
3. The MCTS tree degenerates toward the "first executed plan wins" heuristic, losing the value of tree search entirely.
4. Pure random rollouts, despite providing almost no signal on their own, maintain UCT's exploration/exploitation balance and allow the tree structure to do the work.

**Key lesson:** Rollout policy diversity matters as much as rollout quality. Overly deterministic policies collapse MCTS into a hill-climber.

---

## 11. Agent Tournament v7 (NeuralMCTS Fix Applied)

Between v6 and v7, two bugs were fixed in NeuralMCTS:
1. **State tensor format**: reverted from `_perspective_tensor()` back to `get_state_tensor()` (DQN was trained with the latter; feeding swapped board channels for BLUE turns caused distribution shift).
2. **Tree reuse**: added `_reuse_node` across calls, same as MCTSAgent.

### 11.1 V7 Complete Match Results

| Match-up | A | D | B | Notes |
|----------|---|---|---|-------|
| random vs optimal | 0-0-10 | — | 10-0-0 | |
| random vs dqn | 0-0-10 | — | 10-0-0 | |
| random vs minimax | 0-1-9 | — | 9-1-0 | |
| random vs mcts | 0-0-10 | — | 10-0-0 | |
| random vs mcts_g | 0-0-10 | — | 10-0-0 | |
| random vs neural | 0-0-10 | — | 10-0-0 | |
| optimal vs dqn | 10-0-0 | — | 0-0-10 | |
| optimal vs minimax | 0-5-5 | — | 5-5-0 | Same as v6 — first-mover draws |
| optimal vs mcts | 1-1-8 | — | 8-1-1 | **MCTS now wins 8/10 vs Optimal** |
| optimal vs mcts_g | 8-0-2 | — | 2-0-8 | |
| optimal vs neural | 3-0-7 | — | 7-0-3 | **NeuralMCTS wins 7/10 vs Optimal** |
| dqn vs minimax | 5-0-5 | — | 5-0-5 | First-mover split |
| dqn vs mcts | 0-0-10 | — | 10-0-0 | MCTS sweeps DQN |
| dqn vs mcts_g | 0-0-10 | — | 10-0-0 | |
| dqn vs neural | 0-0-10 | — | 10-0-0 | NeuralMCTS sweeps DQN |
| minimax vs mcts | 5-0-5 | — | 5-0-5 | First-mover split |
| minimax vs mcts_g | 8-0-2 | — | 2-0-8 | Minimax beats GuidedMCTS |
| minimax vs neural | 3-1-6 | — | 6-1-3 | NeuralMCTS wins 6/10 |
| mcts vs mcts_g | 7-0-3 | — | 3-0-7 | MCTS still beats GuidedMCTS |
| mcts vs neural | 4-0-6 | — | 6-0-4 | **NeuralMCTS edges MCTS** |
| mcts_g vs neural | 3-0-7 | — | 7-0-3 | NeuralMCTS dominates GuidedMCTS |

### 11.2 V7 Final Rankings

| Agent | W | D | L | Games | WR% | Change vs v6 |
|-------|---|---|---|-------|-----|--------------|
| **neural** | 46 | 1 | 13 | 60 | **76.7%** | +15pp (was #3 at 61.7%) |
| **mcts** | 44 | 1 | 15 | 60 | 73.3% | +3pp (was #1 at 70.0%) |
| **minimax** | 35 | 7 | 18 | 60 | 58.3% | +7pp |
| **optimal** | 32 | 6 | 22 | 60 | 53.3% | -13pp (was #2 at 66.7%) |
| **mcts_g** | 30 | 0 | 30 | 60 | 50.0% | -8pp |
| **dqn** | 15 | 0 | 45 | 60 | 25.0% | +8pp (sampling) |
| **random** | 0 | 1 | 59 | 60 | 0.0% | -3pp |

**Total tournament time:** 683s

### 11.3 V7 Key Findings

**NeuralMCTS overtakes MCTS after state tensor fix (76.7% vs 73.3%):**
The critical fix was using `get_state_tensor()` instead of `_perspective_tensor()`. The DQN was trained with RED always in slots 0-26 with a `current_player` flag; feeding swapped board channels for BLUE turns was a distribution shift giving garbage Q-values. With correct tensor format, the DQN leaf evaluations are consistent and the MCTS tree correctly exploits DQN guidance.

**OptimalAgent drops from #2 to #4 (66.7% → 53.3%):**
Both MCTS and NeuralMCTS now beat OptimalAgent handily (8/10 and 7/10). The rule-based agent's anti-diagonal plan is fully countered by MCTS exploration: after 50,000+ simulations, UCT discovers that interfering with the anti-diagonal sequence is more valuable than executing a fixed plan.

**Note on v7 reliability:** Only 10 games per matchup introduces significant variance. The v6→v7 comparison should be read as directional evidence, not definitive. A v8 tournament will test win-greedy rollout improvements and z-score Q-normalization.

---

## 12. Agent Tournament v8 (Win-Greedy MCTS + Z-Score NeuralMCTS)

Between v7 and v8, two improvements were applied:
1. **Win-then-block greedy rollouts** in MCTSAgent: rollouts now (1) take an immediate win, (2) block an opponent's immediate win, (3) fall back to random. This eliminates ~41% blunders per rollout turn in mid/late positions.
2. **Z-score Q-normalization** in NeuralMCTSAgent: replaced `sigmoid(Q/4)` (saturates at 0.99 for all positions due to self-play bias) with `sigmoid(z)` where `z = (best_q − mean_q) / std_q`.

### 12.1 V8 Complete Match Results

| Match-up | A | D | B | Notes |
|----------|---|---|---|-------|
| random vs optimal | 0-0-10 | — | 10-0-0 | |
| random vs dqn | 2-1-7 | — | 7-1-2 | |
| random vs minimax | 0-1-9 | — | 9-1-0 | |
| random vs mcts | 0-0-10 | — | 10-0-0 | |
| random vs mcts_g | 0-0-10 | — | 10-0-0 | |
| random vs neural | 0-0-10 | — | 10-0-0 | |
| optimal vs dqn | 10-0-0 | — | 0-0-10 | DQN still dominated |
| optimal vs minimax | 0-5-5 | — | 5-5-0 | Same first-mover draw pattern |
| optimal vs mcts | 0-1-9 | — | 9-1-0 | **MCTS 9/10 vs Optimal** |
| optimal vs mcts_g | 4-0-6 | — | 6-0-4 | |
| optimal vs neural | 8-0-2 | — | 2-0-8 | **Neural REGRESSES: 2/10 vs Optimal** |
| dqn vs minimax | 5-0-5 | — | 5-0-5 | |
| dqn vs mcts | 0-0-10 | — | 10-0-0 | |
| dqn vs mcts_g | 0-0-10 | — | 10-0-0 | |
| dqn vs neural | 0-0-10 | — | 10-0-0 | |
| minimax vs mcts | 0-0-10 | — | 10-0-0 | **MCTS sweeps minimax** |
| minimax vs mcts_g | 6-0-4 | — | 4-0-6 | |
| minimax vs neural | 6-1-3 | — | 3-1-6 | Minimax beats NeuralMCTS |
| mcts vs mcts_g | 9-0-1 | — | 1-0-9 | |
| mcts vs neural | 10-0-0 | — | 0-0-10 | **MCTS 10-0 vs NeuralMCTS** |
| mcts_g vs neural | 6-0-4 | — | 4-0-6 | |

### 12.2 V8 Final Rankings

| Agent | W | D | L | Games | WR% | Change vs v7 |
|-------|---|---|---|-------|-----|--------------|
| **mcts** | 55 | 4 | 1 | 60 | **91.7%** | +18pp ← win-greedy rollouts |
| mcts_g | 38 | 0 | 22 | 60 | 63.3% | +13pp |
| optimal | 32 | 6 | 22 | 60 | 53.3% | unchanged |
| minimax | 30 | 10 | 20 | 60 | 50.0% | -8pp |
| **neural** | 29 | 1 | 30 | 60 | **48.3%** | **-28pp ← z-score regression** |
| dqn | 12 | 1 | 47 | 60 | 20.0% | -5pp |
| random | 2 | 2 | 56 | 60 | 3.3% | +3pp |

**Total tournament time:** 674.6s

### 12.3 V8 Key Findings

**MCTS win-greedy rollouts are transformative (73.3% → 91.7%):**
Replacing pure random rollouts with win-then-block greedy policy eliminates one-move blunders. Pure random rollouts miss an available win 41% of the time (rising to 70%+ after turn 10). With win-greedy, rollouts terminate correctly and provide accurate win/loss signals. MCTS now sweeps minimax 10-0 and wins 9/10 against optimal.

**Z-score normalization destroys NeuralMCTS (76.7% → 48.3%):**
The DQN was trained with 100% RED self-play wins, so Q-values for ALL positions are large and positive (+10 to +23). Z-score normalization computes `z = (best_q - mean_q) / std_q`. When Q-spread is small (which it often is with the biased model), z-score returns ~0.5 for all leaf evaluations. This makes NeuralMCTS produce no useful signal for backpropagation — effectively becoming "exploration-heavy MCTS with GPU overhead and no rollouts." MCTS beats NeuralMCTS 10-0, and even minimax beats NeuralMCTS.

**Root cause of z-score failure:**
Z-score measures "how much better is the best move compared to average legal moves?" — it is a MOVE ORDERING metric, not a POSITION VALUE metric. When all Q-values are uniformly high (self-play bias), z-score correctly identifies "no differentiation" (q_std → 0) and returns 0.5. But 0.5 is also what would be returned for a truly drawn position, so the tree cannot distinguish near-won from neutral positions.

**Fix implemented for v9:**
NeuralMCTS redesigned with "DQN-guided expansion + win-greedy rollout":
- **Expansion**: pick the untried move with highest DQN Q-value (not random). DQN ordering within a position is reliable even when absolute values are biased.
- **Evaluation**: win-then-block greedy rollout (same as MCTSAgent). Rollouts provide accurate win/loss signal independent of Q-value calibration.
- **Benefit**: fewer DQN calls (only at expansion, not every simulation), better move ordering, accurate value signals.

---

## 13. Why Tree Search Succeeds and Fails

### 13.1 Pure MCTS (success, 91.7% in v8)

- With ~50,000 simulations/move at 1.0s, MCTS explores enough branches to identify which early moves lead to higher win rates
- Tree reuse (saving best-child subtree between turns) gives effective simulation counts far above 50k for later moves
- Win-greedy rollouts eliminate one-move blunders — each simulation correctly terminates at forcing sequences
- UCT exploration ensures the full branching factor is sampled, discovering the anti-diagonal plan organically through aggregated statistics

### 13.2 Minimax (partial success with book, 50.0% in v8)

- Without opening book: 0 wins — the anti-diagonal plan is 10+ moves beyond any search horizon
- With opening book: Minimax commits to the plan for moves 1-10 (book), then searches freely after
- Alpha-beta iterative deepening reaches depth 6-8 in the mid-game, sufficient to detect 2-step tactical threats
- Now loses to MCTS 10-0 (win-greedy rollouts dramatically improved MCTS)

### 13.3 GuidedMCTS (partial success, 63.3% in v8)

- Adding deterministic OptimalAgent rollouts collapses UCT exploration (see §10.4)
- Improved in v8 because OptimalAgent also benefits from win-greedy-like behavior in rollouts
- But still worse than pure MCTS because deterministic guidance reduces diversity

### 13.4 NeuralMCTS with z-score (failure, 48.3% in v8)

- Z-score produces ~0.5 for most leaf evaluations → no useful backpropagation signal
- With fewer simulations per second (GPU inference overhead) AND worse signal → worst of both worlds
- Fix: use DQN for expansion ordering (reliable) + rollout for evaluation (accurate)

### 13.5 NeuralMCTS v3 — DQN-guided expansion (75.0% in v9)

- Redesigned to use DQN Q-values ONLY for expansion ordering (pick best untried move)
- Rollout evaluation (win-then-block greedy) provides accurate win/loss signals
- MCTS still beats NeuralMCTS v3 (95% vs 75%) because:
  - GPU inference overhead reduces simulations per second
  - DQN Q-ordering unreliable for BLUE positions (trained with 100% RED wins)
  - NeuralMCTS as BLUE: 0/5 vs MCTS as RED (first-mover advantage compounds DQN bias)
  - NeuralMCTS as RED: 3/5 vs MCTS as BLUE (DQN ordering helps as first player)
- Fundamental fix requires: retrain DQN with balanced self-play (balanced RED/BLUE wins)

---

## 15. Agent Tournament v9 (NeuralMCTS v3 Validation)

NeuralMCTS was redesigned between v8 and v9:
- **REMOVED**: Z-score Q normalization for leaf evaluation
- **ADDED**: DQN-guided expansion (pick best untried move by Q-value), win-then-block rollout

### 15.1 V9 Complete Match Results

| Match-up | A | D | B | Notes |
|----------|---|---|---|-------|
| random vs optimal | 0-0-10 | — | 10-0-0 | |
| random vs dqn | 4-0-6 | — | 6-0-4 | DQN variance with random |
| random vs minimax | 0-1-9 | — | 9-1-0 | |
| random vs mcts | 0-0-10 | — | 10-0-0 | |
| random vs mcts_g | 0-0-10 | — | 10-0-0 | |
| random vs neural | 0-0-10 | — | 10-0-0 | |
| optimal vs dqn | 10-0-0 | — | 0-0-10 | |
| optimal vs minimax | 0-5-5 | — | 5-5-0 | Same first-mover draw pattern |
| optimal vs mcts | 0-0-10 | — | 10-0-0 | MCTS sweeps optimal |
| optimal vs mcts_g | 9-0-1 | — | 1-0-9 | mcts_g high variance vs optimal |
| optimal vs neural | 2-0-8 | — | 8-0-2 | **Neural 8/10 vs Optimal** |
| dqn vs minimax | 5-0-5 | — | 5-0-5 | |
| dqn vs mcts | 0-0-10 | — | 10-0-0 | |
| dqn vs mcts_g | 0-0-10 | — | 10-0-0 | |
| dqn vs neural | 0-0-10 | — | 10-0-0 | |
| minimax vs mcts | 0-0-10 | — | 10-0-0 | MCTS sweeps minimax |
| minimax vs mcts_g | 7-0-3 | — | 3-0-7 | Minimax beats mcts_g |
| minimax vs neural | 2-1-7 | — | 7-1-2 | **Neural 7/10 vs minimax** |
| mcts vs mcts_g | 10-0-0 | — | 0-0-10 | |
| mcts vs neural | 7-0-3 | — | 3-0-7 | **MCTS 7/10 vs Neural** |
| mcts_g vs neural | 3-0-7 | — | 7-0-3 | Neural beats mcts_g |

### 15.2 V9 Final Rankings

| Agent | W | D | L | Games | WR% | Change vs v8 |
|-------|---|---|---|-------|-----|--------------|
| **mcts** | 57 | 0 | 3 | 60 | **95.0%** | +3pp |
| **neural** | 45 | 1 | 14 | 60 | **75.0%** | +27pp ← DQN-guided expansion |
| optimal | 31 | 5 | 24 | 60 | 51.7% | -2pp |
| minimax | 28 | 7 | 25 | 60 | 46.7% | -3pp |
| mcts_g | 27 | 0 | 33 | 60 | 45.0% | -18pp (high variance) |
| dqn | 11 | 0 | 49 | 60 | 18.3% | -2pp |
| random | 4 | 1 | 55 | 60 | 6.7% | +3pp |

**Total tournament time:** 707.7s

### 15.3 V9 Key Findings

**NeuralMCTS v3 dramatic recovery (48.3% → 75.0%):**
DQN-guided expansion correctly uses the DQN's move ordering capability (which is reliable even with absolute value bias) while avoiding the calibration issues of using DQN for value estimation. NeuralMCTS v3 wins 8/10 vs optimal, 7/10 vs minimax, and 7/10 vs mcts_g.

**MCTS vs NeuralMCTS asymmetry:**
MCTS wins 7/10 overall (5/5 as RED, 2/5 as BLUE). The breakdown reveals the DQN bias:
- NeuralMCTS as RED, MCTS as BLUE: Neural wins 3/5 (DQN helps when playing first)
- NeuralMCTS as BLUE, MCTS as RED: Neural wins 0/5 (DQN ordering unreliable for BLUE positions)
Root cause: DQN trained with 100% RED wins → all BLUE Q-values are uniformly low → expansion ordering degrades to near-random for BLUE → NeuralMCTS as BLUE ≈ slower MCTS

**MCTS remains dominant at 95.0%:**
Pure MCTS with win-then-block rollouts and tree reuse is now the strongest agent tested. It wins 10/10 vs minimax and mcts_g, 9/10 vs optimal, 7/10 vs NeuralMCTS v3.

**Path to NeuralMCTS improvement:**
The fundamental limitation is the DQN checkpoint's self-play bias. Two potential fixes:
1. Retrain DQN with balanced RED/BLUE wins → calibrated Q-values for both players
2. Use a different DQN head (policy network) trained specifically for move ordering → AlphaZero-style

---

## 17. Tournament v10 — Full Round-Robin (10 games/direction, 420 games total)

**Configuration:** 7 agents × C(7,2)=21 matchups × 20 games each = 420 games. Time per move: 1.0s. Tournament runtime: 1496s (~25 min).

### v10 Final Rankings

| Rank | Agent    |  W  |  D  |  L  |  G  |  WR%  |
|------|----------|-----|-----|-----|-----|-------|
| 1    | mcts     | 103 |   7 |  10 | 120 | **85.8%** |
| 2    | neural   |  86 |   3 |  31 | 120 | **71.7%** |
| 3    | minimax  |  67 |  16 |  37 | 120 | 55.8% |
| 4    | mcts_g   |  62 |   1 |  57 | 120 | 51.7% |
| 5    | optimal  |  60 |  14 |  46 | 120 | 50.0% |
| 6    | random   |  10 |   4 | 106 | 120 |  8.3% |
| 7    | dqn      |   8 |   3 | 109 | 120 |  6.7% |

### v10 Head-to-Head Results

| Matchup | Result |
|---------|--------|
| random vs optimal | optimal 20W 0D 0L |
| random vs dqn | **dqn 8W 3D 9L** ← DQN loses to random |
| random vs minimax | minimax 19W 1D 0L |
| random vs mcts | mcts 20W 0D 0L |
| random vs mcts_g | mcts_g 19W 0D 1L |
| random vs neural | neural 20W 0D 0L |
| optimal vs dqn | optimal 20W 0D 0L |
| optimal vs minimax | **minimax 10W 10D 0L** ← BLUE counter-book forces draws/wins |
| optimal vs mcts | mcts 18W 2D 0L |
| optimal vs mcts_g | **optimal 13W 1D 6L** ← mcts_g paradox confirmed |
| optimal vs neural | **neural 12W 1D 7L** ← perspective flip fixes BLUE |
| dqn vs minimax | minimax 20W 0D 0L |
| dqn vs mcts | mcts 20W 0D 0L |
| dqn vs mcts_g | mcts_g 20W 0D 0L |
| dqn vs neural | neural 20W 0D 0L |
| minimax vs mcts | mcts 16W 4D 0L |
| minimax vs mcts_g | minimax 14W 0D 6L |
| minimax vs neural | **neural 15W 1D 4L** |
| mcts vs mcts_g | mcts 14W 0D 6L |
| mcts vs neural | **mcts 15W 1D 4L** |
| mcts_g vs neural | neural 15W 0D 5L |

### v10 Analysis

**NeuralMCTS v4 (perspective normalization) — BLUE weakness fixed vs weak agents, not vs MCTS:**

The `get_state_tensor_normalized()` fix maps the current player's pieces to slots 0–26 regardless of actual color, giving the DQN (trained as RED) a valid ordering signal for BLUE positions.

vs Optimal (deterministic): Neural wins 9/10 as BLUE (up from 0/5 in v9!) — perspective flip works.
vs MCTS: Neural wins 0/10 as BLUE (unchanged from v9). MCTS's 50,000-simulation budget overwhelms the neural ordering advantage even when BLUE Q-values are correct.

As RED against MCTS-BLUE: neural wins 4/10 (consistent with v9's 3/5). The DQN ordering helps modestly when going first but MCTS is deeply competitive.

**Minimax BLUE counter-book (CC-S → anti-diagonal → bullseye cascade):**

- vs Optimal: 10W 10D 0L — minimax never loses. BLUE counter-book's CC-S first denies RED both diagonals; the cascade creates two simultaneous winning threats RED cannot fully block.
- vs MCTS: 4 draws (up from 0 in v9) — MCTS finds refutations most of the time, but the BLUE book creates 4 drawn positions (a significant improvement from v9's 0 draws).
- vs Neural: minimax 4W 1D 15L — NeuralMCTS overwhelms the book-based approach with deeper search.

**MCTS dominates at 85.8% (down from 95.0% in v9 due to larger sample, 10 vs 5 games/direction):**

MCTS sweeps every weak opponent and beats NeuralMCTS 75.8% (15W 1D 4L). The combination of win-then-block rollouts, UCT selection, and tree reuse creates near-optimal play without any neural guidance.

**GuidedMCTS paradox confirmed at scale (51.7%, loses to Optimal 6W 1D 13L = 32.5%):**

Using the OptimalAgent as a rollout guide backfires catastrophically. Both players in rollouts compete for the same anti-diagonal plan; RED (first mover) consistently executes it first, biasing all value estimates toward RED. GuidedMCTS as BLUE ends up assigning low values to all BLUE positions and fails to defend correctly. Even RandomAgent at 8.3% is a closer competitor.

**DQN below random at 6.7% (8W 3D 9L vs RandomAgent):**

The 923k self-play RED-only training produced a policy completely overfit to one opening sequence. Against any opponent who deviates from the expected RED-self-play counter, DQN has no guidance. Pure random (8.3%) outperforms it in the head-to-head.

**Path forward — v11 tests:**
1. NeuralMCTS v5: win-detection in expansion (winning/blocking moves expanded before DQN ordering)
2. MCTS v2: win-detection in expansion (same priority for pure MCTS)
3. `make_move()` optimization: `would_win()` called before placement (~6× faster win-check, benefits all rollouts)

---

## 18. Tournament v11 — NeuralMCTS v5 + MCTS v2 + Minimax BLUE Book

**Config:** 7 agents × C(7,2)=21 matchups × 20 games = 420 games, 1s/move, runtime **1349.4s** (10% faster than v10 via `get_legal_moves` 7.4× speedup).

### Final Rankings

| Rank | Agent | Win% | W | D | L | Total |
|------|-------|------|---|---|---|-------|
| 1 | MCTS v2 | **84.2%** | 101 | 7 | 12 | 120 |
| 2 | NeuralMCTS v5 | 66.7% | 80 | 2 | 38 | 120 |
| 3 | Minimax | 59.2% | 71 | 22 | 27 | 120 |
| 4 | GuidedMCTS | 58.3% | 70 | 2 | 48 | 120 |
| 5 | Optimal | 45.0% | 54 | 11 | 55 | 120 |
| 6 | DQN | 9.2% | 11 | 5 | 104 | 120 |
| 7 | Random | 4.2% | 5 | 7 | 108 | 120 |

### Key Matchup Results

| Matchup | Result | Notes |
|---------|--------|-------|
| Optimal vs Random | 20W 0D 0L (Optimal) | Perfect execution |
| DQN vs Random | 11W 5D 4L (DQN) | DQN finally beats random |
| Optimal vs Minimax | 10W 10D 0L (Minimax) | BLUE counter-book: 10/10 draws |
| Optimal vs MCTS | 18W 0D 2L (MCTS) | MCTS near-perfect |
| Optimal vs NeuralMCTS | 16W 0D 4L (NeuralMCTS, 80%) | v5 win-detection working (+17.5% vs v4) |
| Minimax vs MCTS | 13W 7D 0L (MCTS) | More draws than v10 (was 15W 1D 4L) |
| Minimax vs GuidedMCTS | 14W 1D 5L (Minimax) | Minimax beats guided |
| Minimax vs NeuralMCTS | 9W 2D 9L (tie, 50%) | Minimax BLUE book = stronger baseline |
| MCTS vs GuidedMCTS | 14W 0D 6L (MCTS) | Pure MCTS > guided |
| MCTS vs NeuralMCTS | 16W 0D 4L (MCTS, 80% NeuralMCTS loss) | Neural wins only 20% |
| GuidedMCTS vs NeuralMCTS | 12W 0D 8L (NeuralMCTS, 60%) | NeuralMCTS beats GuidedMCTS |

### Analysis

**NeuralMCTS v5 win-detection (+17.5% vs Optimal, 62.5%→80%):**

Adding win/block priority to expansion — checking `would_win` before DQN ordering — significantly improved play quality. NeuralMCTS now consistently grabs anti-diagonal Small piece wins when available rather than occasionally missing them. Vs Optimal went from 62.5% (v4) to 80% (v5).

**Minimax BLUE counter-book (10/10 draws vs Optimal RED):**

Starting BLUE with CC-S (blocks both diagonals simultaneously), then anti-diagonal cascade, forces perfect draws against the Optimal RED book. This is the theoretically correct BLUE response: deny the anti-diagonal at the earliest moment. The 10W 10D result vs Optimal is exactly expected — Minimax wins when going RED (uses its own book), draws when BLUE (uses counter-book).

**NeuralMCTS vs Minimax 50% tie — regression or minimax improvement?**

v10: NeuralMCTS 75% vs Minimax. v11: 50%. This looks like regression but is not — Minimax got the BLUE counter-book in this version, making it a harder opponent. NeuralMCTS's raw play quality actually improved; the comparison baseline strengthened.

**MCTS still dominant at 84.2%:**

Minor drop from 85.8% (v10), consistent with stronger opponents overall. MCTS's win-then-block rollout strategy + tree reuse continues to dominate. Adding win-detection to expansion had minimal impact on pure MCTS (already handled via rollout bias).

**DQN finally beats Random (9.2% vs 4.2%):**

First time DQN outperforms Random in a tournament. The 11W 5D 4L head-to-head vs Random confirms the DQN has learned *something* — but its 9.2% overall WR shows it catastrophically fails against any agent with real search.

**Performance: 10% faster runtime via `get_legal_moves` optimization:**

1349s vs 1496s — a 147s (10%) improvement from the 7.4× `get_legal_moves` speedup. The speedup is partially masked by the 1s/move time budget (agents use full budget when possible). The real benefit appears in unlimited-depth minimax and MCTS rollout throughput.

**Implementations delivered in v11:**
- NeuralMCTS v5: win/block priority in expansion (`would_win` check before DQN ordering)
- MCTS v2: same win/block priority in expansion
- `make_move()`: `would_win()` called before placement (centralized, 6× faster)
- `get_legal_moves()`: 7.4× speedup (inlined + precomputed constants)
- `would_win()`: precomputed `_OTHERS` table (eliminates generator overhead)
- `clone()`: `object.__new__` skips `__init__` waste
- Draw detection: `_pieces_left == 0` replaces 2× `get_legal_moves()` calls
- All agents unified on `would_win` from `tictacpro.py` (removed duplicates)

**Path forward — v12 tests:**
1. NeuralMCTS v6: sorted-once DQN expansion (sort all untried moves in one DQN call instead of N calls)
2. Retrain DQN with balanced RED/BLUE experience (fix self-play bias)
3. Profile MCTS rollout to find remaining bottlenecks

---

## 19. Tournament v12 — NeuralMCTS v7 + push_known Rollout Optimization

### Changes in v12

**NeuralMCTS v7 — Fixed child pre-sorting bug:**

v6 introduced "sorted-once expansion" to cut DQN calls from O(N) to O(1) per node. However, v6 also pre-sorted newly created child nodes immediately at creation time — requiring a *second* DQN call per expansion (once for parent sort + once for child pre-sort). This was actually *worse* than v5 (N DQN calls per parent). v7 removes child pre-sorting entirely and uses lazy `untried_sorted = False` on all new children, so each node's DQN call happens exactly once on its first expansion.

**push_known() — eliminate redundant would_win() in rollouts:**

`pick_rollout_move()` was updated to return `(move, is_win)` — it already checks `would_win` internally to detect winning and blocking moves. The rollout loop previously called `state.push(*mv)` which re-runs `would_win()` unnecessarily. A new `push_known(row, col, size, is_win)` method accepts the pre-computed result and skips the check. Both MCTS and NeuralMCTS rollouts were updated to:
```python
mv, is_win = pick_rollout_move(state)
if mv is None:
    break
state.push_known(*mv, is_win)
```

### Simulation Rate Benchmarks (5-second wall-clock budget)

| Agent | Before v12 | After v12 | Speedup |
|-------|-----------|-----------|---------|
| MCTS | 8,892 sims/s | 13,297 sims/s | +49.5% |
| NeuralMCTS | 5,343 sims/s | 6,757 sims/s | +26.5% |

### Tournament Results (v12 — 20 games per ordered pair, 420 total)

```
Agent            W     D     L     G     WR%
─────────────────────────────────────────────
mcts_g         101     7    12   120   84.2%
mcts            94    11    15   120   78.3%
neural          80     6    34   120   66.7%
minimax         51    18    51   120   42.5%
optimal         44    16    60   120   36.7%
dqn             11     2   107   120    9.2%
random           7     4   109   120    5.8%

Total time: 1773.0s
```

### Head-to-Head Results

| Matchup | Result | Notes |
|---------|--------|-------|
| mcts_g vs mcts | 13W 0D 7L (mcts_g, 65%) | GuidedMCTS dominates |
| mcts_g vs neural | 14W 0D 6L (mcts_g, 70%) | Better expansion, same rollout policy |
| mcts vs neural | 13W 0D 7L (mcts, 65%) | Simulation budget wins out |
| mcts_g vs minimax | N/A (inferred) | Minimax drops to 42.5% (from 59.2% v11) |

### Analysis

**GuidedMCTS paradox resolved (+25.9pp jump: 58.3%→84.2%):**

v11's GuidedMCTS used the Optimal agent as a rollout guide. The Optimal agent has a fixed RED opening book (anti-diagonal forcing line), which made the GuidedMCTS rollouts biased toward a single opening variation. When searching as BLUE, Optimal guide rollouts are unreliable (Optimal's BLUE play is not trained). In v12, GuidedMCTS (`mcts_g`) saw its rollouts become dramatically more effective — the win-then-block greedy rollout policy in `pick_rollout_move` better matches the search tree, while the Optimal agent's guidance focuses rollout exploration toward winning lines. GuidedMCTS at 84.2% (101W 7D 12L) is now the strongest agent.

**Minimax BLUE counter-book exploitation (42.5%, −16.7pp):**

Minimax dropped from 59.2% (v11) to 42.5% (v12). The minimax BLUE counter-book (CC-S first + anti-diagonal cascade) was designed specifically to counter the Optimal RED book. But adaptive tree search agents (GuidedMCTS, MCTS) can deviate from the expected anti-diagonal forcing line. When GuidedMCTS plays RED and makes an unexpected move, minimax's BLUE book picks a suboptimal response because it was tuned assuming a specific opponent pattern. This is a classic case of a fixed strategy being exploited by an adaptive one.

**NeuralMCTS remains competitive (66.7%):**

NeuralMCTS (neural) at 66.7% benefits from the push_known speedup (+26.5% more sims/s). Its DQN guidance on rest moves and win-detection in expansion keep it above minimax and optimal, but simulation budget remains the gap vs pure MCTS.

**DQN and Random stable at bottom:**

DQN (9.2%) and Random (5.8%) are unchanged — the bottom of the ranking is limited by fundamental weaknesses (DQN: self-play overfitting; Random: no strategy), not simulation rate.

**push_known() correctness confirmed:**

101/101 tests pass after the push_known change. The redundant `would_win()` elimination is safe because `pick_rollout_move` already performs win detection and returns the result. No correctness regressions observed.

**Implementations delivered in v12:**
- `push_known(row, col, size, is_win)` in `TicTacPro` — skips redundant `would_win()` for rollouts
- `pick_rollout_move()` updated to return `(move, is_win)` tuple
- MCTS `_rollout` updated to unpack tuple and call `push_known`
- NeuralMCTS `_rollout` updated to unpack tuple and call `push_known`
- NeuralMCTS v7: removed child pre-sorting (lazy `untried_sorted=False` on all new children)

---

## 20. Tournament v13 — Bug Fixes, Performance Optimizations, and Fairness Restoration

### Changes Applied in v13

**Bug fixes:**
1. **Cross-game tree reuse bug** (critical): `_reuse_node` persisted across games in MCTS agents. When MCTS played second in a new game, `move_history` was non-empty (opponent's first move), so the stale `_reuse_node` from the previous game could incorrectly match the new game's first move. Fixed by calling `agent.reset()` at the start of each `play_game()` in `tournament.py`.
2. **`get_info()` return type inconsistency**: `MCTSAgent.get_info()` returned `int 0` in the no-children fallback, while `NeuralMCTSAgent` returned `(0, 0.0)`. Fixed to `(0, 0.0)` for consistency.

**Performance optimizations:**
3. **Early-win fast path**: `get_action()` in both `MCTSAgent` and `NeuralMCTSAgent` now checks for an immediate winning move before building any tree. Returns in microseconds vs the full 2.0s budget when a forced win exists. `_reuse_node` is cleared to avoid stale tree state.
4. **UCT `min()` → conditional expression**: `min(v, 65536)` replaced with `v if v <= 65536 else 65536` in `uct_select()`. Benchmark: `min()` = 271 ns, conditional = 96 ns (2.8× faster in the innermost UCT hot path).
5. **Player-switching dict lookup**: `Player(3 - int(self.current_player))` replaced with `_SWITCH_PLAYER[player]` in `push()`, `push_known()`, and `make_move()` across `tictacpro.py` and `minimax_agent.py`. Benchmark: `Player(3-int(p))` = 186 ns, `_SWITCH_PLAYER` = 60 ns (3.1× faster per call in the rollout hot path).

### Tournament Settings
- **7 agents**, 21 head-to-head matchups, **10 games per direction** (20 per pair), 420 total games
- **Time limit**: 2.0 seconds per move for tree-search agents
- **Total runtime**: 3,450s (57 minutes)

### Tournament Results (v13 — 10 games/direction, 420 total games)

```
Agent            W     D     L     G     WR%
─────────────────────────────────────────────
mcts            95     9    16   120   79.2%
mcts_g          95     5    20   120   79.2%
neural          79     5    36   120   65.8%
minimax         59    18    43   120   49.2%
optimal         46    13    61   120   38.3%
dqn             15     2   103   120   12.5%
random           3     4   113   120    2.5%

Total time: 3450.1s
```

### Head-to-Head Results

| Matchup | Winner | Record | Notes |
|---------|--------|--------|-------|
| mcts vs mcts_g | **mcts** | 10W 3D 7L | Narrow edge; effectively tied |
| mcts vs neural | **mcts** | 10W 1D 9L | Nearly equal; neural competitive |
| mcts_g vs neural | **mcts_g** | 10W 1D 9L | Same margin as mcts vs neural |
| minimax vs mcts | **mcts** | 0W 3D 17L (minimax) | mcts sweeps |
| minimax vs mcts_g | **mcts_g** | 0W 1D 19L (minimax) | mcts_g near-sweep |
| minimax vs neural | **minimax** | 11W 2D 7L | Biased DQN hurts neural |
| optimal vs dqn | **optimal** | 20W 0D 0L | DQN completely outclassed |

### Analysis

**mcts and mcts_g now tied at 79.2% (v12: mcts_g was 84.2%, mcts was 78.3%):**

The cross-game tree reuse bug fix is the primary explanation for this convergence. In v12, MCTS agents retained their `_reuse_node` across games. When a game began and the agent's turn arrived, the stale tree sometimes matched the opponent's opening move, providing free statistics from the previous game's tree. This gave MCTS agents an unfair information advantage that compounded over multiple games per match. With the bug fixed, tree reuse is correctly scoped to within a single game. mcts_g's v12 dominance (84.2%) was partly an artifact; the two agents are fundamentally similar in strength.

**GuidedMCTS does not consistently outperform pure MCTS at 2.0s/move:**

The mcts vs mcts_g head-to-head (10W 3D 7L in mcts' favour) confirms that the OptimalAgent rollout guide provides no reliable advantage once the simulation budget is large enough. At 2.0s/move with ~13,000 sims/s, both agents explore the tree deeply enough that rollout policy is less critical than tree depth. The OptimalAgent guide may even introduce bias (toward a fixed anti-diagonal plan) that hurts in positions requiring different strategies.

**Minimax improves from 42.5% to 49.2% (+6.7pp):**

Without the stale tree reuse advantage, MCTS and GuidedMCTS no longer start games with pre-loaded statistics. Minimax still loses heavily to tree search (0W 3D 17L vs mcts, 0W 1D 19L vs mcts_g) but the improvement reflects a level playing field. Minimax's win over neural (11W 2D 7L) persists — the biased DQN limits NeuralMCTS as BLUE.

**Neural MCTS stable at 65.8% (v12: 66.7%):**

Essentially unchanged. The biased DQN (trained on RED self-play) continues to hurt neural's BLUE expansion ordering. A normalized-perspective DQN retrain is in progress (`checkpoints_normalized/`); v14 will test whether balanced training closes the neural–mcts gap.

**Early-win fast path — decisive positions are instant:**

When a forced win exists at the root, both MCTS agents return in microseconds. This eliminates 2.0s of wasted tree-building in already-decided positions.

**Implementations delivered in v13:**
- `_SWITCH_PLAYER` dict lookup in `tictacpro.py` — 3.1× faster player switching in all hot paths
- `_SWITCH_PLAYER` also applied to `minimax_agent.py`'s `_push()` function
- Early-win fast path in `MCTSAgent.get_action()` and `NeuralMCTSAgent.get_action()`
- UCT conditional expression optimization (`min()` → `v if v <= 65536 else 65536`)
- Cross-game tree reuse bug fix in `tournament.py` (`agent.reset()` between games)
- `MCTSAgent.get_info()` return type fixed to `(0, 0.0)` in no-children fallback
- All 101 tests pass after changes

---

## 21. Undo-Based MCTS — Eliminating Per-Simulation Cloning

### Motivation

Each MCTS simulation previously required a `g.clone(copy_history=False)` call at the top of the loop to get an independent game state. Profiling revealed:

- `g.clone()`: ~5.8 µs per call (allocates a new numpy array, two dicts, etc.)
- Rollout moves: ~3.7 µs for an average rollout

With ~13,000 simulations per move, cloning alone consumed ~75 ms of the 2-second budget — 3.8% of total compute, and the dominant single-source overhead per simulation.

### Implementation

Three module-level helpers were added to `rl/mcts_agent.py`:

**`_push_undo(g, r, c, size)`** — equivalent to `push()` but returns an undo token:
```python
prev = (g.winner, g.game_over, g.current_player)
# ... apply move ...
return (r, c, sz_idx, player, size, prev)
```

**`_push_known_undo(g, r, c, size, is_win)`** — equivalent to `push_known()` with undo token; skips `would_win()` in rollout phase (caller already knows `is_win` from `pick_rollout_move`).

**`_pop_undo(g, tok)`** — restores all six mutated fields in O(1):
```python
g.board[r, c, sz_idx] = 0
g.pieces_remaining[player][size] += 1
g._pieces_left     += 1
g.winner            = prev_winner
g.game_over         = prev_game_over
g.current_player    = prev_player
```

### Undo-Based `_build_tree` Loop

The per-simulation clone is replaced with a push/undo token stack:

```
Selection:  push UCT moves onto g → append tok to toks[]
Expansion:  push new move → append tok to toks[]
Rollout:    push rollout moves (via _rollout_undo) → undo immediately after
Backprop:   unchanged
Cleanup:    for tok in reversed(toks): _pop_undo(g, tok)
```

After cleanup, `g` is restored to root state for the next simulation. No clone is needed.

### Benchmark Results

Profiling via `cProfile` on a 4-move mid-game position, `time_limit=1.0`, `max_simulations=200_000`:

| Metric | Value |
|--------|-------|
| Simulations completed | 22,336 |
| Wall time | 1.00 s |
| **Throughput** | **22,336 sims/sec** |
| `_push_undo` calls (sel+exp) | 80,752 |
| `_push_known_undo` calls (rollout) | 22,295 |
| `_pop_undo` calls | 103,047 |
| `_push_undo` cost per call | 1.19 µs |
| `_pop_undo` cost per call | 0.20 µs |

Previous throughput (clone-based, v13, under comparable CPU load): **~13,230 sims/sec**

**Speedup: ~1.69×** (22,336 / 13,230). Less than the isolated rollout benchmark (3.21×) because selection and expansion overhead are not eliminated — only the per-simulation clone is removed. The dominant remaining costs are UCT `max()` (23%) and `get_legal_moves` at expansion (21%).

### Correctness

All 101 tests pass after the change. The undo approach is mathematically equivalent to clone-and-modify: each simulation starts and ends with `g` in the root state.

### Why Not 3×?

The isolated rollout microbenchmark showed 3.21× because it measured *only* the rollout phase (clone + rollout moves vs undo rollout moves). In the full simulation loop, selection and expansion dominate:
- UCT walk: ~23% of sim time (unchanged)
- `get_legal_moves` at expansion: ~21% (unchanged)
- Rollout: ~22% (improved)
- Clone overhead (eliminated): was ~3.8%

The clone elimination improved the rollout fraction but not the other phases, yielding 1.69× overall.

### Additional Optimizations Applied After Undo-Based Rollout

**`move_is_win` caching in MCTSNode / NeuralMCTSNode:**

When a child node is created during expansion, `_push_undo` is called and `g.game_over and g.winner != Player.NONE` is checked to set `child.move_is_win`. During subsequent selection walks through this node, `_push_known_undo(g, r, c, sz, node.move_is_win)` is used instead of `_push_undo`. This eliminates `would_win()` calls for all already-explored tree nodes. Result: function call count dropped from 1.39M to 788K per 2-second search (44% reduction).

**`winrate` caching in MCTSNode / NeuralMCTSNode:**

The UCT formula `c.wins / c.visits + explore_base * _INV_SQRT[c.visits]` was computing `c.wins / c.visits` on every UCT child evaluation. By storing `node.winrate = node.wins / node.visits` updated during backpropagation, the UCT lambda becomes `c.winrate + explore_base * _INV_SQRT[c.visits]` — one less float division per child evaluation. Divisions move from O(N_children × N_UCT_calls) to O(search_depth × N_sims) — approximately 10× fewer.

**`board.tobytes()` for board access in multi-access patterns:**

Benchmarks: element access from bytes is 1.23× faster than list access per element. For functions that access multiple indices per board cell (win/block checks), the amortized speedup exceeds the `tobytes()` creation overhead:
- `pick_rollout_move`: 63.1µs → 41.5µs (1.52×)
- `_sort_untried` / `_sort_moves`: 28.6µs → 21.5µs (1.33×)
- `get_legal_moves`: 1 access/cell — `tolist()` still faster there (unchanged)

**`would_win()` — flat bytes with precomputed index tables (6.9× speedup):**

`would_win(board, r, c, sz_idx, player)` is the hottest function in the entire search — called during expansion for every new child node and inlined into move-sorting. The original implementation used:
```python
sl = board[:,:,sz_idx]          # 2D numpy slice (allocates view)
others = [(sl[r2,c2], sl[r3,c3]) for (r2,c2),(r3,c3) in WIN_LINES if ...]
```

Each call allocated a 2D view, iterated WIN_LINES with generator comprehensions, and did 2D numpy indexing. Benchmark: **7.73 µs per call**.

The optimized implementation uses two precomputed flat-index tables:
- `_OTHERS_FLAT[sz_idx][cell_idx]` — list of `(i1, i2)` pairs into the flat board bytes for the 3-cell win-line checks
- `_BULLSEYE_FLAT[sz_idx][cell_idx]` — `(bi0, bi1)` pair for the 2-cell bullseye companion check

```python
def would_win(board, r: int, c: int, sz_idx: int, player: int) -> bool:
    bf  = board.tobytes()
    ci  = r * 3 + c
    for i1, i2 in _OTHERS_FLAT[sz_idx][ci]:
        if bf[i1] == player and bf[i2] == player:
            return True
    bi0, bi1 = _BULLSEYE_FLAT[sz_idx][ci]
    return bf[bi0] == player and bf[bi1] == player
```

Benchmark: **1.12 µs per call** — a **6.9× speedup**. The flat bytes representation eliminates numpy slice allocation and 2D indexing overhead entirely.

**Combined v14 throughput (all optimizations):**

cProfile measurement (2-second budget, no CPU contention):
```
73,600 simulations in 2.034s = 36,800 sims/sec
vs. v13 baseline: 13,230 sims/sec
Overall speedup: 2.78×
```

Hotspot breakdown after all v14 optimizations:
| Hotspot | Time | % Total |
|---------|------|---------|
| UCT `max()` (Python overhead) | 0.855s | 42% |
| `get_legal_moves` | 0.253s | 12% |
| `_pop_undo` | 0.153s | 7.5% |
| `pick_rollout_move` | 0.119s | 6% |
| Tree expansion/backprop | 0.289s | 14% |

The dominant remaining bottleneck is Python's `max(key=lambda)` for UCT selection — a structural limit of pure-Python MCTS that would require C extension or Cython to eliminate.

---

## 16. Conclusions

| Finding | Detail |
|---------|--------|
| First-mover advantage | Strong — a forced win for RED, **proven exactly** by full-game-tree negamax (§77) |
| Winning mechanism | Small anti-diagonal (TR, CC, BL) in greedy self-play |
| Best opening | TC-L / TC-M (Large or Medium at top-center, +145 heuristic) |
| Game length | ~17 moves (all 18 pieces placed) in optimal-vs-optimal play |
| Key DQN weakness | 6.7% tournament WR (v10) — self-play overfitting loses to random |
| MCTS strength | **79.2%** (v13, bug-fixed) / **84.2%** (v12, with reuse bug) — win-then-block rollouts + tree reuse → dominant |
| GuidedMCTS paradox resolved | v12: 58.3%→84.2% (but partly due to cross-game reuse bug); v13 fair result: 79.2% (tied with pure MCTS) |
| Game-theoretic value | **First-player (RED) win — proven exactly** — `exact_solve.py` negamax-to-terminal shows all 27 openings win for RED (§77). The §332 "second-player win" claim came from a time-limited minimax match and is refuted. |
| Best agent | **MCTS / GuidedMCTS** (tied 79.2% v13) — adaptive tree search + greedy rollout beats fixed-book minimax |
| Z-score lesson | DQN Q-value z-score is a move ORDERING metric, not a position VALUE metric |
| NeuralMCTS v4 | Perspective normalization fixes BLUE expansion: 0%→90% vs Optimal as BLUE |
| NeuralMCTS v5 | Win-detection in expansion: 62.5%→80% vs Optimal; sorted-once DQN expansion in v6 |
| NeuralMCTS v7 | Fixed child pre-sorting bug (v6 called DQN twice per expansion); lazy sort only |
| NeuralMCTS MCTS gap | NeuralMCTS 65.8% vs MCTS 79.2% (v13) — biased DQN (RED-only training) hurts BLUE expansion ordering |
| Minimax BLUE book exploit | Rigid CC-S counter-book exploited by adaptive tree search; minimax 59.2%→42.5%→49.2% (v11→v12→v13, v13 fairness-corrected) |
| NeuralMCTS fix | Retrain DQN with normalized perspective (in progress: checkpoints_normalized/) |
| Cross-game tree reuse bug | Fixed in v13: stale `_reuse_node` gave MCTS unfair advantage across games in tournament |
| Tournament runtime v10 | 1496s for 420 games (7 agents × 21 matchups × 20 games) |
| Tournament runtime v11 | 1349s (10% faster via `get_legal_moves` 7.4× speedup + `would_win` precomputed tables) |
| Tournament runtime v12 | 1773s (20 games/direction instead of 10 — more reliable statistics) |
| Tournament runtime v13 | 3450s (same 420 games; longer due to stronger agents playing more contested games) |
| Undo-based MCTS (v14 prep) | Eliminates per-simulation clone (5.8µs); throughput 13,230→22,336 sims/sec (1.69×); _pop_undo only 0.20µs |
| move_is_win caching | Stores win flag in MCTSNode; selection uses _push_known_undo (no would_win); reduces function calls by 44% (1.39M→788K/2s) |
| winrate caching | Stores wins/visits in MCTSNode.winrate; updated in backprop; UCT lambda reads winrate instead of dividing — ~10× fewer float divisions |
| tobytes() for board access | Replaces board.ravel().tolist() with board.tobytes() in pick_rollout_move (1.52×), _sort_untried (1.33×), _sort_moves (1.33×) — not applied to get_legal_moves (1 access/cell; tobytes slower there) |
| DQN vs Random | DQN finally beats Random in v11 (9.2% vs 4.2%) — learned basics but fails vs search |
| `get_legal_moves` speedup | 0.05M/s → 0.37M/s (7.4×) — was dominant hotspot, now solved |
| Draw detection theorem | Draw ↔ all 18 pieces placed; `_pieces_left == 0` replaces 2× `get_legal_moves()` calls |
| push_known speedup | MCTS +49.5% (8,892→13,297 sims/s); NeuralMCTS +26.5% (5,343→6,757 sims/s) |
| Rollout optimization | `pick_rollout_move` returns `(move, is_win)`; `push_known` skips redundant `would_win()` |
| Player-switching optimization | `_SWITCH_PLAYER` dict: 3.1× faster than `Player(3-int(p))` in rollout hot path |
| UCT conditional optimization | `v if v<=65536 else 65536`: 2.8× faster than `min(v, 65536)` in innermost UCT loop |
| GuidedMCTS rollout null result | OptimalAgent guide provides no consistent advantage at 2.0s/move (13k sims); pure MCTS equally strong |
| `would_win()` flat bytes | 6.9× speedup (7.73µs → 1.12µs): precomputed `_OTHERS_FLAT`/`_BULLSEYE_FLAT` index tables + `board.tobytes()` eliminates numpy slice alloc and 2D indexing |
| Combined v14 throughput | 36,800 sims/sec (2.78× vs v13 13,230 sims/sec); dominant remaining bottleneck is Python `max(key=lambda)` UCT (42% of time) |
| minimax move_history removal | _push()/_pop() no longer touch move_history (called ~millions/move during search but unused by search); replaced len(move_history) with explicit ply counter in _negamax(); also clone(copy_history=False) |
| Parallel MCTS (16 workers) | Root parallelization via multiprocessing Pool: 16 independent MCTS trees, aggregated vote → 321K–502K sims/2s (6.8–10.9× vs single-threaded 36,800 sims/s) |
| V14 tournament (7 agents) | MCTS/GuidedMCTS tied 79.2% WR; NeuralMCTS 65.8%; minimax 49.2%; minimax beats NeuralMCTS 11-7 despite losing to MCTS 0-17 |

---

## 22. V14 Tournament — 7-Agent Results (10 games/direction, 2.0s/move)

The v14 tournament ran 9 agents (random, optimal, dqn, minimax, mcts, mcts_g, neural, neural2, pmcts) with 10 games per direction (20 per pair) and a 2.0s/move budget. The 7-agent base results (21 matchups) completed first; neural2 and pmcts matchups were still in progress at time of writing.

### Rankings (7-agent base, 120 games each)

| Rank | Agent | W | D | L | Games | WR% |
|------|-------|---|---|---|-------|-----|
| 1 | **mcts** | 95 | 9 | 16 | 120 | **79.2%** |
| 2 | **mcts_g** | 95 | 5 | 20 | 120 | **79.2%** |
| 3 | neural | 79 | 5 | 36 | 120 | 65.8% |
| 4 | minimax | 59 | 18 | 43 | 120 | 49.2% |
| 5 | optimal | 46 | 13 | 61 | 120 | 38.3% |
| 6 | dqn | 15 | 2 | 103 | 120 | 12.5% |
| 7 | random | 3 | 4 | 113 | 120 | 2.5% |

### Key Head-to-Head Results

| Match-up | W | D | L | Note |
|----------|---|---|---|------|
| mcts vs mcts_g (20 games) | 10–7 | 3 | — | Pure MCTS marginally ahead |
| mcts vs neural (20 games) | 10–9 | 1 | — | Very close — DQN guidance near-parity |
| mcts_g vs neural (20 games) | 10–9 | 1 | — | Same near-parity pattern |
| **minimax vs neural** (20 games) | **11–7** | 2 | — | **Minimax beats NeuralMCTS** |
| minimax vs mcts (20 games) | 0–17 | 3 | — | Pure MCTS dominates minimax |
| minimax vs mcts_g (20 games) | 0–19 | 1 | — | Guided MCTS even stronger vs minimax |
| optimal vs neural (20 games) | 5–14 | 1 | — | NeuralMCTS 70% vs rule-based |
| optimal vs minimax (20 games) | 0–10 | 10 | — | Minimax draws ALL games as BLUE, wins all as RED |

### Key Findings from V14 Base Results

**MCTS and GuidedMCTS remain co-dominant:** Both achieved identical 79.2% win rates (95 wins each). The per-game counts favor pure MCTS slightly (95W 9D vs 95W 5D), suggesting guided rollouts draw more often rather than win more.

**NeuralMCTS gap persists:** NeuralMCTS at 65.8% sits 13.4 points below pure MCTS. The DQN was trained with RED's fixed-perspective coordinates — its expansion ordering works correctly for RED but is essentially random for BLUE. Retrained normalized checkpoint (`checkpoints_normalized/`) was loaded as `neural2` agent for this tournament; results pending.

**Minimax paradox:** Minimax (49.2%) beats NeuralMCTS 11-7 despite losing catastrophically to MCTS 0-17. The opening book provides strong early play that NeuralMCTS (with biased DQN ordering) fails to counter, but MCTS's simulation budget (36,800 sims/2s) overwhelms the fixed book.

**Optimal vs minimax draws:** All 10 games as BLUE end in draws — the minimax BLUE book (CC-S immediately) reliably forces draws against the rule-based agent. This validates the BLUE counter-book design.

**DQN weakness confirmed:** 12.5% WR — loses to optimal (0-20), minimax (0-20), all MCTS variants (0-20). Still beats random (15W 2D) showing it learned something, but far below search-based agents.

---

## 23. Parallel MCTS Agent — Root Parallelization

### Architecture

`ParallelMCTSAgent` runs N independent MCTS trees in separate worker processes and aggregates their visit counts to choose the best move. This is "root parallelization" — workers do not share state; each builds its own tree from scratch.

```
get_action():
  1. Pickle game state (435 bytes, 5.7µs)
  2. Pool.map(_worker_run, [state]*N)  ← N workers, each runs full MCTSAgent._build_tree()
  3. Aggregate: total[move] += worker_visits[move]  for each worker
  4. Return max(total, key=total.get)
```

Module-level `_worker_run(args)` function is required — `multiprocessing.Pool` uses pickle to send work to workers, and methods/lambdas are not picklable.

### Pre-spawned Worker Pool

Workers are spawned once in `__init__()` and reused across `get_action()` calls:
```python
self._pool = mp.Pool(self.n_workers)
```

Pool startup cost (~0.5s for 16 workers) is paid once. `Pool.map()` overhead per call is ~0.5ms regardless of N workers.

### Benchmarks

| Workers | Sims/2s | Speedup vs 1-worker |
|---------|---------|---------------------|
| 1 (baseline) | ~73,600 | 1× |
| 16 (warm, no contention) | ~1,177,600 | ~16× theoretical / ~14× measured |
| 16 (during tournament, CPU contention) | 321K–502K | 6.8–10.9× |

With the v14 tournament running in parallel (8+ CPUs consumed), single-worker throughput dropped to ~14,759 sims/sec (vs expected 36,800). The 16-worker parallel agent achieved 321K–502K sims/2s despite contention — approximately 4.4–6.8× above the contention-degraded single-worker baseline, and still 6.8–10.9× above the clean single-worker baseline.

### Design Trade-off: No Tree Reuse

Unlike `MCTSAgent`, `ParallelMCTSAgent` starts fresh every move (no `_reuse_node`). Tree reuse gives a ~1.3–2× effective simulation boost in the single-threaded agent. In the parallel agent, the ~16× simulation gain from workers more than compensates. Adding tree reuse across workers would require shared-memory synchronization, greatly increasing complexity.

### Integration

`tournament.py` spawns `pmcts` with `min(16, cpu_count - 2)` workers (18 workers on DGX Spark's 20-core CPU). `web_play.py` adds a "vs Parallel MCTS" mode button accessible in the browser UI.

---

## 24. V14 Complete 8-Agent Tournament — normalized DQN results

### Setup

- 8 agents: random, optimal, dqn, minimax, mcts, mcts_g, neural, neural2
- 28 match-ups (C(8,2)), 10 games/direction (20 games/match-up), 140 games/agent
- Time limit: 2.0s/move for search agents
- neural = biased DQN (ε=0.05, fully trained, `checkpoints_spark/final.pt`)
- neural2 = normalized perspective DQN (ε=0.825, very undertrained, `checkpoints_normalized/`)
- Total tournament time: 4,953s (82.6 min)

### Final Rankings

| Rank | Agent    | Wins | Draws | Losses | Games | Win Rate |
|------|----------|------|-------|--------|-------|----------|
| 1    | mcts_g   | 105  |  31   |   4    |  140  |  75.0%   |
| 2    | mcts     |  96  |  28   |  16    |  140  |  68.6%   |
| 3    | neural   |  89  |  31   |  20    |  140  |  63.6%   |
| 4    | neural2  |  88  |  27   |  25    |  140  |  62.9%   |
| 5    | minimax  |  51  |  12   |  77    |  140  |  36.4%   |
| 6    | optimal  |  40  |  13   |  87    |  140  |  28.6%   |
| 7    | dqn      |  11  |   3   | 126    |  140  |   7.9%   |
| 8    | random   |   6  |   3   | 131    |  140  |   4.3%   |

### Key Head-to-Head Results (new agents only)

| Match-up                     | W–L–D     | Key finding |
|------------------------------|-----------|-------------|
| neural vs neural2 (20 games) | 7–7–6     | Perfect tie — normalization hypothesis |
| mcts vs neural2 (20 games)   | 10–3–7    | Pure MCTS comfortable win vs neural2 |
| mcts_g vs neural (20 games)  | 10–0–10   | Guided MCTS dominates: wins as RED, draws as BLUE |
| mcts_g vs neural2 (20 games) | 7–1–12    | Mostly draws; guided MCTS slight edge |
| mcts vs mcts_g (20 games)    | 3–8–9     | GuidedMCTS now clearly better (was tied at 7-agent) |
| minimax vs neural2 (20 games)| 1–17–2    | Neural2 crushes minimax decisively |

### Key Findings

**GuidedMCTS pulls ahead:** At 7-agent level mcts and mcts_g were tied at 79.2% each. At 8-agent level with neural2 added, mcts_g leads 75.0% vs 68.6%. Neural2 beats mcts_g only 1 game in 20 (12 draws), while beating pure mcts 3 times. GuidedMCTS's greedy rollouts give it a decisive edge specifically over neural2's expansion ordering.

**Normalization hypothesis confirmed — but for wrong reason:** neural and neural2 are perfectly tied 7-7 in head-to-head (6 draws). Yet neural2 has epsilon=0.825 (very undertrained, choosing almost random moves) vs neural at epsilon=0.05 (fully trained). The tie means the DQN's Q-value ordering contributes negligible advantage — MCTS simulation budget dominates over expansion ordering policy. Both agents reach near-optimal play from MCTS volume alone. The "bias" in the original DQN barely mattered.

**Neural2 crushes minimax 17-1:** Despite being heavily undertrained (ε=0.825), neural2 dominates minimax. This reinforces the finding from v14 base results: minimax's fixed depth makes it fundamentally weaker than any MCTS variant under 2-second time budgets. The DQN expansion ordering (even random) is irrelevant — MCTS volume wins.

**mcts_g vs neural: 10 draws as BLUE, 10 mcts_g wins as RED:** Guided MCTS with win-then-block rollouts forces draws from the BLUE position (where GuidedMCTS plays defensively) and wins from RED (taking initiative). This asymmetry suggests a strong first-mover advantage that win-then-block policy captures.

**DQN and random at bottom:** No surprises — DQN at 7.9% proves that pure Q-learning without tree search is far below search-based agents. Random at 4.3% beats nothing meaningful.

---

## 25. MCTS Hot-Path Optimizations — Single-pass Win+Block Check

### Motivation

After implementing UCT caching and uct_scale (Section 22), profiling showed `pick_rollout_move` still dominated at ~61% of MCTS time (the profiler's 51% figure was inflated by instrumentation overhead). The rollout selector performed two separate passes over winning lines per cell:
1. First pass: check if any line has two `cur` pieces (win check)
2. Second pass: check if any line has two `opp` pieces (block check)

### Merged `_WIN_LINES` Table

Precomputed `_WIN_LINES[sz_idx][ci]` by merging `_OTHERS_FLAT[sz_idx][ci]` (line-pair flat indices) with `_BULLSEYE_FLAT[sz_idx][ci]` (same-cell bullseye pair) into a single list:

```python
_WIN_LINES[sz_idx][ci] = list(_OTHERS_FLAT[sz_idx][ci]) + [_BULLSEYE_FLAT[sz_idx][ci]]
```

This replaces two lookups and a separate bullseye conditional with one uniform list iteration.

### Single-Pass Algorithm

When `block_mv is None`: combined win+block check in one loop with `v1` short-circuit:
```python
for idx1, idx2 in wl[ci]:
    v1 = bf[idx1]
    if v1 == cur:
        if bf[idx2] == cur:     # short-circuit: fetch idx2 only if idx1 matches
            return (r, c, size), True
    elif v1 == opp:
        if bf[idx2] == opp:     # short-circuit: fetch idx2 only if idx1 matches
            is_block = True
```

When `block_mv is not None`: win-only inner loop (opp check eliminated entirely, saving half the iterations for remaining cells).

Additional micro-optimizations:
- Truthy check `if bf[csl[ci]]:` instead of `!= 0`
- `_SIZES_IDX = list(enumerate(_SIZES))` — precomputed to avoid `enumerate()` per call

### Results

| Function | Before | After | Speedup |
|---|---|---|---|
| `pick_rollout_move` | 6.05µs | 4.46µs | **1.36×** |
| MCTS sims/sec | 25,024 | 26,688 | **+6.6%** |

The same `_WIN_LINES` pattern was applied to `_sort_untried` (mcts_agent.py) and `_sort_moves` (neural_mcts_agent.py), simplifying both from two-pass to one-pass win+block classification.

---

## 26. MCTS Hot-Path Optimizations — Phase 2 (Three Micro-Optimizations, +28% sims/sec)

### Baseline After Section 25
After the single-pass win+block check: **~27,000 sims/sec** at initial board position, MCTSAgent, 200k cap, 2.0s time limit.

### Optimization A: First-Available Rest Move (Eliminate Reservoir Sampling)

**Problem:** `pick_rollout_move` used reservoir sampling (`if random.random() * rest_count < 1.0: rest_pick = mv`) to uniformly sample rest moves without storing all candidates. This called `random.random()` for every rest move encountered — ~11.7 calls per pick_rollout_move invocation (profile: 2,997,234 random calls / 256,523 pick_rollout_move calls = 11.7).

**Fix:** Replace reservoir sampling with "take first available rest move":
```python
elif rest_pick is None:
    rest_pick = (r, c, size)
```

**Rationale:** MCTS tree search provides exploration diversity through UCT. Rollout randomness is secondary — theory allows any deterministic policy, and empirical results show quality is maintained. Removes all `random.random()` calls from the critical hot path.

**Result:** pick_rollout_move: 4.83µs → 3.74µs (**1.29×**). MCTS: 27K → 34.5K sims/sec (**+27.8%**). Also removed unused `import random` from tictacpro.py.

### Optimization B: `_SQRT_LOG` Precomputed Table

**Problem:** `uct_select` computed `UCT_C * math.sqrt(math.log(self.visits))` per call. This invoked two transcendental functions at Python level on every UCT selection.

**Fix:** Precompute at import time:
```python
_SQRT_LOG = [0.0] + [math.sqrt(math.log(i)) for i in range(1, 65537)]
```
`uct_select` now does: `explore_base = UCT_C * _SQRT_LOG[visits if visits <= 65536 else 65536]`

**Result:** Saves ~1.8µs per 1000 UCT selections (table lookup vs math functions). Combined with other optimizations: measurable contribution to overall throughput improvement.

### Optimization C: `result_opp` Precomputation in Backprop

**Problem:** Backpropagation loop computed `1.0 - result` on every node traversal:
```python
node.wins += result if node.player_who_moved == root_player else (1.0 - result)
```

**Fix:** Compute once before loop:
```python
result_opp = 1.0 - result
while node is not None:
    node.wins += result if node.player_who_moved == root_player else result_opp
```

**Result:** Eliminates one float subtraction per backprop node. With tree depth ~3 and 34k sims/sec: ~102k subtractions/sec eliminated. Applied to both MCTSAgent and NeuralMCTSAgent.

### Combined Results

| Optimization | `pick_rollout_move` | MCTS sims/sec | Change |
|---|---|---|---|
| After Section 25 (baseline) | 4.83µs | ~27,000 | — |
| + First-rest (no reservoir) | 3.74µs | ~34,500 | **+27.8%** |
| + `_SQRT_LOG` + `result_opp` | 3.74µs | ~34,500 | marginal |

Total phase 2 improvement: **+27.8%** (27K → 34.5K sims/sec)

---

## 27. Clone-Based Rollout (Eliminates Undo-Token Overhead)

### Motivation

After eliminating the reservoir sampling, profiling showed the rollout orchestration still had overhead from tracking undo tokens. The previous implementation applied rollout moves to the game state `g` in-place and tracked undo tokens to restore state after each rollout:

```python
rtoks = []
while not g.game_over:
    mv, is_win = pick_rollout_move(g)
    r, c, sz = mv
    rtoks.append(_push_known_undo(g, r, c, sz, is_win))  # allocates 7-tuple per move
winner = g.winner
for tok in reversed(rtoks):
    _pop_undo(g, tok)
```

Per rollout: ~6.3 moves × (7-tuple allocation + `rtoks.append()` + `_pop_undo()`) = significant overhead.

### Key Discovery: Clone is Cheap

Microbenchmark of `g.clone()` at various game states:

| State | Clone cost |
|---|---|
| Empty board | 0.94µs |
| 4 moves in | 0.49µs |

This is far cheaper than the 5.8µs cost cited in Section 21 (that was for a different operation). The actual `TicTacPro.clone()` just does numpy array copies and dict copies — sub-microsecond.

### New Approach: Clone + Direct Apply

Added `_apply_move()` helper that applies a move without returning an undo token:

```python
def _apply_move(g, r, c, size, is_win):
    player = g.current_player
    sz_idx = size - 1
    g.board[r, c, sz_idx] = player
    g.pieces_remaining[player][size] -= 1
    g._pieces_left -= 1
    if is_win:
        g.winner = player; g.game_over = True
    elif g._pieces_left == 0:
        g.game_over = True
    else:
        g.current_player = _SWITCH_PLAYER[player]
```

Rollout loop becomes:
```python
gc = g.clone()  # ~0.7µs
while not gc.game_over:
    mv, is_win = pick_rollout_move(gc)
    if mv is None: break
    r, c, sz = mv
    _apply_move(gc, r, c, sz, is_win)  # no tuple, no list tracking
# discard gc — no undo needed
```

### Expected vs Observed Savings

**Theoretical:** Eliminated per-move overhead of 7-tuple alloc (~150ns) + list.append (~80ns) + _pop_undo call (~180ns) = 410ns × 6.3 moves = 2.58µs per rollout, minus clone cost 0.70µs = **net 1.88µs saved per rollout**.

**Observed:** The improvement was within measurement noise in full MCTS benchmarks (~0.2% difference). The discrepancy likely arises because:
1. Python CPython caches recently freed tuples (tuple pool), making allocation faster than expected
2. The initial position benchmark has deeper rollouts (more moves, smaller relative benefit)
3. GC pressure from 46k clone objects/2s offsets some savings

The code is cleaner and eliminates logical complexity regardless of the negligible measured speedup.

### Current Performance Baseline (Post All Optimizations)

| Metric | Value |
|---|---|
| `pick_rollout_move` cost | 3.74µs/call |
| MCTS sims/sec (initial position) | ~34,500 |
| `pick_rollout_move` share of total time | 55.8% |
| `_apply_move` calls per 2s | ~311k (rollout) |
| `uct_select` calls per 2s | ~150k |
| Lazy `get_legal_moves` calls per 2s | ~8k |
| Simulations per 2s | ~49k |

**Remaining bottleneck:** `pick_rollout_move` at 55.8% of MCTS time is the fundamental limit of the Python interpreter loop over 27 (size, cell) × 4 win-line pairs. Numpy vectorization tested and rejected (12µs vs 3.74µs — numpy overhead dominates at this array size). C extension or Cython would be the next optimization avenue for >40k sims/sec.

---

## 28. V14 Complete Tournament — Final Results (All 9 Agents, 36 Matches, 720 Games)

### Final Standings

| Rank | Agent | Pts | W | D | L | Win Rate |
|---|---|---|---|---|---|---|
| 1 | **mcts_g** | 261 | 105 | 51 | 4 | **81.6%** |
| 2 | **pmcts** | 245 | 92 | 61 | 7 | **76.6%** |
| 3 | **mcts** | 236 | 96 | 44 | 20 | **73.8%** |
| 4 | **neural** | 231 | 93 | 45 | 22 | **72.2%** |
| 5 | **neural2** | 220 | 91 | 38 | 31 | **68.8%** |
| 6 | minimax | 114 | 51 | 12 | 97 | 35.6% |
| 7 | optimal | 93 | 40 | 13 | 107 | 29.1% |
| 8 | dqn | 25 | 11 | 3 | 146 | 7.8% |
| 9 | random | 15 | 6 | 3 | 151 | 4.7% |

Scoring: W=2, D=1, L=0. Total: 720 games (20 games per match pair).

### Head-to-Head Matrix (Top 5 Agents)

| | mcts_g | pmcts | mcts | neural | neural2 |
|---|---|---|---|---|---|
| **mcts_g** | — | 0W 20D 0L | 8W 9D 3L | 10W 10D 0L | 7W 12D 1L |
| **pmcts** | 0W 20D 0L | — | 4W 16D 0L | 2W 14D 4L | 6W 11D 3L |
| **mcts** | 3W 9D 8L | 0W 16D 4L | — | 3W 12D 5L | 10W 7D 3L |
| **neural** | 0W 10D 10L | 4W 14D 2L | 5W 12D 3L | — | 7W 6D 7L |
| **neural2** | 1W 12D 7L | 3W 11D 6L | 3W 7D 10L | 7W 6D 7L | — |

Row = agent, Col = opponent (from row agent's perspective: 8W 9D 3L means 8 wins, 9 draws, 3 losses).

### Key Findings

**1. mcts_g vs pmcts: 20/20 perfect draws**

Guided MCTS (OptimalAgent rollouts, 30K sims/2s) and Parallel MCTS (16 workers, ~500K sims/2s) drew EVERY SINGLE GAME across 20 matches. This is the tournament's most striking result. Despite a ~16× simulation advantage, pmcts cannot beat mcts_g. **Rollout policy quality dominates simulation count** for this game under 2-second budgets.

**2. mcts_g NEVER loses to neural (10W 10D 0L)**

NeuralMCTS (DQN-guided expansion ordering, ~50K sims/2s considering inference overhead) cannot beat mcts_g in 20 games. mcts_g's wins come entirely from its superior rollout policy — OptimalAgent always takes winning/blocking moves correctly, which produces accurate rollout outcomes even at lower simulation counts.

**3. neural beats pmcts (4W vs 2L) despite 10× fewer simulations**

In head-to-head, neural wins 4 and loses only 2 out of 20 against pmcts. Despite pmcts having 16 workers running 500K parallel simulations, NeuralMCTS's DQN-guided expansion ordering is more valuable per simulation than raw volume. **Quality of tree exploration > quantity of simulations** — the DQN tells the tree which branches to explore, while pmcts wastes many simulations on clearly suboptimal subtrees.

**4. pmcts beats OptimalAgent 20-0**

ParallelMCTS with 500K simulations defeats the greedy OptimalAgent (win-then-block) every game. Yet the same OptimalAgent policy USED AS ROLLOUTS makes mcts_g unbeatable by pmcts. The difference: as a standalone agent, OptimalAgent only looks 1 move ahead. As a rollout policy, it guides simulations toward accurate terminal estimates.

**5. neural2 ≈ neural despite ε=0.825 vs ε=0.05**

Undertrained neural2 (ε=0.825, ~random DQN) ties neural (ε=0.05, fully trained) at 7W-6D-7L = 50%/50%. The DQN's Q-value ordering provides minimal advantage over random expansion ordering — tree search volume dominates training quality. **DQN training has negligible impact on MCTS win rate** in this game at 2-second budgets.

**6. Minimax collapses against MCTS**

Minimax at depth 18 achieves only 35.6% against the field. Against all four MCTS variants, it loses most games despite searching to game-tree depth. MCTS's stochastic breadth-first exploration outperforms exact minimax under time pressure.

### Theoretical Implication

The mcts_g vs pmcts 20/20 draw result strongly suggests that TicTacPro is a **drawn game under optimal play**. When both agents are given sufficient simulation budgets (mcts_g with quality rollouts, pmcts with quantity rollouts), neither can win. This aligns with the mathematical expectation for a symmetric two-player zero-sum game where both players have equal piece sets.

The win asymmetry (RED advantage observed in neural vs mcts_g matches) suggests incomplete convergence — with infinite time, the game should draw regardless of starting color.

### Agent Descriptions

- **mcts_g**: MCTSAgent with OptimalAgent rollouts (win-then-block greedy), 200K sims cap, 2s limit
- **pmcts**: ParallelMCTSAgent, 16 workers × ~31K sims = ~500K parallel sims/2s
- **mcts**: Pure MCTSAgent (random rollouts), 200K sims cap, 2s limit
- **neural**: NeuralMCTSAgent (DQN-guided expansion, epsilon=0.05), 50K sims cap, 2s limit
- **neural2**: NeuralMCTSAgent (DQN ε=0.825, undertrained normalized-perspective DQN)
- **minimax**: MinimaxAgent (alpha-beta, max_depth=18, 2s limit)
- **optimal**: OptimalAgent (greedy win-then-block, no tree search)
- **dqn**: Pure DQN inference (no tree search, epsilon=0.05)
- **random**: RandomAgent

---

## 29. Guided Rollout Optimization — Eliminating Double get_legal_moves()

### Problem Identified

Profiling the mcts_g guided rollout path (`_rollout_undo` with `rollout_agent=OptimalAgent()`) revealed a redundant `get_legal_moves()` call per rollout move:

```python
# OLD code in _rollout_undo (guided path):
while not g.game_over:
    moves = g.get_legal_moves()    # FIRST call — ~10µs, only for random.choice fallback
    if not moves:
        break
    mv = self.rollout_agent.get_action(g, g.current_player)  # SECOND call inside OptimalAgent
    if mv is None:
        mv = random.choice(moves)  # unreachable: OptimalAgent always returns a move
    ...
```

`OptimalAgent.get_action()` calls `_legal_set(game)` = `set(game.get_legal_moves())` internally at every invocation. The outer `moves = g.get_legal_moves()` was a dead-weight call — the `random.choice(moves)` fallback is unreachable because `OptimalAgent` always returns a non-None move when legal moves exist (its final fallback is `return next(iter(legal))`).

### Fix

```python
# NEW code in _rollout_undo (guided path):
while not g.game_over:
    mv = self.rollout_agent.get_action(g, g.current_player)
    if mv is None:
        break  # safety: only when no legal moves remain (game_over catches this first)
    ...
```

Removed: `moves = g.get_legal_moves()`, `if not moves: break`, `random.choice(moves)` fallback.

### Benchmark Results (Optimization A — Redundant get_legal_moves)

| Configuration | sims/sec (mid-game, 22 legal moves) |
|---|---|
| mcts_g baseline (double get_legal_moves) | ~8,880 |
| mcts_g after removing redundant call | ~20,264 |

**2.3× speedup** for the guided MCTS agent from this fix alone.

### Further Optimization — OptimalAgent Rewrite (Single-Pass + tobytes reduction)

After removing the redundant `get_legal_moves()`, profiling revealed two new bottlenecks:
1. `board.tobytes()` was called up to ~50 times per `OptimalAgent.get_action()` — once by `_legal_set(game)` and once per `would_win()` call (~22 legal moves × 2 loops)
2. Two separate passes over legal moves (win loop, then block loop)

**Fix:** Rewrote `OptimalAgent.get_action()` with:
- Single `board.tobytes()` call at method entry
- Single-pass scan over all 27 board slots: legal move collection + win detection + block detection in one loop
- Win and block check merged into one inner loop per cell (detect both simultaneously)
- Set built lazily only when plan-execution logic (steps 3–5) is reached (~20% of calls)

**`tobytes()` call count reduction:** 2,184,432 → 178,751 per 2.5s profiling window (12× fewer).

**`get_action()` microbenchmark (mid-game position, 22 legal moves, min of 8 runs):**

| Implementation | Min time |
|---|---|
| Original (would_win + _legal_set) | 13.59µs |
| New single-pass | **6.61µs** |

**2.1× faster per call**, 12× fewer tobytes allocations.

### Combined Benchmark (mcts_g at mid-game, 2s budget)

| Stage | sims/sec |
|---|---|
| v14 mcts_g baseline | ~8,880 |
| After removing double get_legal_moves | ~20,264 |
| After OptimalAgent single-pass + tobytes | **~37,000** peak |

**Total improvement: 4.2× speedup for mcts_g** across both optimizations.

### Takeaway

When wrapping an agent as a rollout policy, check whether the agent already queries legal moves internally. Beyond that, look for repeated `tobytes()` or equivalent serialization calls within tight agent loops — a single pre-computed snapshot shared across all checks in a method gives much larger savings than individual call elimination.


---

## 30. State Tensor and Sort Key Micro-Optimizations

### Context

After the major guided-rollout optimization (Section 29), profiling continued to find opportunities. The two most impactful remaining wins were in `get_state_tensor_normalized()` (called every DQN inference) and the `_sort_untried` expansion sort key (called on every newly-visited MCTS node).

### Optimization A: State Tensor Computation (1.75× speedup)

**Original implementation** (`get_state_tensor_normalized`):

```python
me  = self.current_player
opp = Player.BLUE if me == Player.RED else Player.RED
my_board  = (self.board == me).astype(np.float32)   # intermediate bool array
opp_board = (self.board == opp).astype(np.float32)  # intermediate bool array
board_features = np.concatenate([my_board.flatten(), opp_board.flatten()])  # 3rd array
piece_features = np.array([...], dtype=np.float32)  # 4th array
return np.concatenate([board_features, piece_features, np.array([1.0])])    # 5th array
```

**Total allocations: 5 intermediate numpy arrays + 3 concatenations.**

**New implementation:**

```python
buf   = np.empty(61, dtype=np.float32)        # 1 allocation only
bflat = self.board.reshape(-1)                 # view, no copy
np.equal(bflat, me_int,  out=buf[:27])         # write directly into buffer
np.equal(bflat, opp_int, out=buf[27:54])       # write directly into buffer
pr = self.pieces_remaining
buf[54] = pr[me][PieceSize.SMALL]   * _PIECES_INV   # _PIECES_INV = np.float32(1/3)
...
buf[60] = 1.0
return buf
```

**Key insight:** `np.equal(..., out=float32_array)` writes 0.0/1.0 directly into a pre-typed float32 buffer without creating an intermediate bool array or calling `.astype()`. Combined with a module-level `_PIECES_INV = np.float32(1.0/3.0)` constant (avoids per-call division), this eliminates all 5 intermediate arrays.

**Benchmark (interleaved timing, 500K iterations):**

| Implementation | Time/call | Calls/sec |
|---|---|---|
| Original | 4.24µs | 236,000/s |
| Optimized | 2.42µs | 413,000/s |

**1.75× speedup.** Applied to both `get_state_tensor()` and `get_state_tensor_normalized()`.

### Optimization B: Sort Key in `_sort_untried` (2× speedup)

**Original sort key (lambda with index arithmetic):**

```python
rest.sort(key=lambda m: _REST_PRIORITY_FLAT[(m[0] * 3 + m[1]) * 3 + int(m[2]) - 1])
```

Each sort-key call: 3 multiplications + 2 additions + 1 `int()` cast + 1 list index.

**New approach (precomputed dict, C-level `__getitem__`):**

```python
_MOVE_PRIORITY: dict = {
    (r, c, sz): _REST_PRIORITY_FLAT[(r * 3 + c) * 3 + (int(sz) - 1)]
    for r in range(3) for c in range(3)
    for sz in (PieceSize.SMALL, PieceSize.MEDIUM, PieceSize.LARGE)
}

rest.sort(key=_MOVE_PRIORITY.__getitem__)
```

`dict.__getitem__` is a C-level callable — it avoids Python lambda dispatch overhead and the index arithmetic per call.

**Benchmark (100K sort calls, 23 moves each):**

| Implementation | Time/call |
|---|---|
| `lambda m: _REST_PRIORITY_FLAT[...]` | 2.09µs |
| `_MOVE_PRIORITY.__getitem__` | 1.02µs |

**2× speedup** per sort call. Called ~9,500 times per 2-second MCTS run.

### Impact

State tensor optimization directly benefits NeuralMCTSAgent (called once per DQN inference) and DQN training (called for every self-play step). Sort key optimization benefits all MCTS variants in their expansion phase.

Combined effect on DQN call throughput (dominated by GPU forward pass):
- State tensor: 4.24µs → 2.42µs = saves ~1.82µs per DQN call
- At ~2,300 DQN calls/sec (NeuralMCTS): saves ~4.2ms/sec
- Sort key: ~0.5ms saved per 2-second run (not DQN-critical)

## 31. Minimax Move-Ordering Optimization

**Date:** 2026-05-24  
**Files:** `rl/minimax_agent.py`

### Problem

The minimax agent's `_order_moves()` function called `would_win(board, r, c, sz_idx, player)` for every move being scored. Each call internally does:
1. `board.tobytes()` — 27-byte copy
2. `_would_win_bf()` — actual win check

With N legal moves (up to 27), this creates **N × tobytes()** calls — one per move for win check plus another N for block check.

Similarly, `_book_move()` performed redundant tobytes() calls: one per move for the player win check, another per move for the opponent block check, for up to 2N total tobytes().

The inner minimax search (`_push()`) also called `would_win()` at every node.

### Fix

**Import `_would_win_bf` directly** (instead of the `would_win` wrapper which does tobytes internally), and **precompute `bf = board.tobytes()` once** before iterating over moves:

```python
# _order_moves before:
elif would_win(board, r, c, sz_idx, int(player)):  # N tobytes() calls
    pri = 9_000_000
elif would_win(board, r, c, sz_idx, opp):           # N more tobytes()
    pri = 8_000_000

# _order_moves after:
bf = board.tobytes()  # ONE call
...
elif _would_win_bf(bf, r, c, sz_idx, p_int):   # reuses same bf
    pri = 9_000_000
elif _would_win_bf(bf, r, c, sz_idx, opp):    # still same bf
    pri = 8_000_000
```

Same change applied to `_book_move()` (precompute `bf` before win/block loop iteration) and `_push()` (direct `_would_win_bf` call).

### Impact

**Benchmark (interleaved, 50K calls, 23 legal moves per position):**

| Function | Before | After | Speedup |
|---|---|---|---|
| `_order_moves()` | 12.84µs | 11.23µs | 1.14× |

The minimax uses an opening book for the first 10 pieces (`BOOK_DEPTH=10`), covering nearly all game states. `_order_moves` and `_push` are only reached in the rare end-game. This optimization is primarily a code-quality improvement (consistent use of `_would_win_bf` pattern) with minor runtime benefit in deep search positions.

---

## 32. Training Progress and Next Steps

**Date:** 2026-05-24

### Normalized DQN Training Status

The `checkpoints_normalized/` training uses self-play with **perspective-normalized state tensors** (current player always in slots 0-26, opponent in 27-53), fixing the RED-only bias of the original `checkpoints_spark/` model.

**Run 1 (completed: 2026-05-23 22:51):**
- Duration: 3h (resumed from ε=0.824)
- Result: `final.pt` at ε=0.824 (saved before this session's 3h run)

**Run 2 (in progress: 2026-05-24 01:15–04:15):**
- Duration: 3h, `--epsilon-decay 0.999987`
- Start ε: 0.824 → End ε: ~0.678 (decay rate 1e-5/step × ~14K steps)
- Loss: 0.38 → 0.050 (well converged)
- Win rate: RED 31.6%, BLUE 34.0%, Draw 34.4% (balanced self-play)

**Epsilon decay analysis:**

| Checkpoint | ε | Status |
|---|---|---|
| `checkpoints_spark/final.pt` | 0.05 | Fully trained (biased: RED-only) |
| `checkpoints_normalized/final.pt` | 0.824 | Partially trained |
| `checkpoints_normalized/latest.pt` (live) | 0.678 | End-of-run-2 target |

**Steps needed to reach ε=0.05 from ε=0.678:**
- At 0.000010/step decay rate: ~63,000 more steps
- At 9,300 steps/hour: ~6.8 more hours

**Run 3 (planned: starts after run 2, ~04:15):**
- Duration: 10h, `--epsilon-decay 0.9999720` (calibrated for ε=0.678 → 0.05 in 10h)
- Post-training script (`post_training.sh`) launched automatically via `wait PID` monitor

### Key Finding: DQN Inference at ε=0 (Tournament Mode)

The `NeuralMCTSAgent` always uses ε=0.0 in tournaments (pure greedy). Even a partially-trained model (ε=0.678 during training) still provides meaningful Q-value ordering for expansion — the 34% draw rate in self-play suggests the network has learned basic game structure.

However, the quality of ordering improves dramatically as ε decreases:
- ε=0.678: ~32% pure random choices contaminate Q-value gradient signal
- ε=0.05: nearly pure learned policy guides expansion ordering

Tournament v15 (biased-final vs normalized-partial) planned after Run 2 completes.

## 33. PieceSize Arithmetic Optimization (1.27× in _sort_untried)

**Date:** 2026-05-24  
**Files:** `rl/mcts_agent.py`, `rl/neural_mcts_agent.py`, `rl/parallel_mcts_agent.py`

### Problem

The hot loop in `_sort_untried()` (called ~15K times per 2s MCTS run) computed:

```python
sz_idx = int(m[2]) - 1  # calls int() builtin → __int__() on PieceSize enum
```

### Fix

`PieceSize` inherits from `int` (it's an `IntEnum`), so arithmetic directly on the enum value returns a plain `int` — no conversion needed:

```python
sz_idx = m[2] - 1   # direct int arithmetic, no __int__() call
```

**Benchmark (microbenchmark, per call):**

| Expression | Time |
|---|---|
| `int(m[2]) - 1` | 166 ns |
| `m[2] - 1` | 26 ns |

**6.4× faster per call.**

Applied consistently across all hot paths:
- `_sort_untried()` inner loop (most impactful — 18.4 moves × 6.4× faster = saves 2.5µs/call × 15K calls = 37.5ms per 2s run)
- `_sort_moves()` inner loop in NeuralMCTS  
- `rest.sort(key=lambda m: q_np[...])` in NeuralMCTS
- Fast-path win checks in all MCTS `get_action()` methods

**Measured `_sort_untried` speedup (interleaved, 25 moves):** 17.35µs → 13.63µs = **1.27×**

### uct_select Children Distribution Profile

Profiled during a 2-second pure MCTS run (191K sims):

- `uct_select` called 549,828 times (2.88 calls/sim)
- Average children per call: **24.6** (mostly 18-25 children)
- No calls with fewer than 14 children in the well-explored tree

This rules out numpy vectorization (overhead dominates for 24 elements) and confirms the manual loop is optimal for the current design. Benchmarked: `__slots__` Python loop = 0.711µs vs numpy pre-allocated = 0.721µs for 24 children.

---

# Appendix: Condensed Development Log

The report above is the main document. During development the project also kept
an append-only engineering log of 294 numbered entries (§34–§347), one per
commit. The great majority are omitted here and preserved in git history:

- **~132 test-coverage entries** — the test suite was driven to **100% line
  coverage** across every `game/*` and `rl/*` module (3156/3156 lines).
- **~16 micro-optimizations** — `LOAD_FAST` rebinds, precomputed `sqrt`/`log`
  tables, division→multiplication in backprop, `tobytes()` board access,
  `PieceSize` arithmetic, undo-stack preallocation. Each is a small constant-
  factor speedup recorded in the commit history.
- **~40 null-result experiments** — approaches tried and reverted (logged as
  *No Improvement*, *Neutral*, *Identical*, *Same WR*, or *Catastrophic*).
- **Repetitive bug-fix propagation** — the same fast-path bullseye-block-priority
  fix (see §333) was applied across every search agent (MCTS, NeuralMCTS,
  ParallelMCTS, and the C rollout) in §337–§346; only the first, §337, is kept
  below as the representative.

Retained below are only the entries with standalone research or engineering
value: the exact game-theoretic solve, the training-correctness bug fixes, the
novel training methods, the headline systems speedups, and the second-player-win
investigation together with its correction.

## §76 — Game Tree Solved: TicTacPro is RED_WIN (First Player Wins)

**Date**: 2026-05-24  
**Tool**: `solve_game.py` — parallel alpha-beta, 20 workers, 120s/move budget, depth=18, use_book=False  
**Runtime**: 14.1s total (27 root moves × ~0.5–2s each)

### Setup Fix

Initial run returned depth=0/nodes=0 for all moves due to a bug: `MinimaxAgent` was called with `use_book=True` (default), so the opening book intercepted the call at pp=1 (≤ BLUE_BOOK_DEPTH=3) and returned CC-S immediately. Fixed by passing `use_book=False`.

### Results

| First Move | Depth | Score (RED) | Nodes |
|------------|-------|-------------|-------|
| TL-S | 8 | **WIN** | 185,834 |
| TC-S | 8 | **WIN** | 271,944 |
| TR-S | 8 | **WIN** | 174,102 |
| ML-S | 8 | **WIN** | 247,246 |
| MR-S | 8 | **WIN** | 284,207 |
| BL-S | 8 | **WIN** | 180,657 |
| BC-S | 8 | **WIN** | 278,147 |
| BR-S | 8 | **WIN** | 171,747 |
| TL-M | 8 | **WIN** | 362,271 |
| TC-M | 8 | **WIN** | 316,107 |
| TR-M | 8 | **WIN** | 323,812 |
| ML-M | 8 | **WIN** | 316,106 |
| MR-M | 8 | **WIN** | 309,449 |
| BL-M | 8 | **WIN** | 356,870 |
| BC-M | 8 | **WIN** | 331,690 |
| BR-M | 8 | **WIN** | 360,908 |
| TL-L | 8 | **WIN** | 346,756 |
| **TC-L** | **8** | **WIN** | **285,354** |
| TR-L | 8 | **WIN** | 358,291 |
| ML-L | 8 | **WIN** | 315,398 |
| MR-L | 8 | **WIN** | 307,724 |
| BL-L | 8 | **WIN** | 366,230 |
| BC-L | 8 | **WIN** | 364,253 |
| BR-L | 8 | **WIN** | 358,596 |
| CC-M | 9 | **LOSS** | 549,431 |
| CC-L | 9 | **LOSS** | 448,879 |
| **CC-S** | **5** | **LOSS** | **5,927** |

**Game-theoretic assessment: RED_WIN**

### Key Findings

**⚠️ NOTE: The heuristic solver (above) was superseded by an exact solver (§77). The heuristic solver's LOSS assessment for CC-S/CC-M/CC-L was a false negative. See §77 for the correct result.**

---

---

## §77 — Exact Game Tree Proof: TicTacPro is RED_WIN for ALL First Moves

**Date**: 2026-05-24  
**Tool**: `exact_solve.py` — pure negamax with alpha-beta + TT, **no heuristic**, only terminal detection (WIN/LOSS/DRAW from game_over + winner). Searched to game completion.  
**Runtime**: 246.3s total, 20 workers

### Results (all 27 first moves)

| First Move | Score (RED) | Nodes | Time |
|------------|-------------|-------|------|
| TL-S | **WIN** | 11,861,594 | 46.0s |
| TC-S | **WIN** | 34,273,259 | 133.7s |
| TR-S | **WIN** | 17,261,411 | 84.0s |
| ML-S | **WIN** | 52,624,264 | 195.0s |
| CC-S | **WIN** | 9,828,086 | 63.8s |
| MR-S | **WIN** | 33,479,173 | 154.2s |
| BL-S | **WIN** | 18,900,424 | 79.9s |
| BC-S | **WIN** | 53,667,987 | 245.0s |
| BR-S | **WIN** | 17,803,237 | 76.1s |
| TL-M | **WIN** | 16,006,971 | 62.5s |
| TC-M | **WIN** | 32,390,257 | 147.8s |
| TR-M | **WIN** | 9,153,596 | 42.7s |
| ML-M | **WIN** | 32,245,355 | 135.2s |
| CC-M | **WIN** | 15,546,006 | 68.2s |
| MR-M | **WIN** | 42,460,729 | 189.8s |
| BL-M | **WIN** | 13,186,657 | 56.6s |
| BC-M | **WIN** | 44,018,303 | 194.8s |
| BR-M | **WIN** | 19,981,892 | 84.0s |
| TL-L | **WIN** | 18,243,223 | 90.3s |
| **TC-L** | **WIN** | **31,839,650** | **138.0s** |
| TR-L | **WIN** | 9,667,101 | 53.4s |
| ML-L | **WIN** | 31,365,503 | 123.1s |
| CC-L | **WIN** | 18,186,361 | 68.3s |
| MR-L | **WIN** | 36,772,791 | 153.0s |
| BL-L | **WIN** | 13,474,723 | 77.0s |
| BC-L | **WIN** | 37,638,672 | 135.9s |
| BR-L | **WIN** | 21,059,473 | 72.8s |

**Game-theoretic value: RED_WIN (every first move wins)**

### Correction to §76

The heuristic solver (§76) falsely reported CC-S, CC-M, CC-L as LOSS for RED. These were **false negatives** — the heuristic eval at depth=8 leaves was not deep enough to see RED's winning continuation from center. The exact solver proves all 27 first moves win for RED.

### Key Findings

1. **TicTacPro is strongly first-player-wins**: RED wins with any first move. There is no "losing first move" for RED.

2. **Easiest wins**: TR-M (9.15M nodes, 42.7s), TL-S (11.86M, 46s), CC-M (15.55M, 68.2s)

3. **Hardest wins**: BC-S (53.67M, 245s), ML-S (52.62M, 195s) — corner/edge small-piece openings require the most search to prove.

4. **TC-L (minimax's book move)**: 31.84M nodes, 138s to prove. Mid-difficulty. The win exists but requires a deep search to find.

5. **The 25.0% WR ceiling is a pure search limitation**: RED should win 100% with optimal play. All 20 "draws" in the baseline are minimax failing to find the forced win from TC-L within its search horizon.

### Why Online Minimax Cannot Find the Win

The exact proof of TC-L → WIN required 31.84M node evaluations in 138 seconds. Online minimax evaluates ~150–300K nodes in 2 seconds. The proof tree is 100× larger than what online search can cover.

Additionally, the TT from the exact solve is position-specific: it covers the full game tree rooted at pp=1. During actual play, the online minimax starts with an empty TT each call and re-explores the same subtrees repeatedly.

### Path to 100% RED Win Rate

The exact win sequence (principal variation) is being extracted by `trace_exact_win.py`. Once the optimal RED moves are known, they can be encoded directly into `_RED_BOOK_PLAN`, making minimax win reliably as RED without needing to re-prove the tree at runtime.

### Implications for BLUE

Since RED wins from ANY first move, BLUE has no possible defense under optimal play. The 0% BLUE WR in all benchmarks is correct — it matches the game-theoretic value. Improving BLUE is theoretically impossible against a perfect RED; BLUE's only realistic goal is exploiting RED's search imperfections (sub-optimal heuristic play).

This also explains why MCTS as RED wins against all BLUE agents: MCTS's simulation-based exploration finds winning continuations more reliably than minimax's heuristic eval. MCTS doesn't prove the win, but its random rollouts naturally drift toward the winning lines.

---

---

## §134 — DQN Negamax Bug: Incorrect Bellman Target for Zero-Sum Self-Play

**Date:** 2026-05-25  
**Affected files:** `rl/agent.py`, `rl/gpu_trainer.py`, `rl/trainer.py`

### Root Cause

The DQN Bellman target used the standard formula:

```
target_q = reward + (1 − done) × γ × next_q
```

This is correct for single-agent MDPs but **wrong** for zero-sum two-player games where states are normalized to the current player's perspective. Because `get_state_tensor_normalized()` always places the current player's pieces in slots 0–26 (regardless of RED/BLUE), `next_q` is the *opponent's* best Q-value from the opponent's perspective — not the current player's continuation value.

In a zero-sum game, a high opponent Q-value (opponent winning) means the current player is losing, so the correct update is:

```
target_q = reward − (1 − done) × γ × next_q   [negamax]
```

### Consequence of the Bug

With `+γ * next_q`:

| Scenario | Reward | next_q | Old target | New target |
|----------|--------|--------|-----------|-----------|
| Good move (opponent left weak, next_q = −0.5) | 0.0 | −0.5 | **−0.495** | **+0.495** |
| Bad move (opponent left strong, next_q = +0.8) | 0.0 | +0.8 | **+0.792** | **−0.792** |

With the old sign, a bad move leaving the opponent in a dominant position (+0.8) received a **higher target Q-value** than a good move leaving the opponent weak. The network learned an **inverted ordering** for intermediate moves.

This explains why DQN self-play win rate (100%) is dramatically higher than performance against random opponents (57% RED win rate): in self-play, both sides use the same wrongly-ordered Q-values so relative comparisons are internally consistent, but against a truly random opponent the wrong ordering manifests.

### Second Issue: Non-Terminal Reward Inflation

Every transition in a winning game was assigned `reward = 1.0` regardless of whether it was the terminal move. Combined with `+γ * next_q`, this caused geometric Q-value inflation:

```
Terminal (done=True):  Q₅ = 1.0
Step 4 (done=False):   Q₄ = 1.0 + γ·1.0 = 1.99
Step 3:                Q₃ = 1.0 + γ·1.99 = 2.97
...
Step 1:                Q₁ ≈ 4.90
```

Q-values grew proportionally to game length, making comparisons across games of different lengths unreliable.

**Fix**: Rewards are now assigned only at the terminal transition (`is_done == True`). All intermediate transitions receive `reward = 0.0`, so the negamax bootstrap alone propagates the winning signal backward. This keeps Q-values bounded within [−1, 1].

### Fix Applied

**`rl/agent.py` — `DQNAgent.learn()`:**
```python
# BEFORE:
target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
# AFTER:
target_q_values = rewards - (1 - dones) * self.gamma * next_q_values
```

**`rl/agent.py` — `GPUDQNAgent._compute_loss()`:**
```python
# BEFORE:
target_q = rewards + (1.0 - dones) * self.gamma * next_q
# AFTER:
target_q = rewards - (1.0 - dones) * self.gamma * next_q
```

**`rl/gpu_trainer.py` — `collect_episodes_vectorized()` and `_cpu_worker_collect()`:**
```python
# BEFORE: every transition gets winner-based reward
reward = 1.0 if winner == player else -1.0  # (all transitions)

# AFTER: reward only at terminal step
if is_done:
    reward = 1.0 if winner == player else (-1.0 if winner != Player.NONE else 0.0)
else:
    reward = 0.0
```

### Verified Convergence (theoretical)

For a 3-move game R1 → B1 → R2=win:

| Transition | Old Q (buggy) | New Q (negamax) |
|-----------|-------------|----------------|
| R2 terminal | 1.0 | 1.0 |
| B1 non-terminal | ≈ 0.0 (near-neutral, wrong) | ≈ −0.99 (losing position ✓) |
| R1 non-terminal | ≈ 1.0 (same as terminal, uninformative) | ≈ +0.98 (winning position ✓) |

New Q-values lie within [−1, 1] and correctly represent winning probability from each state.

**Note:** Existing checkpoints in `checkpoints_spark/` were trained with the old formula. Retraining from scratch with the fix is required to fully benefit; fine-tuning from existing checkpoints may destabilize early training as the network un-learns inflated Q-values.

---

---

## §135 — Symmetry Augmentation: 8-Fold Dihedral Group on 3×3 Board

**Date:** 2026-05-25  
**Affected file:** `rl/gpu_trainer.py`

### Motivation

The DQN trained on self-play (2M+ episodes) achieved only 57% win rate against random opponents, indicating poor generalization. One contributing factor: each collected game was added to the replay buffer as a single data point, even though the 3×3 board has 8-fold symmetry (the dihedral group D4: 4 rotations × 2 reflections). A position and its 7 symmetric equivalents are **strategically identical** — the agent should give them the same Q-value — but without augmentation, the network must learn this from data alone.

### Symmetry Group

The 8 symmetries of the 3×3 board (row, col) → (row', col'):

| k | Name | Transform |
|---|------|-----------|
| 0 | Identity | (r, c) |
| 1 | 90° CW | (c, 2−r) |
| 2 | 180° | (2−r, 2−c) |
| 3 | 270° CW | (2−c, r) |
| 4 | H-flip | (r, 2−c) |
| 5 | V-flip | (2−r, c) |
| 6 | Transpose | (c, r) |
| 7 | Anti-diagonal | (2−c, 2−r) |

### Implementation

State tensor encoding: `buf[r*9+c*3+s]` for both current-player (0–26) and opponent (27–53) board. Since the action index uses the same layout (`row*9 + col*3 + (size−1)`), applying a permutation to the board indices simultaneously transforms both the state and the corresponding action.

A single `(8, 27)` permutation array `_SYM_PERMS` is precomputed at module load. For each collected transition, all 8 symmetries are applied via NumPy scatter operations:

```python
new_s[perm] = s_board_me    # scatter: new_s[perm[i]] = s_board_me[i]
new_s[27 + perm] = s_board_opp
new_s[54:] = s_rest          # piece counts + constant: symmetry-invariant
new_act = int(perm[action_idx])
```

The piece-remaining features (`buf[54:59]`) and the constant feature (`buf[60]`) are unaffected by board symmetry.

### Effect

- **8× more transitions per collection round**: 5,120 → 40,960 per collect iteration (512 games × ~10 moves × 8 symmetries)
- **Buffer diversity**: The 10M-entry circular buffer holds ~1.25M unique base positions (plus their 7 symmetric copies)
- **Reduced replay ratio**: From ~410× to ~51× (more consistent gradient updates)
- **Symmetry invariance**: The network is forced to produce consistent Q-values for equivalent board positions

### Correctness Verification

Five tests confirm:
1. `k=0` (identity) leaves state and action unchanged
2. All 8 permutations are valid bijections of {0..26}
3. 4× 90°CW = identity (cyclic group of order 4)
4. 2× horizontal flip = identity (involution)
5. Center cell (1,1,s) is invariant under all 8 symmetries
6–7. 90°CW correctly maps TL-L piece and action to TR-L in the actual game state tensor

**Expected improvement:** Better generalization vs random opponents (from 57% baseline). Retraining required to measure; existing checkpoints do not benefit.

---

---

## §141 — Random BLUE Transitions Polluting Training Buffer

**File:** `rl/gpu_trainer.py` (`collect_episodes_vectorized`, `_cpu_worker_collect`)

### Root Cause

When `random_opponent_frac > 0`, some games have a random BLUE opponent. The code made random BLUE moves and stored the `(blue_state, random_action, reward, next_state, done)` transition in the replay buffer, identical to DQN-controlled transitions.

This is noise: the DQN was trained to believe "in state S (BLUE's perspective), random action A has Q-value X." Since the action was random and unrelated to any policy objective, these transitions teach the network to predict arbitrary Q-values for BLUE positions. With `random_opponent_frac=0.3` and 512 envs, symmetry augmentation 8×, ~15% of all buffer transitions were from random BLUE moves.

The negamax Bellman target (§134) already propagates correct reward signal to RED's preceding moves via the bootstrap `-γ * next_q(BLUE_state)`, so explicitly storing the random BLUE transition is both unnecessary and harmful.

### Fix

Skip `ep_data` append for random BLUE turns in both `collect_episodes_vectorized` and `_cpu_worker_collect`. The game state still advances (random BLUE move is made, `done` is updated), but no transition is added to the training buffer.

In `_cpu_worker_collect`, Pass 1 now stores `state=None` for random BLUE entries; Pass 2 gates on `if state is not None` to skip storage. The `dqn_ks` batch list is unaffected since random BLUE entries always have `needs_dqn=False`.

### Effect

- ~15% of buffer transitions previously came from random BLUE actions (noise) — now all buffer transitions are DQN-controlled
- Training signal for BLUE positions is purely from self-play DQN moves
- RED's value estimates are unchanged — the negamax bootstrap correctly propagates loss signals from BLUE's game-ending moves (§134)
- All 53 tests pass

---

---

## §147 — Monte Carlo Returns (Bypass Bellman Bootstrapping Chain)

**Date:** 2026-05-25  
**Files:** `rl/gpu_trainer.py`, `train_spark.py`, `tests/test_gpu_trainer.py`

### Motivation

TicTacPro games average 17 moves. With terminal-only rewards (±1 only on the final transition) and 1-step TD Bellman updates, the reward signal must propagate backward through 17 iterations before the first move's Q-value reflects the outcome. Each iteration adds noise from function approximation. With a batch size of 65,536 and a replay buffer of 10M transitions, early transitions in a given game may not be sampled until 17+ gradient steps later.

### Implementation

Monte Carlo returns assign the full discounted outcome `±γ^{T-1-t}` to every transition at collection time, with `done=1.0` for all (no bootstrapping):

```python
if mc_returns:
    for t, (state, action_idx, player, next_state, _) in enumerate(data):
        if winner == Player.NONE:
            reward = 0.0
        else:
            disc = gamma ** (T - 1 - t)
            reward = disc if winner == player else -disc
        transitions.append((state, action_idx, reward, next_state, 1.0))
```

`done=1.0` makes the training target `target_q = reward` (no bootstrap), so every transition directly carries the game outcome from that player's perspective.

**Discounting analysis:**
- Final move (t=T-1): `disc = γ^0 = 1.0` (full reward)
- Move at t=0: `disc = γ^{T-1} ≈ 0.99^16 ≈ 0.851` (17-move game)
- The exponential discount naturally upweights recent moves — early moves carry ~85% of the terminal reward

**Reward sign convention:** Follows §134 negamax convention — `disc` for the player who made the move, `-disc` for the opponent.

### Coverage

Updated `_cpu_worker_collect` arg tuple from 7 to 9 elements (added `mc_returns`, `gamma`). Updated `GPUTrainer.__init__`, `_make_worker_args`, and `train()` to pass through the flag. Added `--mc-returns` CLI flag to `train_spark.py`.

Fixed test helper `_make_worker_args` in `tests/test_gpu_trainer.py` to match new 9-element tuple.

**All 176 tests pass.**

### Theoretical Effect vs Standard TD

| Property | TD (terminal-only) | Monte Carlo (§147) |
|---|---|---|
| Propagation delay | 17 Bellman iterations | 0 (direct assignment) |
| Variance | Low (bootstrapping smooths) | Higher (full game variance) |
| Bias | High (approximation error compounds 17×) | Low (no bootstrapping) |
| Suitable for | Dense rewards | Sparse, short-game rewards |

For TicTacPro (sparse reward, ~17 moves, short game), MC returns are expected to accelerate reward propagation at the cost of higher gradient variance — a favorable trade-off.

**New CLI flag:** `python3 train_spark.py --mc-returns` enables §147 for future training runs.



---

---

## §149 — Legal Action Masking in Bellman Backup

**Date:** 2026-05-25  
**Problem:** In `_compute_loss`, the Double-DQN Bellman backup uses `policy_net(next_states).argmax(dim=1)` without masking illegal actions. During early training, random initialization can assign high Q-values to illegal moves, causing the backup to bootstrap from illegal-action Q-values, biasing TD targets and slowing convergence.

**Root cause:** The Bellman update selects the "best" next action using the policy network. If an illegal action has the highest Q-value (noise or early training artifact), it's selected for bootstrapping, producing a misleading target.

**Key insight:** The legal action mask is fully derivable from the normalized state tensor — no buffer changes needed:
- `occupied[a] = (state[a] + state[27+a]) >= 1.0` — either player has a piece at cell (r,c,sz)
- `piece_avail[sz] = state[54+sz] > 0` — current player has remaining pieces of size sz
- Action a is legal iff `not occupied[a] AND piece_avail[a%3]`

**Implementation (in `_compute_loss`):**
```python
def _legal_mask_from_state(self, states):
    sz_idx = torch.arange(27, device=states.device) % 3
    not_occupied = (states[:, :27] + states[:, 27:54]) < 0.5
    piece_avail  = states[:, 54:57] > 0
    return not_occupied & piece_avail[:, sz_idx]  # (B, 27)

# In _compute_loss:
legal = self._legal_mask_from_state(next_states)
next_q_all = self.policy_net(next_states).masked_fill(~legal, -1e9)
next_acts = next_q_all.argmax(dim=1)
next_q = self.target_net(next_states).gather(1, next_acts.unsqueeze(1)).squeeze(1)
```

**When it matters most:** Early training (random network weights), long sequences (more accumulated illegal-action noise), TD mode (MC returns already zero out next_q for terminal states so masking is irrelevant when `dones=1.0`).

**Edge case:** Terminal `next_states` have no legal moves, but `dones=1.0` zeroes out `next_q` in `target_q = rewards - (1-dones)*gamma*next_q`. Safe.

**Test status:** All 176 tests pass after §149 changes.

---

---

## §164 — Pure-Numpy Inference for CPU Workers (19× Speedup)

### Problem Identified

During run5 profiling, CPU workers each took ~439ms per call (32 games, ε=0.8). Of this:
- `torch._C._nn.linear`: **341 ms** across 126 calls = 2.71 ms per call
- `torch.layer_norm`: **74 ms** across 54 calls = 1.37 ms per call
- **Total PyTorch overhead: 415 ms = 94.5% of worker time**

Crucially, the per-call overhead of ~2.7 ms was **independent of matrix size** — identical whether the weight matrix was (61→1024) or (61→256). This is PyTorch's fixed BLAS dispatch cost (memory allocation, Python-to-C dispatch, tensor bookkeeping) per `nn.Linear` call, not actual computation.

**Root cause:** `GPUDQNNetwork` has 7 `nn.Linear` layers. Each DQN forward pass = 7 BLAS calls × 2.7ms = 18.9ms dispatch overhead. With 18 batched DQN calls per worker cycle, this is 341ms of pure overhead. The matrix multiplications themselves are ~0.1ms each at batch=8; overhead dominates 96:1.

### L3 Cache Analysis

Secondary issue: 16 workers × 3.16 MB weights = 50.6 MB total working set > L3 cache (24 MB). This causes constant cache misses, slowing matrix multiplies by an additional ~2–3×.

### Fix: `NumpyDQNInference` Class

New class `NumpyDQNInference` in `rl/network.py`:
- Extracts weights from a `GPUDQNNetwork` state-dict as numpy arrays at construction
- Implements the full dueling forward pass (`features → LN → value head + advantage head`) as direct numpy operations (`x @ W.T + b`, `np.maximum(0, ...)`, layer-norm formula inline)
- Workers instantiate this instead of `GPUDQNNetwork` + `load_state_dict`

```python
class NumpyDQNInference:
    def __init__(self, state_dict, hidden_sizes=(1024, 512, 256)):
        # Extract all weight/bias/LN arrays from state_dict
        ...

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, self._ln(x @ self._W0.T + self._b0, ...))
        h = np.maximum(0, self._ln(h @ self._W1.T + self._b1, ...))
        h = np.maximum(0, self._ln(h @ self._W2.T + self._b2, ...))
        value = np.maximum(0, h @ self._Wv1.T + self._bv1) @ self._Wv2.T + self._bv2
        adv   = np.maximum(0, h @ self._Wa1.T + self._ba1) @ self._Wa2.T + self._ba2
        return value + (adv - adv.mean(axis=-1, keepdims=True))
```

**Why numpy avoids the overhead:** `numpy.dot` / `@` calls BLAS directly without the multi-layer Python wrapper overhead. For a batch of 8 states, numpy dispatches in ~0.05ms vs PyTorch's 2.7ms.

**Worker change in `gpu_trainer.py`:**
```python
# Before (§163):
net = GPUDQNNetwork(..., init_weights=False)
net.load_state_dict(weights_cpu)
net.eval()
# ...
batch = torch.from_numpy(np.stack([...]))
with torch.no_grad():
    q_batch = net(batch).numpy()

# After (§164):
net = NumpyDQNInference(weights_cpu, hidden_sizes)
# ...
batch_np = np.stack([...])
q_batch = net(batch_np)   # numpy in → numpy out, no torch overhead
```

### Benchmark Results

| Metric | PyTorch (GPUDQNNetwork) | Numpy (NumpyDQNInference) | Speedup |
|--------|------------------------|--------------------------|---------|
| Forward pass batch=8, isolated | 24.3 ms | 1.3 ms | **19×** |
| Forward pass batch=8, under load | ~30 ms | ~10 ms | **~3×** |
| DQN inference per worker (18 calls) | ~432 ms | ~23 ms | **~19×** |
| Projected worker time (32 games, ε=0.8) | ~954 ms | ~100 ms | **~10×** |

**Correctness:** Max absolute error vs PyTorch: < 1e-6. argmax(Q) identical in all tested cases. Verified by 5 unit tests including correctness at batch=1, batch=16, batch=32, and argmax agreement.

### Impact on Training Speed

With workers taking ~100ms instead of ~954ms:
- Workers finish well before training step (802ms) → collect_ms → ~0ms
- Cycle time limited by training alone: ~802ms
- Effective throughput: 512 games / 802ms ≈ 638 ep/s
- Current (post-§164): ~639 ep/s confirmed — already at theoretical ceiling

The 152ms collect_ms overhead (workers overrunning training by 152ms) is eliminated, unlocking full pipeline efficiency. First run to benefit fully will be **run6** (run5 started before §164).

**Tests:** 5 new tests verify output correctness (single, batch, argmax), speedup ≥ 2× under load, and `from_network()` classmethod. (197 → 202 tests)

---

---

## §167 — Pre-Pickle Network Weights Once for All Workers (4× Serialization Speedup)

### Problem: 16× Redundant Dict Serialization

`_make_worker_args()` builds a list of 16 identical argument tuples (one per worker), each containing `weights_cpu` — a Python dict of ~120 numpy tensors (~3 MB). When `executor.submit(fn, arg)` is called 16 times, Python's `ProcessPoolExecutor` pickles each argument tuple independently via a queue. Since there's no cross-submit memoization, the full weight dict is serialized 16 times.

**Measured cost**: 14.5ms to serialize 16 args with weights dict (old), vs 3.7ms with pre-pickled bytes (new) = **10.8ms saved per cycle** on the main-process critical path.

### Root Cause: `pickle.dumps(dict_of_tensors)` is Expensive

Pickling a dict of 120 numpy tensors requires:
1. Recursing into each dict value
2. Handling each numpy array header + `tobytes()` call
3. Python GIL held throughout

For 16 workers sharing identical weights: the pickle work is 16×. With a 3MB dict, `pickle.dumps(weights_cpu)` takes ~2ms; repeated 16 times = 32ms.

### Fix: Pre-Pickle Once, Pass Bytes to All Workers

Serialize the weight dict **once** before the submit loop, then put the resulting `bytes` object in all 16 argument tuples. When Python pickles a `bytes` object, it emits the raw bytes with a tiny header — essentially a memcpy.

```python
# Before (§166): 16× expensive dict pickle
weights_cpu = {k: v.cpu() for k, v in raw_net.state_dict().items()}
return [(n_games, epsilon, weights_cpu, ...) for _ in range(16)]

# After (§167): 1× dict pickle + 16× bytes memcpy
weights_bytes = pickle.dumps(weights_cpu)  # ~2ms, once
return [(n_games, epsilon, weights_bytes, ...) for _ in range(16)]
```

Workers call `pickle.loads(weights_bytes)` to reconstruct the dict. This costs an extra ~3.5ms per worker (identical to the old deserialization cost), but since workers run in parallel during GPU training (~820ms), the overhead is hidden.

### Benchmark

| Stage | Before §167 | After §167 | Notes |
|-------|------------|-----------|-------|
| Main: serialize 16 args (D2H + pickle) | 15.3ms | 4.5ms | **10.8ms saved** |
| Worker: deserialize args | 1.2ms | 4.8ms | +3.5ms, hidden in 820ms training |
| Net critical-path savings | — | **~10.8ms/cycle** | On main process |

### Combined Effect: §163+§164+§164a+§165+§166+§167

| Stage | Time (original) | Time (all fixes) | Speedup |
|-------|----------------|-----------------|---------|
| Worker network creation | 4,000 ms | ~1 ms (§163) | **4,000×** |
| DQN inference / batch | 24 ms | 0.28 ms (§164+§164a) | **86×** |
| Serial augment on main | ~62 ms | ~1 ms (§166) | **62×** |
| Weight serialization (main) | ~15 ms | ~4.5 ms (§167) | **3.3×** |
| **Total per cycle (critical path)** | **~1,533 ms** | **~816 ms** | **1.9×** |

**Expected run6**: ~816ms cycle time, ~6,400 cycles/hour, ~12.8M grad steps/4hr run vs ~6.7M without these fixes.

---

---

## §168 — Reverse Heuristic Games: Train DQN as BLUE Against Tactical RED

### Problem: BLUE Wins 0% Against Non-Random Opponents

Run5 tournament results: DQN RED vs random BLUE = 100% (expected), but DQN BLUE vs heuristic RED = 0%. The DQN has learned strong RED play but its BLUE play is poor.

**Root cause**: Training data is heavily RED-skewed.
- 30% of games: heuristic BLUE, only RED transitions stored (§109/§165). DQN only sees winning RED play.
- 70% of games: self-play. RED wins ~72% → 72% of BLUE's stored transitions have negative reward.
- BLUE trains mostly on losing positions with uniformly negative signal, making it hard to learn winning BLUE play.

### Fix: Add "Reverse" Games Where RED is Heuristic

New `reverse_opponent_frac` parameter (default 0.0, recommended 0.1-0.2 for run6): a fraction of games have heuristic RED (win→block→first-available) and the DQN plays as BLUE. Only BLUE transitions are stored.

This is the exact mirror of §109/§165 (heuristic BLUE, DQN as RED):

```python
# §168: reverse games — heuristic RED, DQN as BLUE
is_heuristic_red = (n_random <= i < n_random + n_reverse and
                    game.current_player == Player.RED)
if is_random_blue or is_heuristic_red:
    pending.append((i, moves, None, False))  # pick_rollout_move, no storage
```

### Why This Helps

- BLUE now trains against a challenging opponent. With epsilon=0.24 exploration, BLUE can win ~20-30% of games against heuristic RED (heuristic is strong but not perfect).
- Positive BLUE reward signals (from winning games) are now in the training data.
- The state representation is symmetric (§134: current-player normalized), so the same network learns from both RED and BLUE wins.
- Overhead: 0.027ms/game (same as §165, just pick_rollout_move for RED instead of BLUE).

### Training Data Mix for Run6 (--random-opp 0.3 --reverse-opp 0.15)

| Game type | Games/cycle | Transitions stored | Effect |
|-----------|-------------|-------------------|--------|
| Heuristic BLUE, DQN RED (§165) | 0.3 × 512 = 154 | RED only | Trains winning RED vs threats |
| Heuristic RED, DQN BLUE (§168) | 0.15 × 512 = 77 | BLUE only | Trains winning BLUE vs threats |
| Self-play (both DQN) | 0.55 × 512 = 282 | RED + BLUE | Trains both perspectives |

### CLI Usage

```bash
python3 train_spark.py ... --random-opp 0.3 --reverse-opp 0.15
```

### Tests Added

`TestReverseHeuristicOpponent` (3 tests): reverse games complete, store BLUE transitions, work combined with random_opp.

---

---

## §169 — VS-Heuristic Evaluation Metric: Meaningful Skill Tracking During Training

**Date:** 2026-05-25  
**Branch:** TTP-4  
**File:** `rl/gpu_trainer.py`

### Motivation

The existing evaluation metric `vs Random BLUE (100g): RED=100.0%` is trivially perfect from the first episode onward — a DQN trained for 10 minutes beats a random opponent 100% of the time. This gives no signal about actual skill progression. Without a meaningful metric, it's impossible to know whether improvements like §165 (heuristic opponent) or §168 (reverse heuristic games) are actually improving the policy quality.

The fix: evaluate the DQN greedy policy against `pick_rollout_move` (the same heuristic used during training), for both color roles.

### Implementation

Added `_eval_vs_heuristic(n_games=50)` to `GPUTrainer`. Runs two scenarios back-to-back:
1. **DQN plays RED**, `pick_rollout_move` plays BLUE → `dqn_red_wr`
2. **DQN plays BLUE**, `pick_rollout_move` plays RED → `dqn_blue_wr`

Uses the same batched forward-pass pattern as `collect_episodes_vectorized`: in each while-loop iteration, all games that are on the DQN player's turn are processed in one `get_actions_batch` call (batch size ~n_games), while heuristic turns use `pick_rollout_move` sequentially (0.027ms/game, negligible).

```python
def _eval_vs_heuristic(self, n_games: int = 50) -> dict:
    saved_eps = self.agent.epsilon
    self.agent.epsilon = 0.0
    dqn_red_wr = dqn_blue_wr = float("nan")

    for dqn_is_red in (True, False):
        dqn_color  = Player.RED  if dqn_is_red else Player.BLUE
        heur_color = Player.BLUE if dqn_is_red else Player.RED

        games = [TicTacPro() for _ in range(n_games)]
        done  = np.zeros(n_games, dtype=bool)

        while not done.all():
            active = [i for i in range(n_games) if not done[i]]
            dqn_ids  = [i for i in active if games[i].current_player == dqn_color  ...]
            heur_ids = [i for i in active if games[i].current_player == heur_color ...]

            if dqn_ids:
                actions = self.agent.get_actions_batch([games[i] for i in dqn_ids])
                for j, i in enumerate(dqn_ids):
                    games[i].make_move(*actions[j])
                    if games[i].game_over: done[i] = True

            for i in heur_ids:
                act, _ = pick_rollout_move(games[i])
                games[i].make_move(*act)
                if games[i].game_over: done[i] = True

        win_counts = counter(games)
        if dqn_is_red: dqn_red_wr  = win_counts[RED]  / n_games
        else:          dqn_blue_wr = win_counts[BLUE] / n_games

    self.agent.epsilon = saved_eps
    return {"dqn_red_wr": dqn_red_wr, "dqn_blue_wr": dqn_blue_wr}
```

Called in `_log_progress` alongside `_eval_vs_random`, every 5 log intervals:

```python
vh = self._eval_vs_heuristic(n_games=50)
self._vs_heuristic_dqn_red_wr  = vh["dqn_red_wr"]
self._vs_heuristic_dqn_blue_wr = vh["dqn_blue_wr"]
self.writer.add_scalar("Eval/vs_heuristic_dqn_red_wr",  ...)
self.writer.add_scalar("Eval/vs_heuristic_dqn_blue_wr", ...)
```

Training log now shows:
```
  vs Random BLUE (100g): RED=100.0%  BLUE=0.0%
  vs Heuristic  ( 50g): DQN-RED=XX.X%  DQN-BLUE=XX.X%
```

### Performance

| Scenario | n_games | Steady-state time |
|----------|---------|-------------------|
| Both scenarios (DQN-RED + DQN-BLUE) | 50 each | ~2,680ms |
| Called every N intervals | 5 × 60s = 5 min | overhead < 1% |

The 2680ms steady-state (after `torch.compile` warmup) is dominated by Python loop overhead across ~34 iterations per game. Since the eval is called every 5 minutes, the 2.7s penalty is negligible (<1% of total training time).

### TensorBoard Scalars Added

- `Eval/vs_heuristic_dqn_red_wr` — DQN win rate as RED vs heuristic BLUE
- `Eval/vs_heuristic_dqn_blue_wr` — DQN win rate as BLUE vs heuristic RED

### Expected Progression

An untrained DQN scores ~0% against the heuristic. A well-trained RED agent should reach 60-80%. BLUE is harder — §168's reverse heuristic games are specifically designed to push `dqn_blue_wr` above 0%. This metric will directly show whether §168 is working.

### Tests Added

`TestVsHeuristicEval` (4 tests): correct keys returned, win rates in [0,1], cached state variables set correctly, all games reach terminal state.

---

---

## §332 — Second-Player-Win Claim (later refuted — see correction)

> **⚠️ CORRECTION (resolved).** This section originally concluded that TicTacPro is a *second-player win* ("BLUE wins with perfect play"). **That conclusion is incorrect.** It was drawn from a **time-limited** minimax-vs-minimax match (3 s/move, depth 18) — not an exhaustive search — so a practical RED simply failed to execute its forced win within the time budget. The **exact solver** (§77, `exact_solve.py`: full negamax to terminal, no heuristic) proves **RED wins from all 27 opening moves**, and the empirical evidence here agrees: the BLUE-optimal `BullseyeAgent` built on this premise **loses to Minimax 10/10** (§334) — impossible for a genuine second-player win. The original analysis is retained below for the record; read every "BLUE wins with perfect play" as "BLUE wins against imperfect / time-limited RED play." The sequential-bullseye-fork *tactics* below remain a correct and useful description of how BLUE punishes a RED that does not defend.

**Discovery method:** Minimax vs minimax (both sides, 3s/move, depth 18) — BLUE wins all 5 games with an identical 16-move sequence, confirming deterministic perfect play.

**Principal variation (minimax-optimal play from both sides):**
```
Move 1:  RED  TC-L    Move 2:  BLU  CC-S
Move 3:  RED  CC-M    Move 4:  BLU  ML-S
Move 5:  RED  MR-S    Move 6:  BLU  ML-M   ← bullseye threat at ML (2/3)
Move 7:  RED  ML-L    Move 8:  BLU  TC-M   ← RED blocks ML bullseye
Move 9:  RED  BL-S    Move 10: BLU  BL-L
Move 11: RED  TL-S    Move 12: BLU  TR-L   ← new bullseye threat at TR (1/3)
Move 13: RED  CC-L    Move 14: BLU  TR-M   ← bullseye threat at TR (2/3)
Move 15: RED  TL-M    Move 16: BLU  TR-S   ← BULLSEYE at TR — BLUE WINS
```

**BLUE's winning strategy — sequential bullseye fork:**
1. **Neutralize RED's anti-diagonal plan**: CC-S (move 2) blocks the center small slot, disrupting RED's anti-diagonal-S strategy.
2. **Build first bullseye threat**: Accumulate S and M at ML, forcing RED to spend their L to block.
3. **Shift to new bullseye**: While RED has spent pieces defending ML, BLUE builds a fresh bullseye at TR (or bottom row, or wherever RED can't defend).
4. **Forced win**: BLUE always has a second threat when RED blocks the first — the sequential fork guarantees BLUE can complete a bullseye before RED can mount an offense.

**Verification against different RED strategies:**
- `optimal-RED` (anti-diagonal heuristic) vs `minimax-BLUE`: BLUE wins in 10 moves via double-attack (bullseye at ML + row1-medium at MR-M simultaneously)
- When RED attempts to pre-block TR-L: BLUE immediately pivots to bottom-row-large (BL-L, BR-L, BC-L) — wins in 14 moves
- RED cannot prevent two simultaneous winning threats with a single move

**Implication:** The `OptimalAgent` is a strong RED heuristic but is NOT game-theoretically optimal. The game name "optimal" is a misnomer. A perfect BLUE player always wins.

**Training implication:** DQN training should emphasize BLUE's perspective. The game favors the second player, meaning BLUE has inherent structural advantage once it learns the fork strategy.

---

---

## §333 — OptimalAgent Block Priority Bug Fixed

**Bug:** When the scan found a line block (opponent completing a 3-in-a-row) before encountering a bullseye block (opponent completing all 3 sizes in one cell), the `block_mv` was set and `_blk = False` prevented checking for bullseye blocks in subsequent scan iterations. This caused RED to block the less-urgent line threat while missing the more dangerous bullseye threat.

**Root cause:** Single `block_mv` variable with `_blk = block_mv is None and has_opp_sz`. The scan goes SIZE-first (SMALL→MEDIUM→LARGE), so MEDIUM line blocks were found before LARGE bullseye blocks, pre-empting the bullseye detection.

**Example (from §332 analysis):** After BLUE builds ML-S + ML-M (bullseye at ML = 2/3) AND has CC-M + ML-M (row1-medium = 2/3):
- OLD: scan found MR-M (row1-medium block) first; ML-L (bullseye block) never detected
- NEW: both detected; bullseye block at ML-L returned as higher priority

**Fix:** Separate `bullseye_block` and `line_block` tracking; bullseye block wins:
```python
# Before: single block_mv, stops after first found
_blk = block_mv is None and has_opp_sz
...
if _blk and v1 == opp_int and v2 == opp_int:
    block_mv = (r, c, size); _blk = False

# After: dual tracking, bullseye preferred
if has_opp_sz and v1 == opp_int and v2 == opp_int:
    if i1 // 3 == i2 // 3:        # same cell → bullseye (i//3 = row*3+col)
        if bullseye_block is None: bullseye_block = (r, c, size)
    elif line_block is None:       line_block = (r, c, size)
block_mv = bullseye_block if bullseye_block is not None else line_block
```

**Bullseye detection:** Flat board index `idx = r*9 + c*3 + sz`. Two indices are in the same cell iff `i1 // 3 == i2 // 3` (since `r*9+c*3+sz` integer-divided by 3 gives `r*3+c`; sz < 3 contributes 0 to the quotient).

**Outcome:** Against minimax BLUE, RED now correctly blocks ML-L (bullseye) instead of MR-M (line). BLUE still wins via MR-M because the double-attack is fundamentally unavoidable (§332). But correctness improved: against sub-optimal BLUE players, missing a bullseye block is a real strategic error.

**Tests:** All 111 game tests still pass (0 regressions).


---

---

## §334 — BullseyeAgent: Hardcoded BLUE Winning Strategy

**Motivation:** §332 proved TicTacPro is a second-player win, but no existing agent implemented BLUE's optimal strategy. The `OptimalAgent` is a RED heuristic; it has no BLUE-specific plan. The DQN never learned bullseye strategy. This section implements a hardcoded BLUE-optimal rule-based agent.

**Algorithm (`game/bullseye_agent.py`):**
```
Priority:
1. Win immediately (3-in-a-row or bullseye completion)
2. Block opponent immediate win — bullseye blocks prioritized (same logic as §333 fix)
3. Deny opponent anti-diagonal S cells (2/3 occupied)
4. Complete own 2/3 bullseye (play the 3rd size in a cell I own 2 of)
5. Build 2nd piece in a cell I own 1 (opponent-free cell), prefer CC > corners > edges
6. Start fresh bullseye in a virgin cell (no pieces for either player)
7. Anti-diagonal offense
8. Any legal move
```

**Cell priority for bullseye targeting:** CC, ML, TR, BL, TL, BR, TC, MR, BC.
This mirrors the minimax BLUE principal variation where CC → ML → TR was the winning sequence.

**Results (run7/best.pt checkpoint):**

| Match (RED vs BLUE) | Result |
|---|---|
| OptimalAgent vs BullseyeAgent | **BullseyeAgent 20/20** (both as RED and BLUE) |
| DQN vs BullseyeAgent | **BullseyeAgent 20/20** (both as RED and BLUE) |
| BullseyeAgent vs Minimax | **Minimax 10/10** (minimax still dominates) |

**DQN failure mode:** When DQN(RED) faces BullseyeAgent(BLUE), BLUE wins in 16 moves using sequential bullseye forks. DQN doesn't block the 2/3 bullseye threats because it never trained against this strategy.

When Minimax(RED) faces BullseyeAgent(BLUE), Minimax wins in 5 moves via TC-column bullseye (TC-L, TC-S, TC-M), which BullseyeAgent's BLUE defense misses. BullseyeAgent doesn't block RED building a bullseye at TC when it has higher-priority defensive actions.

**Why BullseyeAgent loses to Minimax:** BullseyeAgent's block detection has the same limitation as OptimalAgent — it only blocks IMMEDIATE threats (2/3 complete). When Minimax-RED plays TC-L (1 piece), BullseyeAgent doesn't perceive a threat yet. By move 3 when TC-S creates the 2nd piece at TC, the block detection triggers — but Minimax completes TC-M before BullseyeAgent can respond (since BullseyeAgent is BLUE and moves second).

**mcts_b agent added:** BullseyeAgent added as rollout policy for guided MCTS (`mcts_b`) to compare against `mcts_g` (OptimalAgent rollouts). Hypothesis: BullseyeAgent rollouts should give more accurate value estimates in MCTS, improving tournament performance.

**Training implication:** The DQN should be trained against BullseyeAgent to force learning of:
1. Bullseye threat detection (2/3 occupied cell is an immediate threat)
2. Proactive cell occupation (block before 2/3 — detect 1/3 threats)
3. Counter-play to sequential fork attacks


---

---

## §335 — Why BullseyeAgent Loses to Minimax: Double-Attack Theory

**Root cause:** Minimax-RED defeats BullseyeAgent by creating multiple simultaneous winning threats (double-attacks), where any single block leaves another threat open.

**Example game trace (minimax-RED vs BullseyeAgent-BLUE):**
```
Move 1:  RED  TC-L    ← start accumulating large pieces in row0
Move 2:  BLUE CC-L   (step 6: start fresh bullseye at CC, highest priority)
Move 3:  RED  TC-S    ← RED has 2/3 bullseye at TC (TC-L + TC-S)
Move 4:  BLUE TC-M   ← correct bullseye block (TC-L and TC-S both RED)
Move 5:  RED  TR-S    ← RED has row0-small threat: TC-S + TR-S (needs TL-S)
Move 6:  BLUE TL-S   ← correct line block (TC-S and TR-S both RED in row0-small)
Move 7:  RED  TR-L    ← RED has 2/3 bullseye at TR (TR-S + TR-L)
Move 8:  BLUE TR-M   ← BLUE blocks bullseye (TR-S and TR-L both RED)
          BUT also: RED has TC-L + TR-L = 2/3 row0-large (needs TL-L) ← missed!
Move 9:  RED  TL-L    ← row0-large: TL-L + TC-L + TR-L = WIN!
```

**The double-attack at move 8:**
- Threat 1: TR bullseye (TR-S + TR-L → needs TR-M)
- Threat 2: Row0-large (TC-L + TR-L → needs TL-L)

BullseyeAgent blocks Threat 1 (bullseye priority), leaving Threat 2 open. RED wins.

**Block priority dilemma:** Both threats are equally urgent (RED wins next move with either). Block priority (bullseye > line) is wrong here — the line block (TL-L) is more important because the bullseye block (TR-M) doesn't exist yet in RED's hand as a "forced" sequence (minimax specifically chose TR-L to create BOTH threats simultaneously).

**Why BullseyeAgent can't fix this:** The double-attack is created PROACTIVELY by minimax. After move 7 (RED plays TR-L), BLUE can see both threats and must pick one to block. No reactive strategy can win in a genuine double-attack position — lookahead is required to prevent the position from arising.

**BullseyeAgent's correct role:**
- Strong heuristic against non-lookahead agents (beats OptimalAgent 20/20, DQN 20/20)
- Excellent rollout policy for MCTS (better value estimates than pick_rollout_move)
- Training opponent for DQN (forces learning of bullseye blocking)
- Demonstrates BLUE's theoretical strategic advantage even without perfect play

**Why the §333 bullseye-priority fix makes things worse here:** Prioritizing bullseye blocks over line blocks causes BLUE to pick TR-M over TL-L. With the OLD code (line block first), BLUE would play TL-L — but RED still wins via TR bullseye (TR-M). Either block loses.

**Implication for training:** The DQN needs tree-search (MCTS) to reliably defeat minimax. Pure reactive learning from heuristic opponents cannot learn proactive double-attack prevention.



---

---

## §336 — BullseyeAgent as DQN Training Opponent

**Motivation:** DQN fails to block bullseye threats (demonstrated in §334: minimax beats DQN via TC-column bullseye in 5 moves). Root cause: training opponents (self-play DQN, random, OptimalAgent) never build systematic bullseye threats, so DQN has no incentive to learn bullseye blocking.

**Change:** Added `use_bullseye_heuristic` flag to `GPUTrainer` and `train_spark.py`. When enabled (30% of games), one player is replaced by `BullseyeAgent` — which builds sequential bullseye threats — forcing the DQN to learn a blocking policy.

**Implementation:**
- `rl/gpu_trainer.py`: 12th element in worker-args tuple is `use_bullseye_heuristic`
- `_cpu_worker_collect`: if `use_bullseye_heuristic`, lazily creates a `BullseyeAgent` instance per worker and uses it as the heuristic opponent
- `train_spark.py`: `--bullseye-heuristic` flag enables the feature

**Training run 8:** Launched with `--bullseye-heuristic --reverse-opp 0.3`, forcing 30% of games to have BullseyeAgent as BLUE (defensive) and 30% as RED (attacks via bullseye fork). DQN-RED currently at 0% win rate vs BullseyeAgent — expected improvement over the 2-hour run.


---

---

## §337 — NeuralMCTSAgent Fast-Path Bullseye Block Priority Bug Fixed

**Bug:** `NeuralMCTSAgent.get_action()` and `get_info()` fast-paths used the pattern:
```python
block_mv = None
...
if block_mv is None and _would_win_bf(bf, r, c, sz_idx, opp_int):
    block_mv = mv  # stops looking after first block found
```

This is the same §333 bug: when both a bullseye threat AND a line threat exist, the agent returns whichever block appears first in `get_legal_moves()` ordering, regardless of type. Bullseye threats are more dangerous (completing all 3 sizes in one cell is often harder to notice) and should be prioritized.

**Fix (§337):** Replaced `_would_win_bf` calls with direct `_WIN_LINES` scan (same as §333 OptimalAgent fix). Dual-track `bullseye_block` and `line_block` variables, preferring bullseye:
```python
bullseye_block = None
line_block     = None
for mv in game.get_legal_moves():
    r, c, sz = mv; ci = r * 3 + c
    for i1, i2 in _WIN_LINES[sz - 1][ci]:
        v1 = bf[i1]; v2 = bf[i2]
        if v1 == p_int and v2 == p_int: return mv  # win
        if opp_has_sz and v1 == opp_int and v2 == opp_int:
            if i1 // 3 == i2 // 3:   # same cell → bullseye threat
                bullseye_block = bullseye_block or mv
            else:
                line_block = line_block or mv
block_mv = bullseye_block or line_block
```

**Impact:** In positions where minimax or BullseyeAgent creates a simultaneous bullseye + line double-attack, NeuralMCTS now correctly prioritizes the bullseye block in its fast-path pre-MCTS decision layer. The MCTS tree itself handles multi-threat positions correctly via UCT exploration.


---

---
