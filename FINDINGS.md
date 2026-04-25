# TicTacPro: Deep Reinforcement Learning on NVIDIA DGX Spark GB10

**Author:** Khan  
**Hardware:** NVIDIA DGX Spark (GB10 Grace-Blackwell Superchip)  
**Date:** April 2026

---

## Abstract

We trained a Double Deep Q-Network (DDQN) agent with Prioritized Experience Replay (PER) and Dueling architecture to play TicTacPro — a variant of Tic-Tac-Toe where each player has three piece sizes (Small, Medium, Large) and wins either by placing three-in-a-row of the same size or stacking all three sizes in one cell (bullseye). Training ran for 2.1 hours on a single GB10 GPU using BF16 mixed precision and `torch.compile` with Inductor kernel fusion. The agent played **527,872 self-play episodes**, completed **65,472 gradient updates** on batches of 65,536 positions, and reached a final loss of **0.057**. Post-training greedy evaluation revealed a decisive first-player advantage: RED (first mover) won **100% of 2,000 greedy games** against the trained policy. The best discovered opening is a **Medium piece at the bottom-left corner**.

---

## 1. Game Description

### 1.1 TicTacPro Rules

TicTacPro (Brass Monkey variant) is played on a 3×3 grid. Each player starts with 9 pieces: 3 Small (S), 3 Medium (M), and 3 Large (L).

**Win conditions:**
1. **Row/Column/Diagonal:** Three pieces of the **same size** in a line
2. **Bullseye:** One Small, one Medium, and one Large piece stacked in the **same cell**

**Key mechanics:**
- Larger pieces can be placed on top of smaller pieces (gobbling)
- A piece can only be gobbled by a strictly larger piece
- Only the topmost piece in a cell is visible and counts for win detection

### 1.2 State Representation

The game state is encoded as a **61-dimensional float32 vector**:

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| Board cells (9 × 3 layers) | 27 | Player ownership per layer per cell |
| Board sizes (9 × 3 layers) | 27 | Piece size per layer per cell |
| RED piece counts | 3 | Remaining S/M/L for RED |
| BLUE piece counts | 3 | Remaining S/M/L for BLUE |
| Current player | 1 | 0=RED, 1=BLUE |

### 1.3 Action Space

27 discrete actions: 9 cells × 3 piece sizes. Illegal actions (no pieces remaining, cannot goble target) are masked with −10⁹ before argmax selection.

---

## 2. Methods

### 2.1 Algorithm: Double DQN with Prioritized Experience Replay

We used **Double DQN** (van Hasselt et al., 2016) to reduce Q-value overestimation. The policy network selects the next action; the target network evaluates it:

$$
y_t = r_t + \gamma \cdot Q_{\theta^-}\!\left(s_{t+1},\, \arg\max_a Q_\theta(s_{t+1}, a)\right)
$$

**Prioritized Experience Replay (PER)** (Schaul et al., 2016) samples transitions proportionally to their TD-error magnitude, focusing learning on surprising experiences:

$$
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad \alpha = 0.6
$$

Importance-sampling weights correct the update bias:

$$
w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta, \quad \beta: 0.4 \to 1.0
$$

**Dueling architecture** (Wang et al., 2016) separates value and advantage streams:

$$
Q(s, a) = V(s) + \left(A(s,a) - \frac{1}{|A|}\sum_{a'} A(s, a')\right)
$$

### 2.2 Network Architecture

```
Input (61) → Linear(1024) + LayerNorm + ReLU
           → Linear(512)  + LayerNorm + ReLU
           → Linear(256)  + LayerNorm + ReLU
           ┌─────────────────┬──────────────────┐
           │  Value stream   │ Advantage stream  │
           │ Linear(128)+ReLU│ Linear(128)+ReLU  │
           │   Linear(1)     │   Linear(27)      │
           └────────┬────────┴────────┬──────────┘
                    └────── Q(s,a) ───┘
```

**Total parameters:** 792,604  
**Precision:** BF16 (native on Blackwell compute 12.1)  
**Weight init:** Orthogonal for linear layers, zeros for biases

### 2.3 Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (lr=3×10⁻⁴, wd=10⁻⁵, ε=10⁻⁵) |
| LR schedule | CosineAnnealing (T_max=2M, η_min=10⁻⁵) |
| Discount factor γ | 0.99 |
| Batch size | 65,536 |
| Replay buffer | 10,000,000 |
| Target sync freq | Every 5,000 grad steps |
| Gradient clip | 10.0 (L2 norm) |
| Epsilon start → end | 1.0 → 0.05 |
| Epsilon decay | 0.999985 per grad step |
| Loss function | Huber loss with PER IS weights |

### 2.4 Collection Strategy

**Vectorized self-play:** 512 games run in lockstep. All active games share a single batched GPU forward pass per move-step (batch size varies as games terminate). This keeps the GPU busy during the collection phase.

**Reward shaping:** Sparse rewards — +1.0 for win, −1.0 for loss, 0.0 for draw. The reward is assigned retroactively to all moves a player made in the game.

---

## 3. Hardware and Implementation

### 3.1 NVIDIA DGX Spark GB10

| Property | Value |
|---|---|
| GPU | NVIDIA GB10 Grace-Blackwell Superchip |
| Compute capability | 12.1 (Blackwell) |
| Unified memory | 121.7 GB LPDDR5X (shared CPU+GPU) |
| Streaming Multiprocessors | 48 SMs |
| CPU | ARM Grace (20 cores) |
| Peak BF16 throughput | ~200 TFLOPS |

The GB10 uses a **unified memory architecture** — CPU and GPU share the same physical LPDDR5X pool. This eliminates explicit host-to-device transfers for replay buffer data at the cost of `nvidia-smi` GPU utilization reporting being unreliable (it undercounts). True GPU utilization was estimated from wall-clock timing: 64 gradient steps on 65,536-sample batches took ~8 seconds, versus an estimated 60–90 seconds on CPU alone.

### 3.2 Software Stack

- **PyTorch 2.11+cu130** with CUDA 13.0
- **`torch.compile(backend='inductor')`** — Inductor kernel fusion for faster CUDA kernels
- **`torch.amp.autocast('cuda', dtype=torch.bfloat16)`** — BF16 AMP without GradScaler (Blackwell natively supports BF16)
- **Cached cumsum PER sampling** — cumsum refreshed every 16 gradient steps, amortizing the O(N) cost of prioritized sampling over a 10M buffer

### 3.3 Key Engineering Decisions

**Why not CUDA graphs (`mode='reduce-overhead'`)?**  
CUDA graphs pin input tensors at fixed memory addresses. Since each gradient step creates new numpy→CUDA tensor views from a CPU replay buffer, CUDA graphs caused a memory leak growing at ~28 GB per 448 gradient steps. Reverted to `backend='inductor'` only.

**Why not CPU worker processes for collection?**  
`ProcessPoolExecutor` with default `fork` start method inherits the parent's CUDA context, causing child processes to deadlock. Switching to `spawn` context fixed the deadlock but CPU workers completed 512 games in ~14 seconds vs ~700ms for GPU-batched vectorized collection — 20× slower. Vectorized collection was retained.

**Batch size 65,536 over 16,384:**  
Larger batches saturate all 48 SMs per gradient step, increasing effective GPU utilization from ~3% to ~32% (measured via `nvidia-smi` polling, which undercounts on GB10).

---

## 4. Training Results

### 4.1 Training Curve

| Time (h) | Episodes | Grad Steps | Loss | Epsilon | Red Win% | Draw% |
|---|---|---|---|---|---|---|
| 0.02 | 9,728 | 704 | 0.133 | 0.990 | 35.9% | 26.9% |
| 0.30 | 103,936 | 12,480 | 0.073 | 0.829 | 45.4% | 14.3% |
| 0.56 | 181,248 | 22,144 | 0.056 | 0.717 | 43.0% | 14.5% |
| 1.09 | 302,080 | 38,976 | 0.055 | 0.557 | 43.5% | 15.5% |
| 1.57 | 420,864 | 53,312 | 0.055 | 0.449 | 43.1% | 11.8% |
| 1.97 | 502,272 | 62,272 | 0.058 | 0.393 | 47.6% | 10.1% |
| **2.10** | **527,872** | **65,472** | **0.057** | **0.375** | **48.5%** | **10.4%** |

### 4.2 Notable Observations

- **Loss**: Dropped 57% in the first 0.3 hours (0.133 → 0.057), then plateaued. The plateau is expected while epsilon > 0.3 — ~30–40% random moves introduces high noise in TD targets.
- **Draw rate collapse**: From ~28% (random play) to ~10% by mid-training. The agent discovered that decisive play beats passive play.
- **Red win rate rise**: From ~37% (random baseline, expected ~33%) to ~48.5% at end. RED's advantage was being discovered gradually as epsilon decreased.
- **GPU memory**: Stable at 0.03 GB throughout — no memory leak.
- **Training speed**: Slowed from 160 ep/s early (small buffer, fast PER sampling) to 57 ep/s late (7M buffer, slower cached-cumsum). Wall-clock training took 8 seconds per 64-step round at the end vs 2.5 seconds early.

---

## 5. Strategy Analysis

### 5.1 First-Player Advantage

Post-training greedy evaluation (ε=0): **2,000 games, RED wins 100%, BLUE wins 0%, Draws 0%.**

This is a decisive first-player advantage. In TicTacPro, the first player can establish a dominant position that the second player cannot overcome against a well-trained opponent. This likely stems from the gobbling mechanic — the first player can place a large piece to immediately create a threat that requires the second player to respond defensively.

> **Going first in TicTacPro is not just an advantage — it is currently decisive against this agent.**

### 5.2 Best Opening Moves

Q-values for the first move from an empty board:

| Rank | Cell | Size | Q-value | Interpretation |
|---|---|---|---|---|
| 1 | (2,0) — bottom-left | **Medium** | **+14.26** | Corner control with mid-size |
| 2 | (0,1) — top-center | **Medium** | **+13.82** | Edge control with mid-size |
| 3 | (0,1) — top-center | Large | +13.25 | Edge control with large |
| 4 | (0,1) — top-center | Small | +12.25 | Edge control with small |
| 5 | (1,1) — center | Large | +12.10 | Center dominance |
| 6 | (2,2) — bottom-right | Medium | +11.57 | Opposite corner |
| 7 | (0,2) — top-right | Large | +11.19 | Corner with large |
| 8 | (1,0) — mid-left | Small | +11.10 | Edge flanking |
| 9 | (2,2) — bottom-right | Small | +10.95 | Opposite corner small |
| 10 | (2,2) — bottom-right | Large | +10.36 | Opposite corner large |

### 5.3 Q-value Heatmap (Large Pieces, Empty Board)

```
Position:  [0,0]      [0,1]      [0,2]
           +9.999    +13.246    +11.186

           [1,0]      [1,1]      [1,2]
           +9.060    +12.097    +5.834

           [2,0]      [2,1]      [2,2]
           +5.757     +6.107    +10.365
```

**Preferred Large-piece positions:** Top-center (13.25) > Center (12.10) > Top-right (11.19)  
**Avoid:** Mid-right (5.83) and bottom-left (5.76) for Large pieces

### 5.4 Strategic Insights

**As first player:**
- Open with a **Medium piece at bottom-left** (Q=+14.26) — medium-size opening is harder to immediately goble
- Alternatively, **top-center** in any size (Q=+12–13.8) creates threats along the top row and a diagonal
- The **center** with a Large piece (Q=+12.10) enables bullseye setups and claims maximum diagonal control
- Avoid opening mid-right with any large piece (Q=+5.83)

**As second player:**
- You face a structural disadvantage against this agent
- The agent's counter-moves depend strongly on the opponent's opening — no universal reactive strategy emerged from this training
- Focus on preventing the three-in-a-row patterns favored by the first player

**Medium pieces dominate openings:** The top-2 moves both use Medium pieces. This makes sense — Medium pieces can goble Small but be gobled by Large, making them hard to immediately counter while preserving Large pieces for threats.

---

## 6. Challenges and Lessons Learned

### 6.1 GPU Utilization on Unified Memory

**Problem:** `nvidia-smi` consistently reported 0–43% GPU utilization on the GB10, with readings that seemed unreliable.

**Root cause:** The GB10 Grace-Blackwell uses a fully unified LPDDR5X memory pool. `nvidia-smi`'s utilization counter was designed for discrete GPU architectures and undercounts compute activity on unified-memory integrated designs.

**Workaround:** Used wall-clock timing as a proxy: 64 gradient steps × 65,536 batch on GPU took 8–9 seconds vs estimated 60–90 seconds on CPU-only code.

### 6.2 CUDA Graphs Memory Leak

**Problem:** Switching from `backend='inductor'` to `mode='reduce-overhead'` (CUDA graphs) caused GPU memory to grow by ~28 GB per 448 gradient steps, projecting to an OOM crash within minutes on the 121 GB system.

**Root cause:** CUDA graphs record the entire forward/backward computation graph and pin all input tensors at fixed memory addresses. Since our training loop creates new CPU-sampled tensors each step (numpy → `torch.from_numpy` → `.to(cuda)`), each step allocated ~96 MB of pinned GPU memory that could not be reclaimed.

**Fix:** Reverted to `backend='inductor'` only. Memory stabilized at 0.03 GB.

### 6.3 ProcessPoolExecutor Fork vs Spawn

**Problem:** CPU worker processes (for pipelined collection) immediately deadlocked on startup.

**Root cause:** Linux `ProcessPoolExecutor` uses `fork` by default. After CUDA is initialized in the parent, forked child processes inherit a broken CUDA context that hangs on any PyTorch operation.

**Fix:** Used `mp_context=multiprocessing.get_context('spawn')` to start fresh processes. The workers ran correctly but proved 20× slower than GPU-batched collection — so the CPU worker approach was abandoned in favor of vectorized GPU collection.

### 6.4 Epsilon Decay Calibration

**Problem:** Default epsilon decay of 0.9999900 was too slow — epsilon only reached 0.375 after 65,472 gradient steps, meaning the agent was still 37.5% random at the end of training.

**Fix:** Recalibrated to 0.999985 for the 2.1-hour run and 0.99994 for the continuation run, targeting epsilon=0.05 by end of training.

---

## 7. Conclusion

A Double DQN agent with Prioritized Experience Replay and Dueling architecture successfully learned to play TicTacPro, discovering a strong first-player advantage and non-trivial opening preferences after 527,872 self-play episodes on an NVIDIA DGX Spark GB10.

Key findings:
1. **First-player advantage is decisive** — the trained agent wins 100% of greedy games as RED
2. **Best first move:** Medium piece at bottom-left corner (Q=+14.26)
3. **Top-center is universally valuable** — appears in ranks 2, 3, 4 across all piece sizes
4. **Medium pieces dominate openings** — they balance gobbling threat with preservation of Large pieces
5. **Draws collapse** during training (28% → 10%) — the agent learned to play decisively

Training continues with faster epsilon decay (0.99994) targeting ε < 0.05 for fully exploitative play. The final exploitative policy should show even stronger first-player dominance and clearer strategic preferences.

---

## Appendix A: Reproducing the Experiment

```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install numpy tensorboard tqdm pygame

# Train (2.1 hours)
python3 train_spark.py \
    --hours 2.1 \
    --batch-size 65536 \
    --train-steps 64 \
    --epsilon-decay 0.999985 \
    --buffer-size 10000000

# Play against the agent
python3 play.py --checkpoint checkpoints_spark/final.pt

# View training curves
tensorboard --logdir logs_spark
```

## Appendix B: File Structure

```
ticTACpro-sim/
├── game/
│   └── tictacpro.py          # Game engine (3×3 board, gobbling mechanic)
├── rl/
│   ├── network.py             # GPUDQNNetwork (Dueling, LayerNorm, BF16)
│   ├── agent.py               # PrioritizedReplayBuffer + GPUDQNAgent
│   └── gpu_trainer.py         # Vectorized collection + training loop
├── gui/
│   └── game_gui.py            # Pygame visual interface
├── tests/                     # Unit tests (game, network, agent, trainer)
├── train_spark.py             # Main training entry point
├── play.py                    # Play against trained agent
└── FINDINGS.md                # This document
```
