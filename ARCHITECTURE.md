# Tic Tac Pro RL - System Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   play.py    │  │  analyze.py  │  │ Tensorboard  │      │
│  │  (Pygame GUI)│  │  (Reports)   │  │  (Metrics)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Training & Analysis Layer                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ train_local  │  │  train_dgx   │  │   Strategy   │      │
│  │    .py       │  │     .py      │  │  Analyzer    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   RL Algorithm Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  DQN Agent   │  │   Trainer    │  │   Replay     │      │
│  │  (Policy)    │  │  (Self-play) │  │   Buffer     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Neural Network Layer                       │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │  Dueling DQN │  │   Conv DQN   │                         │
│  │  (FC layers) │  │ (Conv layers)│                         │
│  └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Game Engine Layer                       │
│  ┌──────────────────────────────────────────────────┐       │
│  │              TicTacPro Game Engine               │       │
│  │  • State representation  • Win detection         │       │
│  │  • Move validation      • Piece stacking logic   │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Flow
```
Game State (61 features)
    ↓
DQN Network (Policy)
    ↓
Q-values for all actions (27)
    ↓
Epsilon-greedy selection
    ↓
Execute action in game
    ↓
Observe reward & next state
    ↓
Store in Replay Buffer
    ↓
Sample batch (64-512)
    ↓
Compute TD error
    ↓
Backpropagate & update weights
    ↓
Periodically update Target Network
```

### Playing Flow
```
Human Move
    ↓
Update Game State
    ↓
AI Turn → Get Q-values
    ↓
Select best legal move
    ↓
Execute move
    ↓
Update display with suggestions
    ↓
Check win condition
    ↓
Repeat or end game
```

## Component Details

### 1. Game Engine (`game/tictacpro.py`)

**Responsibilities:**
- Maintain game state (3×3×3 board)
- Validate moves
- Detect wins (same-size lines + bullseyes)
- Convert state to neural network input

**Key Classes:**
- `TicTacPro`: Main game class
- `Player`: Enum (RED, BLUE, NONE)
- `PieceSize`: Enum (SMALL, MEDIUM, LARGE)

**State Representation:**
```python
board: np.ndarray(3, 3, 3)  # [row, col, size_level]
pieces_remaining: dict       # {player: {size: count}}
current_player: Player
game_over: bool
winner: Player
```

### 2. Neural Networks (`rl/network.py`)

**DQN Architecture:**
```
Input(61) → Linear(256) → ReLU → Dropout(0.2)
         → Linear(256) → ReLU → Dropout(0.2)
         
Split into:
├─ Value Stream: Linear(128) → ReLU → Linear(1)
└─ Advantage Stream: Linear(128) → ReLU → Linear(27)

Combine: Q = V + (A - mean(A))
```

**ConvDQN Architecture:**
```
Input(61) → reshape to (6, 3, 3)
         → Conv2d(32) → ReLU
         → Conv2d(64) → ReLU
         → Flatten + concat piece features
         → Linear(256) → ReLU → Dropout
         → Linear(128) → ReLU → Dropout
         → Linear(27)
```

### 3. DQN Agent (`rl/agent.py`)

**Key Components:**
- **Policy Network**: Current Q-function approximation
- **Target Network**: Stable target for TD learning
- **Replay Buffer**: Stores experiences (100K-1M)
- **Optimizer**: Adam with learning rate 0.0001-0.001

**Methods:**
- `get_action()`: Epsilon-greedy action selection
- `store_experience()`: Add to replay buffer
- `learn()`: Batch update using TD learning
- `get_move_suggestions()`: Top-k moves for display

### 4. Trainer (`rl/trainer.py`)

**Training Loop:**
```python
for episode in range(num_episodes):
    game = TicTacPro()
    while not game.game_over:
        action = agent.get_action(game, player)
        reward = game.make_move(action)
        agent.store_experience(state, action, reward, next_state)
        agent.learn()
    
    if episode % eval_freq == 0:
        evaluate_agent()
    
    if episode % save_freq == 0:
        save_checkpoint()
```

**Metrics Logged:**
- Win rates (Red/Blue/Draw)
- Average game length
- Q-value estimates
- Training loss
- Epsilon (exploration rate)

### 5. GUI (`gui/game_gui.py`)

**Features:**
- 500×500 pixel game board
- 3D piece rendering with shadows
- Real-time move suggestions (green circles)
- Piece size selector
- Game status display
- Keyboard shortcuts

**Rendering:**
```
┌─────────────────┬──────────────────┐
│                 │  Piece Selector  │
│   Game Board    │  ┌────────────┐  │
│   (3×3 cells)   │  │ ○ Small    │  │
│                 │  │ ● Medium   │  │
│                 │  │ ⬤ Large    │  │
│                 │  └────────────┘  │
│                 │  Game Info       │
│                 │  Controls Help   │
└─────────────────┴──────────────────┘
```

### 6. Strategy Analyzer (`analysis/strategy_analyzer.py`)

**Analyses Performed:**

1. **Opening Analysis**
   - Sample 1000 games
   - Track first move frequencies
   - Compute Q-values for openings

2. **Position Values**
   - Analyze 1000 random positions
   - Average Q-values per cell/size
   - Generate heatmaps

3. **Win Patterns**
   - Analyze 500 games
   - Track move sequences
   - Identify common patterns

4. **Piece Usage**
   - Count piece type usage
   - Calculate percentages
   - Identify preferences

## Training Strategies

### Local Training (Mac)

**Configuration:**
- Episodes: 10K-50K
- Batch size: 64
- Buffer size: 100K
- Epsilon decay: 0.9995
- Learning rate: 0.001

**Timeline:**
- 10K episodes: 10-20 minutes
- 50K episodes: 1-2 hours

### DGX Training (Multi-GPU)

**Configuration:**
- Episodes: 1M-5M
- Batch size: 256-512
- Buffer size: 500K-1M
- Epsilon decay: 0.99995
- Learning rate: 0.0005

**Timeline:**
- 1M episodes: 2-4 hours
- 5M episodes: 10-20 hours

**Distributed Setup:**
```python
# Each GPU runs independent agent
# Shared policy network (DDP)
# Synchronized updates
# Parallel experience collection
```

## Performance Characteristics

### Computational Complexity

**Per Episode:**
- State representations: O(1)
- Neural network forward: O(n) where n = network size
- Legal move filtering: O(27)
- Experience storage: O(1)
- Batch learning: O(batch_size × network_size)

**Memory Usage:**
- Network weights: ~50 MB
- Replay buffer (100K): ~500 MB
- Replay buffer (1M): ~5 GB
- GPU memory: 2-4 GB per GPU

### Scalability

**Single GPU:**
- ~100-500 episodes/minute
- Depends on: batch size, buffer size, network size

**Multi-GPU (DGX):**
- Linear scaling up to communication overhead
- 4 GPUs: ~3-4× speedup
- 8 GPUs: ~6-7× speedup

## File I/O

### Checkpoints
```python
checkpoint = {
    'policy_net_state_dict': ...,
    'target_net_state_dict': ...,
    'optimizer_state_dict': ...,
    'epsilon': ...,
    'episode': ...,
    'metrics': ...
}
torch.save(checkpoint, 'checkpoint.pt')
```

### Tensorboard Logs
- Loss curves
- Win rate over time
- Average episode length
- Epsilon decay
- Q-value distributions

### Analysis Outputs
- `strategy_guide.md`: Markdown report
- `position_heatmap.png`: Matplotlib figure
- `statistics.json`: Raw data export

## Error Handling

**Graceful Degradation:**
- Missing checkpoint: Start from scratch
- CUDA unavailable: Fall back to CPU
- Interrupted training: Save latest checkpoint
- Invalid moves: Filter from action space

**User Feedback:**
- Progress bars (tqdm)
- Tensorboard real-time metrics
- Console logging
- Error messages with suggestions

## Testing

**Unit Tests (`test_setup.py`):**
1. Import verification
2. Game engine functionality
3. Neural network forward pass
4. Agent action selection
5. Training loop (10 episodes)

**Integration Tests:**
- Full training run (small scale)
- Play against trained agent
- Analysis generation

## Extension Points

**Easy to add:**
- New neural network architectures
- Different RL algorithms (A3C, PPO)
- Alternative reward structures
- Opponent modeling
- Transfer learning

**Modification points:**
- `rl/network.py`: Add new architectures
- `rl/agent.py`: Add new algorithms
- `rl/trainer.py`: Modify training loop
- `game/tictacpro.py`: Add game variants

## Summary

This is a **modular, scalable, production-ready** RL system with:
- Clean separation of concerns
- Extensive documentation
- Flexible configuration
- GPU acceleration
- Distributed training support
- Comprehensive analysis tools

Total codebase: ~2,700 lines of well-documented Python code.

---

*Architecture designed for clarity, performance, and extensibility*
