# Tic Tac Pro - RL Simulation

A complete reinforcement learning simulation for **Tic Tac Pro** by Brass Monkey. Train a Deep Q-Network (DQN) agent to master the game, then play against it and learn optimal strategies through visual analysis.

## What is Tic Tac Pro?

Tic Tac Pro is an advanced version of tic-tac-toe featuring:

- **3×3 board** with piece stacking mechanics
- **18 pieces total**: Each player has 9 pieces (3 small, 3 medium, 3 large)
- **Multiple win conditions**:
  - Three pieces of the **same size** in a row (horizontal/vertical/diagonal)
  - A "**bullseye**" - stacking all three sizes of your color in one cell

This creates significantly more strategic depth than traditional tic-tac-toe!

## Features

### 🎮 Complete Game Engine
- Full Tic Tac Pro rules implementation
- Efficient state representation for neural networks
- Legal move validation and win detection

### 🧠 Deep Q-Learning Agent
- **DQN with Experience Replay** for stable learning
- **Dueling DQN architecture** for better value estimation
- **Target network** to prevent divergence
- Alternative convolutional architecture for spatial features

### 💪 Scalable Training
- **Local training**: Quick experiments on your Mac (10K-100K games)
- **DGX Spark training**: Intensive training on NVIDIA GPUs (millions of games)
- Multi-GPU distributed training support
- Automatic checkpointing and resume capability

### 🎨 Visual Interface
- Beautiful Pygame-based GUI with 3D piece visualization
- Play against the trained AI
- **Real-time move suggestions** showing AI's top choices
- Position evaluation display
- Game replay and analysis

### 📊 Strategy Analysis
- Opening move analysis
- Position value heatmaps
- Win pattern detection
- Piece usage statistics
- **Auto-generated strategy guide** with optimal tactics
- Tensorboard integration for training metrics

## Installation

### Requirements
- Python 3.10 or higher
- NVIDIA GPU (optional, for faster training)

### Setup

1. **Clone or navigate to the repository:**
```bash
cd ticTACpro-sim
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### 1. Test the Game Engine
```bash
python -m game.tictacpro
```

### 2. Train Locally (10K episodes, ~10-20 minutes)
```bash
python train_local.py --episodes 10000
```

### 3. Play Against the AI
```bash
python play.py --checkpoint checkpoints/final.pt
```

### 4. Analyze Strategies
```bash
python analyze.py --checkpoint checkpoints/final.pt
```

## Usage Guide

### Local Training

For quick experiments on your local machine:

```bash
python train_local.py --episodes 50000 --lr 0.001 --epsilon-decay 0.9995
```

**Key parameters:**
- `--episodes`: Number of training episodes (default: 10000)
- `--lr`: Learning rate (default: 0.001)
- `--epsilon-start`: Initial exploration (default: 1.0)
- `--epsilon-end`: Final exploration (default: 0.01)
- `--epsilon-decay`: Exploration decay rate (default: 0.9995)
- `--batch-size`: Training batch size (default: 64)
- `--use-conv`: Use convolutional network architecture
- `--resume`: Resume from checkpoint

**Monitor training:**
```bash
tensorboard --logdir logs
```
Visit http://localhost:6006 to see training metrics.

### DGX Spark Training

For intensive training on your NVIDIA DGX Spark cluster:

```bash
python train_dgx.py --episodes 1000000 --num-gpus 4
```

This will:
- Distribute training across multiple GPUs
- Use larger batch sizes and buffer for better learning
- Save checkpoints more frequently
- Train for millions of episodes

**Key differences from local training:**
- Larger replay buffer (500K vs 100K)
- Bigger batch size (256 vs 64)
- Slower epsilon decay for longer exploration
- Multi-GPU distributed training

**Example DGX training session:**
```bash
# Full training run on all available GPUs
python train_dgx.py \
  --episodes 2000000 \
  --num-gpus -1 \
  --batch-size 512 \
  --buffer-size 1000000 \
  --epsilon-decay 0.999995
```

### Playing Against the AI

```bash
python play.py --checkpoint checkpoints/final.pt --ai-player blue
```

**Controls:**
- **Click** to select piece size and place pieces
- **1/2/3** keys: Select Small/Medium/Large piece
- **S**: Toggle move suggestions (shows AI's recommended moves)
- **R**: Reset game
- **ESC**: Quit

**Tips for learning:**
- Enable move suggestions to see what the AI recommends
- Try to understand *why* the AI prefers certain moves
- Experiment with different piece sizes and positions

### Strategy Analysis

Generate comprehensive strategy analysis:

```bash
python analyze.py --checkpoint checkpoints/final.pt --output-dir analysis
```

This creates:
- `strategy_guide.md`: Detailed strategy guide with optimal tactics
- `position_heatmap.png`: Visual heatmap of position values
- `statistics.json`: Raw statistics for further analysis

**Example output:**
- Top opening moves with Q-values
- Piece size usage patterns
- Win pattern analysis
- Strategic tips derived from AI play

## Project Structure

```
ticTACpro-sim/
├── game/                    # Game engine
│   ├── __init__.py
│   └── tictacpro.py        # Core game logic
├── rl/                      # Reinforcement learning
│   ├── __init__.py
│   ├── network.py          # DQN architectures
│   ├── agent.py            # DQN agent with experience replay
│   └── trainer.py          # Training loop
├── gui/                     # Visual interface
│   ├── __init__.py
│   └── game_gui.py         # Pygame GUI
├── analysis/                # Strategy analysis
│   ├── __init__.py
│   └── strategy_analyzer.py
├── train_local.py          # Local training script
├── train_dgx.py            # DGX distributed training script
├── play.py                 # Play against AI
├── analyze.py              # Strategy analysis script
├── requirements.txt        # Dependencies
├── checkpoints/            # Saved model checkpoints
├── logs/                   # Tensorboard logs
└── README.md              # This file
```

## How It Works

### Game Representation

The game state is represented as:
- **Board**: 3×3×3 tensor (row, col, size_level)
- **Pieces remaining**: 6 values (3 sizes × 2 players)
- **Current player**: 1 value

Total: **61 features** fed to the neural network

### Neural Network

**Dueling DQN Architecture:**
```
Input (61 features)
    ↓
Feature Extraction (256 → 256 → ReLU + Dropout)
    ↓
Split into two streams:
    ├─ Value Stream → State value V(s)
    └─ Advantage Stream → Action advantages A(s,a)
    ↓
Combine: Q(s,a) = V(s) + (A(s,a) - mean(A))
    ↓
Output: 27 Q-values (9 positions × 3 sizes)
```

### Training Process

1. **Self-play**: Agent plays against itself
2. **Experience collection**: Store (state, action, reward, next_state) tuples
3. **Replay**: Sample random batches from experience buffer
4. **Learning**: Update policy network to minimize TD error
5. **Target update**: Periodically sync target network
6. **Exploration decay**: Gradually reduce random exploration

### Reward Structure

- **Win**: +1.0
- **Loss**: -1.0
- **Draw**: 0.0
- **Illegal move**: Prevented (not in action space)

Rewards are assigned retroactively after game completion to all moves by each player.

## Training Tips

### For Best Results:

1. **Start with local training** to verify everything works:
   ```bash
   python train_local.py --episodes 5000
   ```

2. **Monitor training metrics** in Tensorboard:
   - Win rates should balance around 50% (self-play equilibrium)
   - Average game length should stabilize
   - Loss should generally decrease

3. **For serious training, use DGX**:
   - Train for at least 500K episodes
   - Use multiple GPUs for faster training
   - Slower epsilon decay allows better exploration

4. **Experiment with hyperparameters**:
   - Learning rate: Try 0.0001 to 0.001
   - Epsilon decay: Slower = more exploration
   - Batch size: Larger = more stable but slower

### Hyperparameter Tuning Guide

| Parameter | Local (Quick) | DGX (Intensive) | Purpose |
|-----------|---------------|-----------------|---------|
| Episodes | 10K-50K | 500K-2M | Total training games |
| Learning rate | 0.001 | 0.0005 | Update step size |
| Epsilon decay | 0.9995 | 0.99995 | Exploration rate |
| Batch size | 64 | 256-512 | Samples per update |
| Buffer size | 100K | 500K-1M | Experience memory |

## Advanced Usage

### Resume Training

If training is interrupted:
```bash
python train_local.py --resume checkpoints/latest.pt --episodes 10000
```

### Custom Network Architecture

Use convolutional architecture for spatial features:
```bash
python train_local.py --use-conv --episodes 20000
```

### Evaluate Against Random Player

Add to your training script:
```python
from rl.trainer import Trainer

# After training
trainer.play_against_random(num_games=100, agent_player=Player.RED)
```

### Export for Analysis

The strategy analyzer can export raw statistics:
```python
from analysis.strategy_analyzer import StrategyAnalyzer

analyzer = StrategyAnalyzer(agent)
analyzer.export_statistics('my_stats.json')
```

## Troubleshooting

### "No module named 'pygame'"
```bash
pip install pygame
```

### "CUDA out of memory"
Reduce batch size:
```bash
python train_local.py --batch-size 32
```

### Training is very slow on Mac
This is expected with CPU training. Consider:
1. Reduce number of episodes
2. Use smaller buffer size
3. Use the DGX Spark cluster for intensive training

### GUI window doesn't appear
Ensure you have a display (not running headless):
```bash
# On macOS, should work by default
# On Linux, ensure DISPLAY is set
echo $DISPLAY
```

## Performance Benchmarks

### Local (MacBook Pro M1)
- **Training speed**: ~100-200 episodes/minute
- **10K episodes**: ~10-20 minutes
- **50K episodes**: ~1-2 hours

### DGX Spark (4× A100 GPUs)
- **Training speed**: ~5000-10000 episodes/minute (distributed)
- **1M episodes**: ~2-4 hours
- **5M episodes**: ~10-20 hours

## Future Improvements

Potential enhancements:
- [ ] AlphaZero-style Monte Carlo Tree Search
- [ ] Multi-agent tournament play
- [ ] OpenAI Gym environment wrapper
- [ ] Web-based interface
- [ ] Pre-trained model zoo
- [ ] Human vs AI leaderboard

## License

This project is for educational purposes. Tic Tac Pro is a product of Brass Monkey.

## Acknowledgments

- **Brass Monkey** for creating Tic Tac Pro
- **OpenAI** for DQN research
- **PyTorch** team for the deep learning framework
- **Pygame** community for the visualization library

## Contact

For questions or issues, please open an issue on GitHub or contact the developer.

---

**Happy Training! 🎮🧠🚀**

*Master Tic Tac Pro with the power of reinforcement learning!*
