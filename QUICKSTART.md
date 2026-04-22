# Quick Start Guide

Get up and running with Tic Tac Pro RL in 5 minutes!

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test the setup
python test_setup.py
```

## Option 1: Quick Local Training (20 minutes)

Perfect for testing and learning on your Mac:

```bash
# Train for 10K episodes (~10-20 minutes)
python train_local.py --episodes 10000

# Play against the trained AI
python play.py

# Analyze strategies
python analyze.py
```

## Option 2: Intensive DGX Training (Recommended)

For serious training on your NVIDIA DGX Spark:

```bash
# Train for 1M episodes on all available GPUs
python train_dgx.py --episodes 1000000 --num-gpus -1

# This will take 2-4 hours but produce a much stronger agent

# Then play and analyze
python play.py --checkpoint checkpoints_dgx/final.pt
python analyze.py --checkpoint checkpoints_dgx/final.pt
```

## Training Comparison

| Method | Time | Episodes | Strength | Best For |
|--------|------|----------|----------|----------|
| Local Quick | 10-20 min | 10K | Beginner | Testing setup |
| Local Full | 1-2 hours | 50K | Intermediate | Learning |
| DGX Standard | 2-4 hours | 1M | Advanced | Competitive play |
| DGX Intensive | 10-20 hours | 5M | Expert | Optimal strategies |

## Common Commands

### Training
```bash
# Quick test
python train_local.py --episodes 5000

# Full local training
python train_local.py --episodes 50000

# DGX training (use all GPUs)
python train_dgx.py --episodes 1000000

# Resume training
python train_local.py --resume checkpoints/latest.pt --episodes 10000
```

### Playing
```bash
# Play as Red (AI is Blue)
python play.py

# Play as Blue (AI is Red)
python play.py --ai-player red

# Play without move suggestions
python play.py --no-suggestions
```

### Analysis
```bash
# Full analysis
python analyze.py

# Quick analysis (fewer games)
python analyze.py --opening-games 500 --pattern-games 250
```

### Monitoring
```bash
# Watch training progress in real-time
tensorboard --logdir logs

# For DGX training
tensorboard --logdir logs_dgx
```

## GUI Controls

While playing:
- **Click**: Select piece size and place on board
- **1/2/3**: Select Small/Medium/Large piece
- **S**: Toggle move suggestions (AI recommendations)
- **R**: Reset game
- **ESC**: Quit

## Tips for Success

### Training Tips
1. **Start small**: Test with 5K episodes first
2. **Monitor Tensorboard**: Watch for stable win rates (~50%)
3. **Use DGX for serious training**: 1M+ episodes recommended
4. **Save checkpoints**: Training can be interrupted and resumed

### Playing Tips
1. **Enable suggestions**: Press 'S' to see AI's top moves
2. **Study the suggestions**: Try to understand *why* the AI chooses certain moves
3. **Control the center**: Often has highest strategic value
4. **Watch for bullseyes**: Don't let opponent stack all 3 sizes in one cell
5. **Plan ahead**: Consider which piece sizes you'll need later

### Strategy Learning
1. **Analyze before playing**: Run `analyze.py` to understand optimal strategies
2. **Read the strategy guide**: Check `analysis/strategy_guide.md`
3. **Study the heatmap**: See which positions are most valuable
4. **Practice**: Play multiple games to internalize patterns

## File Locations

After training, you'll find:

```
ticTACpro-sim/
├── checkpoints/          # Local training checkpoints
│   ├── final.pt         # Best model for local training
│   └── latest.pt        # Resume from here if interrupted
├── checkpoints_dgx/      # DGX training checkpoints
│   └── final.pt         # Best model for DGX training
├── logs/                 # Training logs (view with Tensorboard)
├── analysis/            # Strategy analysis outputs
│   ├── strategy_guide.md      # Auto-generated strategy guide
│   ├── position_heatmap.png   # Position value visualization
│   └── statistics.json        # Raw statistics
```

## Troubleshooting

### "No module named 'X'"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory" (during training)
```bash
python train_local.py --batch-size 32
```

### Training is very slow
- Use DGX Spark for faster training
- Or reduce episodes: `--episodes 5000`

### GUI doesn't show
- Make sure you have a display
- Try reinstalling pygame: `pip install --upgrade pygame`

## Next Steps

After your first successful training:

1. **Compare different training lengths**:
   - Train 10K, 50K, and 100K episodes
   - Play against each to feel the difference

2. **Experiment with hyperparameters**:
   - Try different learning rates: `--lr 0.0001` vs `--lr 0.001`
   - Adjust exploration: `--epsilon-decay 0.999` vs `--epsilon-decay 0.9995`

3. **Use DGX for production training**:
   - Run overnight: 1-5M episodes
   - Generate comprehensive strategy guide

4. **Share your results**:
   - Save your best checkpoint
   - Share strategy insights with others

## Getting Help

- **Test setup**: `python test_setup.py`
- **Full documentation**: See [README.md](README.md)
- **Game rules**: Search "Tic Tac Pro Brass Monkey" online

---

**You're ready to go! Start with:**
```bash
python train_local.py --episodes 10000
```

Good luck mastering Tic Tac Pro! 🎮🧠🚀
