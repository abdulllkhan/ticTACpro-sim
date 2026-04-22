#!/usr/bin/env python3
"""
Quick test script to verify the installation and setup

This script tests all major components of the Tic Tac Pro RL system.
"""

import sys


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False

    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False

    try:
        import pygame
        print(f"  ✓ Pygame {pygame.__version__}")
    except ImportError as e:
        print(f"  ✗ Pygame import failed: {e}")
        return False

    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ✗ Matplotlib import failed: {e}")
        return False

    return True


def test_game_engine():
    """Test the game engine"""
    print("\nTesting game engine...")

    try:
        from game.tictacpro import TicTacPro, PieceSize, Player

        game = TicTacPro()
        print("  ✓ Game initialization")

        # Test moves
        legal_moves = game.get_legal_moves()
        print(f"  ✓ Legal moves: {len(legal_moves)} available")

        # Make a move
        success = game.make_move(0, 0, PieceSize.SMALL)
        print(f"  ✓ Move execution: {'Success' if success else 'Failed'}")

        # Test state
        state = game.get_state_tensor()
        print(f"  ✓ State representation: {state.shape}")

        return True

    except Exception as e:
        print(f"  ✗ Game engine test failed: {e}")
        return False


def test_neural_network():
    """Test the neural network"""
    print("\nTesting neural network...")

    try:
        import torch
        from rl.network import DQNNetwork, ConvDQNNetwork

        # Test standard network
        net = DQNNetwork()
        dummy_input = torch.randn(4, 61)
        output = net(dummy_input)
        print(f"  ✓ DQN Network: input {dummy_input.shape} -> output {output.shape}")

        # Test conv network
        conv_net = ConvDQNNetwork()
        output_conv = conv_net(dummy_input)
        print(f"  ✓ Conv DQN Network: input {dummy_input.shape} -> output {output_conv.shape}")

        return True

    except Exception as e:
        print(f"  ✗ Neural network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent():
    """Test the DQN agent"""
    print("\nTesting DQN agent...")

    try:
        from rl.agent import DQNAgent
        from game.tictacpro import TicTacPro, Player

        agent = DQNAgent()
        print("  ✓ Agent initialization")

        game = TicTacPro()
        action = agent.get_action(game, Player.RED, epsilon=0.5)
        print(f"  ✓ Action selection: {action}")

        # Test move suggestions
        suggestions = agent.get_move_suggestions(game, Player.RED, top_k=3)
        print(f"  ✓ Move suggestions: {len(suggestions)} moves")

        return True

    except Exception as e:
        print(f"  ✗ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer():
    """Test the trainer (brief training)"""
    print("\nTesting trainer (10 episodes)...")

    try:
        from rl.agent import DQNAgent
        from rl.trainer import Trainer

        agent = DQNAgent(epsilon_start=1.0, epsilon_end=0.5)
        trainer = Trainer(agent, checkpoint_dir='test_checkpoints', log_dir='test_logs')

        # Short training
        trainer.train(num_episodes=10)
        print("  ✓ Training completed")

        # Cleanup test files
        import shutil
        shutil.rmtree('test_checkpoints', ignore_errors=True)
        shutil.rmtree('test_logs', ignore_errors=True)

        return True

    except Exception as e:
        print(f"  ✗ Trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Tic Tac Pro RL - Setup Test")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Game Engine", test_game_engine),
        ("Neural Network", test_neural_network),
        ("DQN Agent", test_agent),
        ("Trainer", test_trainer),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nUnexpected error in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<40} {status}")
        if not result:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Train an agent: python train_local.py --episodes 10000")
        print("  2. Play against AI: python play.py")
        print("  3. Analyze strategies: python analyze.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nTry:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
