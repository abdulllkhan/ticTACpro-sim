"""Unit tests for DQN network architectures."""

import pytest
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rl.network import DQNNetwork, ConvDQNNetwork, GPUDQNNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDQNNetwork:
    def test_output_shape_single(self):
        net = DQNNetwork()
        x = torch.randn(61)
        out = net(x)
        assert out.shape == (27,)

    def test_output_shape_batch(self):
        net = DQNNetwork()
        x = torch.randn(32, 61)
        out = net(x)
        assert out.shape == (32, 27)

    def test_action_index_roundtrip(self):
        net = DQNNetwork()
        for row in range(3):
            for col in range(3):
                for size in [1, 2, 3]:
                    idx = net.get_action_index(row, col, size)
                    r2, c2, s2 = net.get_action_from_index(idx)
                    assert (r2, c2, s2) == (row, col, size)

    def test_action_index_range(self):
        net = DQNNetwork()
        for row in range(3):
            for col in range(3):
                for size in [1, 2, 3]:
                    idx = net.get_action_index(row, col, size)
                    assert 0 <= idx < 27


class TestConvDQNNetwork:
    def test_output_shape(self):
        net = ConvDQNNetwork()
        x = torch.randn(4, 61)
        out = net(x)
        assert out.shape == (4, 27)

    def test_single_state(self):
        net = ConvDQNNetwork()
        x = torch.randn(61)
        out = net(x)
        assert out.shape == (27,)


class TestGPUDQNNetwork:
    def test_output_shape_batch(self):
        net = GPUDQNNetwork().to(DEVICE)
        x = torch.randn(64, 61).to(DEVICE)
        out = net(x)
        assert out.shape == (64, 27)

    def test_output_shape_single(self):
        net = GPUDQNNetwork().to(DEVICE)
        x = torch.randn(1, 61).to(DEVICE)
        out = net(x)
        assert out.shape == (1, 27)

    def test_large_batch(self):
        net = GPUDQNNetwork().to(DEVICE)
        x = torch.randn(16384, 61).to(DEVICE)
        out = net(x)
        assert out.shape == (16384, 27)

    def test_no_nan_in_output(self):
        net = GPUDQNNetwork().to(DEVICE)
        x = torch.randn(128, 61).to(DEVICE)
        out = net(x)
        assert not torch.isnan(out).any()

    def test_dueling_architecture_q_values_differ(self):
        net = GPUDQNNetwork().to(DEVICE)
        x = torch.randn(4, 61).to(DEVICE)
        out = net(x)
        # Each state should produce different Q-values across actions
        for i in range(4):
            assert out[i].std() > 0

    def test_action_index_roundtrip(self):
        net = GPUDQNNetwork()
        for row in range(3):
            for col in range(3):
                for size in [1, 2, 3]:
                    idx = net.get_action_index(row, col, size)
                    r2, c2, s2 = net.get_action_from_index(idx)
                    assert (r2, c2, s2) == (row, col, size)

    def test_custom_hidden_sizes(self):
        net = GPUDQNNetwork(hidden_sizes=(256, 128, 64)).to(DEVICE)
        x = torch.randn(8, 61).to(DEVICE)
        out = net(x)
        assert out.shape == (8, 27)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_bf16_autocast(self):
        net = GPUDQNNetwork().cuda()
        x = torch.randn(128, 61).cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = net(x)
        assert not torch.isnan(out.float()).any()
