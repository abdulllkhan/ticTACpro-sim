"""Unit tests for DQN network architectures."""

import pytest
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from rl.network import DQNNetwork, ConvDQNNetwork, GPUDQNNetwork, NumpyDQNInference

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

    def test_action_index_roundtrip(self):
        """Lines 183, 187-190: ConvDQNNetwork get_action_index / get_action_from_index."""
        net = ConvDQNNetwork()
        for row in range(3):
            for col in range(3):
                for size in [1, 2, 3]:
                    idx = net.get_action_index(row, col, size)
                    assert 0 <= idx < 27
                    r2, c2, s2 = net.get_action_from_index(idx)
                    assert (r2, c2, s2) == (row, col, size)


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


class TestNumpyDQNInference:
    """§191: NumpyDQNInference must match GPUDQNNetwork output exactly."""

    def _make_pair(self, hidden_sizes=(256, 128, 64)):
        net = GPUDQNNetwork(hidden_sizes=hidden_sizes)
        net.eval()
        with torch.no_grad():
            sd = {k: v.cpu() for k, v in net.state_dict().items()}
        numpy_net = NumpyDQNInference(sd, hidden_sizes=hidden_sizes)
        return net, numpy_net

    def test_batch_output_matches_gpu(self):
        net, np_net = self._make_pair()
        x = np.random.randn(16, 61).astype(np.float32)
        with torch.no_grad():
            torch_out = net(torch.from_numpy(x)).numpy()
        numpy_out = np_net(x)
        np.testing.assert_allclose(numpy_out, torch_out, rtol=1e-4, atol=1e-5)

    def test_single_input_matches_gpu(self):
        net, np_net = self._make_pair()
        x = np.random.randn(61).astype(np.float32)
        with torch.no_grad():
            torch_out = net(torch.from_numpy(x).unsqueeze(0)).numpy()[0]
        numpy_out = np_net(x)
        assert numpy_out.shape == (27,)
        np.testing.assert_allclose(numpy_out, torch_out, rtol=1e-4, atol=1e-5)

    def test_output_shape_batch(self):
        _, np_net = self._make_pair()
        out = np_net(np.random.randn(8, 61).astype(np.float32))
        assert out.shape == (8, 27)

    def test_output_shape_single(self):
        _, np_net = self._make_pair()
        out = np_net(np.random.randn(61).astype(np.float32))
        assert out.shape == (27,)

    def test_no_nan_in_output(self):
        _, np_net = self._make_pair()
        out = np_net(np.random.randn(32, 61).astype(np.float32))
        assert not np.isnan(out).any()

    def test_from_network_factory(self):
        net = GPUDQNNetwork(hidden_sizes=(256, 128, 64))
        net.eval()
        np_net = NumpyDQNInference.from_network(net)
        x = np.random.randn(4, 61).astype(np.float32)
        with torch.no_grad():
            torch_out = net(torch.from_numpy(x)).numpy()
        numpy_out = np_net(x)
        np.testing.assert_allclose(numpy_out, torch_out, rtol=1e-4, atol=1e-5)

    def test_dueling_q_values_nonzero_variance(self):
        _, np_net = self._make_pair()
        out = np_net(np.random.randn(4, 61).astype(np.float32))
        for i in range(4):
            assert out[i].std() > 0

    def test_action_index_roundtrip(self):
        """Lines 338, 341-344: NumpyDQNInference get_action_index / get_action_from_index."""
        _, np_net = self._make_pair()
        for row in range(3):
            for col in range(3):
                for size in [1, 2, 3]:
                    idx = np_net.get_action_index(row, col, size)
                    assert 0 <= idx < 27
                    r2, c2, s2 = np_net.get_action_from_index(idx)
                    assert (r2, c2, s2) == (row, col, size)
