import unittest

import torch

from thetamem import ThetaMemLayer


class ThetaMemLayerTest(unittest.TestCase):
    def test_forward_shape_all_states(self):
        x = torch.randn(2, 19, 64)
        for state in ("hadamard", "concat", "outer"):
            torch.manual_seed(0)
            layer = ThetaMemLayer(
                64, heads=2, key_dim=16, value_dim=32, state=state, chunk=8
            )
            out = layer(x)
            self.assertEqual(out.shape, x.shape)

    def test_forward_all_updates(self):
        x = torch.randn(2, 19, 64)
        for update in ("sum", "second_pass", "multi_pass", "delta"):
            torch.manual_seed(0)
            layer = ThetaMemLayer(
                64, heads=2, key_dim=16, value_dim=32, update=update, chunk=8
            )
            self.assertEqual(layer(x).shape, x.shape)

    def test_forward_value_centers(self):
        x = torch.randn(2, 19, 64)
        for center in ("running_mean", "exact_mean"):
            torch.manual_seed(0)
            layer = ThetaMemLayer(
                64, heads=2, key_dim=16, value_dim=32,
                update="multi_pass", passes=2, value_center=center, chunk=8,
            )
            self.assertEqual(layer(x).shape, x.shape)
            self.assertEqual(layer.memory.value_center, center)

    def test_position_toggle(self):
        x = torch.randn(2, 19, 64)
        torch.manual_seed(0)
        with_rope = ThetaMemLayer(64, heads=2, key_dim=16, value_dim=32)
        torch.manual_seed(0)
        without = ThetaMemLayer(
            64, heads=2, key_dim=16, value_dim=32, position="none"
        )
        self.assertFalse(torch.allclose(with_rope(x), without(x)))

    def test_linear_and_direct_hadamard_frontends(self):
        x = torch.randn(1, 12, 64)
        linear = ThetaMemLayer(
            64,
            heads=2,
            key_dim=16,
            value_dim=32,
            qk_ops=("conv", "rope"),
            value_ops=("conv",),
            chunk=8,
        )
        direct = ThetaMemLayer(
            64,
            heads=2,
            key_dim=16,
            value_dim=32,
            qk_projection="direct_hadamard",
            qk_ops=("conv", "rope", "l2"),
            value_ops=("conv", "silu"),
            chunk=8,
        )
        direct_with_legacy_state = ThetaMemLayer(
            64,
            heads=2,
            key_dim=16,
            value_dim=32,
            state="hadamard",
            qk_projection="direct_hadamard",
            qk_ops=(),
            value_ops=(),
        )
        self.assertEqual(linear.q_proj.out_features, 2 * 16)
        self.assertEqual(direct.q_proj.out_features, 2 * 2 * 16)
        self.assertEqual(len(direct.memory.lift.branch_weights), 0)
        self.assertEqual(
            len(direct_with_legacy_state.memory.lift.branch_weights), 0
        )
        projected = direct.q_proj(x).view(1, 12, 2, 2, 16)
        expected = (projected[:, :, :, 0] * projected[:, :, :, 1]).transpose(
            1, 2
        )
        torch.testing.assert_close(
            direct._project_qk(x, direct.q_proj), expected
        )
        self.assertEqual(linear(x).shape, x.shape)
        self.assertEqual(direct(x).shape, x.shape)

    def test_gradients_finite(self):
        x = torch.randn(2, 19, 64, requires_grad=True)
        layer = ThetaMemLayer(64, heads=2, key_dim=16, value_dim=32, chunk=8)
        layer(x).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, parameter in layer.named_parameters():
            self.assertIsNotNone(parameter.grad, msg=name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), msg=name)

    def test_causality(self):
        # Changing a future token must not change past outputs.
        torch.manual_seed(1)
        layer = ThetaMemLayer(
            64, heads=2, key_dim=16, value_dim=32, chunk=8, position="none"
        ).double()
        x = torch.randn(1, 12, 64, dtype=torch.float64)
        out_a = layer(x)
        x_mod = x.clone()
        x_mod[:, -1] += 10.0
        out_b = layer(x_mod)
        torch.testing.assert_close(out_a[:, :-4], out_b[:, :-4])

    def test_state_size_reports_conv_cache(self):
        layer = ThetaMemLayer(128, heads=4, key_dim=32, value_dim=64)
        self.assertEqual(layer.state_size, 4 * 32 * 64 + 1536)
        no_conv = ThetaMemLayer(
            128,
            heads=4,
            key_dim=32,
            value_dim=64,
            qk_ops=(),
            value_ops=(),
        )
        self.assertEqual(
            no_conv.state_size, 4 * no_conv.memory.state_size
        )
        with self.assertRaises(ValueError):
            ThetaMemLayer(qk_ops=("conv", "conv"))


if __name__ == "__main__":
    unittest.main()
