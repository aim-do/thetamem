import unittest

import torch

from thetamem import (
    branch,
    concat,
    hadamard,
    key,
    key_part,
    normalize,
    outer,
    state_lift,
)
from thetamem.lift import LiftModule


class LiftAlgebraTest(unittest.TestCase):
    def test_hadamard_requires_two_flat_factors(self):
        with self.assertRaises(ValueError):
            hadamard(branch(8))
        with self.assertRaises(TypeError):
            hadamard(outer(key(), key()), key())

    def test_outer_accepts_arbitrary_order_but_not_nesting(self):
        with self.assertRaises(ValueError):
            outer(branch(4))
        with self.assertRaises(TypeError):
            outer(outer(branch(4), branch(4)), branch(4))
        spec = outer(branch(2), branch(3), branch(4), branch(5))
        module = LiftModule(spec, heads=2, key_dim=8)
        self.assertEqual(module.axis_widths, (2, 3, 4, 5))
        self.assertEqual(module.state_shape(6), (2, 3, 4, 5, 6))

    def test_concat_rejects_outer(self):
        with self.assertRaises(TypeError):
            concat(outer(branch(4), branch(4)), key())

    def test_state_lift_names(self):
        for name in ("hadamard", "concat", "outer"):
            state_lift(name)
        with self.assertRaises(ValueError):
            state_lift("kronecker")

    def test_hadamard_width_mismatch_fails_at_materialization(self):
        spec = hadamard(branch(8), branch(16))
        with self.assertRaises(ValueError):
            LiftModule(spec, heads=2, key_dim=8)


class LiftModuleTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 2, 5, 8, dtype=torch.float64)

    def _module(self, spec, **kwargs):
        module = LiftModule(spec, heads=2, key_dim=8, **kwargs)
        return module.double()

    def test_hadamard_widths_and_values(self):
        module = self._module(state_lift("hadamard"))
        self.assertEqual(module.axis_widths, (8,))
        self.assertEqual(module.feature_width, 8)
        (features,) = module(self.x)
        w_a, w_b = module.branch_weights
        expected = torch.einsum("hfk,bhtk->bhtf", w_a, self.x) * torch.einsum(
            "hfk,bhtk->bhtf", w_b, self.x
        )
        torch.testing.assert_close(features, expected)

    def test_concat_width_is_sum(self):
        module = self._module(state_lift("concat"))
        self.assertEqual(module.axis_widths, (16,))
        (features,) = module(self.x)
        torch.testing.assert_close(features[..., :8], self.x)

    def test_outer_returns_factors(self):
        module = self._module(state_lift("outer", feature_width=4))
        self.assertEqual(module.axis_widths, (4, 4))
        self.assertEqual(module.feature_width, 16)
        self.assertEqual(module.state_shape(6), (4, 4, 6))
        factors = module(self.x)
        self.assertEqual(len(factors), 2)
        self.assertEqual(factors[0].shape, (2, 2, 5, 4))

    def test_generalized_combination(self):
        spec = outer(
            hadamard(branch(8), branch(8)),
            concat(branch(3), key()),
        )
        module = self._module(spec)
        self.assertEqual(module.axis_widths, (8, 11))
        self.assertEqual(len(module.branch_weights), 3)

    def test_split_local_normalization_and_four_factor_outer(self):
        spec = outer(
            hadamard(
                normalize(key_part(0, 2), "center", "l2"),
                normalize(key_part(1, 2), "l1"),
            ),
            key_part(0, 2),
            key_part(1, 2),
            branch(3, source=normalize(key_part(0, 2), "rms")),
        )
        module = self._module(spec)
        factors = module(self.x)
        left = self.x[..., :4]
        left = left - left.mean(-1, keepdim=True)
        left = left / left.norm(dim=-1, keepdim=True).clamp_min(module.eps)
        right = self.x[..., 4:]
        right = right / right.abs().sum(-1, keepdim=True).clamp_min(module.eps)
        torch.testing.assert_close(factors[0], left * right)
        torch.testing.assert_close(factors[1], self.x[..., :4])
        torch.testing.assert_close(factors[2], self.x[..., 4:])
        source = self.x[..., :4]
        source = source * torch.rsqrt(
            source.square().mean(-1, keepdim=True) + module.eps
        )
        projected = torch.einsum(
            "hfi,bhti->bhtf", module.branch_weights[0], source
        )
        torch.testing.assert_close(factors[3], projected)
        self.assertEqual(module.axis_widths, (4, 4, 4, 3))
        self.assertEqual(module.branch_weights[0].shape, (2, 3, 4))

    def test_independent_branches(self):
        module = self._module(state_lift("hadamard"))
        w_a, w_b = module.branch_weights
        self.assertFalse(torch.equal(w_a, w_b))

    def test_feature_norm_l2(self):
        module = self._module(state_lift("hadamard"), feature_norm="l2")
        (features,) = module(self.x)
        norms = features.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms))


if __name__ == "__main__":
    unittest.main()
