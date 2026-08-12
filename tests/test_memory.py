import unittest

import torch

import thetamem.scan as scan
from thetamem import ThetaMemory, branch, concat, hadamard, key, outer


def _qkv(batch=2, heads=4, time=17, key_dim=8, value_dim=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, time, key_dim, generator=generator)
    k = torch.randn(batch, heads, time, key_dim, generator=generator)
    v = torch.randn(batch, heads, time, value_dim, generator=generator)
    return q, k, v


class ThetaMemoryTest(unittest.TestCase):
    def test_state_kinds_shapes(self):
        q, k, v = _qkv()
        expected_shapes = {
            "hadamard": (8, 6),
            "concat": (16, 6),
            "outer": (8, 8, 6),
        }
        for state, shape in expected_shapes.items():
            memory = ThetaMemory(8, 6, 4, state=state)
            self.assertEqual(memory.state_shape, shape)
            out = memory(q, k, v)
            self.assertEqual(out.shape, v.shape)

    def test_backends_agree_per_state(self):
        q, k, v = _qkv()
        for state in ("hadamard", "concat", "outer"):
            torch.manual_seed(1)
            naive = ThetaMemory(8, 6, 4, state=state, backend="naive").double()
            torch.manual_seed(1)
            chunked = ThetaMemory(
                8, 6, 4, state=state, backend="chunked", chunk=5
            ).double()
            out_naive = naive(q.double(), k.double(), v.double())
            out_chunked = chunked(q.double(), k.double(), v.double())
            torch.testing.assert_close(out_naive, out_chunked)

    def test_updates_run_and_differ(self):
        q, k, v = _qkv()
        reads = {}
        for update in ("sum", "second_pass", "delta"):
            torch.manual_seed(2)
            memory = ThetaMemory(8, 6, 4, state="hadamard", update=update)
            reads[update] = memory(q, k, v)
            self.assertEqual(reads[update].shape, v.shape)
        self.assertFalse(torch.allclose(reads["sum"], reads["second_pass"]))
        self.assertFalse(torch.allclose(reads["sum"], reads["delta"]))

    def test_second_pass_state_size_doubles(self):
        base = ThetaMemory(8, 6, 4, state="hadamard")
        corrected = ThetaMemory(8, 6, 4, state="hadamard", update="second_pass")
        self.assertEqual(corrected.state_size, 2 * base.state_size)

    def test_custom_lift(self):
        q, k, v = _qkv()
        spec = outer(hadamard(branch(8), branch(8)), concat(branch(4), key()))
        memory = ThetaMemory(8, 6, 4, lift=spec)
        self.assertEqual(memory.state_shape, (8, 12, 6))
        out = memory(q, k, v)
        self.assertEqual(out.shape, v.shape)

    def test_second_pass_mix_at_init_is_mostly_base(self):
        q, k, v = _qkv()
        torch.manual_seed(3)
        corrected = ThetaMemory(8, 6, 4, state="hadamard", update="second_pass")
        weight = torch.sigmoid(corrected.mix_logit)
        torch.testing.assert_close(weight, torch.full_like(weight, 0.9))

    def test_delta_uses_raw_learned_strength(self):
        q, k, v = _qkv(batch=1, heads=2, time=5)
        memory = ThetaMemory(
            8, 6, 2, state="hadamard", update="delta", backend="naive"
        )
        initial = torch.sigmoid(memory.strength_logit)
        torch.testing.assert_close(
            initial,
            torch.full_like(initial, 1.0 / (memory.feature_width + 1)),
        )
        memory.strength_logit.data.zero_()
        q_factors = memory.lift(q)
        k_factors = memory.lift(k)
        expected = scan.delta_read(
            q_factors,
            k_factors,
            v,
            torch.full((1, 2, 5), 0.5),
            backend="naive",
        )
        torch.testing.assert_close(memory(q, k, v), expected)

    def test_rejects_unknown_arguments(self):
        with self.assertRaises(ValueError):
            ThetaMemory(8, 6, 4, update="erase")
        with self.assertRaises(ValueError):
            ThetaMemory(8, 6, 4, backend="triton")
        with self.assertRaises(ValueError):
            ThetaMemory(8, 6, 4, state="dense")
        with self.assertRaises(ValueError):
            ThetaMemory(8, 6, 4, value_center="divide")
        with self.assertRaises(ValueError):
            ThetaMemory(8, 6, 4, update="multi_pass", passes=0)

    def test_gradients_reach_branches(self):
        q, k, v = _qkv()
        for update in ("sum", "second_pass", "multi_pass", "delta"):
            torch.manual_seed(4)
            memory = ThetaMemory(8, 6, 4, state="hadamard", update=update)
            memory(q, k, v).square().sum().backward()
            for weight in memory.lift.branch_weights:
                self.assertIsNotNone(weight.grad)
                self.assertTrue(torch.isfinite(weight.grad).all())


class MultiPassMemoryTest(unittest.TestCase):
    def test_initialization(self):
        memory = ThetaMemory(
            8, 6, 4, state="hadamard", update="multi_pass", passes=3
        )
        strengths = torch.sigmoid(memory.pass_strength_logit)
        torch.testing.assert_close(
            strengths, torch.full_like(strengths, 0.1)
        )

    def test_one_pass_matches_second_pass_initialization(self):
        q, k, v = _qkv()
        torch.manual_seed(5)
        second = ThetaMemory(
            8, 6, 4, state="outer", update="second_pass", backend="naive"
        ).double()
        torch.manual_seed(5)
        repeated = ThetaMemory(
            8, 6, 4, state="outer", update="multi_pass", passes=1,
            backend="naive",
        ).double()
        torch.testing.assert_close(
            second(q.double(), k.double(), v.double()),
            repeated(q.double(), k.double(), v.double()),
        )

    def test_state_size_counts_passes(self):
        base = ThetaMemory(8, 6, 4, state="hadamard")
        multi = ThetaMemory(
            8, 6, 4, state="hadamard", update="multi_pass", passes=3
        )
        self.assertEqual(multi.state_size, 4 * base.state_size)

    def test_backends_agree(self):
        q, k, v = _qkv()
        torch.manual_seed(5)
        naive = ThetaMemory(
            8, 6, 4, state="outer", update="multi_pass", passes=2,
            backend="naive",
        ).double()
        torch.manual_seed(5)
        chunked = ThetaMemory(
            8, 6, 4, state="outer", update="multi_pass", passes=2,
            backend="chunked", chunk=5,
        ).double()
        torch.testing.assert_close(
            naive(q.double(), k.double(), v.double()),
            chunked(q.double(), k.double(), v.double()),
        )

    def test_pass_parameters_learn(self):
        q, k, v = _qkv()
        memory = ThetaMemory(
            8, 6, 4, state="hadamard", update="multi_pass", passes=2
        )
        memory(q, k, v).square().sum().backward()
        self.assertTrue(torch.isfinite(memory.pass_strength_logit.grad).all())


class ValueCenterTest(unittest.TestCase):
    def test_shift_equivariance(self):
        # Centered reads commute with a constant value shift; raw reads do
        # not (the shift rides the signed cross-talk).
        q, k, v = _qkv()
        shift = 3.0
        for update in ("sum", "second_pass", "multi_pass", "delta"):
            for center in ("running_mean", "exact_mean"):
                torch.manual_seed(6)
                memory = ThetaMemory(
                    8, 6, 4, state="hadamard", update=update,
                    value_center=center,
                ).double()
                base = memory(q.double(), k.double(), v.double())
                shifted = memory(q.double(), k.double(), v.double() + shift)
                torch.testing.assert_close(shifted, base + shift)
        torch.manual_seed(6)
        raw = ThetaMemory(8, 6, 4, state="hadamard").double()
        base = raw(q.double(), k.double(), v.double())
        shifted = raw(q.double(), k.double(), v.double() + shift)
        self.assertFalse(torch.allclose(shifted, base + shift))

    def test_mass_matches_reference(self):
        # value_center="exact_mean" == raw read minus mean * mass read plus mean.
        q, k, v = _qkv()
        torch.manual_seed(7)
        memory = ThetaMemory(
            8, 6, 4, state="hadamard", value_center="exact_mean", backend="naive"
        ).double()
        qd, kd, vd = q.double(), k.double(), v.double()
        result = memory(qd, kd, vd)
        q_factors = memory.lift(qd)
        k_factors = memory.lift(kd)
        raw = scan.sum_read(q_factors, k_factors, vd, backend="naive")
        ones = torch.ones_like(vd[..., :1])
        mass = scan.sum_read(q_factors, k_factors, ones, backend="naive")
        counts = torch.arange(1, vd.shape[2] + 1, dtype=vd.dtype).view(
            1, 1, -1, 1
        )
        mean = vd.cumsum(2) / counts
        torch.testing.assert_close(result, raw - mean * mass + mean)

    def test_backends_agree_with_centering(self):
        q, k, v = _qkv()
        for center in ("running_mean", "exact_mean"):
            torch.manual_seed(8)
            naive = ThetaMemory(
                8, 6, 4, state="outer", value_center=center, backend="naive"
            ).double()
            torch.manual_seed(8)
            chunked = ThetaMemory(
                8, 6, 4, state="outer", value_center=center,
                backend="chunked", chunk=5,
            ).double()
            torch.testing.assert_close(
                naive(q.double(), k.double(), v.double()),
                chunked(q.double(), k.double(), v.double()),
            )

    def test_state_size_accounting(self):
        base = ThetaMemory(8, 6, 4, state="hadamard")
        running = ThetaMemory(
            8, 6, 4, state="hadamard", value_center="running_mean"
        )
        mass = ThetaMemory(8, 6, 4, state="hadamard", value_center="exact_mean")
        self.assertEqual(running.state_size, base.state_size + 6 + 1)
        # One ones channel of width 8 plus the running value sum and count.
        self.assertEqual(mass.state_size, base.state_size + 8 + 6 + 1)


if __name__ == "__main__":
    unittest.main()
