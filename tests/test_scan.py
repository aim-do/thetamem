import unittest
from unittest import mock

import torch

import thetamem.scan as scan


def _random_factors(widths, batch=2, heads=2, time=33, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(batch, heads, time, width, generator=generator, dtype=torch.float64)
        for width in widths
    )


def _random_values(batch=2, heads=2, time=33, dim=6, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, heads, time, dim, generator=generator, dtype=torch.float64)


def _loop_sum(q_factors, k_factors, values):
    """Independent sequential reference: explicit state, one token at a time."""
    query = scan.flatten_factors(q_factors)
    keys = scan.flatten_factors(k_factors)
    batch, heads, time, width = keys.shape
    state = torch.zeros(batch, heads, width, values.shape[-1], dtype=values.dtype)
    outputs = []
    for t in range(time):
        state = state + torch.einsum("bhw,bhv->bhwv", keys[:, :, t], values[:, :, t])
        outputs.append(torch.einsum("bhw,bhwv->bhv", query[:, :, t], state))
    return torch.stack(outputs, dim=2)


def _loop_second_pass(q_factors, k_factors, values):
    query = scan.flatten_factors(q_factors)
    keys = scan.flatten_factors(k_factors)
    batch, heads, time, width = keys.shape
    value_dim = values.shape[-1]
    base = torch.zeros(batch, heads, width, value_dim, dtype=values.dtype)
    correction = torch.zeros_like(base)
    base_reads, correction_reads = [], []
    for t in range(time):
        key_t = keys[:, :, t]
        prediction = torch.einsum("bhw,bhwv->bhv", key_t, base)
        residual = values[:, :, t] - prediction
        base = base + torch.einsum("bhw,bhv->bhwv", key_t, values[:, :, t])
        correction = correction + torch.einsum("bhw,bhv->bhwv", key_t, residual)
        base_reads.append(torch.einsum("bhw,bhwv->bhv", query[:, :, t], base))
        correction_reads.append(
            torch.einsum("bhw,bhwv->bhv", query[:, :, t], correction)
        )
    return torch.stack(base_reads, dim=2), torch.stack(correction_reads, dim=2)


def _loop_multi_pass(q_factors, k_factors, values, strengths):
    """Direct triangular-matrix reference for causal Richardson."""
    query = scan.flatten_factors(q_factors)
    keys = scan.flatten_factors(k_factors)
    lower = torch.matmul(keys, keys.transpose(-1, -2)).tril(-1)
    iterate = values
    for index in range(strengths.shape[0]):
        strength = strengths[index].view(1, -1, 1, 1)
        iterate = iterate + strength * (
            values - iterate - torch.matmul(lower, iterate)
        )
    read_score = torch.matmul(query, keys.transpose(-1, -2)).tril()
    return torch.matmul(read_score, iterate)


class SumScanTest(unittest.TestCase):
    def test_chunked_matches_loop_flat(self):
        qf, kf = _random_factors([8], seed=0), _random_factors([8], seed=2)
        values = _random_values()
        expected = _loop_sum(qf, kf, values)
        for chunk in (4, 8, 33, 64):
            result = scan.sum_read(qf, kf, values, backend="chunked", chunk=chunk)
            torch.testing.assert_close(result, expected)
        naive = scan.sum_read(qf, kf, values, backend="naive")
        torch.testing.assert_close(naive, expected)

    def test_chunked_matches_loop_factored(self):
        qf = _random_factors([4, 5], seed=3)
        kf = _random_factors([4, 5], seed=4)
        values = _random_values()
        expected = _loop_sum(qf, kf, values)
        for chunk in (4, 16, 33, 128):
            result = scan.sum_read(qf, kf, values, backend="chunked", chunk=chunk)
            torch.testing.assert_close(result, expected)
        torch.testing.assert_close(
            scan.sum_read(qf, kf, values, backend="naive"), expected
        )

    def test_four_factor_outer(self):
        qf = _random_factors([3, 4, 2, 2], seed=5)
        kf = _random_factors([3, 4, 2, 2], seed=6)
        values = _random_values()
        expected = _loop_sum(qf, kf, values)
        result = scan.sum_read(qf, kf, values, backend="chunked", chunk=8)
        torch.testing.assert_close(result, expected)

    def test_low_precision_carry_uses_fp32_matmuls(self):
        generator = torch.Generator().manual_seed(0)
        past_keys = torch.randn(
            1, 1, 2, 32, generator=generator
        ).bfloat16()
        past_values = torch.randn(
            1, 1, 2, 3, generator=generator
        ).bfloat16()
        final_query = torch.randn(
            1, 1, 1, 32, generator=generator
        ).bfloat16()
        keys = torch.cat((past_keys, torch.zeros_like(past_keys[:, :, :1])), 2)
        values = torch.cat(
            (past_values, torch.zeros_like(past_values[:, :, :1])), 2
        )
        query = torch.cat((torch.zeros_like(past_keys), final_query), 2)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            state = scan._state_update(past_keys, past_values, torch.float32)
            carry = scan._state_read(final_query, state, torch.float32)
            result = scan.sum_read(
                (query,), (keys,), values, backend="chunked", chunk=2
            )
            factor_a = torch.randn(
                1, 1, 2, 4, generator=generator
            ).bfloat16()
            factor_b = torch.randn(
                1, 1, 2, 8, generator=generator
            ).bfloat16()
            promoted_flat = scan._flatten_state_factors(
                (factor_a, factor_b), torch.float32
            )
        expected_state = torch.matmul(
            past_keys.float().transpose(-1, -2), past_values.float()
        )
        expected_carry = torch.matmul(final_query.float(), expected_state)
        expected = expected_carry.bfloat16()
        self.assertEqual(state.dtype, torch.float32)
        torch.testing.assert_close(state, expected_state)
        torch.testing.assert_close(carry, expected_carry)
        torch.testing.assert_close(result[:, :, 2:3], expected)
        torch.testing.assert_close(
            promoted_flat,
            scan.flatten_factors((factor_a.float(), factor_b.float())),
        )

    def test_factored_equals_flattened(self):
        qf = _random_factors([4, 5], seed=7)
        kf = _random_factors([4, 5], seed=8)
        values = _random_values()
        factored = scan.sum_read(qf, kf, values, backend="chunked", chunk=8)
        flat = scan.sum_read(
            (scan.flatten_factors(qf),),
            (scan.flatten_factors(kf),),
            values,
            backend="chunked",
            chunk=8,
        )
        torch.testing.assert_close(factored, flat)

    def test_gradients_match_between_backends(self):
        qf = _random_factors([6], seed=9)
        kf = _random_factors([6], seed=10)
        values = _random_values(seed=11)
        grads = {}
        for backend in ("naive", "chunked"):
            q = tuple(f.clone().requires_grad_(True) for f in qf)
            k = tuple(f.clone().requires_grad_(True) for f in kf)
            v = values.clone().requires_grad_(True)
            out = scan.sum_read(q, k, v, backend=backend, chunk=8)
            out.square().sum().backward()
            grads[backend] = (q[0].grad, k[0].grad, v.grad)
        for a, b in zip(grads["naive"], grads["chunked"]):
            torch.testing.assert_close(a, b)


class SecondPassTest(unittest.TestCase):
    def test_matches_loop_flat(self):
        qf, kf = _random_factors([8], seed=12), _random_factors([8], seed=13)
        values = _random_values(seed=14)
        base_ref, corr_ref = _loop_second_pass(qf, kf, values)
        for backend, chunk in (("naive", 0), ("chunked", 8), ("chunked", 64)):
            kwargs = {"backend": backend}
            if chunk:
                kwargs["chunk"] = chunk
            base, corr = scan.second_pass_read(qf, kf, values, **kwargs)
            torch.testing.assert_close(base, base_ref)
            torch.testing.assert_close(corr, corr_ref)

    def test_matches_loop_factored(self):
        qf = _random_factors([4, 3], seed=15)
        kf = _random_factors([4, 3], seed=16)
        values = _random_values(seed=17)
        base_ref, corr_ref = _loop_second_pass(qf, kf, values)
        base, corr = scan.second_pass_read(
            qf, kf, values, backend="chunked", chunk=8
        )
        torch.testing.assert_close(base, base_ref)
        torch.testing.assert_close(corr, corr_ref)

    def test_residual_excludes_own_write(self):
        # A single repeated key: the first residual must equal the value
        # itself (empty prefix), the second must be the value difference.
        key = torch.ones(1, 1, 2, 3, dtype=torch.float64)
        values = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]], dtype=torch.float64)
        base, corr = scan.second_pass_read((key,), (key,), values, backend="naive")
        # correction state after t=0 holds v_0; read at t=0 gives |k|^2 * v_0
        torch.testing.assert_close(corr[:, :, 0], values[:, :, 0] * 3.0)


class MultiPassTest(unittest.TestCase):
    def _strengths(self, passes, heads=2, seed=7):
        generator = torch.Generator().manual_seed(seed)
        return torch.rand(
            passes, heads, generator=generator, dtype=torch.float64
        )

    def test_naive_matches_loop_flat(self):
        qf, kf = _random_factors([8], seed=10), _random_factors([8], seed=11)
        values = _random_values(seed=12)
        strengths = self._strengths(3)
        expected = _loop_multi_pass(qf, kf, values, strengths)
        result = scan.multi_pass_read(qf, kf, values, strengths, backend="naive")
        torch.testing.assert_close(result, expected)

    def test_chunked_matches_loop_factored(self):
        qf = _random_factors([4, 5], seed=13)
        kf = _random_factors([4, 5], seed=14)
        values = _random_values(seed=15)
        strengths = self._strengths(2)
        expected = _loop_multi_pass(qf, kf, values, strengths)
        for chunk in (4, 16, 33, 128):
            result = scan.multi_pass_read(
                qf, kf, values, strengths, backend="chunked", chunk=chunk
            )
            torch.testing.assert_close(result, expected)

    def test_chunked_matches_loop_flat(self):
        qf, kf = _random_factors([8], seed=16), _random_factors([8], seed=17)
        values = _random_values(seed=18)
        strengths = self._strengths(2)
        expected = _loop_multi_pass(qf, kf, values, strengths)
        for chunk in (4, 8, 33, 64):
            result = scan.multi_pass_read(
                qf, kf, values, strengths, backend="chunked", chunk=chunk
            )
            torch.testing.assert_close(result, expected)

    def test_one_unit_pass_is_second_pass(self):
        qf = _random_factors([8], seed=19)
        kf = _random_factors([8], seed=20)
        values = _random_values(seed=21)
        strengths = torch.ones(1, 2, dtype=torch.float64)
        result = scan.multi_pass_read(
            qf, kf, values, strengths, backend="chunked", chunk=8
        )
        _, correction = scan.second_pass_read(
            qf, kf, values, backend="chunked", chunk=8
        )
        torch.testing.assert_close(result, correction)

    def test_one_token_identity_does_not_overcount(self):
        key = torch.ones(1, 1, 1, 1, dtype=torch.float64)
        values = torch.tensor([[[[3.0, -2.0]]]], dtype=torch.float64)
        strengths = torch.tensor([[0.25], [0.5], [0.75]], dtype=torch.float64)
        for backend in ("naive", "chunked"):
            result = scan.multi_pass_read(
                (key,), (key,), values, strengths, backend=backend, chunk=1
            )
            torch.testing.assert_close(result, values)


class ReplaySolverTest(unittest.TestCase):
    def test_cg_matches_lstsq(self):
        generator = torch.Generator().manual_seed(30)
        keys = torch.randn(1, 1, 9, 4, generator=generator, dtype=torch.float64)
        values = torch.randn(
            1, 1, 9, 2, generator=generator, dtype=torch.float64
        )
        state = scan.replay_fit(
            (keys,), values, solver="cg", iterations=4, tolerance=0.0
        )
        result = scan.replay_read((keys,), state)
        solution = torch.linalg.lstsq(keys[0, 0], values[0, 0]).solution
        expected = torch.matmul(keys[0, 0], solution)[None, None]
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_richardson_matches_lstsq_for_orthonormal_design(self):
        generator = torch.Generator().manual_seed(31)
        raw = torch.randn(7, 3, generator=generator, dtype=torch.float64)
        keys = torch.linalg.qr(raw, mode="reduced").Q[None, None]
        values = torch.randn(
            1, 1, 7, 2, generator=generator, dtype=torch.float64
        )
        state = scan.replay_fit(
            (keys,), values, solver="richardson", iterations=1, strength=1.0
        )
        result = scan.replay_read((keys,), state)
        solution = torch.linalg.lstsq(keys[0, 0], values[0, 0]).solution
        expected = torch.matmul(keys[0, 0], solution)[None, None]
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_outer_solvers_never_flatten_factors(self):
        factor_a = torch.tensor(
            [[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3,
            dtype=torch.float64,
        )[None, None]
        factor_b = torch.eye(3, dtype=torch.float64).repeat(2, 1)[None, None]
        values = torch.arange(12, dtype=torch.float64).reshape(1, 1, 6, 2)
        expected_shape = (1, 1, 2, 3, 2)
        with mock.patch.object(
            scan, "flatten_factors", side_effect=AssertionError("flattened")
        ):
            richardson = scan.replay_fit(
                (factor_a, factor_b),
                values,
                solver="richardson",
                iterations=2,
                strength=1.0,
            )
            cg = scan.replay_fit(
                (factor_a, factor_b), values, solver="cg", iterations=2
            )
            heavy_ball = scan.replay_fit(
                (factor_a, factor_b),
                values,
                solver="heavy_ball",
                iterations=2,
                strength=1.0,
                momentum=0.0,
            )
            for state in (richardson, cg, heavy_ball):
                self.assertEqual(state.shape, expected_shape)
                torch.testing.assert_close(
                    scan.replay_read((factor_a, factor_b), state), values
                )

    def test_second_delta_sweep_reuses_and_improves_state(self):
        keys = torch.tensor(
            [[1.0, 0.0], [2**-0.5, 2**-0.5], [0.0, 1.0],
             [-2**-0.5, 2**-0.5]],
            dtype=torch.float64,
        )[None, None]
        target_state = torch.tensor([[2.0], [-1.0]], dtype=torch.float64)
        values = torch.matmul(keys[0, 0], target_state)[None, None]
        errors = []
        for sweeps in (1, 2):
            state = scan.replay_fit(
                (keys,), values, solver="delta", iterations=sweeps
            )
            prediction = scan.replay_read((keys,), state)
            errors.append((prediction - values).square().mean())
        self.assertLess(errors[1], errors[0])


class DeltaTest(unittest.TestCase):
    def test_chunked_matches_naive(self):
        qf, kf = _random_factors([8], seed=18), _random_factors([8], seed=19)
        values = _random_values(seed=20)
        beta = torch.full((2, 2, 33), 0.5, dtype=torch.float64)
        expected = scan.delta_read(qf, kf, values, beta, backend="naive")
        for chunk in (4, 16, 33, 128):
            result = scan.delta_read(
                qf, kf, values, beta, backend="chunked", chunk=chunk
            )
            torch.testing.assert_close(result, expected)

    def test_repeated_key_overwrites(self):
        # With beta = 1 / ||phi||^2, writing the same key twice must return
        # the latest value, not the sum of both.
        key = torch.full((1, 1, 2, 4), 0.5, dtype=torch.float64)
        norm_sq = key[:, :, 0].square().sum(-1)
        beta = (1.0 / norm_sq).expand(1, 1, 2)
        values = torch.tensor([[[[1.0, 5.0], [3.0, -2.0]]]], dtype=torch.float64)
        result = scan.delta_read((key,), (key,), values, beta, backend="naive")
        torch.testing.assert_close(result[:, :, 1], values[:, :, 1])

    def test_gradients_flow(self):
        qf = tuple(f.requires_grad_(True) for f in _random_factors([6], seed=21))
        kf = tuple(f.requires_grad_(True) for f in _random_factors([6], seed=22))
        values = _random_values(seed=23).requires_grad_(True)
        beta = torch.full((2, 2, 33), 0.3, dtype=torch.float64, requires_grad=True)
        out = scan.delta_read(qf, kf, values, beta, backend="chunked", chunk=8)
        out.sum().backward()
        self.assertTrue(torch.isfinite(qf[0].grad).all())
        self.assertTrue(torch.isfinite(beta.grad).all())


class FlaBackendTest(unittest.TestCase):
    def test_fla_matches_naive_or_is_unavailable(self):
        try:
            import fla  # noqa: F401
        except ImportError:
            self.skipTest("flash-linear-attention not installed")
        if not torch.cuda.is_available():
            self.skipTest("fla kernels require a GPU")
        qf, kf = _random_factors([8], seed=24), _random_factors([8], seed=25)
        values = _random_values(seed=26)
        expected = scan.sum_read(qf, kf, values, backend="naive")
        result = scan.sum_read(
            tuple(f.cuda().float() for f in qf),
            tuple(f.cuda().float() for f in kf),
            values.cuda().float(),
            backend="fla",
        )
        torch.testing.assert_close(
            result.cpu().double(), expected, rtol=1e-3, atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
