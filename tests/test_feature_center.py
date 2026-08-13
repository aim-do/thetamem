import unittest
from unittest import mock

import torch
import torch.nn.functional as F

import thetamem.scan as scan
from thetamem import ThetaMemLayer, ThetaMemory, state_lift
from thetamem.lift import LiftModule


def _outer_modules(*, key_dim=6, heads=2, width=3):
    """Matching raw and final-feature-centered outer lifts."""
    spec = state_lift("outer", feature_width=width)
    raw = LiftModule(
        spec, heads=heads, key_dim=key_dim, feature_norm="none"
    ).double()
    centered = LiftModule(
        spec, heads=heads, key_dim=key_dim, feature_norm="center"
    ).double()
    centered.load_state_dict(raw.state_dict())
    return raw, centered


def _outer_problem(*, time=13):
    generator = torch.Generator().manual_seed(41)
    query = torch.randn(
        2, 2, time, 6, generator=generator, dtype=torch.float64
    )
    keys = torch.randn(
        2, 2, time, 6, generator=generator, dtype=torch.float64
    )
    values = torch.randn(
        2, 2, time, 4, generator=generator, dtype=torch.float64
    )
    _, lift = _outer_modules()
    q_factors = lift(query)
    k_factors = lift(keys)
    return q_factors, k_factors, values


class FinalFeatureCenterTest(unittest.TestCase):
    def test_flat_center_is_applied_after_the_complete_lift(self):
        torch.manual_seed(40)
        x = torch.randn(2, 2, 7, 6, dtype=torch.float64)
        spec = state_lift("hadamard", feature_width=5)
        raw = LiftModule(
            spec, heads=2, key_dim=6, feature_norm="none"
        ).double()
        centered = LiftModule(
            spec, heads=2, key_dim=6, feature_norm="center"
        ).double()
        centered.load_state_dict(raw.state_dict())

        (raw_feature,) = raw(x)
        (actual,) = centered(x)
        expected = raw_feature - raw_feature.mean(-1, keepdim=True)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(-1), torch.zeros_like(actual[..., 0]))

    def test_outer_center_is_global_lazy_and_preserves_factor_axes(self):
        generator = torch.Generator().manual_seed(42)
        x = torch.randn(
            2, 2, 7, 6, generator=generator, dtype=torch.float64
        )
        raw, centered = _outer_modules(width=3)
        raw_factors = raw(x)
        centered_factors = centered(x)

        raw_flat = scan.flatten_factors(raw_factors)
        actual = scan.flatten_factors(centered_factors)
        expected = raw_flat - raw_flat.mean(-1, keepdim=True)
        separately_centered = scan.flatten_factors(
            tuple(factor - factor.mean(-1, keepdim=True) for factor in raw_factors)
        )

        self.assertTrue(getattr(centered_factors, "centered", False))
        self.assertEqual(len(centered_factors), 2)
        self.assertEqual(centered.axis_widths, (3, 3))
        self.assertEqual(centered.state_shape(4), (3, 3, 4))
        self.assertEqual(centered.feature_width, 9)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(-1), torch.zeros_like(actual[..., 0]))
        self.assertFalse(torch.allclose(actual, separately_centered))

    def test_outer_memory_center_keeps_state_shape_and_size(self):
        raw = ThetaMemory(
            6, 4, 2, state="outer", feature_width=3, feature_norm="none"
        )
        centered = ThetaMemory(
            6, 4, 2, state="outer", feature_width=3, feature_norm="center"
        )
        self.assertEqual(centered.state_shape, (3, 3, 4))
        self.assertEqual(centered.state_shape, raw.state_shape)
        self.assertEqual(centered.state_size, raw.state_size)


class CenteredOuterScanTest(unittest.TestCase):
    def test_sum_second_and_multi_pass_match_explicit_centered_flat(self):
        q_factors, k_factors, values = _outer_problem()
        flat_q = (scan.flatten_factors(q_factors),)
        flat_k = (scan.flatten_factors(k_factors),)

        sum_expected = scan.sum_read(flat_q, flat_k, values, backend="naive")
        second_expected = scan.second_pass_read(
            flat_q, flat_k, values, backend="naive"
        )
        strengths = torch.tensor(
            [[0.15, 0.20], [0.10, 0.25]], dtype=torch.float64
        )
        multi_expected = scan.multi_pass_read(
            flat_q, flat_k, values, strengths, backend="naive"
        )

        for backend in ("naive", "chunked"):
            kwargs = {"backend": backend, "chunk": 5}
            torch.testing.assert_close(
                scan.sum_read(q_factors, k_factors, values, **kwargs),
                sum_expected,
            )
            second_actual = scan.second_pass_read(
                q_factors, k_factors, values, **kwargs
            )
            for actual, expected in zip(second_actual, second_expected):
                torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(
                scan.multi_pass_read(
                    q_factors, k_factors, values, strengths, **kwargs
                ),
                multi_expected,
            )

    def test_factorized_chunk_paths_do_not_materialize_centered_feature(self):
        q_factors, k_factors, values = _outer_problem()
        strengths = torch.full((2, 2), 0.1, dtype=torch.float64)

        with mock.patch.object(
            scan, "flatten_factors", side_effect=AssertionError("flattened")
        ):
            scan.sum_read(
                q_factors, k_factors, values, backend="chunked", chunk=5
            )
            scan.second_pass_read(
                q_factors, k_factors, values, backend="chunked", chunk=5
            )
            scan.multi_pass_read(
                q_factors,
                k_factors,
                values,
                strengths,
                backend="chunked",
                chunk=5,
            )

    def test_delta_materialization_matches_explicit_centered_flat(self):
        q_factors, k_factors, values = _outer_problem(time=11)
        flat_q = (scan.flatten_factors(q_factors),)
        flat_k = (scan.flatten_factors(k_factors),)
        beta = torch.full((2, 2, 11), 0.2, dtype=torch.float64)
        expected = scan.delta_read(
            flat_q, flat_k, values, beta, backend="naive"
        )
        for backend in ("naive", "chunked"):
            torch.testing.assert_close(
                scan.delta_read(
                    q_factors,
                    k_factors,
                    values,
                    beta,
                    backend=backend,
                    chunk=4,
                ),
                expected,
            )

    def test_replay_state_and_read_match_explicit_centered_flat(self):
        _, k_factors, values = _outer_problem(time=8)
        flat = scan.flatten_factors(k_factors)
        explicit_factors = (flat,)

        factored_state = scan.replay_fit(
            k_factors,
            values,
            solver="richardson",
            iterations=3,
            strength=0.05,
        )
        explicit_state = scan.replay_fit(
            explicit_factors,
            values,
            solver="richardson",
            iterations=3,
            strength=0.05,
        )

        self.assertEqual(factored_state.shape, (2, 2, 3, 3, 4))
        torch.testing.assert_close(
            factored_state.reshape_as(explicit_state), explicit_state
        )
        torch.testing.assert_close(
            scan.replay_read(k_factors, factored_state),
            scan.replay_read(explicit_factors, explicit_state),
        )
        torch.testing.assert_close(
            factored_state.sum(dim=(2, 3)),
            torch.zeros_like(factored_state[:, :, 0, 0]),
        )


class ValueCoordinateCenterTest(unittest.TestCase):
    def test_value_ops_center_is_ordered_coordinate_centering(self):
        layer = ThetaMemLayer(
            16,
            heads=2,
            key_dim=4,
            value_dim=5,
            qk_ops=(),
            value_ops=("silu", "center"),
            position="none",
        ).double()
        generator = torch.Generator().manual_seed(43)
        values = torch.randn(
            2, 2, 7, 5, generator=generator, dtype=torch.float64
        )
        actual = layer._apply_value_ops(values)
        transformed = F.silu(values)
        expected = transformed - transformed.mean(-1, keepdim=True)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(-1), torch.zeros_like(actual[..., 0]))

    def test_coordinate_center_is_distinct_from_temporal_value_center(self):
        coordinate = ThetaMemLayer(
            16,
            heads=2,
            key_dim=4,
            value_dim=5,
            qk_ops=(),
            value_ops=("center",),
            value_center="none",
            position="none",
        )
        temporal = ThetaMemLayer(
            16,
            heads=2,
            key_dim=4,
            value_dim=5,
            qk_ops=(),
            value_ops=(),
            value_center="exact_mean",
            position="none",
        )
        values = torch.arange(1, 71, dtype=torch.float32).reshape(1, 2, 7, 5)

        coordinate_values = coordinate._apply_value_ops(values)
        temporal_frontend_values = temporal._apply_value_ops(values)

        torch.testing.assert_close(
            coordinate_values.sum(-1),
            torch.zeros_like(coordinate_values[..., 0]),
        )
        torch.testing.assert_close(temporal_frontend_values, values)
        self.assertEqual(coordinate.memory.value_center, "none")
        self.assertEqual(temporal.memory.value_center, "exact_mean")


if __name__ == "__main__":
    unittest.main()
