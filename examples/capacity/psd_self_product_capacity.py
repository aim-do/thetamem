"""Grouped PSD self-product versus a signed degree-two product-code surrogate.

This is an idealized geometry check, not a library or training benchmark.
The two-factor signed lift uses independent unit-vector factors of widths
10 and 14.  The grouped positive-semidefinite (PSD) self-product comparator
starts from a unit vector split into two width-16 groups and uses

    psi(x) = svec(x_1 x_1^T + x_2 x_2^T),

whose query/key score is the sum of four squared cross-group dot products.
Both maps are degree two.  At four heads and value width 64 their complete
lifted-memory states are nearly equal after including the PSD read's mass
vector:
35,840 versus 35,360 floats.  Adding the 1,536-float frontend cache shared by
the recorded comparison shell gives full layer states of 37,376 and 36,896.

The simulation reports (1) interference power and its reciprocal pSNR,
(2) mass-normalized PSD and raw signed read MSE after one learned scalar
rescale, and (3) one, two, and three cyclic delta sweeps on unit-norm lifted
features.  One delta sweep is a causal sequence of writes; later sweeps
require replaying the records and are an offline ceiling.  Delta reads are
unnormalized and do not use the additive-read mass denominator.
The comparison matches polynomial degree and recurrent state, not raw key
width or learned projection count.

This exact grouped geometry matches the Sigma2 construction of Ghriss and
Chakraborty (2026); the script uses the broader PSD/self-product terminology
because that geometric property, not a particular architecture name, is the
comparison being isolated.
"""

from __future__ import annotations

import argparse

import numpy as np


SEED = 20260812
VALUE_WIDTH = 64
HEADS = 4
SHARED_LAYER_CACHE = 1_536
SIGNED_WIDTHS = (10, 14)
PSD_GROUP_WIDTH = 16
TRIALS_BY_RECORDS = ((32, 200), (64, 150), (128, 100), (256, 60))


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def signed_gram(rng: np.random.Generator, records: int) -> np.ndarray:
    """Gram of an independent two-factor signed outer-product code."""
    left = unit(rng.standard_normal((records, SIGNED_WIDTHS[0])))
    right = unit(rng.standard_normal((records, SIGNED_WIDTHS[1])))
    return (left @ left.T) * (right @ right.T)


def grouped_psd_gram(rng: np.random.Generator, records: int) -> np.ndarray:
    """Grouped PSD self-product Gram, up to its irrelevant global scale."""
    width = PSD_GROUP_WIDTH
    keys = unit(rng.standard_normal((records, 2 * width))).reshape(
        records, 2, width
    )
    first, second = keys[:, 0], keys[:, 1]
    first_first = first @ first.T
    first_second = first @ second.T
    second_second = second @ second.T
    return (
        first_first**2
        + first_second**2
        + first_second.T**2
        + second_second**2
    )


def interference_power(gram: np.ndarray) -> float:
    """Mean distractor power after making every matched score one."""
    normalized = gram / np.diag(gram)[:, None]
    return float(np.mean(np.sum(normalized**2, axis=1) - 1.0))


def oracle_scaled_mse(read: np.ndarray, values: np.ndarray) -> float:
    """Relative MSE after a per-trial target-dependent oracle rescale.

    The scale minimizing the error is computed from the target values of this
    trial, so it is an oracle: it removes a global gain a trained output
    projection would absorb anyway, but it is not attainable online. Report it
    as a diagnostic. The assumption-light number of this script is the
    interference power ratio, which needs no fitted quantity at all.
    """
    denominator = float(read @ read)
    scale = float(read @ values) / denominator if denominator > 0.0 else 0.0
    return float(np.mean((scale * read - values) ** 2) / np.mean(values**2))


def delta_sweeps(
    gram: np.ndarray,
    values: np.ndarray,
    sweeps: int = 3,
) -> list[float]:
    """Cyclic Widrow-Hoff/Kaczmarz writes, evaluated after each replay."""
    norms = np.sqrt(np.diag(gram))
    unit_gram = gram / (norms[:, None] * norms[None, :])
    prediction = np.zeros_like(values)
    errors: list[float] = []
    for _ in range(sweeps):
        for index in range(len(values)):
            residual = values[index] - prediction[index]
            prediction += residual * unit_gram[:, index]
        errors.append(
            float(np.mean((prediction - values) ** 2) / np.mean(values**2))
        )
    return errors


def main(seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)
    signed_features = SIGNED_WIDTHS[0] * SIGNED_WIDTHS[1]
    psd_features = PSD_GROUP_WIDTH * (PSD_GROUP_WIDTH + 1) // 2
    signed_state_per_head = signed_features * VALUE_WIDTH
    psd_state_per_head = psd_features * (VALUE_WIDTH + 1)
    signed_memory_state = HEADS * signed_state_per_head
    psd_memory_state = HEADS * psd_state_per_head
    signed_layer_state = signed_memory_state + SHARED_LAYER_CACHE
    psd_layer_state = psd_memory_state + SHARED_LAYER_CACHE

    print("Theory surrogate: random unit keys and iid N(0,1) scalar values")
    print("Matched: degree and recurrent state. Not matched: raw key/projection width.")
    print(
        f"signed: {SIGNED_WIDTHS[0]}x{SIGNED_WIDTHS[1]}={signed_features} "
        f"features, memory state={signed_state_per_head} floats/head"
    )
    print(
        f"grouped PSD self-product: groups {PSD_GROUP_WIDTH}+{PSD_GROUP_WIDTH}, "
        f"packed features={psd_features}, numerator+mass state="
        f"{psd_state_per_head} floats/head"
    )
    print(
        f"four-head memory state: signed={signed_memory_state}, "
        f"PSD={psd_memory_state}"
    )
    print(
        f"full layer state with shared {SHARED_LAYER_CACHE}-float cache: "
        f"signed={signed_layer_state}, PSD={psd_layer_state}"
    )
    print(
        f"full-layer ratio signed/PSD = "
        f"{signed_layer_state / psd_layer_state:.4f}\n"
    )

    aggregate: dict[int, np.ndarray] = {}
    for records, trials in TRIALS_BY_RECORDS:
        measurements = []
        for _ in range(trials):
            values = rng.standard_normal(records)
            signed = signed_gram(rng, records)
            psd = grouped_psd_gram(rng, records)
            signed_delta = delta_sweeps(signed, values)
            psd_delta = delta_sweeps(psd, values)
            measurements.append(
                (
                    interference_power(signed),
                    interference_power(psd),
                    oracle_scaled_mse(signed @ values, values),
                    oracle_scaled_mse(
                        (psd @ values) / psd.sum(axis=1), values
                    ),
                    signed_delta[0],
                    psd_delta[0],
                    signed_delta[1],
                    psd_delta[1],
                    signed_delta[2],
                    psd_delta[2],
                )
            )
        aggregate[records] = np.mean(measurements, axis=0)

    print("Interference power (lower is better); pSNR = 1 / power:")
    print(
        f"{'records':>7} {'P signed':>10} {'P PSD':>10} {'PSD/signed':>12} "
        f"{'pSNR signed':>12} {'pSNR PSD':>10}"
    )
    for records, _ in TRIALS_BY_RECORDS:
        signed_power, psd_power = aggregate[records][:2]
        print(
            f"{records:>7} {signed_power:>10.4f} {psd_power:>10.4f} "
            f"{psd_power / signed_power:>12.2f} "
            f"{1.0 / signed_power:>12.3f} {1.0 / psd_power:>10.3f}"
        )

    print("\nRelative MSE (lower is better):")
    print(
        f"{'records':>7} {'add s':>8} {'add PSD':>8} {'ratio':>6} | "
        f"{'d1 s':>8} {'d1 PSD':>8} {'ratio':>6} | "
        f"{'d2 s':>8} {'d2 PSD':>8} {'ratio':>6} | "
        f"{'d3 s':>8} {'d3 PSD':>8} {'ratio':>6}"
    )
    for records, _ in TRIALS_BY_RECORDS:
        row = aggregate[records]
        print(
            f"{records:>7} {row[2]:>8.4f} {row[3]:>8.4f} "
            f"{row[3] / row[2]:>6.2f} | "
            f"{row[4]:>8.4f} {row[5]:>8.4f} {row[5] / row[4]:>6.2f} | "
            f"{row[6]:>8.4f} {row[7]:>8.4f} {row[7] / row[6]:>6.2f} | "
            f"{row[8]:>8.4f} {row[9]:>8.4f} {row[9] / row[8]:>6.2f}"
        )

    print(
        "\nReading: under these ideal independent-code assumptions, the "
        "nonnegative PSD/self-product Gram has about 2.6x the random "
        "interference power "
        "at nearly equal complete state. The MSE ratio is smaller at high "
        "occupancy because both reads approach their error ceiling. Delta "
        "reduces interference for both; sweeps two and three require offline "
        "replay and are not a one-pass streaming result."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(seed=args.seed)
