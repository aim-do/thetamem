"""Physical state width is not always the functional rank of a lift.

This is a theory surrogate, not a model benchmark.  It compares three
bias-free quadratic maps with the same 16 x 16 physical feature array:

* same source: ``(X A) outer (X B)`` for one width-8 source X;
* split source: the outer product of two disjoint width-16 source blocks;
* independent factors: two independently sampled width-16 codebooks.

The first map is made only of homogeneous quadratics in eight variables, so
its rank is at most 8 * 9 / 2 = 36 no matter how wide A and B are.  The other
two maps can use all 256 physical coordinates.  The finite random matrix
check below makes that distinction visible without training anything.
"""

from __future__ import annotations

import argparse

import numpy as np


SEED = 20260812
RECORDS = 320
SOURCE_WIDTH = 8
FACTOR_WIDTH = 16


def outer_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Flatten one outer-product feature per record."""
    return (left[:, :, None] * right[:, None, :]).reshape(len(left), -1)


def numerical_rank(features: np.ndarray) -> int:
    """Matrix rank with NumPy's standard scale-aware SVD tolerance."""
    singular_values = np.linalg.svd(features, compute_uv=False)
    tolerance = (
        max(features.shape)
        * np.finfo(singular_values.dtype).eps
        * singular_values[0]
    )
    return int(np.count_nonzero(singular_values > tolerance))


def main(seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)

    source = rng.standard_normal((RECORDS, SOURCE_WIDTH))
    project_left = rng.standard_normal((SOURCE_WIDTH, FACTOR_WIDTH))
    project_right = rng.standard_normal((SOURCE_WIDTH, FACTOR_WIDTH))
    same_source = outer_features(
        source @ project_left,
        source @ project_right,
    )

    split_source = rng.standard_normal((RECORDS, 2 * FACTOR_WIDTH))
    split = outer_features(
        split_source[:, :FACTOR_WIDTH],
        split_source[:, FACTOR_WIDTH:],
    )

    independent = outer_features(
        rng.standard_normal((RECORDS, FACTOR_WIDTH)),
        rng.standard_normal((RECORDS, FACTOR_WIDTH)),
    )

    physical_width = FACTOR_WIDTH**2
    same_source_ceiling = SOURCE_WIDTH * (SOURCE_WIDTH + 1) // 2
    rows = (
        ("same-source linear branches", same_source, same_source_ceiling),
        ("disjoint split source", split, physical_width),
        ("independent factor codebooks", independent, physical_width),
    )

    print(
        "Theory surrogate: bias-free quadratic lifts, random Gaussian "
        f"sources, seed={seed}"
    )
    print(
        f"Every lift below has {FACTOR_WIDTH}x{FACTOR_WIDTH} = "
        f"{physical_width} physical coordinates and {RECORDS} records."
    )
    print(f"{'construction':>30} {'observed rank':>15} {'rank ceiling':>14}")
    for name, features, ceiling in rows:
        print(f"{name:>30} {numerical_rank(features):>15} {ceiling:>14}")

    print("\nProduction-scale accounting (algebraic ceilings, not benchmarks):")
    print("  same source d=32 -> branch(32) outer branch(32): 1024 cells, rank <= 528")
    print("  same source d=32 -> branch(10) outer branch(14):  140 cells, ceiling does not bind")
    print("  split source d=64 -> 32+32 outer split:          1024 cells, rank <= 1024")
    print(
        "\nReading: an outer array always allocates its physical cells, but "
        "two linear branches of the same small source need not span them. "
        "A sufficiently wide disjoint split, or genuinely independent factor "
        "codes, removes this particular quadratic-rank bottleneck."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(seed=args.seed)
