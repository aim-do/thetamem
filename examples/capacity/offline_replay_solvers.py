"""Offline replay solvers for a fixed lifted-key least-squares problem.

This script deliberately does *not* model the library's causal multi-state
update.  Every iteration rereads the complete feature matrix, computes the
residual for every stored record, and writes all residuals back.  It therefore
requires retaining or reconstructing the records and is an offline/prefill
fit, not fixed-state streaming inference.

The comparison uses full-residual Richardson iteration with unit and damped
step sizes, heavy-ball momentum, and conjugate gradient on the normal
equations. The unit step eventually diverges when the largest lifted Gram
eigenvalue exceeds two. For the random normalized design, heavy-ball uses the
Marchenko-Pastur schedule eta=1 and beta=T/state; this is a model-based choice,
not a universal safe default. CG chooses its scalars from global dot products.
"""

from __future__ import annotations

import argparse

import numpy as np


SEED = 20260812
FACTOR_WIDTH = 32
PASSES = 20
TRIALS = 5
MARKS = (1, 2, 3, 5, 10, 20)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def lifted_keys(
    rng: np.random.Generator,
    records: int,
) -> np.ndarray:
    left = unit(rng.standard_normal((records, FACTOR_WIDTH)))
    right = unit(rng.standard_normal((records, FACTOR_WIDTH)))
    return (left[:, :, None] * right[:, None, :]).reshape(records, -1)


def relative_error(read: np.ndarray, values: np.ndarray) -> float:
    return float(np.mean((read - values) ** 2) / np.var(values))


def richardson_replay(
    features: np.ndarray,
    values: np.ndarray,
    passes: int,
    step: float,
) -> list[float]:
    state = np.zeros(features.shape[1])
    errors = []
    for _ in range(passes):
        residual = values - features @ state
        state += step * (features.T @ residual)
        errors.append(relative_error(features @ state, values))
    return errors


def heavy_ball_replay(
    features: np.ndarray,
    values: np.ndarray,
    passes: int,
    step: float,
    momentum: float,
) -> list[float]:
    state = np.zeros(features.shape[1])
    velocity = np.zeros_like(state)
    errors = []
    for _ in range(passes):
        residual = values - features @ state
        velocity = momentum * velocity + step * (features.T @ residual)
        state += velocity
        errors.append(relative_error(features @ state, values))
    return errors


def conjugate_gradient_replay(
    features: np.ndarray,
    values: np.ndarray,
    passes: int,
) -> list[float]:
    """CG on Phi.T Phi s = Phi.T v using replayed Phi/Phi.T products."""
    state = np.zeros(features.shape[1])
    residual = features.T @ values
    direction = residual.copy()
    residual_power = float(residual @ residual)
    errors = []
    for _ in range(passes):
        mapped = features.T @ (features @ direction)
        denominator = float(direction @ mapped)
        if denominator <= np.finfo(features.dtype).eps:
            last_error = (
                errors[-1]
                if errors
                else relative_error(features @ state, values)
            )
            errors.extend([last_error] * (passes - len(errors)))
            break
        step = residual_power / denominator
        state += step * direction
        residual -= step * mapped
        next_power = float(residual @ residual)
        errors.append(relative_error(features @ state, values))
        if next_power <= np.finfo(features.dtype).eps:
            errors.extend([errors[-1]] * (passes - len(errors)))
            break
        direction = residual + (next_power / residual_power) * direction
        residual_power = next_power
    return errors


def main(seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)
    state_width = FACTOR_WIDTH**2
    print(
        "Offline replay surrogate: every pass rereads all records; "
        "this is not ThetaMemory's causal multi-state update."
    )
    for load in (0.25, 0.50):
        records = int(load * state_width)
        runs = {
            name: np.zeros(PASSES)
            for name in ("unit", "damped", "heavy_ball", "cg")
        }
        largest_eigenvalue = None
        for trial in range(TRIALS):
            values = rng.standard_normal(records)
            features = lifted_keys(rng, records)
            if trial == 0:
                largest_eigenvalue = float(
                    np.linalg.eigvalsh(features @ features.T)[-1]
                )
            runs["unit"] += np.asarray(
                richardson_replay(features, values, PASSES, step=1.0)
            ) / TRIALS
            runs["damped"] += np.asarray(
                richardson_replay(features, values, PASSES, step=0.5)
            ) / TRIALS
            runs["heavy_ball"] += np.asarray(
                heavy_ball_replay(
                    features,
                    values,
                    PASSES,
                    step=1.0,
                    momentum=load,
                )
            ) / TRIALS
            runs["cg"] += np.asarray(
                conjugate_gradient_replay(features, values, PASSES)
            ) / TRIALS

        print(
            f"\nload T/state={load:.2f}: {records} records, "
            f"{state_width} state coordinates, first-draw lambda_max="
            f"{largest_eigenvalue:.3f}"
        )
        print(
            f"{'pass':>5} {'Richardson 1.0':>16} "
            f"{'Richardson 0.5':>16} {'heavy-ball':>14} {'CG':>12}"
        )
        for mark in MARKS:
            print(
                f"{mark:>5} {runs['unit'][mark - 1]:>16.4g} "
                f"{runs['damped'][mark - 1]:>16.4g} "
                f"{runs['heavy_ball'][mark - 1]:>14.4g} "
                f"{runs['cg'][mark - 1]:>12.4g}"
            )

    print(
        "\nReading: unit Richardson is unstable once lambda_max > 2. "
        "Damped Richardson, tuned heavy-ball, and CG solve the offline "
        "replay problem, but "
        "these curves make no convergence claim about the library's causal "
        "multi-state update."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(seed=args.seed)
