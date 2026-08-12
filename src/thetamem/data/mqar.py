"""Multi-query associative recall (MQAR) data generation.

Reimplements the MQAR synthetic of Arora, Eyuboglu, et al., "Zoology:
Measuring and improving recall in efficient language models" (2023), with one
reproducibility fix carried over from our benchmark runs: filler tokens come
from a local generator seeded by the data seed, never from the global RNG, so
two differently sized models prepared in one process still see identical data.

Each example writes ``kv_pairs`` unique key/value pairs, then queries a subset
of the keys at gaps drawn from a power-law over the remaining positions
(``power_a = 1.0`` is uniform; the default ``0.01`` is strongly long-range).
Labels are ``-100`` everywhere except the position after each query.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["generate", "mixture", "TRAIN_MIXTURE", "EVAL_SLICES"]

#: (seq_len, num_examples, kv_pairs) — the training mixture used by the
#: recorded MQAR runs (maximum training length 256).
TRAIN_MIXTURE = (
    (64, 100_000, 4),
    (128, 20_000, 8),
    (256, 20_000, 16),
    (256, 20_000, 32),
    (256, 20_000, 64),
)

#: (seq_len, num_examples, kv_pairs) — the two evaluation slices this
#: project reports: the hardest in-distribution length and the 4x length
#: extrapolation slice.
EVAL_SLICES = (
    (256, 1_000, 64),
    (1024, 1_000, 256),
)


def generate(
    num_examples: int,
    seq_len: int,
    kv_pairs: int,
    *,
    vocab_size: int = 8_192,
    power_a: float = 0.01,
    seed: int = 0,
    random_fillers: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one MQAR segment.

    Returns ``(inputs, labels)`` of shape ``[num_examples, seq_len]``; label
    positions that are not answers hold ``-100``.
    """
    if seq_len % 2 != 0:
        raise ValueError("seq_len must be even")
    if vocab_size <= seq_len:
        raise ValueError("vocab_size must exceed seq_len")
    if kv_pairs * 4 > seq_len:
        raise ValueError("seq_len must fit the key/value pairs and queries")

    rng = np.random.default_rng(seed)
    context_size = kv_pairs * 2

    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys = np.stack(
        [rng.choice(key_choices, size=kv_pairs, replace=False) for _ in range(num_examples)]
    )
    values = np.stack(
        [rng.choice(value_choices, size=kv_pairs, replace=False) for _ in range(num_examples)]
    )

    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    space = (seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()
    gaps = np.stack(
        [rng.choice(space, size=kv_pairs, replace=False, p=p) for _ in range(num_examples)]
    )

    queries = np.zeros((num_examples, seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, gaps * 2, values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, gaps * 2 + context_size + 1, values=values, axis=1)

    inputs = torch.from_numpy(examples[:, :-1].copy())
    labels = torch.from_numpy(labels[:, 1:].copy())

    if random_fillers:
        # Local generator, never the global RNG: matched comparisons must not
        # depend on how much randomness model construction consumed.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        filler = torch.randint(
            vocab_size, size=inputs.shape, generator=generator, dtype=inputs.dtype
        )
        mask = inputs == 0
        inputs[mask] = filler[mask]
    return inputs, labels


def mixture(
    segments: tuple[tuple[int, int, int], ...] = TRAIN_MIXTURE,
    *,
    vocab_size: int = 8_192,
    seed: int = 0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Generate a list of ``(inputs, labels)`` segments.

    Segments keep their own sequence lengths; batches must not mix segments.
    """
    return [
        generate(
            num_examples,
            seq_len,
            kv_pairs,
            vocab_size=vocab_size,
            seed=seed + index,
        )
        for index, (seq_len, num_examples, kv_pairs) in enumerate(segments)
    ]
