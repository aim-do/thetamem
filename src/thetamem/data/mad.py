"""MAD synthetic tasks: fuzzy in-context recall and selective copying.

Reimplemented from the task definitions of Poli, Thomas, Massaroli, et al.,
"Mechanistic Design and Scaling of Hybrid Architectures" (2024; the MAD
benchmark, MIT-licensed). Two departures from the reference code, both about
reproducibility rather than the task: every random draw goes through one
seeded ``numpy`` generator (the reference draws blank positions in selective
copying from the global RNG), and batch generation is vectorized where the
task allows it. Task semantics, vocabularies, and label conventions follow
the reference.

Both tasks return ``(inputs, targets)`` of shape ``[num_examples, seq_len]``
with ``-100`` at ignored target positions.

- **Fuzzy in-context recall** (multi-query): keys and values are variable-
  length token motifs (1 to ``k_motif``/``v_motif`` tokens at training time,
  maximum-size keys at test time). Whenever a key motif repeats, the target
  is its value motif. Partially overlapping motifs make the addresses fuzzy.
- **Selective copying**: content tokens are scattered among blanks; after the
  copy marker the model must reproduce the content tokens in order.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import torch

__all__ = ["fuzzy_recall", "selective_copy", "generate", "TASKS"]

IGNORE_INDEX = -100
TASKS = ("fuzzy", "selective")


def _motif_tables(
    vocab: np.ndarray, max_size: int, rng: np.random.Generator
) -> dict[int, list[tuple[int, ...]]]:
    tables: dict[int, list[tuple[int, ...]]] = {}
    for size in range(1, max_size + 1):
        motifs = list(permutations(vocab.tolist(), size))
        rng.shuffle(motifs)
        tables[size] = motifs
    return tables


def _pick(rng: np.random.Generator, items: list) -> tuple[int, ...]:
    return tuple(items[int(rng.integers(len(items)))])


def _fuzzy_instance(
    rng: np.random.Generator,
    *,
    vocab_size: int,
    seq_len: int,
    k_motif: int,
    v_motif: int,
    train: bool,
) -> tuple[np.ndarray, np.ndarray]:
    pad_token = vocab_size - 1
    non_special = vocab_size - 1
    key_vocab = np.arange(non_special // 2)
    value_vocab = np.arange(non_special // 2, non_special)

    if train:
        keys = _motif_tables(key_vocab, k_motif, rng)
    else:
        keys = {k_motif: _motif_tables(key_vocab, k_motif, rng)[k_motif]}
    values = _motif_tables(value_vocab, v_motif, rng)

    key_sizes = list(keys.keys())
    value_sizes = list(values.keys())

    probe_key_size = (
        int(key_sizes[int(rng.integers(len(key_sizes)))]) if train else k_motif
    )
    probe_value_size = int(value_sizes[int(rng.integers(len(value_sizes)))])
    probe_key = _pick(rng, keys[probe_key_size])
    probe_value = _pick(rng, values[probe_value_size])
    probe_size = probe_key_size + probe_value_size
    probe_idx = int(rng.integers(seq_len - 2 * probe_size))
    probe_added = False

    kv_map: dict[int, dict[tuple[int, ...], tuple[int, ...]]] = {
        size: {} for size in key_sizes
    }
    seen: set[tuple[int, ...]] = set()
    inputs: list[int] = []
    targets: list[int] = []

    while len(inputs) < seq_len - probe_size - (k_motif + v_motif):
        if len(inputs) >= probe_idx and not probe_added:
            inputs.extend(probe_key)
            inputs.extend(probe_value)
            targets.extend([IGNORE_INDEX] * probe_size)
            kv_map[probe_key_size][probe_key] = probe_value
            seen.add(probe_key)
            probe_added = True
            continue
        key_size = int(key_sizes[int(rng.integers(len(key_sizes)))])
        value_size = int(value_sizes[int(rng.integers(len(value_sizes)))])
        key = _pick(rng, keys[key_size])
        inputs.extend(key)
        if key == probe_key:
            value = probe_value
            probe_added = True
        elif key in kv_map[key_size]:
            value = kv_map[key_size][key]
        else:
            value = _pick(rng, values[value_size])
            kv_map[key_size][key] = value
        inputs.extend(value)
        targets.extend([IGNORE_INDEX] * len(key))
        if key in seen:
            targets.extend(value)
        else:
            targets.extend([IGNORE_INDEX] * len(value))
        seen.add(key)

    inputs.extend(probe_key)
    inputs.extend(probe_value)
    targets.extend([IGNORE_INDEX] * probe_key_size)
    targets.extend(probe_value)

    inputs_arr = np.asarray(inputs, dtype=np.int64)
    targets_arr = np.asarray(targets, dtype=np.int64)
    if len(inputs_arr) < seq_len + 1:
        pad = seq_len + 1 - len(inputs_arr)
        inputs_arr = np.concatenate(
            [np.full(pad, pad_token, dtype=np.int64), inputs_arr]
        )
        targets_arr = np.concatenate(
            [np.full(pad, IGNORE_INDEX, dtype=np.int64), targets_arr]
        )
    if train:
        return inputs_arr[:-1], inputs_arr[1:]
    return inputs_arr[:-1], targets_arr[1:]


def fuzzy_recall(
    num_examples: int,
    *,
    seq_len: int = 128,
    vocab_size: int = 16,
    k_motif: int = 3,
    v_motif: int = 3,
    train: bool = True,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuzzy in-context recall (multi-query)."""
    rng = np.random.default_rng(seed)
    pairs = [
        _fuzzy_instance(
            rng,
            vocab_size=vocab_size,
            seq_len=seq_len,
            k_motif=k_motif,
            v_motif=v_motif,
            train=train,
        )
        for _ in range(num_examples)
    ]
    inputs = torch.from_numpy(np.stack([p[0] for p in pairs]))
    targets = torch.from_numpy(np.stack([p[1] for p in pairs]))
    return inputs, targets


def selective_copy(
    num_examples: int,
    *,
    seq_len: int = 256,
    vocab_size: int = 16,
    tokens_to_copy: int = 16,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selective copying: reproduce scattered content tokens in order."""
    if seq_len <= 2 * tokens_to_copy + 1:
        raise ValueError("seq_len must exceed 2 * tokens_to_copy + 1")
    rng = np.random.default_rng(seed)
    copy_token = vocab_size - 1
    blank_token = vocab_size - 2
    vocab = np.arange(vocab_size - 2)
    blanks = seq_len - 2 * tokens_to_copy - 1

    inputs = np.empty((num_examples, seq_len), dtype=np.int64)
    targets = np.full((num_examples, seq_len), IGNORE_INDEX, dtype=np.int64)
    for row in range(num_examples):
        content = rng.choice(vocab, size=tokens_to_copy, replace=True)
        insert_at = rng.integers(0, tokens_to_copy, size=blanks)
        body = np.insert(content, np.sort(insert_at), blank_token)
        inputs[row, : blanks + tokens_to_copy] = body
        inputs[row, blanks + tokens_to_copy] = copy_token
        inputs[row, blanks + tokens_to_copy + 1 :] = blank_token
        targets[row, blanks + tokens_to_copy + 1 :] = content
    return torch.from_numpy(inputs), torch.from_numpy(targets)


def generate(
    task: str,
    num_examples: int,
    *,
    train: bool = True,
    seed: int = 0,
    **overrides,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a MAD task split by name (``"fuzzy"`` or ``"selective"``)."""
    if task == "fuzzy":
        return fuzzy_recall(num_examples, train=train, seed=seed, **overrides)
    if task == "selective":
        return selective_copy(num_examples, seed=seed, **overrides)
    raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
