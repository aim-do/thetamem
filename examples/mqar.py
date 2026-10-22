"""Multi-query associative recall (MQAR) with the canonical state kinds.

Protocol (matching the recorded runs): vocabulary 8,192; training mixture of
lengths 64-256 with 4-64 key/value pairs; AdamW, weight decay 0.1, 32 epochs
of per-epoch cosine decay; one learning rate per invocation. ``--eval
default`` scores length 256 (hardest in-distribution slice) and length 1,024
with 256 pairs; ``--eval long`` scores the three long slices of the 2026-08
matched-state suite (1,024/256, 2,048/512, 4,096/512) and their mean.

The matched-state suite's quadratic arm (learning-rate grid
{3.16e-4, 1e-3, 3.16e-3}):

    python examples/mqar.py --state shifted_hadamard --qk-ops conv --position none --eval long --lr 1e-3

The v0.1 launch configurations (learning-rate grid {1e-3, 3.16e-3, 1e-2}):

    python examples/mqar.py --state hadamard --lr 3.16e-3
    python examples/mqar.py --state concat   --lr 1e-3
    python examples/mqar.py --state outer    --lr 1e-3

Add ``--update second_pass`` for one correction,
``--update multi_pass --passes 3`` to repeat it, or ``--update delta`` for
the causal delta reference. Use ``--smoke`` for a short CPU sanity run.
"""

from __future__ import annotations

import argparse

import thetamem
from thetamem.data import mqar

from common import (
    SequenceModel,
    TrainSettings,
    add_frontend_arguments,
    build_memory_layer,
    describe_memory_call,
    parameter_count,
    set_seed,
    train,
)

VOCAB_SIZE = 8_192


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--state", default="hadamard", choices=thetamem.STATE_KINDS)
    parser.add_argument("--update", default="sum", choices=thetamem.UPDATES)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--backend", default="chunked", choices=thetamem.BACKENDS)
    parser.add_argument("--chunk", type=int, default=256)
    add_frontend_arguments(parser)
    parser.add_argument("--eval", default="default", choices=("default", "long"))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--train-batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--smoke", action="store_true", help="tiny CPU sanity run")
    args = parser.parse_args()

    if args.smoke:
        train_spec = ((64, 512, 4), (128, 128, 8))
        args.epochs, args.train_batch = 2, 64
        if args.eval == "long":
            eval_spec = tuple(
                (seq_len, 2, kv_pairs)
                for seq_len, _, kv_pairs in mqar.LONG_EVAL_SLICES
            )
        else:
            eval_spec = ((64, 128, 4), (128, 128, 8))
    else:
        train_spec = mqar.TRAIN_MIXTURE
        eval_spec = mqar.LONG_EVAL_SLICES if args.eval == "long" else mqar.EVAL_SLICES

    set_seed(args.seed)

    print("memory:", describe_memory_call(args))
    print("data: thetamem.data.mqar.mixture(...) /", eval_spec)

    train_segments = mqar.mixture(train_spec, vocab_size=VOCAB_SIZE, seed=args.seed)
    eval_segments = mqar.mixture(
        eval_spec, vocab_size=VOCAB_SIZE, seed=args.seed + 1_000
    )

    model = SequenceModel(VOCAB_SIZE, build_memory_layer(args))
    print(f"parameters: {parameter_count(model):,}")
    print(f"layer streaming state floats (core + caches): {model.blocks[1].mixer.state_size:,}")

    settings = TrainSettings(
        learning_rate=args.lr,
        weight_decay=0.1,
        epochs=args.epochs,
        train_batch=args.train_batch,
        # Full-vocabulary logits at length 4,096 need a small evaluation batch.
        eval_batch=4 if args.eval == "long" else 32,
        seed=args.seed,
    )
    accuracies = train(model, train_segments, eval_segments, settings)
    for (seq_len, _, kv_pairs), accuracy in zip(eval_spec, accuracies):
        print(f"accuracy @ len {seq_len} / kv {kv_pairs}: {accuracy:.4f}")
    if args.eval == "long":
        print(f"long-slice score (unweighted mean): {sum(accuracies) / len(accuracies):.4f}")


if __name__ == "__main__":
    main()
