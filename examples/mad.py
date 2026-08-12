"""MAD synthetics — fuzzy in-context recall and selective copying.

Protocol (matching the recorded runs): 6,400 training and 1,280 test
examples; batch 128; 200 epochs (10,000 optimizer steps); AdamW with weight
decay 0; per-epoch cosine decay; learning-rate grid {5e-4, 1e-3, 3.16e-3},
one run per invocation. Fuzzy recall trains autoregressively and is scored
on re-queried values; selective copying is scored on the copied tokens.

The launch configurations this example exposes:

    python examples/mad.py --task fuzzy     --state outer    --lr 3.16e-3
    python examples/mad.py --task fuzzy     --state concat   --lr 1e-3
    python examples/mad.py --task fuzzy     --state hadamard --lr 3.16e-3
    python examples/mad.py --task selective --state hadamard --lr 1e-3

Add ``--update second_pass`` for one correction or
``--update multi_pass --passes 3`` to repeat it; use ``--smoke`` for a quick
CPU sanity run.
"""

from __future__ import annotations

import argparse

import thetamem
from thetamem.data import mad

from common import (
    SequenceModel,
    TrainSettings,
    build_memory_layer,
    describe_memory_call,
    parameter_count,
    set_seed,
    train,
)

VOCAB_SIZE = 16  # both tasks; already a multiple of eight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="fuzzy", choices=mad.TASKS)
    parser.add_argument("--state", default="hadamard", choices=thetamem.STATE_KINDS)
    parser.add_argument("--update", default="sum", choices=thetamem.UPDATES)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--backend", default="chunked", choices=thetamem.BACKENDS)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train-examples", type=int, default=6_400)
    parser.add_argument("--test-examples", type=int, default=1_280)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--smoke", action="store_true", help="tiny CPU sanity run")
    args = parser.parse_args()

    if args.smoke:
        args.epochs, args.train_examples, args.test_examples = 2, 256, 128

    set_seed(args.seed)

    print("memory:", describe_memory_call(args))
    print(
        f"data: thetamem.data.mad.generate({args.task!r}, "
        f"{args.train_examples}, seed={args.seed})"
    )

    train_split = mad.generate(
        args.task, args.train_examples, train=True, seed=args.seed
    )
    test_split = mad.generate(
        args.task, args.test_examples, train=False, seed=args.seed + 1_000
    )

    model = SequenceModel(VOCAB_SIZE, build_memory_layer(args))
    print(f"parameters: {parameter_count(model):,}")
    print(f"layer streaming state floats (core + caches): {model.blocks[1].mixer.state_size:,}")

    settings = TrainSettings(
        learning_rate=args.lr,
        weight_decay=0.0,
        epochs=args.epochs,
        train_batch=128,
        eval_batch=128,
        seed=args.seed,
    )
    accuracies = train(model, [train_split], [test_split], settings)
    print(f"accuracy on {args.task}: {accuracies[0]:.4f}")


if __name__ == "__main__":
    main()
