"""Shared model shell and training loop for the benchmark examples.

The model follows the protocol of the recorded runs: a tied token embedding,
two pre-RMSNorm residual blocks at width 128 — the first block mixes tokens
with a gated short convolution (a BaseConv-style mixer), the second block is
:class:`thetamem.ThetaMemLayer` — each followed by a bias-free
``SwiGLU(d -> 2d -> d)``, and a final RMSNorm before the tied head.

This is a minimal reproduction harness, not a benchmark framework: it
regenerates data with :mod:`thetamem.data`, trains with AdamW and a cosine
schedule, and reports accuracy over labeled positions. Recorded reference
numbers come from the original benchmark harnesses; this script reproduces
the protocol, not the bit pattern.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from thetamem import ThetaMemLayer


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return normalized * self.weight


class BiasFreeSwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.value(x))


class BaseConvMixer(nn.Module):
    """Gated short-convolution token mixer: ``conv(u) * proj(u) + u``."""

    def __init__(self, width: int, kernel: int = 3) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width)
        self.kernel = kernel
        self.conv = nn.Conv1d(width, width, kernel, groups=width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x.transpose(1, 2), (self.kernel - 1, 0))
        mixed = self.conv(padded).transpose(1, 2)
        return mixed * self.projection(x) + x


class Block(nn.Module):
    def __init__(self, width: int, mixer: nn.Module, hidden: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.mixer = mixer
        self.norm2 = RMSNorm(width)
        self.ffn = BiasFreeSwiGLU(width, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class SequenceModel(nn.Module):
    """Tied-embedding language model: BaseConv block, then a ThetaMem block."""

    def __init__(self, vocab_size: int, memory_layer: ThetaMemLayer, *, d_model: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                Block(d_model, BaseConvMixer(d_model), 2 * d_model),
                Block(d_model, memory_layer, 2 * d_model),
            ]
        )
        self.norm = RMSNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.norm(x), self.embedding.weight)


@dataclass
class TrainSettings:
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    epochs: int = 32
    train_batch: int = 256
    eval_batch: int = 32
    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 50


def _batches(inputs: torch.Tensor, labels: torch.Tensor, batch: int, generator: torch.Generator):
    order = torch.randperm(inputs.shape[0], generator=generator)
    for start in range(0, len(order), batch):
        index = order[start : start + batch]
        yield inputs[index], labels[index]


def evaluate(model: nn.Module, segments, batch: int, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in segments:
            for start in range(0, inputs.shape[0], batch):
                x = inputs[start : start + batch].to(device)
                y = labels[start : start + batch].to(device)
                logits = model(x)
                mask = y != -100
                predictions = logits.argmax(-1)
                correct += int((predictions[mask] == y[mask]).sum())
                total += int(mask.sum())
    model.train()
    return correct / max(total, 1)


def train(model: nn.Module, train_segments, eval_segments, settings: TrainSettings):
    """AdamW + per-epoch cosine decay; returns per-eval-segment accuracy."""
    torch.manual_seed(settings.seed)
    device = settings.device
    model.to(device).train()
    fused = device == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
        fused=fused,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings.epochs, eta_min=0.0
    )
    generator = torch.Generator().manual_seed(settings.seed)
    step = 0
    started = time.time()
    for epoch in range(settings.epochs):
        for inputs, labels in train_segments:
            for x, y in _batches(inputs, labels, settings.train_batch, generator):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.flatten(0, 1), y.flatten(), ignore_index=-100
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step += 1
                if step % settings.log_every == 0:
                    rate = step / (time.time() - started)
                    print(
                        f"epoch {epoch + 1}/{settings.epochs} step {step} "
                        f"loss {loss.item():.4f} ({rate:.1f} steps/s)",
                        flush=True,
                    )
        scheduler.step()
    return [
        evaluate(model, [segment], settings.eval_batch, device)
        for segment in eval_segments
    ]


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def set_seed(seed: int) -> None:
    """Seed model construction as well as the later training loop."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe_memory_call(args) -> str:
    """The exact public API call the example builds — printed for the log."""
    return (
        "thetamem.ThetaMemLayer(128, heads=4, key_dim=32, value_dim=64, "
        f"state={args.state!r}, update={args.update!r}, passes={args.passes}, "
        f"backend={args.backend!r}, chunk={args.chunk})"
    )


def build_memory_layer(args) -> ThetaMemLayer:
    return ThetaMemLayer(
        128,
        heads=4,
        key_dim=32,
        value_dim=64,
        state=args.state,
        update=args.update,
        passes=args.passes,
        backend=args.backend,
        chunk=args.chunk,
    )
