# Algorithms

This page states what the package actually executes. The paper also studies
idealized key geometries and offline limits; those are not automatically
claims about a shipped preset.

## 1. Lifted additive memory

For projected keys `k_i`, values `v_i`, queries `q_t`, and lift `Phi`,

```text
S_t = sum_{i<=t} Phi(k_i) (x) v_i
y_t = <Phi(q_t), S_t>
```

ThetaMem uses a raw read: there is no accumulated-mass denominator. A signed
kernel sum can cross zero, and signed feature coordinates alone do not prove
that its mean is zero. Branch geometry and the data distribution still
matter. `ThetaMemLayer` restores output scale with a per-head RMSNorm.

## 2. Lift algebra and physical cost

- `branch`: `f(x)=Wx`.
- `hadamard`: `phi(x)=f_a(x) * f_b(x)`.
- `concat`: a direct sum of independent state blocks.
- `outer`: `Phi(x)=f_1(x) (x) ... (x) f_m(x)`, with one state axis per
  factor.

For an outer lift, token-pair scores factorize:

```text
<Phi(q),Phi(k)> = product_j <f_j(q),f_j(k)>.
```

This avoids a `C x C x R` tensor inside a chunk, where
`R=product_j F_j`. It does **not** remove the full-state work. Exact carried
state reads and writes cost `Theta(T R V)` and store `Theta(R V)` values.
Factorization reduces projection and chunk-local score formation, not the
arithmetic that touches the recurrent state.

Physical width and functional rank are different. For same-source linear
branches `a=Ak`, `b=Bk`,

```text
vec(a (x) b) = (B (x) A) vec(k (x) k),
rank <= min(F_a F_b, d(d+1)/2).
```

Thus the default `d=F_a=F_b=32` outer allocates 1,024 rows but can span at
most 528 quadratic functions. A disjoint `64 -> 32+32 -> 32x32` split can
generically span all 1,024. See
[`rank_geometry.py`](../examples/capacity/rank_geometry.py).

## 3. Chunked executor

The sequence is divided into chunks of length `C` (default 256).

- Flat lifts issue chunk-local `C x C` score tiles and carry exclusive-prefix
  `K^T V` states between tiles.
- Outer lifts multiply one `C x C` score per factor and carry the tensor
  state sequentially between tiles.
- The `naive` backend evaluates the same causal read as one masked quadratic
  form and is the semantic test oracle.

Carried-state reductions use at least FP32. With `T<=C`, the chunked path
reduces to the local quadratic form; long-context behavior requires multiple
chunks.

## 4. Original second-pass correction

```text
p_t = <Phi(k_t), S_{t-1}>                  # strict-prefix prediction
r_t = v_t - p_t
D_t = sum_{i<=t} Phi(k_i) (x) r_i
y_t = w <Phi(q_t),S_t> + (1-w) <Phi(q_t),D_t>
```

The base pass is finalized before residuals are written. The strict-prefix
boundary is essential: an inclusive first-pass prediction would contain the
token's own value. The correction is stored beside the archive; it does not
erase the base state. Algebraically, this is one correction by the
strict-lower key-overlap Gram: the base state predicts overlap error and the
second state writes that measured residual. It is a useful finite
approximation, not by itself a claim of convergence to a global least-squares
solution.

## 5. Repeating the second pass

`update="multi_pass", passes=P` repeats the causal correction for any
positive `P`. Define the unit-diagonal causal operator

```text
A_c = I + tril(Phi(K) Phi(K)^T, -1)
W_0 = V
W_{p+1} = W_p + eta_p (V - A_c W_p)
y = tril(Phi(Q) Phi(K)^T) W_P.
```

Every pass reads the finalized preceding iterate. Consequently an isolated
key remains `v` rather than being counted again on every pass. With one pass
and `eta=1`, `W_1` is exactly the raw correction from `second_pass`. The
learned per-head `eta_p` values initialize at 0.1, so one default pass exactly
matches `second_pass`'s initial 90/10 base/correction mixture.

This is Richardson iteration for the **causal lower-triangular system**
`A_c W=V`, not for the full-Gram least-squares problem. It preserves causal
prefixes and carries `P+1` full states. Because the triangular operator is
non-normal, a small step is not a universal proof of monotone error. Treat
the result as a learned finite approximation; three to five passes are the
practical range before state and compute dominate.

## 6. Global replay solvers

For a completed context, `replay_fit` approximates the global fitted memory

```text
min_S ||Phi(K) S - V||^2
```

using `solver="richardson"`, `"heavy_ball"`, or `"cg"`. Under their stability
conditions these are finite approximations to the global least-squares
solution. The `"delta"` option performs repeated cyclic delta/Kaczmarz sweeps,
carrying the final state into the next sweep. With unit-norm feature rows (or
a correspondingly stable strength), it can approach interpolation for a
consistent record set; the raw default strength is not a universal
convergence guarantee. For an inconsistent overloaded set it need not equal
least squares. A second sweep is the first practical replay correction, and
three to five iterations are the usual small-budget range; more passes serve
mainly as an offline ceiling.

These solvers do not construct a token-token `T x T` Gram and do not flatten
outer factors. A normal-operator application is a factorized read followed by
a factorized write:

```text
S -> Phi(K) S -> Phi(K)^T(Phi(K) S).
```

The dense state keeps shape `[B,H,F1,...,Fm,V]`, so the product state and its
workspaces still exist and are touched. Every iteration also needs the
completed keys and values again, with a barrier between iterations. This is
an offline/prefill API, not one-pass fixed-state streaming.

- Richardson needs an explicit stable step; `0<eta<2/lambda_max`, not merely
  `eta<1`.
- Heavy-ball adds momentum and is more sensitive to spectral tuning.
- CG is the default. It computes adaptive scalars from global FP32 dot
  products and normally reaches the fitted floor fastest.
- Replayed delta is sequential within each sweep; sweeps after the first are
  offline because they revisit the context.

The deterministic
[`offline_replay_solvers.py`](../examples/capacity/offline_replay_solvers.py)
uses a 1,024-coordinate state. At load `T/R=0.5`, relative MSE after passes
1/3/5 is `0.500/1.16/7.55` for unstable unit Richardson,
`0.362/0.149/0.082` for damped Richardson,
`0.500/0.126/0.036` for model-tuned heavy-ball, and
`0.321/0.063/0.015` for CG. These curves validate the global replay problem
only; they are not a convergence claim for causal `multi_pass`.

## 7. Value centering

Both options remove the value mean and both are subtractive; neither divides
the read. The name says how the mean is obtained:

```text
running_mean: write v_i - mean_i, then add mean_t
exact_mean:   write [v_i;1], then read(v) - mean_t*read(1) + mean_t.
```

The ones channel of `exact_mean` accumulates the signed key mass, and that
mass is subtracted, never divided by — a signed mass can pass through zero.
The grouped PSD/self-product theory surrogate includes its positive
denominator explicitly; see
[`psd_self_product_capacity.py`](../examples/capacity/psd_self_product_capacity.py).

## 8. Delta rule

```text
beta_h = sigmoid(logit_h)                  # init 1/(feature_width+1)
S_t = S_{t-1} + beta_h Phi_t (v_t - S_{t-1}^T Phi_t)^T
y_t = S_t^T Phi(q_t).
```

The causal implementation is one sequential sweep and does not divide by
`||Phi||^2`; normalization is explicit in the frontend or lift. Its existing
backends flatten factorized lifts. To repeat delta without a product-feature
temporary, use `replay_fit(..., solver="delta", iterations=P)`: it retains
factor axes, but later sweeps are offline replays.

## 9. Layer recipe

The default projected-key route is

```text
Q/K: projection -> causal Conv -> SiLU -> RoPE -> L2 -> lift -> memory
V:   projection -> causal Conv -> SiLU -> memory
out: HeadRMSNorm(read) * low-rank SiLU gate -> output projection.
```

All stages are configurable. See [API.md](API.md) for the direct-Hadamard
route and ordered frontend operations.
