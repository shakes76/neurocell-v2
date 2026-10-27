# Draft PR to jax-ml/jax — `_rfft_transpose` without a full complex transform

**Status: draft, not filed.** Written 2026-09-30 against JAX 0.8.0. The patch is
`jax-rfft-transpose.patch`; the body below is the PR description; the tests are meant for
`tests/fft_test.py`. Everything here is validated locally against ROCm and CPU — see
"Testing" — but it has not been run against JAX's own CI, and the numbers are from one
device.

Before filing: check whether `_rfft_transpose` has changed upstream since 0.8.0, re-run the
microbenchmark on the filer's hardware, and search the issue tracker — the honest framing is
"this rebuts a comment in the source", so if that comment has already been revisited the PR
needs rewriting rather than reposting.

---

## Title

`Compute the transpose of RFFT with an IRFFT instead of a padded complex FFT`

---

## Body

### What this changes

`_rfft_transpose` currently takes the adjoint of `RFFT` by transposing a naive
implementation — full complex FFT, then slice — so the transpose becomes *zero-pad the
half-spectrum to full length, run a full complex C2C FFT, take the real part*. The source
says why:

```python
# The transpose of RFFT can't be expressed only in terms of irfft. Instead of
# manually building up larger twiddle matrices (which would increase the
# asymptotic complexity and is also rather complicated), we rely JAX to
# transpose a naive RFFT implementation.
```

**It can be expressed in terms of `irfft`, and doing so is cheaper in both memory and
flops** — no larger twiddle matrices, no asymptotic increase. This PR replaces the rule with
one `IRFFT` of a masked, conjugated cotangent: the mirror of the `_irfft_transpose` that
already sits twenty lines below it.

### Why it is expressible

The key is a constraint the current rule does not exploit: **the transpose of `RFFT` maps
into the reals.** The rule asserts this itself (`assert result.dtype ==
_real_dtype(t.dtype)`). So only the *real part* of the twiddle sum is ever wanted:

```
(A^T t)_j = Re( sum_{k : k_last < m} t_k W^(j.k) ),   W^(j.k) = prod_i exp(-2 pi i j_i k_i / n_i)
          = N * Re( IFFT( conj(pad(t)) ) )
          = N * IFFT( herm( conj(pad(t)) ) )
```

where `herm(z)_k = (z_k + conj(z_{-k})) / 2` is the projection onto Hermitian-symmetric
arrays, `m = n_last // 2 + 1` and `N = prod(fft_lengths)`. Taking the real part commutes into
that projection, and — this is the part that makes it work — `herm` of an array that is zero
beyond bin `m - 1` on the last axis is **supported entirely on the half-spectrum**:

```
herm(z)_k = z_k / 2                        for 1 <= k_last <= m - 2
            (z_k + conj(z_{-k})) / 2        for k_last in {0, Nyquist}
```

because for interior `k_last` the reflected index `n_last - k_last` lands in the zero-padded
region. And a half-spectrum with that symmetry is exactly what `IRFFT` consumes.

The two self-conjugate bins — DC always, Nyquist when `n_last` is even — need `z_{-k}`
reflected through the origin along the **leading** transformed axes as well, because `IRFFT`
assumes Hermitian symmetry only on its last axis and does a full complex inverse on the
others. That reflection is an index permutation, and it is needed on at most two of the `m`
bins.

So the rule becomes: conjugate, halve, symmetrise two bins, one `IRFFT`, scale by `N`.

### Why it is cheaper

Per transform, with `n = n_last`, `m = n // 2 + 1`:

| | current | this PR |
|---|---|---|
| largest temporary | the zero-padded **full** spectrum, `n/m` x the cotangent | the masked half-spectrum, same size as the cotangent |
| transform | one **C2C** of length `n` | one **C2R** of length `n` |
| output | `real(...)` of a full complex array | the C2R output directly |

Two full complex arrays become one half-spectrum, and a complex-to-complex transform becomes
a complex-to-real one. Asymptotic complexity is unchanged at `O(N log N)`; the constants go
down, not up.

### Measured

XLA's `memory_analysis().temp_size_in_bytes` on the gradient of
`lambda x: real(sum(rfftn(x, s=(44,44), axes=(3,4)) * conj(...) * (1 + 0.5j)))` at
`x.shape = (32, 21, 32, 44, 44)`, float32 — a 159 MiB primal:

| | temp | `fft_type` in the optimised HLO |
|---|---|---|
| current | 635.2 MiB | `FFT`, `RFFT` |
| this PR | **249.0 MiB** | `IRFFT`, `RFFT` |

**2.55x less scratch.** The two rules must be timed and compiled in separate processes: an
in-process A/B silently reports the first variant's HLO for both, because the second hits its
trace cache. (Instrumenting the rule with a call counter shows it firing once for the first
variant and zero times for the second. This cost me an afternoon; it is worth a sentence in
the PR in case a reviewer tries to reproduce it the quick way.)

`bench_rfft_transpose.py` in this directory is the standalone reproducer, with no dependency
on the project it came from.

In the application this came from — a 3-D depthwise spectral convolution on a
`(100, 44, 44, 21, 32)` window — it is worth 0.671 GiB of a 9.973 GiB training step, 6.7%,
with step time unchanged at 1080 ms. That is an end-to-end figure on one model and one
device; the 2.55x above is the one a reviewer can check.

### Numerics

**Bitwise identical to the current rule on every shape tested.** Twelve shapes — 1-, 2- and
3-D transforms, even and odd last axis, `n_last` of 1 and 2 as degenerate cases, leading batch
axes — float32, CPU, JAX 0.8.0, compared through `jax.vjp(jnp.fft.rfftn)` with the old route
retained as `_rfft_transpose_via_naive_rfft`:

```
(3,16)/(16,)            (3,17)/(17,)          (3,2)/(2,)
(3,1)/(1,)              (2,5,12,14)/(12,14)   (2,5,12,13)/(12,13)
(2,5,13,13)/(13,13)     (2,3,6,8,10)/(6,8,10) (2,3,6,8,9)/(6,8,9)
(4,44,44,21,8)/(21,8)   (2,8,44,44)/(44,44)   (2,4,44,44,21)/(44,44,21)
```

Relative difference `0.000e+00` on all twelve. The cotangent has independent real and
imaginary parts — a real loss built from `f * conj(f)` feeds a structured cotangent that can
hide a conjugation error in the mask, so that would not be a real test. The rule was also
instrumented with a call counter to confirm it fires once per case rather than being shadowed
by a trace cache.

**I would not claim more than "on every shape tested", though.** These are two different
floating-point routes to the same linear map, and there is no argument that they must round
identically — only the observation that here they do. A reviewer should treat this as a
change that *may* move gradients at the last bit on shapes, dtypes or backends I did not try
(I tested float32 on CPU only), rather than as a guaranteed no-op. For what it is worth, a
`jnp`-level rendering of the same derivation, with `jnp.fft.irfftn` in place of the primitive,
differs from the current rule by 1–3 ulps on similar shapes, so the bitwise agreement here is
a property of this particular expression and not of the algebra.

Two reasons to expect it is not systematically *worse* conditioned: the new route runs one
transform where the old runs one, and it is a C2R of the same length rather than a C2C, so
there is no extra accumulation. I have not attempted a formal error analysis.

If the project would rather not move these values at all, the reasonable outcome is to close
this and keep only the correction to the comment.

### Testing

Added to `tests/fft_test.py` (see `fft_test_additions.py`):

1. `testRfftTransposeAgainstNaive` — the new rule against the old one, which is kept as
   `_rfft_transpose_via_naive_rfft` for the test to call, over 1-, 2- and 3-D transforms,
   even and odd last axis, with leading batch axes. Asserts a ulp bound.
2. `testRfftAdjointIdentity` — `Re(sum(t_k f_k)) == sum(x_j (A^T t)_j)`, in float64, from the
   definition. This is the test that would fail if *both* rules were wrong in the same way.
   Note the pairing is **bilinear, not Hermitian** — no conjugate on `t` — because the rule
   computes the plain transpose, not the adjoint in the Hermitian sense. Pairing `conj(t)`
   with `f` is the natural reflex and checks a different quantity; it fails by order one.
3. `testRfftGradAgainstFiniteDifference` — float64 central difference, to catch a sign
   convention that both rules might share.

Locally: the three above pass over 11 shapes, and JAX's existing `fft_test.py` passes
unchanged. Also exercised indirectly by 633 tests in the project this came from, including
gradient checks of a spectral convolution against a NumPy float64 oracle.

### Not covered by this PR

- `_irfft_transpose` is already efficient (`RFFT` plus a mask) and is untouched.
- The forward `RFFT` path is untouched.
- Only the *trailing* transform axes case exists at this level, because that is the only
  thing the `fft` primitive expresses. Callers who pad before transforming still pay the
  pad's own transpose, which is correct and unrelated.

---

## What has and has not been run

| | status |
|---|---|
| the exact `lax` ops in the patch, against the old route, 12 shapes | **run** — bitwise, float32, CPU |
| adjoint identity and finite difference on the patch, 12 shapes | **run** — pass |
| the three added tests | **run** in this form, not yet inside `fft_test.py`'s harness |
| JAX's own `fft_test.py` against a patched tree | **not run** |
| float64 | **not run** through the patch (only through the `jnp`-level rendering) |
| GPU/TPU backends through this path | **not run** |
| the 2.55x scratch figure | **run**, ROCm, one device |

The first draft of the patch used `lax.slice_in_dim`, which does not exist —
`jax._src.lax.lax` exports neither `slice_in_dim` nor `slice`. It died with `AttributeError`
on the first multi-dimensional case. Worth recording because it is the kind of error that
review does not catch and execution does.

---

## Implementation note — a refinement I have not validated

The patch reflects the **whole** cotangent along the leading transformed axes and then
multiplies by a 0/1 indicator that keeps the result only on the two self-conjugate bins. That
allocates one array the size of the cotangent, which XLA appears to fuse into the subsequent
multiply-add (the post-patch HLO shows no extra full-size buffer), but that is a fusion
observation and not a guarantee.

Slicing the one or two special bins out first, reflecting only those, and concatenating would
make the reflection act on a `1/m` slice and remove the dependence on fusion entirely. It is
very likely the better version. **I have not implemented or tested it**, so it is not in the
patch; it belongs in review, or in a follow-up once someone has run it.

---

## Files in this directory

| file | what it is |
|---|---|
| `jax-rfft-transpose.md` | this document: the PR body and everything a filer needs to check first |
| `jax-rfft-transpose.patch` | the diff against `jax/_src/lax/fft.py` |
| `fft_test_additions.py` | the three tests, in `tests/fft_test.py`'s idiom |
| `bench_rfft_transpose.py` | the standalone reproducer for the 2.55x, no project dependency |

The same derivation ships in this repository as `neurocell.convs._rfftn` — a `custom_vjp`
that applies it to one call site rather than globally — with `tests/unit_rfft.py` and
`docs/FFT_MEMORY.md` sections 2 and 8. **If this PR lands upstream, `convs._rfftn` becomes
redundant and should be deleted rather than kept** (D-065).

---

*Copyright (c) 2026 Shakes Chandra. This draft is for submission to jax-ml/jax under that
project's Apache-2.0 licence and CLA; the repository it sits in is LGPL v3 (see LICENSE),
which does not apply to the contributed patch.*
