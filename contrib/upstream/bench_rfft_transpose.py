"""Standalone reproducer for the 2.55x figure in the `_rfft_transpose` PR.

No dependency on the project this came from: `jax` only. Compiles the gradient of an `rfftn`
and reads XLA's own memory analysis, so it does not execute the program and needs no
particular amount of device memory.

    python bench_rfft_transpose.py lean
    python bench_rfft_transpose.py stock

**One variant per process, and this matters.** Running both in one interpreter reports the
first variant's HLO for the second, because the second hits its trace cache and the patched
rule is never invoked. The `rfft_transposes` count in the output is the guard: it must be 1.
An in-process A/B that reports identical numbers for both variants has not measured anything.

Copyright 2026 The JAX Authors. (Contributed under Apache-2.0.)
"""

from __future__ import annotations

import json
import math
import sys
from functools import partial

import jax
import jax.numpy as jnp
from jax._src.lax import fft as lax_fft
from jax._src.lax import lax, slicing
from jax._src.lax.fft import FftType, _real_dtype, fft

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "lean"

#: A shape where the difference is visible: a 159 MiB float32 primal, transformed over its
#: two trailing axes. Any real-input FFT gradient at a few hundred MiB shows the same thing.
SHAPE = (32, 21, 32, 44, 44)
LENGTHS = (44, 44)


# ---------------------------------------------------------------------------
# The proposed rule, exactly as in jax-rfft-transpose.patch
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnums=1)
def rfft_transpose_lean(t, fft_lengths):
    n = fft_lengths[-1]
    m = n // 2 + 1
    assert t.shape[-1] == m, (t.shape, fft_lengths)
    is_odd = n % 2

    full = partial(lax.full_like, t, dtype=t.dtype)
    reflected_bins = lax.concatenate(
        [
            full(1.0, shape=(1,)),
            full(0.0, shape=(m - 2 + is_odd,)),
            full(1.0, shape=(1 - is_odd,)),
        ],
        dimension=0,
    )
    reflected_bins = lax.expand_dims(reflected_bins, range(t.ndim - 1))

    z = lax.conj(t)
    reflected = z
    for axis in range(t.ndim - len(fft_lengths), t.ndim - 1):
        d = t.shape[axis]
        rev = lax.rev(reflected, dimensions=(axis,))
        reflected = lax.concatenate(
            [
                slicing.slice_in_dim(rev, d - 1, d, axis=axis),
                slicing.slice_in_dim(rev, 0, d - 1, axis=axis),
            ],
            dimension=axis,
        )

    half = lax.full_like(t, 0.5)
    h = half * (z + reflected_bins * lax.conj(reflected))
    out = fft(h, FftType.IRFFT, fft_lengths)
    out = out * lax.full_like(out, math.prod(fft_lengths))
    assert out.dtype == _real_dtype(t.dtype), (out.dtype, t.dtype)
    return out


# ---------------------------------------------------------------------------
fired: list = []
_chosen = rfft_transpose_lean if VARIANT == "lean" else lax_fft._rfft_transpose


def _counted(t, fft_lengths):
    fired.append(tuple(t.shape))
    return _chosen(t, fft_lengths)


lax_fft._rfft_transpose = _counted

d = len(LENGTHS)
axes = tuple(range(len(SHAPE) - d, len(SHAPE)))


def loss(x):
    f = jnp.fft.rfftn(x, s=LENGTHS, axes=axes)
    return jnp.real(jnp.sum(f * jnp.conj(f) * (1.0 + 0.5j)))


compiled = jax.jit(jax.grad(loss)).lower(jnp.zeros(SHAPE, jnp.float32)).compile()
ma = compiled.memory_analysis()
text = compiled.as_text()
kinds = sorted(
    {ln.split("fft_type=")[1].split(",")[0] for ln in text.splitlines() if "fft_type=" in ln}
)

print(
    json.dumps(
        {
            "variant": VARIANT,
            "device": jax.local_devices()[0].device_kind,
            "jax": jax.__version__,
            "shape": list(SHAPE),
            "fft_lengths": list(LENGTHS),
            "primal_MiB": round(math.prod(SHAPE) * 4 / 2**20, 1),
            "temp_MiB": round(ma.temp_size_in_bytes / 2**20, 1),
            "fft_kinds": kinds,
            "rfft_transposes": len(fired),  # must be 1; see the module docstring
        },
        sort_keys=True,
    )
)
