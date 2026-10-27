"""Tests to add to JAX's `tests/fft_test.py` alongside the `_rfft_transpose` patch.

Not part of the neurocell library and not collected by its suite (`python_files = unit_*.py`).
Written in `fft_test.py`'s idiom -- `jtu.JaxTestCase`, `jtu.sample_product`, `jtu.rand_default`
-- and validated in an equivalent standalone form; see the "What has and has not been run"
table in `jax-rfft-transpose.md` before filing.

All three deliberately avoid the shape of test that would pass vacuously:

* `testRfftTransposeAgainstNaiveRfft` compares the new rule against the old one, which the
  patch keeps as `_rfft_transpose_via_naive_rfft`. Without keeping the old route there is
  nothing to compare against and the test would only check the new rule against itself.
* The cotangent has independent real and imaginary parts. A real loss built from
  `f * conj(f)` feeds a cotangent proportional to `f`, which is structured enough to hide a
  conjugation error in the mask -- and did, in an earlier version of this work.
* `testRfftAdjointIdentity` and `testRfftGradAgainstFiniteDifference` step outside JAX's own
  `vjp` machinery, so an error *shared* by both rules would still fail. That is the whole
  reason they are here rather than only the first test.

Copyright 2026 The JAX Authors. (Contributed under Apache-2.0; see the licence note in
`jax-rfft-transpose.md`.)
"""

import numpy as np

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax._src.lax import fft as lax_fft

from absl.testing import absltest

jax.config.parse_flags_with_absl()


# (input shape, fft_lengths). One to three transform axes, both parities of the last one,
# leading batch axes, and n_last of 1 and 2 as degenerate cases -- n_last = 2 has both of
# its bins self-conjugate, and n_last = 1 has only DC, so the mask degenerates in opposite
# directions and both must still be right.
RFFT_TRANSPOSE_SHAPES = [
    ((3, 16), (16,)),
    ((3, 17), (17,)),
    ((3, 2), (2,)),
    ((3, 1), (1,)),
    ((2, 5, 12, 14), (12, 14)),
    ((2, 5, 12, 13), (12, 13)),
    ((2, 5, 13, 13), (13, 13)),
    ((2, 3, 6, 8, 10), (6, 8, 10)),
    ((2, 3, 6, 8, 9), (6, 8, 9)),
    ((2, 8, 44, 44), (44, 44)),
    ((2, 4, 44, 44, 21), (44, 44, 21)),
]


def _axes(ndim, d):
    """The fft primitive transforms the trailing `d` axes."""
    return tuple(range(ndim - d, ndim))


def _asymmetric_cotangent(rng, shape, dtype):
    real_dtype = np.finfo(dtype).dtype if np.issubdtype(dtype, np.complexfloating) else dtype
    return (rng(shape, real_dtype) + 1j * rng(shape, real_dtype)).astype(dtype)


class RfftTransposeTest(jtu.JaxTestCase):
    @jtu.sample_product(
        shape_lengths=RFFT_TRANSPOSE_SHAPES,
        dtype=[np.float32, np.float64],
    )
    def testRfftTransposeAgainstNaiveRfft(self, shape_lengths, dtype):
        """The IRFFT-based rule against the route it replaces.

        Asserts agreement to a few ulps rather than bitwise. The two were observed bitwise
        equal on every shape here in float32 on CPU, but they are different floating-point
        routes to the same linear map and nothing guarantees that across dtypes and
        backends, so pinning bitwise would make this test a tripwire for unrelated changes.
        """
        if dtype == np.float64 and not jax.config.jax_enable_x64:
            self.skipTest("float64 requires x64")
        shape, lengths = shape_lengths
        d = len(lengths)
        axes = _axes(len(shape), d)
        rng = jtu.rand_default(self.rng())

        x = rng(shape, dtype)
        spectrum = jnp.fft.rfftn(x, s=lengths, axes=axes)
        ct = _asymmetric_cotangent(rng, spectrum.shape, spectrum.dtype)

        def pull(rule):
            original = lax_fft._rfft_transpose
            lax_fft._rfft_transpose = rule
            try:
                _, f_vjp = jax.vjp(lambda a: jnp.fft.rfftn(a, s=lengths, axes=axes), x)
                (g,) = f_vjp(ct)
                return np.asarray(jax.block_until_ready(g))
            finally:
                lax_fft._rfft_transpose = original

        new = pull(lax_fft._rfft_transpose)
        old = pull(lax_fft._rfft_transpose_via_naive_rfft)

        self.assertEqual(new.dtype, x.dtype)
        self.assertEqual(new.shape, x.shape)
        scale = max(float(np.max(np.abs(old))), np.finfo(dtype).tiny)
        ulps = float(np.max(np.abs(new - old))) / float(np.spacing(np.asarray(scale, dtype)))
        self.assertLessEqual(ulps, 16.0, f"{ulps:.1f} ulps between the two transpose rules")

    @jtu.sample_product(shape_lengths=RFFT_TRANSPOSE_SHAPES)
    @jtu.skip_on_devices("tpu")  # float64
    def testRfftAdjointIdentity(self, shape_lengths):
        """`Re(sum_k t_k f_k) == sum_j x_j (A^T t)_j`, from the definition.

        The pairing on the spectrum side is **bilinear, not Hermitian** -- no conjugate on
        `t`. That is not a choice: the rule computes the plain transpose, so with
        `A_{kj} = W^(j.k)` and `x` real,

            sum_j x_j (A^T t)_j = Re( sum_k t_k sum_j x_j W^(j.k) ) = Re( sum_k t_k f_k ).

        Pairing `conj(t)` with `f` is the natural reflex, checks a different quantity, and
        fails by order one.

        float64, because in float32 the two sides of a length-40000 sum agree to about five
        digits and the test would have to be loosened until it proved nothing.
        """
        if not jax.config.jax_enable_x64:
            self.skipTest("float64 requires x64")
        shape, lengths = shape_lengths
        axes = _axes(len(shape), len(lengths))
        rng = jtu.rand_default(self.rng())

        x = rng(shape, np.float64)
        f = jnp.fft.rfftn(x, s=lengths, axes=axes)
        ct = _asymmetric_cotangent(rng, f.shape, np.complex128)

        lhs = float(jnp.real(jnp.sum(ct * f)))
        _, f_vjp = jax.vjp(lambda a: jnp.fft.rfftn(a, s=lengths, axes=axes), x)
        (adj,) = f_vjp(ct)
        rhs = float(jnp.sum(x * adj))
        self.assertAllClose(lhs, rhs, rtol=1e-11, atol=1e-11)

    def testRfftGradAgainstFiniteDifference(self):
        """Against arithmetic rather than against JAX: a convention both rules could share.

        float64 throughout -- a central difference in float32 has no significant digits left
        after the subtraction.
        """
        if not jax.config.jax_enable_x64:
            self.skipTest("float64 requires x64")
        shape, lengths, axes = (2, 6, 8), (6, 8), (1, 2)
        x = jtu.rand_default(self.rng())(shape, np.float64)

        def loss(a):
            f = jnp.fft.rfftn(a, s=lengths, axes=axes)
            w = jnp.arange(f.size, dtype=a.dtype).reshape(f.shape) * (1.0 + 0.25j)
            return jnp.real(jnp.sum(f * w))

        g = jax.grad(loss)(x)
        eps = 1e-6
        for idx in ((0, 0, 0), (1, 3, 5), (0, 5, 7)):
            fd = (loss(x.at[idx].add(eps)) - loss(x.at[idx].add(-eps))) / (2 * eps)
            self.assertAllClose(float(fd), float(g[idx]), rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
