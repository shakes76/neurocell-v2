'''
Unit and numerical parity tests for `neurocell/kernels_greens.py` (Milestone 3:
Hoisted 64x64 Overlap-Save 2D RFFT and Hybrid NCHW/RFFT Green's function backends).

@author Shakes
'''
import os
import sys
import unittest
from functools import partial

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONTRIB_DIR = os.path.join(ROOT_DIR, "contrib")
for p in (ROOT_DIR, CONTRIB_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
import jax.numpy as jnp
import optax

import convs as legacy_convs
from neurocell.topology import ChipTopology2D
from neurocell.params import VNNParams, init_vnn_params
from neurocell.kernels_greens import (
    conv2d_batched_nchw,
    hoist_filter_rfft2,
    conv2d_roi_rfft_step,
    conv2d_valid_rfft_precomputed,
    hoist_filter_rfft2_zero_fwd,
    conv2d_hybrid_roi_step,
)
from neurocell.model import VonNeumannNetwork


def _rel_l2_error(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(a) + 1e-12))


class TestKernelsGreensRFFT(unittest.TestCase):
    """Verifies forward and custom_vjp backward parity of RFFT and Hybrid Green's kernels."""

    def test_valid_rfft_matches_legacy_fast_conv_and_nchw(self):
        """
        Verifies that `conv2d_valid_rfft_precomputed` matches `conv2d_batched_nchw`
        and legacy `contrib/convs.py` (`_fast_conv`) in both forward and backward passes.
        """
        B, H, W = 32, 62, 62
        M, Kh, Kw = 16, 23, 13
        roi_shape = (H - Kh + 1, W - Kw + 1)

        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(101), 3)
        cells = jax.random.normal(k1, (B, H, W), dtype=jnp.float32)
        filters_oihw = jax.random.normal(k2, (M, 1, Kh, Kw), dtype=jnp.float32)
        filters_ihwo = jnp.transpose(filters_oihw, (1, 2, 3, 0))
        cotangent = jax.random.normal(k3, (B, M, roi_shape[0], roi_shape[1]), dtype=jnp.float32)

        # 1. Legacy _fast_conv (via vmap over B, returning (B, H_v, W_v, M))
        legacy_fn = jax.vmap(
            partial(
                legacy_convs._fast_conv,
                filters=filters_ihwo,
                dn=("NHWC", "IHWO", "NHWC"),
                window_strides=(1, 1),
                padding_mode="VALID",
            )
        )
        out_legacy_bhwm = legacy_fn(cells[:, :, :, jnp.newaxis])
        out_legacy_bmhw = jnp.transpose(out_legacy_bhwm, (0, 3, 1, 2))

        # 2. Direct NCHW batched convolution
        out_nchw = conv2d_batched_nchw(
            cells[:, jnp.newaxis, :, :], filters_oihw, padding="VALID"
        )

        # 3. Hoisted 64x64 Overlap-Save RFFT VALID convolution
        ker_freq = hoist_filter_rfft2(filters_oihw, kernel_size=(Kh, Kw), fft_shape=(64, 64))
        out_rfft = conv2d_valid_rfft_precomputed(
            cells, ker_freq, roi_shape=roi_shape, kernel_size=(Kh, Kw), fft_shape=(64, 64)
        )

        self.assertLess(_rel_l2_error(out_nchw, out_legacy_bmhw), 1e-6)
        self.assertLess(_rel_l2_error(out_rfft, out_nchw), 1e-6)
        self.assertLess(float(jnp.max(jnp.abs(out_rfft - out_nchw))), 5e-5)

        # Check backward gradients (dX, dG) against IEEE FP32 reference (Precision.HIGHEST)
        def loss_nchw_fp32(c, f):
            y = conv2d_batched_nchw(
                c[:, jnp.newaxis, :, :],
                f,
                padding="VALID",
                precision=jax.lax.Precision.HIGHEST,
            )
            return jnp.sum(y * cotangent)

        def loss_rfft(c, f):
            kf = hoist_filter_rfft2(f, kernel_size=(Kh, Kw), fft_shape=(64, 64))
            y = conv2d_valid_rfft_precomputed(
                c, kf, roi_shape=roi_shape, kernel_size=(Kh, Kw), fft_shape=(64, 64)
            )
            return jnp.sum(y * cotangent)

        dx_nchw, dg_nchw = jax.grad(loss_nchw_fp32, argnums=(0, 1))(cells, filters_oihw)
        dx_rfft, dg_rfft = jax.grad(loss_rfft, argnums=(0, 1))(cells, filters_oihw)

        self.assertLess(_rel_l2_error(dx_nchw, dx_rfft), 1e-5)
        self.assertLess(_rel_l2_error(dg_nchw, dg_rfft), 1e-5)

    def test_roi_rfft_step_forward_and_custom_vjp_parity(self):
        """
        Verifies `conv2d_roi_rfft_step` + `hoist_filter_rfft2` on the `(40, 50)` ROI
        against `conv2d_batched_nchw(..., padding="SAME")` for both `need_dx=True` and `False`.
        """
        B, roi_h, roi_w = 64, 40, 50
        M, Kh, Kw = 16, 23, 13
        pad_hw = ((Kh - 1) // 2, (Kw - 1) // 2)

        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(202), 3)
        roi_cells = jax.random.normal(k1, (B, roi_h, roi_w), dtype=jnp.float32)
        filters_oihw = jax.random.normal(k2, (M, 1, Kh, Kw), dtype=jnp.float32)
        dy = jax.random.normal(k3, (B, M, roi_h, roi_w), dtype=jnp.float32)

        def ref_fn(c, f):
            y = conv2d_batched_nchw(c[:, jnp.newaxis, :, :], f, padding="SAME")
            return jnp.sum(y * dy), y

        def rfft_fn(c, f, need_dx=True):
            kf = hoist_filter_rfft2(f, kernel_size=(Kh, Kw), fft_shape=(64, 64))
            y = conv2d_roi_rfft_step(
                c,
                kf,
                roi_shape=(roi_h, roi_w),
                pad_hw=pad_hw,
                fft_shape=(64, 64),
                need_dx=need_dx,
            )
            return jnp.sum(y * dy), y

        (dx_ref, dg_ref), y_ref = jax.grad(ref_fn, argnums=(0, 1), has_aux=True)(
            roi_cells, filters_oihw
        )
        (dx_rfft, dg_rfft), y_rfft = jax.grad(
            partial(rfft_fn, need_dx=True), argnums=(0, 1), has_aux=True
        )(roi_cells, filters_oihw)

        self.assertLess(_rel_l2_error(y_ref, y_rfft), 1e-6)
        self.assertLess(float(jnp.max(jnp.abs(y_ref - y_rfft))), 5e-5)
        self.assertLess(_rel_l2_error(dx_ref, dx_rfft), 1e-5)
        self.assertLess(_rel_l2_error(dg_ref, dg_rfft), 1e-5)

        # When need_dx=False, dG must still match while dX is identically zero
        (dx_zero, dg_rfft_nodx), _ = jax.grad(
            partial(rfft_fn, need_dx=False), argnums=(0, 1), has_aux=True
        )(roi_cells, filters_oihw)
        self.assertEqual(float(jnp.max(jnp.abs(dx_zero))), 0.0)
        self.assertLess(_rel_l2_error(dg_ref, dg_rfft_nodx), 1e-5)

    def test_hybrid_roi_step_forward_and_custom_vjp_parity(self):
        """
        Verifies `conv2d_hybrid_roi_step` + `hoist_filter_rfft2_zero_fwd` on the `(40, 50)` ROI.
        """
        B, roi_h, roi_w = 64, 40, 50
        M, Kh, Kw = 16, 23, 13
        pad_hw = ((Kh - 1) // 2, (Kw - 1) // 2)

        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(303), 3)
        roi_cells = jax.random.normal(k1, (B, roi_h, roi_w), dtype=jnp.float32)
        filters_oihw = jax.random.normal(k2, (M, 1, Kh, Kw), dtype=jnp.float32)
        dy = jax.random.normal(k3, (B, M, roi_h, roi_w), dtype=jnp.float32)

        def ref_fn(c, f):
            y = conv2d_batched_nchw(c[:, jnp.newaxis, :, :], f, padding="SAME")
            return jnp.sum(y * dy), y

        def hyb_fn(c, f, need_dx=True):
            token = hoist_filter_rfft2_zero_fwd(f, kernel_size=(Kh, Kw), fft_shape=(64, 64))
            f_sg = jax.lax.stop_gradient(f)
            y = conv2d_hybrid_roi_step(
                c,
                f_sg,
                token,
                pad_hw=pad_hw,
                fft_shape=(64, 64),
                need_dx=need_dx,
            )
            return jnp.sum(y * dy), y

        (dx_ref, dg_ref), y_ref = jax.grad(ref_fn, argnums=(0, 1), has_aux=True)(
            roi_cells, filters_oihw
        )
        (dx_hyb, dg_hyb), y_hyb = jax.grad(
            partial(hyb_fn, need_dx=True), argnums=(0, 1), has_aux=True
        )(roi_cells, filters_oihw)

        # Forward and dX are bit-exact with NCHW direct!
        self.assertEqual(float(jnp.max(jnp.abs(y_ref - y_hyb))), 0.0)
        self.assertLess(_rel_l2_error(dx_ref, dx_hyb), 1e-6)
        self.assertLess(_rel_l2_error(dg_ref, dg_hyb), 1e-5)

    def test_vnn_alu_rfft_hoisted_and_hybrid_end_to_end_parity(self):
        """
        Verifies end-to-end forward and backward gradient parity of `VonNeumannNetwork`
        across `"nchw_direct"`, `"rfft_hoisted"`, and `"nchw_rfft_bwd"` with non-zero biases.
        """
        topo = ChipTopology2D.alu_chip(
            n=8, kernel_size=(23, 13), directions=16, depth=4, gap=5, ctrl_gap=5
        )
        params0 = init_vnn_params(jax.random.PRNGKey(42), topo, legacy_compat=True)
        b_nonzero = jax.random.normal(jax.random.PRNGKey(77), topo.grid_shape) * 0.25
        params = VNNParams(
            cell_params=params0.cell_params.at[1].set(b_nonzero),
            greens_filters=params0.greens_filters,
        )

        B = 32
        k_in, k_tgt = jax.random.split(jax.random.PRNGKey(999))
        inputs = jax.random.randint(k_in, (B, 3, 8, 1), 0, 2).astype(jnp.float32)
        targets = jax.random.randint(k_tgt, (B, 8), 0, 2).astype(jnp.float32)

        model_nchw = VonNeumannNetwork(topo, backend="nchw_direct")
        model_rfft = VonNeumannNetwork(topo, backend="rfft_hoisted")
        model_hyb = VonNeumannNetwork(topo, backend="nchw_rfft_bwd")

        # 1. Check return_universe=True parity
        out_nchw, univ_nchw = model_nchw.apply(params, inputs, return_universe=True)
        out_rfft, univ_rfft = model_rfft.apply(params, inputs, return_universe=True)
        out_hyb, univ_hyb = model_hyb.apply(params, inputs, return_universe=True)

        self.assertLess(float(jnp.max(jnp.abs(out_nchw - out_rfft))), 2e-6)
        self.assertLess(float(jnp.max(jnp.abs(univ_nchw - univ_rfft))), 2e-6)
        self.assertEqual(float(jnp.max(jnp.abs(out_nchw - out_hyb))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(univ_nchw - univ_hyb))), 0.0)

        # 2. Check end-to-end MSE loss and parameter gradients (W, B, S, G)
        def make_loss(model):
            def loss_fn(p):
                logits = jnp.squeeze(model.apply(p, inputs, return_universe=False), axis=-1)
                return optax.squared_error(logits, targets).mean()
            return loss_fn

        l_nchw, g_nchw = jax.value_and_grad(make_loss(model_nchw))(params)
        l_rfft, g_rfft = jax.value_and_grad(make_loss(model_rfft))(params)
        l_hyb, g_hyb = jax.value_and_grad(make_loss(model_hyb))(params)

        self.assertLess(abs(float(l_nchw - l_rfft)), 1e-6)
        self.assertEqual(abs(float(l_nchw - l_hyb)), 0.0)

        for g_test in (g_rfft, g_hyb):
            self.assertLess(_rel_l2_error(g_nchw.weights, g_test.weights), 1e-5)
            self.assertLess(_rel_l2_error(g_nchw.biases, g_test.biases), 1e-5)
            self.assertLess(_rel_l2_error(g_nchw.states, g_test.states), 1e-5)
            self.assertLess(_rel_l2_error(g_nchw.greens_filters, g_test.greens_filters), 5e-5)


if __name__ == "__main__":
    unittest.main()
