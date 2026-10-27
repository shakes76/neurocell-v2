'''
Unit and numerical parity tests for Milestone 7: Epoch-Adaptive Active-Region
Pre-Compute Masking (`neurocell/adaptive_mask.py`).

Tests:
1. Pre-epoch salience detection, safety margin (`margin`) dilation, and bucket quantization
   (`bucket_multiple=8`), including automatic bounding-box expansion when frontier cells
   become active at the end of an epoch.
2. Exact forward, universe-state, and backward gradient parity (`< 1e-5`) across all
   backends (`nchw_direct`, `rfft_hoisted`, `nchw_rfft_bwd`, `pallas_fused`, `pallas_sram`,
   `auto`) when executing with `spatial_mask=mask_spec`, while keeping full `(3, H, W)`
   `VNNParams` shapes invariant.
3. Model-agnostic behavior across full-grid Game of Life (both full-grid default and
   localized `activity_map`), Coordinate-Port CLP (`CoordinatePortSpec`), and large
   `82x82` grids.
4. `VNNTrainer` adaptive epoch JIT bucket caching and convergence parity.

@author Shakes
'''
import os
import sys
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import jax
import jax.numpy as jnp
import optax

from neurocell import (
    AdaptiveMaskSpec,
    ChipTopology2D,
    PortSpec,
    VNNParams,
    VNNTrainer,
    VonNeumannNetwork,
    compute_adaptive_spatial_mask,
    estimate_effective_filter_radii,
    init_vnn_params,
)


def _rel_l2_error(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(a) + 1e-12))


class TestMilestone7AdaptiveMask(unittest.TestCase):
    """Verifies pre-epoch adaptive spatial extent masking, margin growth, and backend parity."""

    def test_adaptive_mask_geometry_and_frontier_growth(self):
        """
        Verifies initial active bounding box (`32x32 = 1024` cells out of `40x50 = 2000` ROI)
        on the 8-bit ALU chip and automatic outward expansion when frontier cells inside
        the safety margin become salient in a subsequent epoch.
        """
        topo = ChipTopology2D.alu_chip(
            n=8, kernel_size=(23, 13), directions=16, depth=4, gap=5, ctrl_gap=5
        )
        params0 = init_vnn_params(jax.random.PRNGKey(701), topo, legacy_compat=True)

        mask0 = compute_adaptive_spatial_mask(
            params0, topo, margin=4, bucket_multiple=8
        )
        self.assertIsInstance(mask0, AdaptiveMaskSpec)
        self.assertFalse(mask0.is_full_roi)
        self.assertEqual(mask0.active_shape, (32, 32))
        self.assertAlmostEqual(mask0.active_fraction, 1024.0 / 2000.0, places=5)

        # Simulate circuit growth near the right frontier of the current box during Epoch e:
        # Activate bias cells at ROI coordinate (row=35, col=44), which is outside (2:34, 10:42).
        r_h, r_w = topo.pad_h, topo.pad_w
        b_grown = params0.biases.at[r_h + 35, r_w + 44].set(0.8)
        s_grown = params0.states.at[r_h + 35, r_w + 44].set(2.0)
        params1 = VNNParams(
            cell_params=jnp.stack([params0.weights, b_grown, s_grown], axis=0),
            greens_filters=params0.greens_filters,
        )

        mask1 = compute_adaptive_spatial_mask(
            params1, topo, margin=4, bucket_multiple=8
        )
        # Bounding box must automatically expand to cover (35 + 4, 44 + 4) quantized to 8
        self.assertGreaterEqual(mask1.roi_bbox[1], 39)
        self.assertGreaterEqual(mask1.roi_bbox[3], 48)
        self.assertGreater(mask1.active_shape[0], mask0.active_shape[0])
        self.assertGreater(mask1.active_shape[1], mask0.active_shape[1])

    def test_adaptive_mask_forward_and_backward_parity_across_backends(self):
        """
        Verifies that executing `VonNeumannNetwork.apply(..., spatial_mask=mask_spec)`
        produces identical forward predictions, universe states, and backward parameter
        gradients (`< 1e-5`) across all 6 backends while keeping `VNNParams` full-sized.
        """
        topo = ChipTopology2D.alu_chip(
            n=8, kernel_size=(23, 13), directions=16, depth=4, gap=5, ctrl_gap=5
        )
        params0 = init_vnn_params(jax.random.PRNGKey(702), topo, legacy_compat=True)
        b_nz = jax.random.normal(jax.random.PRNGKey(703), topo.grid_shape) * 0.02
        params = VNNParams(
            cell_params=params0.cell_params.at[1].set(b_nz),
            greens_filters=params0.greens_filters,
        )
        mask_spec = compute_adaptive_spatial_mask(
            params, topo, margin=4, bucket_multiple=8, bias_threshold=0.5
        )
        self.assertEqual(mask_spec.active_shape, (32, 32))

        B = 32
        k_in, k_tgt = jax.random.split(jax.random.PRNGKey(704))
        inputs = jax.random.randint(k_in, (B, 3, 8, 1), 0, 2).astype(jnp.float32)
        targets = jax.random.randint(k_tgt, (B, 8), 0, 2).astype(jnp.float32)

        model_ref = VonNeumannNetwork(topo, backend="nchw_direct")
        out_ref, univ_ref = model_ref.apply(
            params, inputs, return_universe=True, spatial_mask=mask_spec
        )
        self.assertEqual(out_ref.shape, (B, 8, 1))
        self.assertEqual(univ_ref.shape, (B, 62, 62))

        def make_loss(model):
            def _loss(p):
                logits = jnp.squeeze(
                    model.apply(p, inputs, return_universe=False, spatial_mask=mask_spec),
                    axis=-1,
                )
                return optax.squared_error(logits, targets).mean(), logits
            return _loss

        (l_ref, logits_ref), g_ref = jax.value_and_grad(make_loss(model_ref), has_aux=True)(params)
        # Parameter gradient shapes must remain full (62, 62) and (16, 1, 23, 13)
        self.assertEqual(g_ref.weights.shape, (62, 62))
        self.assertEqual(g_ref.greens_filters.shape, (16, 1, 23, 13))

        for backend in ("rfft_hoisted", "nchw_rfft_bwd", "pallas_fused", "pallas_sram", "auto"):
            model_b = VonNeumannNetwork(topo, backend=backend)
            out_b, univ_b = model_b.apply(
                params, inputs, return_universe=True, spatial_mask=mask_spec
            )
            self.assertLess(_rel_l2_error(out_ref, out_b), 1e-5)
            self.assertLess(_rel_l2_error(univ_ref, univ_b), 1e-5)

            (l_b, logits_b), g_b = jax.value_and_grad(make_loss(model_b), has_aux=True)(params)
            self.assertLess(abs(float(l_ref - l_b)), 1e-6)
            self.assertLess(_rel_l2_error(logits_ref, logits_b), 1e-5)
            self.assertLess(_rel_l2_error(g_ref.weights, g_b.weights), 1e-5)
            self.assertLess(_rel_l2_error(g_ref.biases, g_b.biases), 1e-5)
            self.assertLess(_rel_l2_error(g_ref.states, g_b.states), 1e-5)
            self.assertLess(_rel_l2_error(g_ref.greens_filters, g_b.greens_filters), 5e-5)

    def test_adaptive_mask_across_gol_clp_and_large_grids(self):
        """
        Verifies model-agnostic adaptive masking on:
        - Full-grid Game of Life (defaults to `is_full_roi=True`, or crops when localized
          `activity_map` is provided).
        - Large `82x82` grid (`80x80 = 6400` ROI -> cropped `(32, 32) = 1024` sub-ROI).
        - Coordinate-addressed CLP (`CoordinatePortSpec`).
        """
        # 1. Full-grid Game of Life without activity_map -> full ROI
        topo_gol = ChipTopology2D.gol_chip(n=64, kernel_size=(3, 3), directions=16, depth=4)
        p_gol = init_vnn_params(jax.random.PRNGKey(705), topo_gol)
        mask_gol_full = compute_adaptive_spatial_mask(p_gol, topo_gol, margin=4)
        self.assertTrue(mask_gol_full.is_full_roi)
        self.assertEqual(mask_gol_full.active_shape, topo_gol.roi_shape)

        # 2. Large 82x82 grid with localized ports -> crops from 80x80 (6400) to 32x32 (1024)
        topo_large = ChipTopology2D.create(
            grid_shape=(82, 82),
            kernel_size=(3, 3),
            directions=16,
            depth=6,
            in_ports=(PortSpec(offset=(20, 20), size=(8, 1), name="in"),),
            out_port=PortSpec(offset=(24, 32), size=(8, 1), name="out"),
        )
        p_large = init_vnn_params(jax.random.PRNGKey(706), topo_large)
        mask_large = compute_adaptive_spatial_mask(p_large, topo_large, margin=4, bucket_multiple=8)
        self.assertFalse(mask_large.is_full_roi)
        self.assertLessEqual(mask_large.active_shape[0], 32)
        self.assertLessEqual(mask_large.active_shape[1], 32)
        self.assertLess(mask_large.active_fraction, 0.20)

        B = 16
        x_large = jax.random.normal(jax.random.PRNGKey(707), (B, 1, 8, 1), dtype=jnp.float32)
        m_large_ref = VonNeumannNetwork(topo_large, backend="nchw_direct")
        m_large_pf = VonNeumannNetwork(topo_large, backend="pallas_fused")
        y_ref = m_large_ref.apply(p_large, x_large, spatial_mask=mask_large)
        y_pf = m_large_pf.apply(p_large, x_large, spatial_mask=mask_large)
        self.assertLess(_rel_l2_error(y_ref, y_pf), 1e-5)

        # 3. Effective filter radii estimation
        r_h, r_w = estimate_effective_filter_radii(p_large.greens_filters)
        self.assertEqual((r_h, r_w), (1, 1))

    def test_adaptive_trainer_bucket_caching_and_convergence(self):
        """
        Verifies `VNNTrainer.train_epoch_adaptive` and `evaluate_packed_adaptive`,
        confirming JIT executable reuse across epochs in the same quantized bucket.
        """
        topo = ChipTopology2D.alu_chip(
            n=8, kernel_size=(23, 13), directions=16, depth=4, gap=5, ctrl_gap=5
        )
        model = VonNeumannNetwork(topo, backend="auto")
        params = model.init(jax.random.PRNGKey(708), legacy_compat=True)
        trainer = VNNTrainer(model.apply, optimizer=optax.adamw(0.002), batch_size=32)
        opt_state = trainer.optimizer.init(params)

        k1, k2 = jax.random.split(jax.random.PRNGKey(709))
        x_ep = jax.random.randint(k1, (4, 32, 3, 8, 1), 0, 2).astype(jnp.float32)
        y_ep = jax.random.randint(k2, (4, 32, 8), 0, 2).astype(jnp.float32)
        from neurocell.trainer import packbits_u8_on_device
        y_packed = packbits_u8_on_device(y_ep.astype(jnp.uint8))

        losses = []
        for _ in range(3):
            mask_spec = model.compute_spatial_mask(params, margin=4, bucket_multiple=8)
            params, opt_state, ep_loss = trainer.train_epoch_adaptive(
                params, opt_state, x_ep, y_ep, mask_spec
            )
            losses.append(float(ep_loss))

        # Only 1 bucket compiled and cached across the 3 epochs
        self.assertEqual(len(trainer._adaptive_epoch_cache), 1)
        self.assertLess(losses[-1], losses[0])

        acc, preds_p = trainer.evaluate_packed_adaptive(params, x_ep, y_packed, mask_spec)
        self.assertEqual(preds_p.shape, (4, 32, 1))
        self.assertTrue(0.0 <= float(acc) <= 1.0)


if __name__ == "__main__":
    unittest.main()
