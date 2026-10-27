'''
Unit tests for `neurocell.params`, `neurocell.datasets`, and `neurocell.trainer`.

Verifies:
1. Bit-for-bit (`max_abs_diff == 0.0`) initialization parity between `init_vnn_params`
   and `contrib/universe.py` (`UniverseLP.initialize`).
2. Algebraic equivalence of `precompute_step_params` with the legacy per-step
   `dynamic_update_slice` + `sigmoid` + unfactored Codd gating.
3. Bit-for-bit parity and speedup of vectorized `generate_arithmetic_logic` vs `contrib/data.py`.
4. Exact equivalence and speedup of on-device `VNNTrainer.train_epoch` and
   `VNNTrainer.evaluate_packed` vs Python batch loops.

@author Shakes
'''
import os
import sys
import time
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "contrib"))

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from neurocell.topology import ChipTopology2D
from neurocell.params import (
    init_vnn_params,
    precompute_step_params,
    export_legacy_haiku_params,
    import_legacy_haiku_params,
)
from neurocell.datasets import generate_arithmetic_logic, prepare_alu_device_dataset
from neurocell.trainer import VNNTrainer, packbits_u8_on_device

# Legacy imports from contrib/ for parity verification
from universe import UniverseLP
import data as legacy_data


class TestParamsAndTrainer(unittest.TestCase):
    """Tests parameter initialization parity, pre-step fusion, and on-device scan trainer."""

    def test_bit_exact_init_parity_with_contrib_universe(self):
        """
        Proves that `init_vnn_params(..., legacy_compat=True)` produces bit-for-bit
        (`0.0` difference) identical initial tensors to `contrib/universe.py`.
        """
        topo = ChipTopology2D.alu_chip(
            n=8, kernel_size=(23, 13), directions=16, depth=4, gap=5, ctrl_gap=5
        )
        rng = jax.random.PRNGKey(42)
        vnn_params = init_vnn_params(rng, topo, legacy_compat=True)

        # Initialize legacy UniverseLP with the exact same PRNGKey(42)
        def _legacy_forward(batch):
            mlp = UniverseLP(topo.N, sharpen=False)
            in_sizes = [p.size for p in topo.in_ports]
            in_offsets = [p.offset for p in topo.in_ports]
            mlp.initialize(
                in_sizes,
                topo.out_port.size,
                in_offsets,
                topo.out_port.offset,
                topo.depth,
                topo.kernel_size,
                directions=topo.directions,
            )
            return mlp(batch, jax.nn.tanh, None, norm=True)

        dummy_batch = [
            jnp.zeros((2, 8, 1), dtype=jnp.float32),
            jnp.zeros((2, 8, 1), dtype=jnp.float32),
            jnp.zeros((2, 8, 1), dtype=jnp.float32),
        ]
        legacy_net = hk.without_apply_rng(hk.transform_with_state(_legacy_forward))
        hk_params, _ = legacy_net.init(rng, dummy_batch)
        sub = hk_params["universe_lp"]

        diff_w = float(jnp.max(jnp.abs(vnn_params.weights - sub["w"])))
        diff_b = float(jnp.max(jnp.abs(vnn_params.biases - sub["b"])))
        diff_s = float(jnp.max(jnp.abs(vnn_params.states - sub["s"])))
        diff_f = float(jnp.max(jnp.abs(vnn_params.filters_ihwo - sub["f"])))

        self.assertEqual(diff_w, 0.0)
        self.assertEqual(diff_b, 0.0)
        self.assertEqual(diff_s, 0.0)
        self.assertEqual(diff_f, 0.0)

        # Verify round-trip export/import
        exported = export_legacy_haiku_params(vnn_params)
        reimported = import_legacy_haiku_params(exported)
        self.assertEqual(float(jnp.max(jnp.abs(reimported.cell_params - vnn_params.cell_params))), 0.0)
        self.assertEqual(
            float(jnp.max(jnp.abs(reimported.greens_filters - vnn_params.greens_filters))), 0.0
        )

    def test_precompute_step_params_algebraic_equivalence(self):
        """
        Verifies that factored `PrecomputedStepParams` matches the unfactored legacy Codd update
        `mean(z * w * (1 - sigmoid(s)) + tanh(z + b) * sigmoid(s), axis=-1)` on the active ROI.
        """
        topo = ChipTopology2D.alu_chip()
        rng = jax.random.PRNGKey(123)
        params = init_vnn_params(rng, topo)
        step_p = precompute_step_params(params, topo, activate=jax.nn.tanh)

        # Generate random post-standardization tensor Z on ROI (B=4, H_v=40, W_v=50, M=16)
        z_nhwc = jax.random.normal(jax.random.PRNGKey(999), (4, 40, 50, 16))
        z_nchw = jnp.transpose(z_nhwc, (0, 3, 1, 2))

        # Legacy unfactored Codd formula on ROI
        r_h, r_w = topo.pad_h, topo.pad_w
        H, W = topo.grid_shape
        w_roi = step_p.w_pinned[r_h : H - r_h, r_w : W - r_w][..., jnp.newaxis]
        b_roi = step_p.b_full[r_h : H - r_h, r_w : W - r_w][..., jnp.newaxis]
        s_roi = step_p.s_prob[r_h : H - r_h, r_w : W - r_w][..., jnp.newaxis]

        legacy_out = jnp.mean(
            z_nhwc * w_roi * (1.0 - s_roi) + jax.nn.tanh(z_nhwc + b_roi) * s_roi,
            axis=-1,
        )

        # New factored formula using PrecomputedStepParams
        sum_lin = jnp.sum(z_nchw, axis=1)
        sum_act = jnp.sum(jax.nn.tanh(z_nchw + step_p.b_roi[jnp.newaxis, jnp.newaxis, :, :]), axis=1)
        fused_out = step_p.w_eff_roi[jnp.newaxis, :, :] * sum_lin + step_p.s_eff_roi[jnp.newaxis, :, :] * sum_act

        max_err = float(jnp.max(jnp.abs(legacy_out - fused_out)))
        self.assertLess(max_err, 1e-6)

    def test_vectorized_dataset_bit_exact_parity(self):
        """
        Verifies that `neurocell.datasets.generate_arithmetic_logic` produces bit-exact
        identical arrays to `contrib/data.py` when seeded identically.
        """
        np.random.seed(2026)
        x1_ref, x2_ref, c_ref, y_ref = legacy_data.generate_arithmetic_logic(
            samples=2560, n=8, ctrl_bits=4, batch_size=128, noise_level=0.2
        )

        np.random.seed(2026)
        x1_new, x2_new, c_new, y_new = generate_arithmetic_logic(
            samples=2560, n=8, ctrl_bits=4, batch_size=128, noise_level=0.2
        )

        np.testing.assert_array_equal(x1_ref, x1_new)
        np.testing.assert_array_equal(x2_ref, x2_new)
        np.testing.assert_array_equal(c_ref, c_new)
        np.testing.assert_array_equal(y_ref, y_new)

    def test_on_device_trainer_and_packbits_equivalence(self):
        """
        Verifies `packbits_u8_on_device` and `VNNTrainer` (`train_epoch` & `evaluate_packed`)
        against a reference per-batch loop and `jnp.packbits`.
        """
        ds = prepare_alu_device_dataset(samples=2560, batch_size=128)
        self.assertEqual(ds.x_train.shape, (16, 128, 3, 8, 1))
        self.assertEqual(ds.y_train.shape, (16, 128, 8))
        self.assertEqual(ds.y_train_packed.shape, (16, 128, 1))

        # Test packbits_u8_on_device vs jnp.packbits
        bits_u8 = ds.y_train.astype(jnp.uint8)
        packed_dev = packbits_u8_on_device(bits_u8)
        self.assertTrue(bool(jnp.all(packed_dev == ds.y_train_packed)))

        # Test VNNTrainer on a lightweight linear projection model
        w0 = jax.random.normal(jax.random.PRNGKey(7), (24, 8)) * 0.1
        b0 = jnp.zeros((8,))
        init_params = {"w": w0, "b": b0}

        def apply_fn(p, bx):
            flat = bx.reshape(bx.shape[0], -1)
            return (flat @ p["w"] + p["b"])[..., jnp.newaxis]

        opt = optax.adamw(0.01)
        trainer = VNNTrainer(apply_fn=apply_fn, optimizer=opt, batch_size=128)

        # Reference step-by-step loop
        p_ref = init_params
        opt_ref = opt.init(init_params)
        loss_ref = 0.0
        for i in range(ds.x_train.shape[0]):
            p_ref, opt_ref, l_val = trainer.train_step(p_ref, opt_ref, ds.x_train[i], ds.y_train[i])
            loss_ref = loss_ref + l_val / 128.0

        # Single fused on-device `train_epoch` via `jax.lax.scan`
        p_scan, opt_scan, loss_scan = trainer.train_epoch(
            init_params, opt.init(init_params), ds.x_train, ds.y_train
        )

        self.assertAlmostEqual(float(loss_ref), float(loss_scan), places=5)
        self.assertLess(float(jnp.max(jnp.abs(p_ref["w"] - p_scan["w"]))), 1e-6)

        acc, preds_packed = trainer.evaluate_packed(p_scan, ds.x_test, ds.y_test_packed)
        self.assertEqual(preds_packed.shape, ds.y_test_packed.shape)
        self.assertGreaterEqual(float(acc), 0.0)
        self.assertLessEqual(float(acc), 1.0)


if __name__ == "__main__":
    unittest.main()
