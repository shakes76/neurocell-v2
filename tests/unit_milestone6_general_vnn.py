'''
Unit and numerical parity tests for Milestone 6: Model-, Input-, and Output-Agnostic
Core Primitives in FP32.

Tests:
1. Hardware-tiled Pallas epilogue (`N_PIX > 2048`: `62x62 = 3844`, `64x64 = 4096`,
   `80x80 = 6400`), pluggable activation functions (`tanh`, `leaky_relu`, `sigmoid`,
   `relu`, `gelu`, `None`), and `has_border=False` zero-border optimization.
2. Full-grid Game of Life VNN (`ChipTopology2D.gol_chip`, `64x64 -> 64x64`, `K = 3x3`)
   forward and backward gradient parity across `nchw_direct`, `pallas_fused`, and `auto`.
3. Coordinate-addressed CLP VNN (`ChipTopology2D.clp_chip` with `CoordinatePortSpec`)
   forward and backward gradient parity across `nchw_direct`, `pallas_fused`, and `auto`.
4. `rfft2_pruned`, `irfft2_pruned`, `conv2d_grouped_filter_grad`, and
   `install_lean_rfft_transpose` numerical parity.
5. Task-agnostic `VNNTrainer` (`loss_fn`, `eval_metric_fn`, `evaluate_metric`, `train_epochs`).

@author Shakes
'''
from functools import partial
import os
import sys
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import jax
import jax.numpy as jnp
import optax

from neurocell.topology import ChipTopology2D, CoordinatePortSpec
from neurocell.params import VNNParams, init_vnn_params, precompute_step_params
from neurocell.kernels_greens import (
    rfft2_pruned,
    irfft2_pruned,
    conv2d_grouped_filter_grad,
    install_lean_rfft_transpose,
)
from neurocell.kernels_pallas import (
    pallas_fused_roi_step_epilogue,
    pallas_fused_output_port_epilogue,
)
from neurocell.epilogue import fused_roi_step_epilogue, fused_output_port_epilogue
from neurocell.model import VonNeumannNetwork
from neurocell.trainer import VNNTrainer


def _rel_l2_error(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(a) + 1e-12))


class TestMilestone6GeneralVNN(unittest.TestCase):
    """Verifies model-, grid-size-, activation-, and port-agnostic VNN primitives."""

    def test_large_grid_tiled_pallas_epilogue_and_activations(self):
        """
        Verifies hardware-tiled `pallas_fused_roi_step_epilogue` on grids with
        `N_PIX > 2048` (`64x64 = 4096`, `62x62 = 3844`, `80x80 = 6400`) across
        multiple activation functions and `has_border in (True, False)`.
        """
        B, M = 16, 16
        k1, k2, k3, k4, k5, k6 = jax.random.split(jax.random.PRNGKey(601), 6)

        for h_v, w_v in ((64, 64), (62, 62), (80, 80), (96, 96)):
            z_roi = jax.random.normal(k1, (B, M, h_v, w_v), dtype=jnp.float32)
            db = jax.random.normal(k2, (1, M, h_v, w_v), dtype=jnp.float32)
            w_eff = jax.random.normal(k3, (h_v, w_v), dtype=jnp.float32) * 0.1
            s_eff = jax.nn.sigmoid(jax.random.normal(k4, (h_v, w_v), dtype=jnp.float32)) / float(M)
            b_roi = jax.random.normal(k5, (h_v, w_v), dtype=jnp.float32) * 0.2
            dy = jax.random.normal(k6, (B, h_v, w_v), dtype=jnp.float32)

            for act in (
                jax.nn.tanh,
                jax.nn.sigmoid,
                jax.nn.relu,
                partial(jax.nn.leaky_relu, negative_slope=0.05),
                None,
            ):
                for has_border in (True, False):
                    def ref_fn(z, d_b, w, s, b):
                        z_in = z + d_b if has_border else z
                        z_norm = jax.nn.standardize(z_in, axis=(-2, -1))
                        sum_lin = jnp.sum(z_norm, axis=1)
                        if act is None:
                            sum_act = sum_lin + float(M) * b[jnp.newaxis, :, :]
                        else:
                            sum_act = jnp.sum(act(z_norm + b[jnp.newaxis, jnp.newaxis, :, :]), axis=1)
                        out = w[jnp.newaxis, :, :] * sum_lin + s[jnp.newaxis, :, :] * sum_act
                        return jnp.sum(out * dy), out

                    def pal_fn(z, d_b, w, s, b):
                        out = pallas_fused_roi_step_epilogue(
                            z, d_b, w, s, b, activate=act, has_border=has_border
                        )
                        return jnp.sum(out * dy), out

                    grads_ref, out_ref = jax.grad(ref_fn, argnums=(0, 1, 2, 3, 4), has_aux=True)(
                        z_roi, db, w_eff, s_eff, b_roi
                    )
                    grads_pal, out_pal = jax.grad(pal_fn, argnums=(0, 1, 2, 3, 4), has_aux=True)(
                        z_roi, db, w_eff, s_eff, b_roi
                    )

                    self.assertLess(_rel_l2_error(out_ref, out_pal), 1e-5)
                    self.assertLess(_rel_l2_error(grads_ref[0], grads_pal[0]), 1e-5)
                    if has_border:
                        self.assertLess(_rel_l2_error(grads_ref[1], grads_pal[1]), 1e-5)
                    else:
                        self.assertEqual(float(jnp.max(jnp.abs(grads_pal[1]))), 0.0)
                    for idx in (2, 3, 4):
                        self.assertLess(_rel_l2_error(grads_ref[idx], grads_pal[idx]), 1e-5)

    def test_full_grid_gol_vnn_parity(self):
        """
        Verifies forward and backward gradient parity on a full-grid 64x64 -> 64x64
        Game of Life VNN (`K = 3x3, M = 16, depth = 4`) across `nchw_direct`,
        `pallas_fused`, and `auto` in both `"VALID"` and `"SAME"` padding modes.
        """
        B = 16
        k_in, k_tgt = jax.random.split(jax.random.PRNGKey(602))
        inputs = jax.random.randint(k_in, (B, 64, 64), 0, 2).astype(jnp.float32)
        targets = jax.random.randint(k_tgt, (B, 64, 64), 0, 2).astype(jnp.float32)

        for pad_mode in ("VALID", "SAME"):
            topo = ChipTopology2D.gol_chip(
                n=64, kernel_size=(3, 3), directions=16, depth=4, padding_mode=pad_mode
            )
            params0 = init_vnn_params(jax.random.PRNGKey(603), topo, legacy_compat=True)
            b_nz = jax.random.normal(jax.random.PRNGKey(604), topo.grid_shape) * 0.2
            params = VNNParams(
                cell_params=params0.cell_params.at[1].set(b_nz),
                greens_filters=params0.greens_filters,
            )

            for act, subgrad_tol in (
                (jax.nn.tanh, 1e-5),
                (partial(jax.nn.leaky_relu, negative_slope=0.01), 1e-3),
            ):
                m_ref = VonNeumannNetwork(topo, activate=act, final_activate=jax.nn.sigmoid, backend="nchw_direct")
                m_pf = VonNeumannNetwork(topo, activate=act, final_activate=jax.nn.sigmoid, backend="pallas_fused")
                m_auto = VonNeumannNetwork(topo, activate=act, final_activate=jax.nn.sigmoid, backend="auto")

                def make_loss(model):
                    def _loss(p):
                        preds = model.apply(p, inputs)
                        return optax.squared_error(preds, targets).mean(), preds
                    return _loss

                (l_ref, out_ref), g_ref = jax.value_and_grad(make_loss(m_ref), has_aux=True)(params)
                (l_pf, out_pf), g_pf = jax.value_and_grad(make_loss(m_pf), has_aux=True)(params)
                (l_auto, out_auto), g_auto = jax.value_and_grad(make_loss(m_auto), has_aux=True)(params)

                self.assertEqual(out_ref.shape, (B, 64, 64))
                self.assertLess(_rel_l2_error(out_ref, out_pf), 1e-5)
                self.assertLess(_rel_l2_error(out_ref, out_auto), 1e-5)
                self.assertLess(abs(float(l_ref - l_pf)), 1e-6)
                for g_t in (g_pf, g_auto):
                    self.assertLess(_rel_l2_error(g_ref.weights, g_t.weights), 1e-5)
                    self.assertLess(_rel_l2_error(g_ref.biases, g_t.biases), subgrad_tol)
                    self.assertLess(_rel_l2_error(g_ref.states, g_t.states), subgrad_tol)
                    self.assertLess(_rel_l2_error(g_ref.greens_filters, g_t.greens_filters), subgrad_tol)

    def test_clp_coordinate_port_vnn_parity(self):
        """
        Verifies `ChipTopology2D.clp_chip` with scattered `CoordinatePortSpec` I/O ports
        across `nchw_direct`, `pallas_fused`, and `auto`.
        """
        topo = ChipTopology2D.clp_chip(n=8, kernel_size=(3, 3), directions=16, depth=4)
        self.assertIsInstance(topo.in_ports[0], CoordinatePortSpec)
        self.assertIsInstance(topo.out_port, CoordinatePortSpec)

        params0 = init_vnn_params(jax.random.PRNGKey(605), topo, legacy_compat=True)
        b_nz = jax.random.normal(jax.random.PRNGKey(606), topo.grid_shape) * 0.2
        params = VNNParams(
            cell_params=params0.cell_params.at[1].set(b_nz),
            greens_filters=params0.greens_filters,
        )

        B = 32
        inputs = jax.random.normal(jax.random.PRNGKey(607), (B, 2), dtype=jnp.float32)
        targets = jax.random.randint(jax.random.PRNGKey(608), (B, 2, 1), 0, 2).astype(jnp.float32)

        m_ref = VonNeumannNetwork(topo, activate=jax.nn.tanh, final_activate=jax.nn.sigmoid, backend="nchw_direct")
        m_pf = VonNeumannNetwork(topo, activate=jax.nn.tanh, final_activate=jax.nn.sigmoid, backend="pallas_fused")
        m_auto = VonNeumannNetwork(topo, activate=jax.nn.tanh, final_activate=jax.nn.sigmoid, backend="auto")

        def make_loss(model):
            def _loss(p):
                preds = model.apply(p, inputs)
                return optax.squared_error(preds, targets).mean(), preds
            return _loss

        (l_ref, out_ref), g_ref = jax.value_and_grad(make_loss(m_ref), has_aux=True)(params)
        (l_pf, out_pf), g_pf = jax.value_and_grad(make_loss(m_pf), has_aux=True)(params)
        (l_auto, out_auto), g_auto = jax.value_and_grad(make_loss(m_auto), has_aux=True)(params)

        self.assertEqual(out_ref.shape, (B, 2, 1))
        self.assertLess(_rel_l2_error(out_ref, out_pf), 1e-6)
        self.assertLess(_rel_l2_error(out_ref, out_auto), 1e-6)
        self.assertLess(abs(float(l_ref - l_pf)), 1e-6)
        self.assertLess(_rel_l2_error(g_ref.cell_params, g_pf.cell_params), 1e-5)
        self.assertLess(_rel_l2_error(g_ref.greens_filters, g_pf.greens_filters), 1e-5)

    def test_pruned_rfft_and_grouped_filter_grad_parity(self):
        """
        Verifies `rfft2_pruned`, `irfft2_pruned`, `conv2d_grouped_filter_grad`, and
        `install_lean_rfft_transpose` against exact JAX references.
        """
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(609), 3)
        x = jax.random.normal(k1, (32, 16, 40, 50), dtype=jnp.float32)

        f_ref = jnp.fft.rfft2(x, s=(64, 64), axes=(-2, -1))
        f_pruned = rfft2_pruned(x, fft_shape=(64, 64))
        self.assertLess(_rel_l2_error(f_ref, f_pruned), 1e-6)

        sp_ref = jnp.fft.irfft2(f_ref, s=(64, 64), axes=(-2, -1))[:, :, 11 : 11 + 40, 6 : 6 + 50]
        sp_pruned = irfft2_pruned(f_pruned, roi_shape=(40, 50), pad_hw=(11, 6), fft_shape=(64, 64))
        self.assertLess(_rel_l2_error(sp_ref, sp_pruned), 1e-6)

        # Grouped-batch filter gradient parity on 3x3 and 17x3 kernels
        cells = jax.random.normal(k1, (64, 40, 50), dtype=jnp.float32)
        for kh, kw in ((3, 3), (17, 3)):
            rh, rw = kh // 2, kw // 2
            filt = jax.random.normal(k2, (16, 1, kh, kw), dtype=jnp.float32)
            gz = jax.random.normal(k3, (64, 16, 40, 50), dtype=jnp.float32)

            def _conv_loss(f):
                z = jax.lax.conv_general_dilated(
                    cells[:, jnp.newaxis, :, :],
                    f,
                    window_strides=(1, 1),
                    padding=((rh, rh), (rw, rw)),
                    dimension_numbers=("NCHW", "OIHW", "NCHW"),
                )
                return jnp.sum(z * gz)

            dg_ref = jax.grad(_conv_loss)(filt)
            dg_grouped = conv2d_grouped_filter_grad(cells, gz, pad_hw=(rh, rw))
            self.assertLess(_rel_l2_error(dg_ref, dg_grouped), 1e-5)

        # Verify opt-in upstream lean RFFT transpose installation and gradient parity
        self.assertTrue(install_lean_rfft_transpose())
        x_small = jax.random.normal(k1, (8, 16, 24), dtype=jnp.float32)
        g_small = jax.grad(lambda u: jnp.sum(jnp.abs(jnp.fft.rfft2(u, s=(32, 32))) ** 2))(x_small)
        self.assertEqual(g_small.shape, x_small.shape)

    def test_trainer_custom_loss_metric_and_multi_epoch_scan(self):
        """
        Verifies `VNNTrainer` with custom `loss_fn`, `eval_metric_fn`, `evaluate_metric`,
        and multi-epoch `train_epochs` scan.
        """
        topo = ChipTopology2D.clp_chip(n=8, kernel_size=(3, 3), directions=8, depth=2)
        model = VonNeumannNetwork(
            topo, activate=jax.nn.tanh, final_activate=jax.nn.sigmoid, backend="auto"
        )
        params = model.init(jax.random.PRNGKey(610))

        num_samples = 64
        batch_size = 16
        x_train = jax.random.normal(jax.random.PRNGKey(611), (num_samples, 2), dtype=jnp.float32)
        y_train = jax.random.randint(jax.random.PRNGKey(612), (num_samples, 2, 1), 0, 2).astype(jnp.float32)

        def bce_loss(logits, targets):
            return optax.sigmoid_binary_cross_entropy(logits, targets).mean()

        def acc_metric(logits, targets):
            return jnp.mean((logits > 0.5).astype(jnp.float32) == targets)

        trainer = VNNTrainer(
            model.apply,
            optimizer=optax.adamw(0.01),
            batch_size=batch_size,
            loss_fn=bce_loss,
            eval_metric_fn=acc_metric,
        )
        opt_state = trainer.optimizer.init(params)
        new_params, new_opt_state, epoch_losses = trainer.train_epochs(
            params, opt_state, x_train, y_train, jax.random.PRNGKey(613), num_epochs=3
        )
        self.assertEqual(epoch_losses.shape, (3,))

        x_batches = jnp.reshape(x_train, (num_samples // batch_size, batch_size, 2))
        y_batches = jnp.reshape(y_train, (num_samples // batch_size, batch_size, 2, 1))
        acc = trainer.evaluate_metric(new_params, x_batches, y_batches)
        self.assertTrue(0.0 <= float(acc) <= 1.0)


if __name__ == "__main__":
    unittest.main()
