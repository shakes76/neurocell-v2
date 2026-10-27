'''
Unit and numerical parity tests for `neurocell/kernels_pallas.py` (Milestone 4:
Custom JAX Pallas / Triton GPU kernels for fused epilogue and SRAM execution).

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

from neurocell.topology import ChipTopology2D
from neurocell.params import VNNParams, init_vnn_params, precompute_step_params
from neurocell.epilogue import fused_roi_step_epilogue, fused_output_port_epilogue
from neurocell.kernels_pallas import (
    pallas_fused_roi_step_epilogue,
    pallas_fused_output_port_epilogue,
)
from neurocell.model import VonNeumannNetwork
from neurocell.trainer import VNNTrainer


def _rel_l2_error(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(a) + 1e-12))


class TestKernelsPallas(unittest.TestCase):
    """Verifies forward and custom_vjp backward parity of Pallas Triton GPU kernels."""

    def test_pallas_fused_roi_step_epilogue_forward_and_vjp_parity(self):
        """
        Verifies `pallas_fused_roi_step_epilogue` against the JAX reference
        `fused_roi_step_epilogue` in both forward and backward passes.
        """
        topo = ChipTopology2D.alu_chip()
        params0 = init_vnn_params(jax.random.PRNGKey(11), topo, legacy_compat=True)
        b_nz = jax.random.normal(jax.random.PRNGKey(12), topo.grid_shape) * 0.25
        params = VNNParams(
            cell_params=params0.cell_params.at[1].set(b_nz),
            greens_filters=params0.greens_filters,
        )
        step_p = precompute_step_params(params, topo, activate=jax.nn.tanh)

        B, M = 64, topo.directions
        roi_h, roi_w = topo.roi_shape
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(101), 3)
        z_roi = jax.random.normal(k1, (B, M, roi_h, roi_w), dtype=jnp.float32)
        delta_border = jax.random.normal(k2, (1, M, roi_h, roi_w), dtype=jnp.float32)
        dy = jax.random.normal(k3, (B, roi_h, roi_w), dtype=jnp.float32)

        def ref_fn(z, db, w, s, b):
            z_tot = z + db
            z_norm = jax.nn.standardize(z_tot, axis=(-2, -1))
            sum_lin = jnp.sum(z_norm, axis=1)
            sum_act = jnp.sum(jnp.tanh(z_norm + b[jnp.newaxis, jnp.newaxis, :, :]), axis=1)
            out = w[jnp.newaxis, :, :] * sum_lin + s[jnp.newaxis, :, :] * sum_act
            return jnp.sum(out * dy), out

        def pal_fn(z, db, w, s, b):
            out = pallas_fused_roi_step_epilogue(z, db, w, s, b)
            return jnp.sum(out * dy), out

        grads_ref, out_ref = jax.grad(ref_fn, argnums=(0, 1, 2, 3, 4), has_aux=True)(
            z_roi, delta_border, step_p.w_eff_roi, step_p.s_eff_roi, step_p.b_roi
        )
        grads_pal, out_pal = jax.grad(pal_fn, argnums=(0, 1, 2, 3, 4), has_aux=True)(
            z_roi, delta_border, step_p.w_eff_roi, step_p.s_eff_roi, step_p.b_roi
        )

        self.assertLess(_rel_l2_error(out_ref, out_pal), 1e-6)
        for g_r, g_p in zip(grads_ref, grads_pal):
            self.assertLess(_rel_l2_error(g_r, g_p), 1e-6)

    def test_pallas_fused_output_port_epilogue_forward_and_vjp_parity(self):
        """
        Verifies `pallas_fused_output_port_epilogue` against `fused_output_port_epilogue`
        for both linear (`final_activate=None`) and nonlinear (`final_activate=jax.nn.tanh`)
        output ports.
        """
        topo = ChipTopology2D.alu_chip()
        params0 = init_vnn_params(jax.random.PRNGKey(21), topo, legacy_compat=True)
        b_nz = jax.random.normal(jax.random.PRNGKey(22), topo.grid_shape) * 0.25
        params = VNNParams(
            cell_params=params0.cell_params.at[1].set(b_nz),
            greens_filters=params0.greens_filters,
        )

        B, M = 64, topo.directions
        roi_h, roi_w = topo.roi_shape
        out_h, out_w = topo.out_port.size
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(202), 3)
        z_final = jax.random.normal(k1, (B, M, roi_h, roi_w), dtype=jnp.float32)
        delta_border = jax.random.normal(k2, (1, M, roi_h, roi_w), dtype=jnp.float32)
        dy_out = jax.random.normal(k3, (B, out_h, out_w), dtype=jnp.float32)

        for final_act in (None, jax.nn.tanh):
            def ref_loss(z, db, p):
                sp = precompute_step_params(p, topo, activate=jax.nn.tanh)
                out = fused_output_port_epilogue(
                    z + db, sp, topo, final_activate=final_act, norm=True
                )
                return jnp.sum(out * dy_out), out

            def pal_loss(z, db, p):
                sp = precompute_step_params(p, topo, activate=jax.nn.tanh)
                out = pallas_fused_output_port_epilogue(
                    z, db, sp, topo, final_activate=final_act
                )
                return jnp.sum(out * dy_out), out

            (gz_ref, gdb_ref, gp_ref), out_ref = jax.grad(
                ref_loss, argnums=(0, 1, 2), has_aux=True
            )(z_final, delta_border, params)
            (gz_pal, gdb_pal, gp_pal), out_pal = jax.grad(
                pal_loss, argnums=(0, 1, 2), has_aux=True
            )(z_final, delta_border, params)

            self.assertLess(_rel_l2_error(out_ref, out_pal), 1e-6)
            self.assertLess(_rel_l2_error(gz_ref, gz_pal), 1e-6)
            self.assertLess(_rel_l2_error(gdb_ref, gdb_pal), 1e-6)
            self.assertLess(_rel_l2_error(gp_ref.cell_params, gp_pal.cell_params), 1e-6)

    def test_vnn_alu_pallas_fused_and_pallas_sram_end_to_end_parity(self):
        """
        Verifies end-to-end forward, universe-state, backward gradient, and multi-step
        AdamW training trajectory parity of `pallas_fused` and `pallas_sram` against `nchw_direct`.
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

        model_ref = VonNeumannNetwork(topo, backend="nchw_direct")
        model_pf = VonNeumannNetwork(topo, backend="pallas_fused")
        model_ps = VonNeumannNetwork(topo, backend="pallas_sram")

        # 1. Forward and universe parity
        out_ref, univ_ref = model_ref.apply(params, inputs, return_universe=True)
        out_pf, univ_pf = model_pf.apply(params, inputs, return_universe=True)
        out_ps, univ_ps = model_ps.apply(params, inputs, return_universe=True)

        self.assertLess(_rel_l2_error(out_ref, out_pf), 1e-6)
        self.assertLess(_rel_l2_error(univ_ref, univ_pf), 1e-6)
        self.assertLess(_rel_l2_error(out_ref, out_ps), 1e-6)
        self.assertLess(_rel_l2_error(univ_ref, univ_ps), 1e-6)

        # 2. Stateless output port forward & gradient parity
        def make_loss(model):
            def loss_fn(p):
                logits = jnp.squeeze(model.apply(p, inputs, return_universe=False), axis=-1)
                return optax.squared_error(logits, targets).mean()
            return loss_fn

        l_ref, g_ref = jax.value_and_grad(make_loss(model_ref))(params)
        l_pf, g_pf = jax.value_and_grad(make_loss(model_pf))(params)
        l_ps, g_ps = jax.value_and_grad(make_loss(model_ps))(params)

        self.assertLess(abs(float(l_ref - l_pf)), 1e-6)
        self.assertLess(abs(float(l_ref - l_ps)), 1e-6)

        for g_test in (g_pf, g_ps):
            self.assertLess(_rel_l2_error(g_ref.weights, g_test.weights), 1e-5)
            self.assertLess(_rel_l2_error(g_ref.biases, g_test.biases), 1e-5)
            self.assertLess(_rel_l2_error(g_ref.states, g_test.states), 1e-5)
            self.assertLess(_rel_l2_error(g_ref.greens_filters, g_test.greens_filters), 5e-5)

        # 3. 5-step AdamW trajectory parity via VNNTrainer
        trainer_ref = VNNTrainer(model_ref.apply, optimizer=optax.adamw(0.001))
        trainer_pf = VNNTrainer(model_pf.apply, optimizer=optax.adamw(0.001))
        p_ref = params
        p_pf = params
        opt_ref = trainer_ref.optimizer.init(p_ref)
        opt_pf = trainer_pf.optimizer.init(p_pf)

        for _ in range(5):
            p_ref, opt_ref, loss_r = trainer_ref.train_step(p_ref, opt_ref, inputs, targets)
            p_pf, opt_pf, loss_p = trainer_pf.train_step(p_pf, opt_pf, inputs, targets)
            self.assertLess(abs(float(loss_r - loss_p)), 1e-5)

        self.assertLess(_rel_l2_error(p_ref.cell_params, p_pf.cell_params), 1e-5)
        self.assertLess(_rel_l2_error(p_ref.greens_filters, p_pf.greens_filters), 5e-5)


if __name__ == "__main__":
    unittest.main()
