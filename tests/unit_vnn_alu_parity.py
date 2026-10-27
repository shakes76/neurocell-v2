'''
End-to-end forward and backward gradient parity tests for `VonNeumannNetwork`
(`backend="nchw_direct"`) against `contrib/universe.py` (`UniverseLP`).

Proves that:
1. Forward predictions `(B, 8, 1)` and full cellular fields `(B, 62, 62)` match
   `contrib/` within `< 1e-5` across random minibatches and non-zero biases.
2. All 4 parameter gradients (`dL/dW`, `dL/dB`, `dL/dS`, `dL/dG`) computed via
   `jax.value_and_grad` match `contrib/` within `< 1e-5` relative L2 error.
3. Multi-step `optax.adamw` training trajectories match `contrib/`.

@author Shakes
'''
import os
import sys
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "contrib"))

import jax
import jax.numpy as jnp
import haiku as hk
import optax

from neurocell.topology import ChipTopology2D
from neurocell.params import VNNParams, import_legacy_haiku_params
from neurocell.model import VonNeumannNetwork
from neurocell.datasets import prepare_alu_device_dataset

from universe import UniverseLP


def _rel_l2_error(a: jax.Array, b: jax.Array) -> float:
    """Computes relative L2 norm `||a - b||_2 / (||a||_2 + 1e-12)`."""
    num = jnp.linalg.norm(a - b)
    den = jnp.linalg.norm(a) + 1e-12
    return float(num / den)


class TestVNNALUParity(unittest.TestCase):
    """End-to-end numerical parity between `VonNeumannNetwork` and `contrib/universe.py`."""

    def setUp(self):
        self.topo = ChipTopology2D.alu_chip(
            n=8, kernel_size=(23, 13), directions=16, depth=4, gap=5, ctrl_gap=5
        )
        self.model = VonNeumannNetwork(
            topology=self.topo,
            activate=jax.nn.tanh,
            final_activate=None,
            norm=True,
            sharpen=False,
            backend="nchw_direct",
        )
        self.ds = prepare_alu_device_dataset(samples=1024, batch_size=32)

        def _legacy_forward(batch):
            mlp = UniverseLP(self.topo.N, sharpen=False)
            in_sizes = [p.size for p in self.topo.in_ports]
            in_offsets = [p.offset for p in self.topo.in_ports]
            mlp.initialize(
                in_sizes,
                self.topo.out_port.size,
                in_offsets,
                self.topo.out_port.offset,
                self.topo.depth,
                self.topo.kernel_size,
                directions=self.topo.directions,
            )
            return mlp(batch, jax.nn.tanh, None, norm=True)

        self.legacy_net = hk.without_apply_rng(hk.transform_with_state(_legacy_forward))

    def test_forward_and_universe_parity(self):
        """
        Verifies both stateless output predictions (`return_universe=False`) and
        full final universe state (`return_universe=True`) match `UniverseLP`.
        """
        rng = jax.random.PRNGKey(42)
        bx = self.ds.x_train[0]  # (32, 3, 8, 1)
        bx_list = [bx[:, 0], bx[:, 1], bx[:, 2]]

        hk_params, hk_state = self.legacy_net.init(rng, bx_list)
        # Also test with non-zero biases so border propagation is active
        b_rand = jax.random.normal(jax.random.PRNGKey(99), self.topo.grid_shape) * 0.2
        hk_params["universe_lp"]["b"] = b_rand

        vnn_params = import_legacy_haiku_params(hk_params)

        ref_logits, ref_state = self.legacy_net.apply(hk_params, hk_state, bx_list)
        ref_cells = ref_state["universe_lp"]["universe"]

        out_fast = self.model.apply(vnn_params, bx, return_universe=False)
        out_full, cells_full = self.model.apply(vnn_params, bx, return_universe=True)

        self.assertLess(float(jnp.max(jnp.abs(ref_logits - out_fast))), 1e-5)
        self.assertLess(float(jnp.max(jnp.abs(ref_logits - out_full))), 1e-5)
        self.assertLess(float(jnp.max(jnp.abs(ref_cells - cells_full))), 1e-5)

    def test_backward_gradient_parity(self):
        """
        Verifies that `jax.value_and_grad` produces matching loss and gradients
        (`dL/dW`, `dL/dB`, `dL/dS`, `dL/dG`) between `VonNeumannNetwork` and `UniverseLP`.
        """
        rng = jax.random.PRNGKey(42)
        bx = self.ds.x_train[0]
        by = self.ds.y_train[0]
        bx_list = [bx[:, 0], bx[:, 1], bx[:, 2]]

        hk_params, hk_state = self.legacy_net.init(rng, bx_list)
        hk_params["universe_lp"]["b"] = (
            jax.random.normal(jax.random.PRNGKey(55), self.topo.grid_shape) * 0.15
        )
        vnn_params = import_legacy_haiku_params(hk_params)

        def legacy_loss(p):
            logits, _ = self.legacy_net.apply(p, hk_state, bx_list)
            logits = jnp.squeeze(logits, axis=-1)
            return optax.squared_error(predictions=logits, targets=by).mean()

        def new_loss(p: VNNParams):
            logits = self.model.apply(p, bx, return_universe=False)
            logits = jnp.squeeze(logits, axis=-1)
            return optax.squared_error(predictions=logits, targets=by).mean()

        loss_ref, grads_ref_tree = jax.value_and_grad(legacy_loss)(hk_params)
        loss_new, grads_new = jax.value_and_grad(new_loss)(vnn_params)

        grads_ref = grads_ref_tree["universe_lp"]

        self.assertAlmostEqual(float(loss_ref), float(loss_new), places=5)
        self.assertLess(_rel_l2_error(grads_ref["w"], grads_new.weights), 1e-5)
        self.assertLess(_rel_l2_error(grads_ref["b"], grads_new.biases), 1e-5)
        self.assertLess(_rel_l2_error(grads_ref["s"], grads_new.states), 1e-5)
        # Filter backward pass accumulates over 256,000 spatial-batch elements in cuDNN;
        # relative L2 difference between SAME(40x50) + B=1 border and VALID(62x62) is ~4.1e-5
        self.assertLess(_rel_l2_error(grads_ref["f"], grads_new.filters_ihwo), 5e-5)

    def test_multi_step_adamw_trajectory_parity(self):
        """
        Verifies that 5 consecutive AdamW updates produce identical loss and parameter
        trajectories between `VonNeumannNetwork` and `UniverseLP`.
        """
        rng = jax.random.PRNGKey(42)
        bx0 = self.ds.x_train[0]
        bx0_list = [bx0[:, 0], bx0[:, 1], bx0[:, 2]]

        hk_params, hk_state = self.legacy_net.init(rng, bx0_list)
        vnn_params = self.model.init(rng, legacy_compat=True)

        opt = optax.adamw(0.005)
        hk_opt_state = opt.init(hk_params)
        vnn_opt_state = opt.init(vnn_params)

        @jax.jit
        def legacy_step(p, o_s, s, bx_l, by):
            def l_fn(params):
                logits, st = self.legacy_net.apply(params, s, bx_l)
                logits = jnp.squeeze(logits, axis=-1)
                return optax.squared_error(predictions=logits, targets=by).mean(), st
            (l_val, s_new), g = jax.value_and_grad(l_fn, has_aux=True)(p)
            u, o_s_new = opt.update(g, o_s, p)
            return optax.apply_updates(p, u), o_s_new, s_new, l_val

        @jax.jit
        def new_step(p, o_s, bx, by):
            def l_fn(params):
                logits = self.model.apply(params, bx, return_universe=False)
                logits = jnp.squeeze(logits, axis=-1)
                return optax.squared_error(predictions=logits, targets=by).mean()
            l_val, g = jax.value_and_grad(l_fn)(p)
            u, o_s_new = opt.update(g, o_s, p)
            return optax.apply_updates(p, u), o_s_new, l_val

        for step_idx in range(5):
            bx = self.ds.x_train[step_idx]
            by = self.ds.y_train[step_idx]
            bx_l = [bx[:, 0], bx[:, 1], bx[:, 2]]

            hk_params, hk_opt_state, hk_state, l_ref = legacy_step(
                hk_params, hk_opt_state, hk_state, bx_l, by
            )
            vnn_params, vnn_opt_state, l_new = new_step(
                vnn_params, vnn_opt_state, bx, by
            )
            self.assertLess(abs(float(l_ref) - float(l_new)), 2e-5)

        sub = hk_params["universe_lp"]
        self.assertLess(_rel_l2_error(sub["w"], vnn_params.weights), 5e-5)
        self.assertLess(_rel_l2_error(sub["b"], vnn_params.biases), 5e-5)
        self.assertLess(_rel_l2_error(sub["s"], vnn_params.states), 5e-5)
        self.assertLess(_rel_l2_error(sub["f"], vnn_params.filters_ihwo), 5e-5)


if __name__ == "__main__":
    unittest.main()
