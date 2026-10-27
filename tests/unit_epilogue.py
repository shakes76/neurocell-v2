'''
Unit tests for `neurocell.kernels_greens` and `neurocell.epilogue`.

Verifies that native batched `NCHW` convolution, factored active-ROI Codd gating,
48% dead-border decomposition, and 8-cell output-port final slicing match the
legacy `jax.vmap(forward._forward_full)` implementation in `contrib/forward.py`.

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

from neurocell.topology import ChipTopology2D
from neurocell.params import VNNParams, init_vnn_params, precompute_step_params
from neurocell.kernels_greens import conv2d_batched_nchw
from neurocell.epilogue import fused_roi_epilogue, fused_output_port_epilogue

import forward as legacy_forward


class TestEpilogueAndBatchedConv(unittest.TestCase):
    """Tests `conv2d_batched_nchw`, `fused_roi_epilogue`, and `fused_output_port_epilogue`."""

    def setUp(self):
        self.topo = ChipTopology2D.alu_chip(
            n=8, kernel_size=(23, 13), directions=16, depth=4, gap=5, ctrl_gap=5
        )
        rng = jax.random.PRNGKey(42)
        base_params = init_vnn_params(rng, self.topo, legacy_compat=True)
        # Use non-zero random biases as well so the dead-border term tanh(B)*sigmoid(S) is non-zero!
        b_nonzero = jax.random.normal(jax.random.PRNGKey(77), self.topo.grid_shape) * 0.25
        self.params = VNNParams(
            cell_params=base_params.cell_params.at[1].set(b_nonzero),
            greens_filters=base_params.greens_filters,
        )
        self.dn = jax.lax.conv_dimension_numbers(
            (1, 62, 62, 1), (1, 23, 13, 16), ("NHWC", "IHWO", "NHWC")
        )

    def test_single_step_roi_epilogue_matches_legacy_forward_full(self):
        """
        Verifies that `conv2d_batched_nchw` + `fused_roi_epilogue` matches
        `jax.vmap(legacy_forward._forward_full)` across the entire (B, 62, 62) grid,
        including the 48% dead border when biases are non-zero.
        """
        B = 8
        cells_bhw = jax.random.normal(jax.random.PRNGKey(101), (B, 62, 62))
        step_p = precompute_step_params(self.params, self.topo, activate=jax.nn.tanh)

        # Legacy _forward_full expects raw pre-sigmoid states (pinned at I/O)
        s_pinned_raw = jnp.where(self.topo.io_state_mask, 1.0, self.params.states)
        w_pinned = step_p.w_pinned
        b_full = step_p.b_full
        f_ihwo = self.params.filters_ihwo

        vmap_legacy = jax.vmap(
            lambda c: legacy_forward._forward_full(
                1, c, w_pinned, b_full, s_pinned_raw, f_ihwo, jax.nn.tanh, True, self.dn, (1, 1)
            )
        )
        ref_next = vmap_legacy(cells_bhw)

        # New batched NCHW conv + ROI epilogue
        z_roi = conv2d_batched_nchw(
            cells_bhw[:, jnp.newaxis, :, :], self.params.greens_filters, padding="VALID"
        )
        c_border = jnp.broadcast_to(step_p.border_2d[jnp.newaxis, :, :], (B, 62, 62))
        new_next = fused_roi_epilogue(
            z_roi, step_p, c_border, self.topo, activate=jax.nn.tanh, norm=True
        )

        max_err = float(jnp.max(jnp.abs(ref_next - new_next)))
        self.assertLess(max_err, 1e-5)

    def test_output_port_epilogue_matches_full_grid_slice(self):
        """
        Verifies that `fused_output_port_epilogue` (which computes Codd gating on only 8 cells)
        matches full-grid `_forward_full` + `dynamic_slice` for both `final_activate=None`
        and `final_activate=jax.nn.sigmoid`.
        """
        B = 8
        cells_bhw = jax.random.normal(jax.random.PRNGKey(202), (B, 62, 62))
        step_p = precompute_step_params(self.params, self.topo, activate=jax.nn.tanh)
        s_pinned_raw = jnp.where(self.topo.io_state_mask, 1.0, self.params.states)
        w_pinned = step_p.w_pinned
        b_full = step_p.b_full
        f_ihwo = self.params.filters_ihwo

        z_roi = conv2d_batched_nchw(
            cells_bhw[:, jnp.newaxis, :, :], self.params.greens_filters, padding="VALID"
        )

        for final_act in (None, jax.nn.sigmoid):
            def _legacy_final(c):
                c_out = legacy_forward._forward_full(
                    1, c, w_pinned, b_full, s_pinned_raw, f_ihwo, final_act, True, self.dn, (1, 1)
                )
                if final_act is not None:
                    c_out = final_act(c_out)
                return jax.lax.dynamic_slice(c_out, self.topo.out_port.offset, self.topo.out_port.size)

            ref_out = jax.vmap(_legacy_final)(cells_bhw)
            new_out = fused_output_port_epilogue(
                z_roi, step_p, self.topo, final_activate=final_act, norm=True
            )
            max_err = float(jnp.max(jnp.abs(ref_out - new_out)))
            self.assertLess(max_err, 1e-5)


if __name__ == "__main__":
    unittest.main()
