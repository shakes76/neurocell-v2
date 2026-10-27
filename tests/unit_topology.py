'''
Unit tests for `neurocell.topology` (`PortSpec` and `ChipTopology2D`).

Verifies compile-time geometry, active ROI bounds, output port coordinate mapping,
and static boolean pinning masks against `contrib/universe.py`.

@author Shakes
'''
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp

from neurocell.topology import PortSpec, ChipTopology2D


class TestTopology(unittest.TestCase):
    """Tests `ChipTopology2D` construction, validation, and mask equivalence."""

    def test_alu_chip_geometry(self):
        """Verifies canonical 62x62 ALU chip dimensions, ports, and ROI bounds."""
        topo = ChipTopology2D.alu_chip(
            n=8, kernel_size=(23, 13), directions=16, depth=4, gap=5, ctrl_gap=5
        )
        self.assertEqual(topo.grid_shape, (62, 62))
        self.assertEqual(topo.H, 62)
        self.assertEqual(topo.W, 62)
        self.assertEqual(topo.N, 62)
        self.assertEqual(topo.kernel_size, (23, 13))
        self.assertEqual(topo.directions, 16)
        self.assertEqual(topo.depth, 4)

        # Kernel radii: (23-1)//2 = 11, (13-1)//2 = 6 -> ROI = (62-22, 62-12) = (40, 50)
        self.assertEqual(topo.pad_h, 11)
        self.assertEqual(topo.pad_w, 6)
        self.assertEqual(topo.roi_shape, (40, 50))
        self.assertEqual(topo.fft_shape, (64, 64))

        # Verify input and output port coordinates against test_jax_vnn_alu_arithmetic.py
        self.assertEqual(topo.in_ports[0], PortSpec(offset=(23, 23), size=(8, 1), name="x1"))
        self.assertEqual(topo.in_ports[1], PortSpec(offset=(32, 23), size=(8, 1), name="x2"))
        self.assertEqual(topo.in_ports[2], PortSpec(offset=(18, 28), size=(1, 8), name="ctrl"))
        self.assertEqual(topo.out_port, PortSpec(offset=(27, 41), size=(8, 1), name="out"))

        # Output offset relative to ROI [11:51, 6:56] is (27-11, 41-6) = (16, 35)
        self.assertEqual(topo.roi_out_offset, (16, 35))

    def test_static_masks_match_dynamic_update_slice(self):
        """Verifies static boolean masks match legacy `dynamic_update_slice` chain exactly."""
        topo = ChipTopology2D.alu_chip()

        # Legacy dynamic_update_slice chain from contrib/universe.py:595-604
        rng = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(rng)
        s_orig = jax.random.normal(k1, (62, 62))
        w_orig = jax.random.normal(k2, (62, 62))

        s_ref = s_orig
        for port in topo.in_ports:
            s_ref = jax.lax.dynamic_update_slice(
                s_ref, jnp.full(port.size, 1.0, dtype=jnp.float32), port.offset
            )
        s_ref = jax.lax.dynamic_update_slice(
            s_ref, jnp.full(topo.out_port.size, 1.0, dtype=jnp.float32), topo.out_port.offset
        )
        w_ref = jax.lax.dynamic_update_slice(
            w_orig, jnp.full(topo.out_port.size, 1.0, dtype=jnp.float32), topo.out_port.offset
        )

        s_masked = jnp.where(topo.io_state_mask, 1.0, s_orig)
        w_masked = jnp.where(topo.out_weight_mask, 1.0, w_orig)

        self.assertEqual(float(jnp.max(jnp.abs(s_ref - s_masked))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(w_ref - w_masked))), 0.0)
        # 8 + 8 + 8 + 8 = 32 pinned I/O cells
        self.assertEqual(int(jnp.sum(topo.io_state_mask)), 32)
        self.assertEqual(int(jnp.sum(topo.out_weight_mask)), 8)

    def test_validation_errors(self):
        """Verifies that invalid kernel parity or out-of-ROI output ports raise ValueError."""
        with self.assertRaises(ValueError):
            ChipTopology2D.create(
                grid_shape=(62, 62),
                kernel_size=(22, 13),  # Even kernel height
                directions=16,
                depth=4,
                in_ports=(PortSpec((23, 23), (8, 1)),),
                out_port=PortSpec((27, 41), (8, 1)),
            )

        with self.assertRaises(ValueError):
            ChipTopology2D.create(
                grid_shape=(62, 62),
                kernel_size=(23, 13),
                directions=16,
                depth=4,
                in_ports=(PortSpec((23, 23), (8, 1)),),
                out_port=PortSpec((2, 41), (8, 1)),  # Outside valid ROI [11:51, 6:56]
            )


if __name__ == "__main__":
    unittest.main()
