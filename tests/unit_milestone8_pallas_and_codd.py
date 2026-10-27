'''
Unit tests for Milestone 8 (`v2.3`):
1. Parallel `(B, M, num_tiles)` Pallas forward/backward tiled epilogue kernels.
2. In-register fused sparse output-port backward scatter (`_make_sparse_port_bwd_kernel`
   and `_make_tiled_sparse_port_bwd_kernel`) for rectangular and coordinate ports.
3. SRAM-resident grouped-batch spatial filter gradients (`use_sram_grouped_conv`) and
   row-pruned backward 2D `irfft` in `pallas_sram`.
4. Temperature-annealed sigmoid and Straight-Through Estimator (STE) discrete Codd state
   hardening (`"continuous"`, `"annealed"`, `"ste_binary"`, `"ste_ternary"` `{0, 1, 2}`).

@author Shakes
'''
import unittest
import jax
import jax.numpy as jnp
import optax

from neurocell import (
    SUPPORTED_CODD_MODES,
    ChipTopology2D,
    CoordinatePortSpec,
    PortSpec,
    VonNeumannNetwork,
    VNNTrainer,
    compute_annealed_temperature,
    compute_codd_state_probs,
    fused_roi_step_epilogue,
    pallas_fused_roi_step_epilogue,
    pallas_standardize_output_port,
    standardize_spatial_nchw,
)


class TestMilestone8PallasAndCodd(unittest.TestCase):
    """Verifies Milestone 8 parallel Pallas kernels, SRAM-grouped conv, and STE Codd hardening."""

    def test_parallel_tiled_pallas_fwd_and_bwd_parity(self):
        """
        Tests `pallas_fused_roi_step_epilogue` on multi-tile grids (`96x96 = 9216 > 8192`)
        across both `B * num_tiles < 128` (`grid = (B, M, num_tiles)` Stage 2) and
        `B * num_tiles >= 128` (`grid = (B, num_tiles)` Stage 2).
        """
        key = jax.random.PRNGKey(101)
        h_v, w_v = 96, 96  # 9,216 pixels -> 9 tiles of 1024
        m_dirs = 8

        for bsz in (4, 16):  # 4 * 9 = 36 < 128 (parallel 3D grid) and 16 * 9 = 144 >= 128
            k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
            z_bmhw = jax.random.normal(k1, (bsz, m_dirs, h_v, w_v), dtype=jnp.float32)
            db_1mhw = jax.random.normal(k2, (1, m_dirs, h_v, w_v), dtype=jnp.float32)
            w_eff = jax.random.normal(k3, (h_v, w_v), dtype=jnp.float32)
            s_eff = jax.random.uniform(k4, (h_v, w_v), dtype=jnp.float32)
            b_roi = jax.random.normal(k5, (h_v, w_v), dtype=jnp.float32)
            uw = jax.random.normal(k6, (bsz, h_v, w_v), dtype=jnp.float32)

            def ref_fn(z, db, w, s, b):
                z_tot = z + db
                z_norm = standardize_spatial_nchw(z_tot)
                out = w[None, :, :] * jnp.sum(z_norm, axis=1) + s[None, :, :] * jnp.sum(
                    jnp.tanh(z_norm + b[None, None, :, :]), axis=1
                )
                return jnp.sum(out * uw)

            def pal_fn(z, db, w, s, b):
                out = pallas_fused_roi_step_epilogue(
                    z, db, w, s, b, activate=jax.nn.tanh, has_border=True
                )
                return jnp.sum(out * uw)

            val_ref, grads_ref = jax.jit(jax.value_and_grad(ref_fn, argnums=(0, 1, 2, 3, 4)))(
                z_bmhw, db_1mhw, w_eff, s_eff, b_roi
            )
            val_pal, grads_pal = jax.jit(jax.value_and_grad(pal_fn, argnums=(0, 1, 2, 3, 4)))(
                z_bmhw, db_1mhw, w_eff, s_eff, b_roi
            )

            rel_val = float(jnp.abs(val_ref - val_pal) / (jnp.abs(val_ref) + 1e-5))
            self.assertLess(rel_val, 1e-5, f"Forward mismatch for bsz={bsz}: {rel_val}")
            for g_r, g_p, name in zip(grads_ref, grads_pal, ("gz", "gdb", "gw", "gs", "gb")):
                rel_err = float(
                    jnp.linalg.norm(g_r - g_p) / (jnp.linalg.norm(g_r) + 1e-5)
                )
                self.assertLess(rel_err, 1e-5, f"Gradient {name} mismatch for bsz={bsz}: {rel_err}")

    def test_in_register_sparse_port_bwd_rectangular_and_coordinate(self):
        """
        Tests `pallas_standardize_output_port` with in-register fused sparse output-port
        backward scatter on both single-tile (`40x50 = 2000`) and multi-tile (`96x96 = 9216`)
        grids for rectangular `PortSpec` and scattered `CoordinatePortSpec` offsets.
        """
        key = jax.random.PRNGKey(202)
        bsz, m_dirs = 8, 8

        for h_v, w_v in ((40, 50), (96, 96)):
            z_bmhw = jax.random.normal(key, (bsz, m_dirs, h_v, w_v), dtype=jnp.float32)
            db_1mhw = jax.random.normal(key, (1, m_dirs, h_v, w_v), dtype=jnp.float32)

            # 1. Rectangular output port (8, 1)
            ro_r, ro_c, out_h, out_w = 10, 12, 8, 1
            uw_rect = jax.random.normal(key, (bsz, m_dirs, out_h, out_w), dtype=jnp.float32)

            def ref_rect(z, db):
                z_norm = standardize_spatial_nchw(z + db)
                return jnp.sum(z_norm[:, :, ro_r : ro_r + out_h, ro_c : ro_c + out_w] * uw_rect)

            def pal_rect(z, db):
                z_out = pallas_standardize_output_port(
                    z, db, roi_out_offset=(ro_r, ro_c), out_size=(out_h, out_w), has_border=True
                )
                return jnp.sum(z_out * uw_rect)

            v_r, (gz_r, gdb_r) = jax.jit(jax.value_and_grad(ref_rect, argnums=(0, 1)))(z_bmhw, db_1mhw)
            v_p, (gz_p, gdb_p) = jax.jit(jax.value_and_grad(pal_rect, argnums=(0, 1)))(z_bmhw, db_1mhw)
            self.assertLess(float(jnp.abs(v_r - v_p) / (jnp.abs(v_r) + 1e-5)), 1e-5)
            self.assertLess(float(jnp.linalg.norm(gz_r - gz_p) / jnp.linalg.norm(gz_r)), 1e-5)
            self.assertLess(float(jnp.linalg.norm(gdb_r - gdb_p) / jnp.linalg.norm(gdb_r)), 1e-5)

            # 2. Coordinate output port (6 scattered points)
            coord_rows = (2, 7, 15, 21, 30, 35)
            coord_cols = (3, 11, 19, 25, 31, 38)
            uw_coord = jax.random.normal(key, (bsz, m_dirs, 6, 1), dtype=jnp.float32)

            def ref_coord(z, db):
                z_norm = standardize_spatial_nchw(z + db)
                gathered = z_norm[:, :, jnp.asarray(coord_rows), jnp.asarray(coord_cols)][..., None]
                return jnp.sum(gathered * uw_coord)

            def pal_coord(z, db):
                z_out = pallas_standardize_output_port(
                    z,
                    db,
                    roi_out_offset=(coord_rows, coord_cols),
                    out_size=(6, 1),
                    has_border=True,
                )
                return jnp.sum(z_out * uw_coord)

            v_rc, (gz_rc, gdb_rc) = jax.jit(jax.value_and_grad(ref_coord, argnums=(0, 1)))(
                z_bmhw, db_1mhw
            )
            v_pc, (gz_pc, gdb_pc) = jax.jit(jax.value_and_grad(pal_coord, argnums=(0, 1)))(
                z_bmhw, db_1mhw
            )
            self.assertLess(float(jnp.abs(v_rc - v_pc) / (jnp.abs(v_rc) + 1e-5)), 1e-5)
            self.assertLess(float(jnp.linalg.norm(gz_rc - gz_pc) / jnp.linalg.norm(gz_rc)), 1e-5)
            self.assertLess(float(jnp.linalg.norm(gdb_rc - gdb_pc) / jnp.linalg.norm(gdb_rc)), 1e-5)

    def test_sram_grouped_conv_and_pruned_rfft_bwd_parity(self):
        """
        Verifies end-to-end forward and backward gradient parity across `nchw_direct`,
        `pallas_fused` (including `use_sram_grouped_conv` on compact adaptive sub-ROIs),
        and `pallas_sram` (with row-pruned backward 2D `irfft`).
        """
        key = jax.random.PRNGKey(303)
        topo = ChipTopology2D.alu_chip(
            n=8, kernel_size=(23, 13), directions=16, depth=4, gap=5, ctrl_gap=5
        )
        m_ref = VonNeumannNetwork(topology=topo, backend="nchw_direct")
        m_fused = VonNeumannNetwork(topology=topo, backend="pallas_fused")
        m_sram = VonNeumannNetwork(topology=topo, backend="pallas_sram")

        params = m_ref.init(key, clear_outside_io=True)
        mask = m_ref.compute_spatial_mask(params, margin=4)
        self.assertFalse(mask.is_full_roi)

        bsz = 32
        inputs = [
            jax.random.uniform(key, (bsz, 8, 1)),
            jax.random.uniform(jax.random.PRNGKey(304), (bsz, 8, 1)),
            jax.random.uniform(jax.random.PRNGKey(305), (bsz, 1, 4)),
        ]
        targets = jax.random.uniform(jax.random.PRNGKey(306), (bsz, 8, 1))

        for use_mask in (False, True):
            sm = mask if use_mask else None
            val_ref, g_ref = jax.jit(
                jax.value_and_grad(lambda p: jnp.mean(jnp.square(m_ref.apply(p, inputs, spatial_mask=sm) - targets)))
            )(params)
            val_fused, g_fused = jax.jit(
                jax.value_and_grad(lambda p: jnp.mean(jnp.square(m_fused.apply(p, inputs, spatial_mask=sm) - targets)))
            )(params)
            val_sram, g_sram = jax.jit(
                jax.value_and_grad(lambda p: jnp.mean(jnp.square(m_sram.apply(p, inputs, spatial_mask=sm) - targets)))
            )(params)

            self.assertLess(float(jnp.abs(val_ref - val_fused)), 1e-5)
            self.assertLess(float(jnp.abs(val_ref - val_sram)), 1e-5)

            for leaf_r, leaf_f, leaf_s in zip(
                jax.tree_util.tree_leaves(g_ref),
                jax.tree_util.tree_leaves(g_fused),
                jax.tree_util.tree_leaves(g_sram),
            ):
                err_f = float(jnp.linalg.norm(leaf_r - leaf_f) / (jnp.linalg.norm(leaf_r) + 1e-5))
                err_s = float(jnp.linalg.norm(leaf_r - leaf_s) / (jnp.linalg.norm(leaf_r) + 1e-5))
                self.assertLess(err_f, 1e-4, f"pallas_fused grad mismatch (mask={use_mask}): {err_f}")
                self.assertLess(err_s, 1e-4, f"pallas_sram grad mismatch (mask={use_mask}): {err_s}")

    def test_codd_modes_forward_quantization_and_backward_ste_gradients(self):
        """
        Verifies `"continuous"`, `"annealed"`, `"ste_binary"`, and `"ste_ternary"` Codd modes:
        - Forward values under `"ste_binary"` lie strictly in `{0.0, 1.0}`.
        - Forward values under `"ste_ternary"` lie strictly in `{0.0, 0.5, 1.0}` (roles `{0, 1, 2}`),
          and post-training state thresholding produces zero forward discrepancy.
        - Backward gradients wrt `params.states` are finite and non-zero across all modes.
        """
        self.assertEqual(
            SUPPORTED_CODD_MODES, ("continuous", "annealed", "ste_binary", "ste_ternary")
        )
        key = jax.random.PRNGKey(404)
        topo = ChipTopology2D.create(
            grid_shape=(16, 16),
            kernel_size=(3, 3),
            directions=4,
            depth=3,
            in_ports=(PortSpec(offset=(4, 2), size=(4, 1)),),
            out_port=PortSpec(offset=(4, 12), size=(4, 1)),
            padding_mode="VALID",
        )
        model = VonNeumannNetwork(topology=topo, backend="pallas_fused")
        params = model.init(key)
        # Set diverse positive, intermediate, and negative raw states
        s_diverse = jax.random.uniform(key, topo.grid_shape, minval=-1.0, maxval=1.2)
        params = params._replace(
            cell_params=params.cell_params.at[2].set(s_diverse)
        )

        # 1. Check effective_states quantization sets
        s_bin = model.effective_states(params, codd_mode="ste_binary", state_temperature=0.3)
        unique_bin = set(float(x) for x in jnp.unique(s_bin))
        self.assertTrue(unique_bin.issubset({0.0, 1.0}), f"Unexpected binary states: {unique_bin}")

        s_tern = model.effective_states(params, codd_mode="ste_ternary", state_temperature=0.3)
        unique_tern = set(float(x) for x in jnp.unique(s_tern))
        self.assertTrue(
            unique_tern.issubset({0.0, 0.5, 1.0}), f"Unexpected ternary states: {unique_tern}"
        )

        # 2. Verify zero post-training thresholding gap under ste_ternary
        x = jax.random.normal(key, (8, 4, 1))
        y = jax.random.normal(jax.random.PRNGKey(405), (8, 4, 1))
        out_tern = model.apply(params, x, codd_mode="ste_ternary", state_temperature=0.25)

        # Replace raw states with thresholded discrete Codd roles {0, 1, 2} mapped into each regime
        roles = VonNeumannNetwork.threshold_states(params.states, active_threshold=0.6)
        s_discrete = jnp.where(roles >= 2.0, 1.0, jnp.where(roles > 0.0, 0.3, -1.0))
        params_disc = params._replace(
            cell_params=params.cell_params.at[2].set(s_discrete)
        )
        out_disc = model.apply(params_disc, x, codd_mode="ste_ternary", state_temperature=0.25)
        max_gap = float(jnp.max(jnp.abs(out_tern - out_disc)))
        self.assertEqual(max_gap, 0.0, f"Expected zero thresholding gap under ste_ternary, got {max_gap}")

        # 3. Verify non-zero smooth STE gradients wrt raw states `S` in all modes
        for mode in SUPPORTED_CODD_MODES:
            grads = jax.jit(
                jax.grad(
                    lambda p: jnp.mean(
                        jnp.square(model.apply(p, x, codd_mode=mode, state_temperature=0.3) - y)
                    )
                )
            )(params)
            grad_s = grads.states
            self.assertTrue(bool(jnp.all(jnp.isfinite(grad_s))), f"Non-finite grad_s for {mode}")
            self.assertGreater(
                float(jnp.linalg.norm(grad_s)), 1e-6, f"Zero grad_s under {mode}"
            )

    def test_trainer_annealed_and_ste_convergence(self):
        """
        Verifies `compute_annealed_temperature` schedules and `VNNTrainer.train_epoch_annealed`
        across multiple temperatures and STE modes without XLA recompilation.
        """
        for sched in ("exponential", "cosine", "linear"):
            t0 = compute_annealed_temperature(0, 10, init_temp=1.0, final_temp=0.1, schedule=sched)
            t_mid = compute_annealed_temperature(5, 10, init_temp=1.0, final_temp=0.1, schedule=sched)
            tf = compute_annealed_temperature(9, 10, init_temp=1.0, final_temp=0.1, schedule=sched)
            self.assertAlmostEqual(t0, 1.0, places=5)
            self.assertAlmostEqual(tf, 0.1, places=5)
            self.assertTrue(0.1 < t_mid < 1.0)

        key = jax.random.PRNGKey(505)
        topo = ChipTopology2D.create(
            grid_shape=(12, 12),
            kernel_size=(3, 3),
            directions=4,
            depth=2,
            in_ports=(PortSpec(offset=(4, 2), size=(2, 1)),),
            out_port=PortSpec(offset=(4, 9), size=(2, 1)),
            padding_mode="VALID",
        )
        model = VonNeumannNetwork(topology=topo, backend="pallas_fused")
        params = model.init(key)
        trainer = VNNTrainer(
            apply_fn=model.apply,
            optimizer=optax.adam(1e-2),
            batch_size=16,
        )
        opt_state = trainer.optimizer.init(params)

        x_ep = jax.random.normal(key, (4, 16, 2, 1))
        y_ep = jnp.sin(x_ep[:, :, :, 0])

        losses = []
        for ep in range(4):
            tau = compute_annealed_temperature(ep, 4, init_temp=1.0, final_temp=0.2)
            params, opt_state, ep_loss = trainer.train_epoch_annealed(
                params,
                opt_state,
                x_ep,
                y_ep,
                state_temperature=tau,
                codd_mode="ste_ternary",
            )
            losses.append(float(ep_loss))

        # Only 1 compiled executable should be cached for ("ste_ternary", None) across all 4 temperatures
        self.assertEqual(len(trainer._annealed_epoch_cache), 1)
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
