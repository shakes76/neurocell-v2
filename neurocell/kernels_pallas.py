'''
Custom JAX Pallas / Triton GPU kernels for Von Neumann Networks (`SM 8.9`).

Provides model-, grid-size-, activation-, and port-geometry-independent Pallas Triton
epilogue and spectral primitives:

1. `pallas_fused_roi_step_epilogue` (`@jax.custom_vjp`):
   - Fuses optional `+ Delta_border` (`has_border=True/False`), spatial mean/variance
     standardization (`mean_{b,m}`, `rstd_{b,m}`), pluggable nonlinear Codd activation
     (`tanh`, `leaky_relu`, `relu`, `sigmoid`, `gelu`, `None`), and `M`-direction
     accumulation in registers/SRAM without materializing `(B, M, H_v, W_v)` intermediates.
   - Automatically dispatches between a single-pass register kernel for `N_PIX <= 2048`
     and a hardware-tiled (`BLOCK_N <= 1024`, `grid = (B, num_tiles)`) two-stage Chan/Welford
     pipeline for arbitrarily large cellular grids (`N_PIX > 2048`, e.g. 64x64 GoL, 80x80,
     or 800x17 MNIST) with zero register spilling.
   - When `has_border=False` (e.g. at step `t = 0` or in `"SAME"`/`"CIRCULAR"` modes),
     skips loading `dborder_ref` and skips the `16.38 MB` `gdb` HBM reduction in backward.

2. `pallas_standardize_output_port` & `pallas_fused_output_port_epilogue` (`@jax.custom_vjp`):
   - Supports arbitrary output port geometries (rectangular `PortSpec`, full-grid outputs,
     and scattered `CoordinatePortSpec`), arbitrary grid sizes (`N_PIX <= 2048` and `> 2048`),
     and `has_border=True/False`.

3. Zero-Pad Fourier Shift, Pruned Separable RFFT, & Small-Kernel Grouped Filter Gradients:
   - `conv2d_hybrid_nopad_roi_step`, `conv2d_pruned_rfft_roi_step`, and
     `conv2d_small_kernel_roi_step`.

Copyright (c) 2026 Shekhar S. Chandra. All rights reserved.
Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

@author Shakes
'''
from functools import partial
import math
import warnings
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pl_triton

warnings.filterwarnings(
    "ignore",
    message=".*The Pallas Triton backend is deprecated.*",
    category=DeprecationWarning,
)

from neurocell.topology import ChipTopology2D, CoordinatePortSpec
from neurocell.params import PrecomputedStepParams
from neurocell.kernels_greens import (
    precompute_filter_rfft2,
    rfft2_pruned,
    irfft2_pruned,
    conv2d_grouped_filter_grad,
)

SINGLE_TILE_MAX_PIX = 8192
MEAN_RSTD_SINGLE_PLANE_MAX_PIX = 8192
TILED_BLOCK_N = 1024


def _next_pow2(n: int) -> int:
    return max(16, 1 << (int(n) - 1).bit_length())


def _select_tile_block_n(n_pix: int, max_block_n: int = TILED_BLOCK_N) -> int:
    """
    Selects a power-of-two tile size `<= max_block_n`. Prefers the largest power of two
    in `(1024, 512, 256, 128)` that evenly divides `n_pix` to avoid padding when possible.
    """
    for cand in (1024, 512, 256, 128):
        if cand <= max_block_n and n_pix % cand == 0:
            return cand
    return max_block_n


def _resolve_activation_spec(activate) -> tuple[str, float]:
    """
    Maps a Python/JAX activation callable or name to a static `(act_name, neg_slope)` pair
    suitable for compilation inside a Pallas Triton kernel.
    """
    if activate is None or activate == "identity":
        return ("identity", 1.0)
    if activate in (jax.nn.tanh, jnp.tanh, "tanh"):
        return ("tanh", 0.0)
    if activate in (jax.nn.sigmoid, "sigmoid"):
        return ("sigmoid", 0.0)
    if activate in (jax.nn.relu, "relu"):
        return ("relu", 0.0)
    if activate in (jax.nn.leaky_relu, "leaky_relu"):
        return ("leaky_relu", 0.01)
    if activate in (jax.nn.gelu, "gelu"):
        return ("gelu", 0.0)
    if isinstance(activate, partial) and activate.func is jax.nn.leaky_relu:
        slope = float(activate.keywords.get("negative_slope", 0.01))
        return ("leaky_relu", slope)
    name = getattr(activate, "__name__", "")
    if name in ("tanh", "sigmoid", "relu", "leaky_relu", "gelu"):
        return (name, 0.01 if name == "leaky_relu" else 0.0)
    raise ValueError(f"Unsupported Pallas activation function: {activate}")


def is_supported_pallas_activation(activate) -> bool:
    """Returns `True` if `activate` can be executed inside the Pallas Triton epilogue."""
    try:
        _resolve_activation_spec(activate)
        return True
    except ValueError:
        return False


def _eval_act_fwd(x: jax.Array, act_name: str, neg_slope: float) -> jax.Array:
    if act_name == "identity":
        return x
    if act_name == "tanh":
        return jnp.tanh(x)
    if act_name == "sigmoid":
        return jax.lax.logistic(x)
    if act_name == "relu":
        return jnp.maximum(x, 0.0)
    if act_name == "leaky_relu":
        return jnp.where(x >= 0.0, x, neg_slope * x)
    if act_name == "gelu":
        c = math.sqrt(2.0 / math.pi)
        u = c * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + jnp.tanh(u))
    raise ValueError(f"Unknown activation '{act_name}'.")


def _eval_act_fwd_and_deriv(
    x: jax.Array, act_name: str, neg_slope: float
) -> tuple[jax.Array, jax.Array]:
    if act_name == "identity":
        return x, jnp.ones_like(x)
    if act_name == "tanh":
        y = jnp.tanh(x)
        return y, 1.0 - y * y
    if act_name == "sigmoid":
        y = jax.lax.logistic(x)
        return y, y * (1.0 - y)
    if act_name == "relu":
        return jnp.maximum(x, 0.0), (x > 0.0).astype(jnp.float32)
    if act_name == "leaky_relu":
        return (
            jnp.where(x >= 0.0, x, neg_slope * x),
            jnp.where(x >= 0.0, 1.0, neg_slope),
        )
    if act_name == "gelu":
        c = math.sqrt(2.0 / math.pi)
        x2 = x * x
        u = c * (x + 0.044715 * x2 * x)
        th = jnp.tanh(u)
        du = c * (1.0 + 3.0 * 0.044715 * x2)
        y = 0.5 * x * (1.0 + th)
        dy = 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du
        return y, dy
    raise ValueError(f"Unknown activation '{act_name}'.")


# ============================================================================
# 1A. Single-Tile Pallas Hidden-Step ROI Epilogue (`N_PIX <= 2048`)
# ============================================================================

def _make_roi_epilogue_fwd_kernel(
    n_pix: int,
    n_pad: int,
    act_name: str = "tanh",
    neg_slope: float = 0.0,
    has_border: bool = True,
):
    inv_n = 1.0 / float(n_pix)

    def _kernel(
        z_ref,          # (1, M, n_pix)
        dborder_ref,    # (1, M, n_pix)
        w_ref,          # (n_pix,)
        s_ref,          # (n_pix,)
        b_ref,          # (n_pix,)
        out_ref,        # (1, n_pix)
        mean_ref,       # (1, M)
        rstd_ref,       # (1, M)
    ):
        idx = jnp.arange(n_pad)
        mask = idx < n_pix
        mask_f = mask.astype(jnp.float32)

        w = pl_triton.load(w_ref.at[:], mask=mask, other=0.0)
        s = pl_triton.load(s_ref.at[:], mask=mask, other=0.0)
        b = pl_triton.load(b_ref.at[:], mask=mask, other=0.0)

        acc_lin = jnp.zeros((n_pad,), dtype=jnp.float32)
        acc_act = jnp.zeros((n_pad,), dtype=jnp.float32)

        m_dirs = z_ref.shape[1]
        for m in range(m_dirs):
            z_m = pl_triton.load(z_ref.at[0, m, :], mask=mask, other=0.0)
            if has_border:
                db_m = pl_triton.load(dborder_ref.at[0, m, :], mask=mask, other=0.0)
                z_tot = z_m + db_m
            else:
                z_tot = z_m

            mean_m = jnp.sum(z_tot) * inv_n
            diff_m = (z_tot - mean_m) * mask_f
            var_m = jnp.sum(diff_m * diff_m) * inv_n
            rstd_m = jax.lax.rsqrt(var_m + 1e-5)
            z_norm = diff_m * rstd_m

            acc_lin = acc_lin + z_norm
            act_m = _eval_act_fwd(z_norm + b, act_name, neg_slope) * mask_f
            acc_act = acc_act + act_m

            mean_ref[0, m] = mean_m
            rstd_ref[0, m] = rstd_m

        out_val = w * acc_lin + s * acc_act
        pl_triton.store(out_ref.at[0, :], out_val, mask=mask)

    return _kernel


def _make_roi_epilogue_bwd_kernel(
    n_pix: int,
    n_pad: int,
    act_name: str = "tanh",
    neg_slope: float = 0.0,
    has_border: bool = True,
):
    inv_n = 1.0 / float(n_pix)

    def _kernel(
        z_ref,          # (1, M, n_pix)
        dborder_ref,    # (1, M, n_pix)
        w_ref,          # (n_pix,)
        s_ref,          # (n_pix,)
        b_ref,          # (n_pix,)
        mean_ref,       # (1, M)
        rstd_ref,       # (1, M)
        gy_ref,         # (1, n_pix)
        gz_ref,         # (1, M, n_pix)
        gw_b_ref,       # (1, n_pix)
        gs_b_ref,       # (1, n_pix)
        gb_b_ref,       # (1, n_pix)
    ):
        idx = jnp.arange(n_pad)
        mask = idx < n_pix
        mask_f = mask.astype(jnp.float32)

        w = pl_triton.load(w_ref.at[:], mask=mask, other=0.0)
        s = pl_triton.load(s_ref.at[:], mask=mask, other=0.0)
        b = pl_triton.load(b_ref.at[:], mask=mask, other=0.0)
        gy = pl_triton.load(gy_ref.at[0, :], mask=mask, other=0.0)

        w_gy = w * gy
        s_gy = s * gy

        acc_lin = jnp.zeros((n_pad,), dtype=jnp.float32)
        acc_act = jnp.zeros((n_pad,), dtype=jnp.float32)
        acc_db = jnp.zeros((n_pad,), dtype=jnp.float32)

        m_dirs = z_ref.shape[1]
        for m in range(m_dirs):
            z_m = pl_triton.load(z_ref.at[0, m, :], mask=mask, other=0.0)
            if has_border:
                db_m = pl_triton.load(dborder_ref.at[0, m, :], mask=mask, other=0.0)
                z_tot = z_m + db_m
            else:
                z_tot = z_m

            mean_m = mean_ref[0, m]
            rstd_m = rstd_ref[0, m]

            z_norm = (z_tot - mean_m) * rstd_m * mask_f
            act_raw, deriv_raw = _eval_act_fwd_and_deriv(z_norm + b, act_name, neg_slope)
            act_m = act_raw * mask_f
            d_act = s_gy * deriv_raw * mask_f

            acc_lin = acc_lin + z_norm
            acc_act = acc_act + act_m
            acc_db = acc_db + d_act

            gz_norm = (w_gy + d_act) * mask_f
            c1 = jnp.sum(gz_norm) * inv_n
            c2 = jnp.sum(gz_norm * z_norm) * inv_n
            gz_m = rstd_m * (gz_norm - c1 - z_norm * c2) * mask_f
            pl_triton.store(gz_ref.at[0, m, :], gz_m, mask=mask)

        pl_triton.store(gw_b_ref.at[0, :], gy * acc_lin, mask=mask)
        pl_triton.store(gs_b_ref.at[0, :], gy * acc_act, mask=mask)
        pl_triton.store(gb_b_ref.at[0, :], acc_db, mask=mask)

    return _kernel


# ============================================================================
# 1B. Hardware-Tiled Pallas Hidden-Step ROI Epilogue (`N_PIX > 2048`)
# ============================================================================

def _make_tiled_chan_stats_fwd_kernel(block_n: int, has_border: bool = True):
    """
    Stage 1 of tiled forward epilogue (`grid = (B, M, num_tiles)`):
    Computes tile sum `s1` and tile centered sum-of-squares `m2` for each `(b, m, t)`
    in parallel across all `B * M * num_tiles` blocks using strictly `block_n <= 1024`
    registers per threadblock.
    """
    inv_block = 1.0 / float(block_n)

    def _kernel(
        z_ref,          # (1, 1, 1, block_n)
        dborder_ref,    # (1, 1, 1, block_n)
        s1_ref,         # (1, 1, 1)
        m2_ref,         # (1, 1, 1)
    ):
        z_m = pl_triton.load(z_ref.at[0, 0, 0, :])
        if has_border:
            db_m = pl_triton.load(dborder_ref.at[0, 0, 0, :])
            z_tot = z_m + db_m
        else:
            z_tot = z_m

        s1_m = jnp.sum(z_tot)
        mu_m = s1_m * inv_block
        diff_m = z_tot - mu_m
        m2_m = jnp.sum(diff_m * diff_m)

        s1_ref[0, 0, 0] = s1_m
        m2_ref[0, 0, 0] = m2_m

    return _kernel


def _make_tiled_codd_fwd_kernel(
    block_n: int,
    act_name: str = "tanh",
    neg_slope: float = 0.0,
    has_border: bool = True,
):
    """
    Stage 2 of tiled forward epilogue (`grid = (B, num_tiles)`):
    Standardizes each tile `t` of size `block_n <= 1024` using `(mean_bm, rstd_bm)`,
    applies Codd activation, and reduces over `M` directions in registers.
    """
    def _kernel(
        z_ref,          # (1, M, 1, block_n)
        dborder_ref,    # (1, M, 1, block_n)
        w_ref,          # (1, block_n)
        s_ref,          # (1, block_n)
        b_ref,          # (1, block_n)
        mean_ref,       # (1, M)
        rstd_ref,       # (1, M)
        out_ref,        # (1, 1, block_n)
    ):
        w = pl_triton.load(w_ref.at[0, :])
        s = pl_triton.load(s_ref.at[0, :])
        b = pl_triton.load(b_ref.at[0, :])

        acc_lin = jnp.zeros((block_n,), dtype=jnp.float32)
        acc_act = jnp.zeros((block_n,), dtype=jnp.float32)

        m_dirs = z_ref.shape[1]
        for m in range(m_dirs):
            z_m = pl_triton.load(z_ref.at[0, m, 0, :])
            if has_border:
                db_m = pl_triton.load(dborder_ref.at[0, m, 0, :])
                z_tot = z_m + db_m
            else:
                z_tot = z_m

            mean_m = mean_ref[0, m]
            rstd_m = rstd_ref[0, m]
            z_norm = (z_tot - mean_m) * rstd_m

            acc_lin = acc_lin + z_norm
            acc_act = acc_act + _eval_act_fwd(z_norm + b, act_name, neg_slope)

        out_val = w * acc_lin + s * acc_act
        pl_triton.store(out_ref.at[0, 0, :], out_val)

    return _kernel


def _make_tiled_bwd_stage1_kernel(
    block_n: int,
    act_name: str = "tanh",
    neg_slope: float = 0.0,
    has_border: bool = True,
):
    """
    Stage 1 of tiled backward epilogue (`grid = (B, num_tiles)`):
    Computes per-tile parameter gradients `(gw_b, gs_b, gb_b)` and partial standardization
    projection scalars `(c1_part, c2_part)` of shape `(B, num_tiles, M)`.
    """
    def _kernel(
        z_ref,          # (1, M, 1, block_n)
        dborder_ref,    # (1, M, 1, block_n)
        w_ref,          # (1, block_n)
        s_ref,          # (1, block_n)
        b_ref,          # (1, block_n)
        mean_ref,       # (1, M)
        rstd_ref,       # (1, M)
        gy_ref,         # (1, 1, block_n)
        gw_b_ref,       # (1, 1, block_n)
        gs_b_ref,       # (1, 1, block_n)
        gb_b_ref,       # (1, 1, block_n)
        c1_part_ref,    # (1, 1, M)
        c2_part_ref,    # (1, 1, M)
    ):
        w = pl_triton.load(w_ref.at[0, :])
        s = pl_triton.load(s_ref.at[0, :])
        b = pl_triton.load(b_ref.at[0, :])
        gy = pl_triton.load(gy_ref.at[0, 0, :])

        w_gy = w * gy
        s_gy = s * gy

        acc_lin = jnp.zeros((block_n,), dtype=jnp.float32)
        acc_act = jnp.zeros((block_n,), dtype=jnp.float32)
        acc_db = jnp.zeros((block_n,), dtype=jnp.float32)

        m_dirs = z_ref.shape[1]
        for m in range(m_dirs):
            z_m = pl_triton.load(z_ref.at[0, m, 0, :])
            if has_border:
                db_m = pl_triton.load(dborder_ref.at[0, m, 0, :])
                z_tot = z_m + db_m
            else:
                z_tot = z_m

            mean_m = mean_ref[0, m]
            rstd_m = rstd_ref[0, m]

            z_norm = (z_tot - mean_m) * rstd_m
            act_m, deriv_m = _eval_act_fwd_and_deriv(z_norm + b, act_name, neg_slope)
            d_act = s_gy * deriv_m

            acc_lin = acc_lin + z_norm
            acc_act = acc_act + act_m
            acc_db = acc_db + d_act

            gz_norm = w_gy + d_act
            c1_part_ref[0, 0, m] = jnp.sum(gz_norm)
            c2_part_ref[0, 0, m] = jnp.sum(gz_norm * z_norm)

        pl_triton.store(gw_b_ref.at[0, 0, :], gy * acc_lin)
        pl_triton.store(gs_b_ref.at[0, 0, :], gy * acc_act)
        pl_triton.store(gb_b_ref.at[0, 0, :], acc_db)

    return _kernel


def _make_tiled_bwd_stage2_kernel(
    block_n: int,
    act_name: str = "tanh",
    neg_slope: float = 0.0,
    has_border: bool = True,
):
    """
    Stage 2 of tiled backward epilogue (`grid = (B, num_tiles)`):
    Computes `gz_m = rstd_m * (gz_norm - c1_m - z_norm * c2_m)` for each tile `(b, t)`.
    """
    def _kernel(
        z_ref,          # (1, M, 1, block_n)
        dborder_ref,    # (1, M, 1, block_n)
        w_ref,          # (1, block_n)
        s_ref,          # (1, block_n)
        b_ref,          # (1, block_n)
        mean_ref,       # (1, M)
        rstd_ref,       # (1, M)
        c1_ref,         # (1, M)
        c2_ref,         # (1, M)
        gy_ref,         # (1, 1, block_n)
        gz_ref,         # (1, M, 1, block_n)
    ):
        w = pl_triton.load(w_ref.at[0, :])
        s = pl_triton.load(s_ref.at[0, :])
        b = pl_triton.load(b_ref.at[0, :])
        gy = pl_triton.load(gy_ref.at[0, 0, :])

        w_gy = w * gy
        s_gy = s * gy

        m_dirs = z_ref.shape[1]
        for m in range(m_dirs):
            z_m = pl_triton.load(z_ref.at[0, m, 0, :])
            if has_border:
                db_m = pl_triton.load(dborder_ref.at[0, m, 0, :])
                z_tot = z_m + db_m
            else:
                z_tot = z_m

            mean_m = mean_ref[0, m]
            rstd_m = rstd_ref[0, m]
            c1_m = c1_ref[0, m]
            c2_m = c2_ref[0, m]

            z_norm = (z_tot - mean_m) * rstd_m
            _, deriv_m = _eval_act_fwd_and_deriv(z_norm + b, act_name, neg_slope)
            d_act = s_gy * deriv_m
            gz_norm = w_gy + d_act
            gz_m = rstd_m * (gz_norm - c1_m - z_norm * c2_m)
            pl_triton.store(gz_ref.at[0, m, 0, :], gz_m)

    return _kernel


def _make_parallel_tiled_bwd_stage2_kernel(
    block_n: int,
    act_name: str = "tanh",
    neg_slope: float = 0.0,
    has_border: bool = True,
):
    """
    Parallel `(B, M, num_tiles)` Stage 2 of tiled backward epilogue:
    Executes each `(b, m, t)` direction-tile in parallel across GPU SMs without
    looping over `m_dirs`, maximizing SM occupancy when `B * num_tiles < 128`.
    """
    def _kernel(
        z_ref,          # (1, 1, 1, block_n)
        dborder_ref,    # (1, 1, 1, block_n)
        w_ref,          # (1, block_n)
        s_ref,          # (1, block_n)
        b_ref,          # (1, block_n)
        mean_ref,       # (1, 1)
        rstd_ref,       # (1, 1)
        c1_ref,         # (1, 1)
        c2_ref,         # (1, 1)
        gy_ref,         # (1, 1, block_n)
        gz_ref,         # (1, 1, 1, block_n)
    ):
        w = pl_triton.load(w_ref.at[0, :])
        s = pl_triton.load(s_ref.at[0, :])
        b = pl_triton.load(b_ref.at[0, :])
        gy = pl_triton.load(gy_ref.at[0, 0, :])

        z_m = pl_triton.load(z_ref.at[0, 0, 0, :])
        if has_border:
            db_m = pl_triton.load(dborder_ref.at[0, 0, 0, :])
            z_tot = z_m + db_m
        else:
            z_tot = z_m

        mean_m = mean_ref[0, 0]
        rstd_m = rstd_ref[0, 0]
        c1_m = c1_ref[0, 0]
        c2_m = c2_ref[0, 0]

        z_norm = (z_tot - mean_m) * rstd_m
        _, deriv_m = _eval_act_fwd_and_deriv(z_norm + b, act_name, neg_slope)
        gz_norm = gy * (w + s * deriv_m)
        gz_m = rstd_m * (gz_norm - c1_m - z_norm * c2_m)
        pl_triton.store(gz_ref.at[0, 0, 0, :], gz_m)

    return _kernel


def _run_tiled_bwd_stage2(
    z_tiled: jax.Array,
    db_tiled: jax.Array,
    w_tiled: jax.Array,
    s_tiled: jax.Array,
    b_tiled: jax.Array,
    mean_bm: jax.Array,
    rstd_bm: jax.Array,
    c1_bm: jax.Array,
    c2_bm: jax.Array,
    gy_tiled: jax.Array,
    block_n: int,
    act_name: str,
    neg_slope: float,
    has_border: bool,
) -> jax.Array:
    """
    Dispatches Stage 2 of the tiled backward epilogue:
    - Uses `grid = (B, M, num_tiles)` (`_make_parallel_tiled_bwd_stage2_kernel`) when
      `B * num_tiles < 128` to saturate all GPU SMs across directions.
    - Uses `grid = (B, num_tiles)` (`_make_tiled_bwd_stage2_kernel`) when
      `B * num_tiles >= 128` so `w, s, b, gy` are loaded once per `(b, t)` and reused
      in registers across `M` directions.
    """
    bsz, m_dirs, num_tiles, _ = z_tiled.shape
    if bsz * num_tiles < 128:
        return pl.pallas_call(
            _make_parallel_tiled_bwd_stage2_kernel(
                block_n, act_name=act_name, neg_slope=neg_slope, has_border=has_border
            ),
            out_shape=jax.ShapeDtypeStruct((bsz, m_dirs, num_tiles, block_n), jnp.float32),
            in_specs=[
                pl.BlockSpec((1, 1, 1, block_n), lambda b, m, t: (b, m, t, 0)),
                pl.BlockSpec((1, 1, 1, block_n), lambda b, m, t: (0, m, t, 0)),
                pl.BlockSpec((1, block_n), lambda b, m, t: (t, 0)),
                pl.BlockSpec((1, block_n), lambda b, m, t: (t, 0)),
                pl.BlockSpec((1, block_n), lambda b, m, t: (t, 0)),
                pl.BlockSpec((1, 1), lambda b, m, t: (b, m)),
                pl.BlockSpec((1, 1), lambda b, m, t: (b, m)),
                pl.BlockSpec((1, 1), lambda b, m, t: (b, m)),
                pl.BlockSpec((1, 1), lambda b, m, t: (b, m)),
                pl.BlockSpec((1, 1, block_n), lambda b, m, t: (b, t, 0)),
            ],
            out_specs=pl.BlockSpec((1, 1, 1, block_n), lambda b, m, t: (b, m, t, 0)),
            grid=(bsz, m_dirs, num_tiles),
        )(z_tiled, db_tiled, w_tiled, s_tiled, b_tiled, mean_bm, rstd_bm, c1_bm, c2_bm, gy_tiled)

    return pl.pallas_call(
        _make_tiled_bwd_stage2_kernel(
            block_n, act_name=act_name, neg_slope=neg_slope, has_border=has_border
        ),
        out_shape=jax.ShapeDtypeStruct((bsz, m_dirs, num_tiles, block_n), jnp.float32),
        in_specs=[
            pl.BlockSpec((1, m_dirs, 1, block_n), lambda b, t: (b, 0, t, 0)),
            pl.BlockSpec((1, m_dirs, 1, block_n), lambda b, t: (0, 0, t, 0)),
            pl.BlockSpec((1, block_n), lambda b, t: (t, 0)),
            pl.BlockSpec((1, block_n), lambda b, t: (t, 0)),
            pl.BlockSpec((1, block_n), lambda b, t: (t, 0)),
            pl.BlockSpec((1, m_dirs), lambda b, t: (b, 0)),
            pl.BlockSpec((1, m_dirs), lambda b, t: (b, 0)),
            pl.BlockSpec((1, m_dirs), lambda b, t: (b, 0)),
            pl.BlockSpec((1, m_dirs), lambda b, t: (b, 0)),
            pl.BlockSpec((1, 1, block_n), lambda b, t: (b, t, 0)),
        ],
        out_specs=pl.BlockSpec((1, m_dirs, 1, block_n), lambda b, t: (b, 0, t, 0)),
        grid=(bsz, num_tiles),
    )(z_tiled, db_tiled, w_tiled, s_tiled, b_tiled, mean_bm, rstd_bm, c1_bm, c2_bm, gy_tiled)


def _compute_tiled_mean_rstd(
    z_tiled: jax.Array,
    db_tiled: jax.Array,
    n_pix: int,
    num_tiles: int,
    block_n: int,
    has_border: bool,
) -> tuple[jax.Array, jax.Array]:
    """
    Computes exact `(mean_bm, rstd_bm)` across `num_tiles` tiles using Chan's parallel
    variance combination formula with `grid = (B, M, num_tiles)`.
    """
    bsz, m_dirs, _, _ = z_tiled.shape
    s1_bmt, m2_bmt = pl.pallas_call(
        _make_tiled_chan_stats_fwd_kernel(block_n, has_border=has_border),
        out_shape=(
            jax.ShapeDtypeStruct((bsz, m_dirs, num_tiles), jnp.float32),
            jax.ShapeDtypeStruct((bsz, m_dirs, num_tiles), jnp.float32),
        ),
        in_specs=[
            pl.BlockSpec((1, 1, 1, block_n), lambda b, m, t: (b, m, t, 0)),
            pl.BlockSpec((1, 1, 1, block_n), lambda b, m, t: (0, m, t, 0)),
        ],
        out_specs=(
            pl.BlockSpec((1, 1, 1), lambda b, m, t: (b, m, t)),
            pl.BlockSpec((1, 1, 1), lambda b, m, t: (b, m, t)),
        ),
        grid=(bsz, m_dirs, num_tiles),
    )(z_tiled, db_tiled)

    inv_n = 1.0 / float(n_pix)
    mean_bm = jnp.sum(s1_bmt, axis=2) * inv_n
    mu_tile = s1_bmt / float(block_n)
    delta_mu = mu_tile - mean_bm[:, :, jnp.newaxis]
    var_bm = jnp.sum(m2_bmt + float(block_n) * delta_mu * delta_mu, axis=2) * inv_n
    rstd_bm = jax.lax.rsqrt(var_bm + 1e-5)
    return mean_bm, rstd_bm


@partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def pallas_fused_roi_step_epilogue(
    z_bmhw: jax.Array,
    delta_border_1mhw: jax.Array,
    w_eff_roi: jax.Array,
    s_eff_roi: jax.Array,
    b_roi: jax.Array,
    activate=jax.nn.tanh,
    has_border: bool = True,
) -> jax.Array:
    """
    Fused Pallas Triton GPU kernel for the hidden-step Von Neumann ROI epilogue:
        `Z_hat_{b,m} = standardize(Z_{b,m} + (Delta_border_{0,m} if has_border else 0))`
        `roi_next_b  = W_eff * sum_m(Z_hat_{b,m}) + S_eff * sum_m(activate(Z_hat_{b,m} + B_roi))`

    Automatically selects the single-pass register kernel when `H_v * W_v <= 2048` and the
    hardware-tiled (`BLOCK_N <= 1024`) two-stage pipeline when `H_v * W_v > 2048`.
    """
    out_hw, _, _ = _pallas_roi_epilogue_fwd_impl(
        z_bmhw,
        delta_border_1mhw,
        w_eff_roi,
        s_eff_roi,
        b_roi,
        activate=activate,
        has_border=has_border,
    )
    return out_hw


def _pallas_roi_epilogue_fwd_impl(
    z_bmhw: jax.Array,
    delta_border_1mhw: jax.Array,
    w_eff_roi: jax.Array,
    s_eff_roi: jax.Array,
    b_roi: jax.Array,
    activate=jax.nn.tanh,
    has_border: bool = True,
):
    bsz, m_dirs, h_v, w_v = z_bmhw.shape
    n_pix = h_v * w_v
    act_name, neg_slope = _resolve_activation_spec(activate)

    if n_pix <= SINGLE_TILE_MAX_PIX:
        n_pad = _next_pow2(n_pix)
        z_flat = jnp.reshape(z_bmhw, (bsz, m_dirs, n_pix))
        db_flat = jnp.reshape(delta_border_1mhw, (1, m_dirs, n_pix))
        w_flat = jnp.reshape(w_eff_roi, (n_pix,))
        s_flat = jnp.reshape(s_eff_roi, (n_pix,))
        b_flat = jnp.reshape(b_roi, (n_pix,))

        out_flat, mean_bm, rstd_bm = pl.pallas_call(
            _make_roi_epilogue_fwd_kernel(
                n_pix, n_pad, act_name=act_name, neg_slope=neg_slope, has_border=has_border
            ),
            out_shape=(
                jax.ShapeDtypeStruct((bsz, n_pix), jnp.float32),
                jax.ShapeDtypeStruct((bsz, m_dirs), jnp.float32),
                jax.ShapeDtypeStruct((bsz, m_dirs), jnp.float32),
            ),
            in_specs=[
                pl.BlockSpec((1, m_dirs, n_pad), lambda i: (i, 0, 0)),
                pl.BlockSpec((1, m_dirs, n_pad), lambda i: (0, 0, 0)),
                pl.BlockSpec((n_pad,), lambda i: (0,)),
                pl.BlockSpec((n_pad,), lambda i: (0,)),
                pl.BlockSpec((n_pad,), lambda i: (0,)),
            ],
            out_specs=(
                pl.BlockSpec((1, n_pad), lambda i: (i, 0)),
                pl.BlockSpec((1, m_dirs), lambda i: (i, 0)),
                pl.BlockSpec((1, m_dirs), lambda i: (i, 0)),
            ),
            grid=(bsz,),
        )(z_flat, db_flat, w_flat, s_flat, b_flat)

        return jnp.reshape(out_flat, (bsz, h_v, w_v)), mean_bm, rstd_bm

    # Hardware-tiled path for N_PIX > 2048
    block_n = _select_tile_block_n(n_pix)
    if n_pix % block_n == 0:
        num_tiles = n_pix // block_n
        z_tiled = jnp.reshape(z_bmhw, (bsz, m_dirs, num_tiles, block_n))
        db_tiled = jnp.reshape(delta_border_1mhw, (1, m_dirs, num_tiles, block_n))
        w_tiled = jnp.reshape(w_eff_roi, (num_tiles, block_n))
        s_tiled = jnp.reshape(s_eff_roi, (num_tiles, block_n))
        b_tiled = jnp.reshape(b_roi, (num_tiles, block_n))

        mean_bm, rstd_bm = _compute_tiled_mean_rstd(
            z_tiled, db_tiled, n_pix=n_pix, num_tiles=num_tiles, block_n=block_n, has_border=has_border
        )

        out_tiled = pl.pallas_call(
            _make_tiled_codd_fwd_kernel(
                block_n, act_name=act_name, neg_slope=neg_slope, has_border=has_border
            ),
            out_shape=jax.ShapeDtypeStruct((bsz, num_tiles, block_n), jnp.float32),
            in_specs=[
                pl.BlockSpec((1, m_dirs, 1, block_n), lambda b, t: (b, 0, t, 0)),
                pl.BlockSpec((1, m_dirs, 1, block_n), lambda b, t: (0, 0, t, 0)),
                pl.BlockSpec((1, block_n), lambda b, t: (t, 0)),
                pl.BlockSpec((1, block_n), lambda b, t: (t, 0)),
                pl.BlockSpec((1, block_n), lambda b, t: (t, 0)),
                pl.BlockSpec((1, m_dirs), lambda b, t: (b, 0)),
                pl.BlockSpec((1, m_dirs), lambda b, t: (b, 0)),
            ],
            out_specs=pl.BlockSpec((1, 1, block_n), lambda b, t: (b, t, 0)),
            grid=(bsz, num_tiles),
        )(z_tiled, db_tiled, w_tiled, s_tiled, b_tiled, mean_bm, rstd_bm)

        return jnp.reshape(out_tiled, (bsz, h_v, w_v)), mean_bm, rstd_bm

    # Fallback for large non-multiple-of-128 grids (e.g. 62x62 = 3844): pad to multiple of block_n
    num_tiles = (n_pix + block_n - 1) // block_n
    n_padded = num_tiles * block_n
    pad_rem = n_padded - n_pix
    z_flat = jnp.reshape(z_bmhw, (bsz, m_dirs, n_pix))
    db_flat = jnp.reshape(delta_border_1mhw, (1, m_dirs, n_pix))

    if n_pix <= MEAN_RSTD_SINGLE_PLANE_MAX_PIX:
        n_pad_pow2 = _next_pow2(n_pix)
        mean_bm, rstd_bm = pl.pallas_call(
            _make_mean_rstd_fwd_kernel(n_pix, n_pad_pow2, has_border=has_border),
            out_shape=(
                jax.ShapeDtypeStruct((bsz, m_dirs), jnp.float32),
                jax.ShapeDtypeStruct((bsz, m_dirs), jnp.float32),
            ),
            in_specs=[
                pl.BlockSpec((1, 1, n_pad_pow2), lambda b, m: (b, m, 0)),
                pl.BlockSpec((1, 1, n_pad_pow2), lambda b, m: (0, m, 0)),
            ],
            out_specs=(
                pl.BlockSpec((1, 1), lambda b, m: (b, m)),
                pl.BlockSpec((1, 1), lambda b, m: (b, m)),
            ),
            grid=(bsz, m_dirs),
        )(z_flat, db_flat)
    else:
        z_tot = z_bmhw + delta_border_1mhw if has_border else z_bmhw
        mean_bm = jnp.mean(z_tot, axis=(-2, -1))
        var_bm = jnp.mean(jnp.square(z_tot - mean_bm[:, :, jnp.newaxis, jnp.newaxis]), axis=(-2, -1))
        rstd_bm = jax.lax.rsqrt(var_bm + 1e-5)

    z_padded = jnp.pad(z_flat, ((0, 0), (0, 0), (0, pad_rem)))
    db_padded = jnp.pad(db_flat, ((0, 0), (0, 0), (0, pad_rem)))
    w_padded = jnp.pad(jnp.reshape(w_eff_roi, (n_pix,)), ((0, pad_rem),))
    s_padded = jnp.pad(jnp.reshape(s_eff_roi, (n_pix,)), ((0, pad_rem),))
    b_padded = jnp.pad(jnp.reshape(b_roi, (n_pix,)), ((0, pad_rem),))

    out_tiled = pl.pallas_call(
        _make_tiled_codd_fwd_kernel(
            block_n, act_name=act_name, neg_slope=neg_slope, has_border=has_border
        ),
        out_shape=jax.ShapeDtypeStruct((bsz, num_tiles, block_n), jnp.float32),
        in_specs=[
            pl.BlockSpec((1, m_dirs, 1, block_n), lambda b, t: (b, 0, t, 0)),
            pl.BlockSpec((1, m_dirs, 1, block_n), lambda b, t: (0, 0, t, 0)),
            pl.BlockSpec((1, block_n), lambda b, t: (t, 0)),
            pl.BlockSpec((1, block_n), lambda b, t: (t, 0)),
            pl.BlockSpec((1, block_n), lambda b, t: (t, 0)),
            pl.BlockSpec((1, m_dirs), lambda b, t: (b, 0)),
            pl.BlockSpec((1, m_dirs), lambda b, t: (b, 0)),
        ],
        out_specs=pl.BlockSpec((1, 1, block_n), lambda b, t: (b, t, 0)),
        grid=(bsz, num_tiles),
    )(
        jnp.reshape(z_padded, (bsz, m_dirs, num_tiles, block_n)),
        jnp.reshape(db_padded, (1, m_dirs, num_tiles, block_n)),
        jnp.reshape(w_padded, (num_tiles, block_n)),
        jnp.reshape(s_padded, (num_tiles, block_n)),
        jnp.reshape(b_padded, (num_tiles, block_n)),
        mean_bm,
        rstd_bm,
    )
    out_flat = jnp.reshape(out_tiled, (bsz, n_padded))[:, :n_pix]
    return jnp.reshape(out_flat, (bsz, h_v, w_v)), mean_bm, rstd_bm


def _pallas_roi_epilogue_fwd_rule(
    z_bmhw: jax.Array,
    delta_border_1mhw: jax.Array,
    w_eff_roi: jax.Array,
    s_eff_roi: jax.Array,
    b_roi: jax.Array,
    activate=jax.nn.tanh,
    has_border: bool = True,
):
    out_hw, mean_bm, rstd_bm = _pallas_roi_epilogue_fwd_impl(
        z_bmhw,
        delta_border_1mhw,
        w_eff_roi,
        s_eff_roi,
        b_roi,
        activate=activate,
        has_border=has_border,
    )
    return out_hw, (
        z_bmhw,
        delta_border_1mhw,
        w_eff_roi,
        s_eff_roi,
        b_roi,
        mean_bm,
        rstd_bm,
    )


def _pallas_roi_epilogue_bwd_rule(activate, has_border: bool, res, gy_bhw: jax.Array):
    z_bmhw, delta_border_1mhw, w_eff_roi, s_eff_roi, b_roi, mean_bm, rstd_bm = res
    bsz, m_dirs, h_v, w_v = z_bmhw.shape
    n_pix = h_v * w_v
    act_name, neg_slope = _resolve_activation_spec(activate)

    if n_pix <= SINGLE_TILE_MAX_PIX:
        n_pad = _next_pow2(n_pix)
        z_flat = jnp.reshape(z_bmhw, (bsz, m_dirs, n_pix))
        db_flat = jnp.reshape(delta_border_1mhw, (1, m_dirs, n_pix))
        w_flat = jnp.reshape(w_eff_roi, (n_pix,))
        s_flat = jnp.reshape(s_eff_roi, (n_pix,))
        b_flat = jnp.reshape(b_roi, (n_pix,))
        gy_flat = jnp.reshape(gy_bhw, (bsz, n_pix))

        gz_flat, gw_b, gs_b, gb_b = pl.pallas_call(
            _make_roi_epilogue_bwd_kernel(
                n_pix, n_pad, act_name=act_name, neg_slope=neg_slope, has_border=has_border
            ),
            out_shape=(
                jax.ShapeDtypeStruct((bsz, m_dirs, n_pix), jnp.float32),
                jax.ShapeDtypeStruct((bsz, n_pix), jnp.float32),
                jax.ShapeDtypeStruct((bsz, n_pix), jnp.float32),
                jax.ShapeDtypeStruct((bsz, n_pix), jnp.float32),
            ),
            in_specs=[
                pl.BlockSpec((1, m_dirs, n_pad), lambda i: (i, 0, 0)),
                pl.BlockSpec((1, m_dirs, n_pad), lambda i: (0, 0, 0)),
                pl.BlockSpec((n_pad,), lambda i: (0,)),
                pl.BlockSpec((n_pad,), lambda i: (0,)),
                pl.BlockSpec((n_pad,), lambda i: (0,)),
                pl.BlockSpec((1, m_dirs), lambda i: (i, 0)),
                pl.BlockSpec((1, m_dirs), lambda i: (i, 0)),
                pl.BlockSpec((1, n_pad), lambda i: (i, 0)),
            ],
            out_specs=(
                pl.BlockSpec((1, m_dirs, n_pad), lambda i: (i, 0, 0)),
                pl.BlockSpec((1, n_pad), lambda i: (i, 0)),
                pl.BlockSpec((1, n_pad), lambda i: (i, 0)),
                pl.BlockSpec((1, n_pad), lambda i: (i, 0)),
            ),
            grid=(bsz,),
        )(z_flat, db_flat, w_flat, s_flat, b_flat, mean_bm, rstd_bm, gy_flat)

        gz = jnp.reshape(gz_flat, (bsz, m_dirs, h_v, w_v))
        gdb = jnp.sum(gz, axis=0, keepdims=True) if has_border else jnp.zeros_like(delta_border_1mhw)
        gw = jnp.reshape(jnp.sum(gw_b, axis=0), (h_v, w_v))
        gs = jnp.reshape(jnp.sum(gs_b, axis=0), (h_v, w_v))
        gb = jnp.reshape(jnp.sum(gb_b, axis=0), (h_v, w_v))
        return gz, gdb, gw, gs, gb

    # Hardware-tiled backward pass for N_PIX > 2048
    block_n = _select_tile_block_n(n_pix)
    inv_n = 1.0 / float(n_pix)
    if n_pix % block_n == 0:
        num_tiles = n_pix // block_n
        z_tiled = jnp.reshape(z_bmhw, (bsz, m_dirs, num_tiles, block_n))
        db_tiled = jnp.reshape(delta_border_1mhw, (1, m_dirs, num_tiles, block_n))
        w_tiled = jnp.reshape(w_eff_roi, (num_tiles, block_n))
        s_tiled = jnp.reshape(s_eff_roi, (num_tiles, block_n))
        b_tiled = jnp.reshape(b_roi, (num_tiles, block_n))
        gy_tiled = jnp.reshape(gy_bhw, (bsz, num_tiles, block_n))

        gw_bt, gs_bt, gb_bt, c1_part, c2_part = pl.pallas_call(
            _make_tiled_bwd_stage1_kernel(
                block_n, act_name=act_name, neg_slope=neg_slope, has_border=has_border
            ),
            out_shape=(
                jax.ShapeDtypeStruct((bsz, num_tiles, block_n), jnp.float32),
                jax.ShapeDtypeStruct((bsz, num_tiles, block_n), jnp.float32),
                jax.ShapeDtypeStruct((bsz, num_tiles, block_n), jnp.float32),
                jax.ShapeDtypeStruct((bsz, num_tiles, m_dirs), jnp.float32),
                jax.ShapeDtypeStruct((bsz, num_tiles, m_dirs), jnp.float32),
            ),
            in_specs=[
                pl.BlockSpec((1, m_dirs, 1, block_n), lambda b, t: (b, 0, t, 0)),
                pl.BlockSpec((1, m_dirs, 1, block_n), lambda b, t: (0, 0, t, 0)),
                pl.BlockSpec((1, block_n), lambda b, t: (t, 0)),
                pl.BlockSpec((1, block_n), lambda b, t: (t, 0)),
                pl.BlockSpec((1, block_n), lambda b, t: (t, 0)),
                pl.BlockSpec((1, m_dirs), lambda b, t: (b, 0)),
                pl.BlockSpec((1, m_dirs), lambda b, t: (b, 0)),
                pl.BlockSpec((1, 1, block_n), lambda b, t: (b, t, 0)),
            ],
            out_specs=(
                pl.BlockSpec((1, 1, block_n), lambda b, t: (b, t, 0)),
                pl.BlockSpec((1, 1, block_n), lambda b, t: (b, t, 0)),
                pl.BlockSpec((1, 1, block_n), lambda b, t: (b, t, 0)),
                pl.BlockSpec((1, 1, m_dirs), lambda b, t: (b, t, 0)),
                pl.BlockSpec((1, 1, m_dirs), lambda b, t: (b, t, 0)),
            ),
            grid=(bsz, num_tiles),
        )(z_tiled, db_tiled, w_tiled, s_tiled, b_tiled, mean_bm, rstd_bm, gy_tiled)

        c1_bm = jnp.sum(c1_part, axis=1) * inv_n
        c2_bm = jnp.sum(c2_part, axis=1) * inv_n

        gz_tiled = _run_tiled_bwd_stage2(
            z_tiled,
            db_tiled,
            w_tiled,
            s_tiled,
            b_tiled,
            mean_bm,
            rstd_bm,
            c1_bm,
            c2_bm,
            gy_tiled,
            block_n=block_n,
            act_name=act_name,
            neg_slope=neg_slope,
            has_border=has_border,
        )

        gz = jnp.reshape(gz_tiled, (bsz, m_dirs, h_v, w_v))
        gdb = jnp.sum(gz, axis=0, keepdims=True) if has_border else jnp.zeros_like(delta_border_1mhw)
        gw = jnp.reshape(jnp.sum(gw_bt, axis=0), (h_v, w_v))
        gs = jnp.reshape(jnp.sum(gs_bt, axis=0), (h_v, w_v))
        gb = jnp.reshape(jnp.sum(gb_bt, axis=0), (h_v, w_v))
        return gz, gdb, gw, gs, gb

    # Fallback for non-divisible large grids
    num_tiles = (n_pix + block_n - 1) // block_n
    n_padded = num_tiles * block_n
    pad_rem = n_padded - n_pix
    z_padded = jnp.pad(jnp.reshape(z_bmhw, (bsz, m_dirs, n_pix)), ((0, 0), (0, 0), (0, pad_rem)))
    db_padded = jnp.pad(jnp.reshape(delta_border_1mhw, (1, m_dirs, n_pix)), ((0, 0), (0, 0), (0, pad_rem)))
    w_padded = jnp.pad(jnp.reshape(w_eff_roi, (n_pix,)), ((0, pad_rem),))
    s_padded = jnp.pad(jnp.reshape(s_eff_roi, (n_pix,)), ((0, pad_rem),))
    b_padded = jnp.pad(jnp.reshape(b_roi, (n_pix,)), ((0, pad_rem),))
    gy_padded = jnp.pad(jnp.reshape(gy_bhw, (bsz, n_pix)), ((0, 0), (0, pad_rem)))

    z_tiled = jnp.reshape(z_padded, (bsz, m_dirs, num_tiles, block_n))
    db_tiled = jnp.reshape(db_padded, (1, m_dirs, num_tiles, block_n))
    w_tiled = jnp.reshape(w_padded, (num_tiles, block_n))
    s_tiled = jnp.reshape(s_padded, (num_tiles, block_n))
    b_tiled = jnp.reshape(b_padded, (num_tiles, block_n))
    gy_tiled = jnp.reshape(gy_padded, (bsz, num_tiles, block_n))

    gw_bt, gs_bt, gb_bt, c1_part, c2_part = pl.pallas_call(
        _make_tiled_bwd_stage1_kernel(
            block_n, act_name=act_name, neg_slope=neg_slope, has_border=has_border
        ),
        out_shape=(
            jax.ShapeDtypeStruct((bsz, num_tiles, block_n), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_tiles, block_n), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_tiles, block_n), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_tiles, m_dirs), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_tiles, m_dirs), jnp.float32),
        ),
        in_specs=[
            pl.BlockSpec((1, m_dirs, 1, block_n), lambda b, t: (b, 0, t, 0)),
            pl.BlockSpec((1, m_dirs, 1, block_n), lambda b, t: (0, 0, t, 0)),
            pl.BlockSpec((1, block_n), lambda b, t: (t, 0)),
            pl.BlockSpec((1, block_n), lambda b, t: (t, 0)),
            pl.BlockSpec((1, block_n), lambda b, t: (t, 0)),
            pl.BlockSpec((1, m_dirs), lambda b, t: (b, 0)),
            pl.BlockSpec((1, m_dirs), lambda b, t: (b, 0)),
            pl.BlockSpec((1, 1, block_n), lambda b, t: (b, t, 0)),
        ],
        out_specs=(
            pl.BlockSpec((1, 1, block_n), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, 1, block_n), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, 1, block_n), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, 1, m_dirs), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, 1, m_dirs), lambda b, t: (b, t, 0)),
        ),
        grid=(bsz, num_tiles),
    )(z_tiled, db_tiled, w_tiled, s_tiled, b_tiled, mean_bm, rstd_bm, gy_tiled)

    c1_bm = jnp.sum(c1_part, axis=1) * inv_n
    c2_bm = jnp.sum(c2_part, axis=1) * inv_n

    gz_tiled = _run_tiled_bwd_stage2(
        z_tiled,
        db_tiled,
        w_tiled,
        s_tiled,
        b_tiled,
        mean_bm,
        rstd_bm,
        c1_bm,
        c2_bm,
        gy_tiled,
        block_n=block_n,
        act_name=act_name,
        neg_slope=neg_slope,
        has_border=has_border,
    )

    gz = jnp.reshape(jnp.reshape(gz_tiled, (bsz, m_dirs, n_padded))[:, :, :n_pix], (bsz, m_dirs, h_v, w_v))
    gdb = jnp.sum(gz, axis=0, keepdims=True) if has_border else jnp.zeros_like(delta_border_1mhw)
    gw = jnp.reshape(jnp.sum(jnp.reshape(gw_bt, (bsz, n_padded))[:, :n_pix], axis=0), (h_v, w_v))
    gs = jnp.reshape(jnp.sum(jnp.reshape(gs_bt, (bsz, n_padded))[:, :n_pix], axis=0), (h_v, w_v))
    gb = jnp.reshape(jnp.sum(jnp.reshape(gb_bt, (bsz, n_padded))[:, :n_pix], axis=0), (h_v, w_v))
    return gz, gdb, gw, gs, gb


pallas_fused_roi_step_epilogue.defvjp(
    _pallas_roi_epilogue_fwd_rule, _pallas_roi_epilogue_bwd_rule
)


# ============================================================================
# 2. Pallas Final-Step Output-Port Standardization & Epilogue
# ============================================================================

def _make_mean_rstd_fwd_kernel(n_pix: int, n_pad: int, has_border: bool = True):
    inv_n = 1.0 / float(n_pix)

    def _kernel(
        z_ref,          # (1, 1, n_pix)
        dborder_ref,    # (1, 1, n_pix)
        mean_ref,       # (1, 1)
        rstd_ref,       # (1, 1)
    ):
        idx = jnp.arange(n_pad)
        mask = idx < n_pix
        mask_f = mask.astype(jnp.float32)

        z_m = pl_triton.load(z_ref.at[0, 0, :], mask=mask, other=0.0)
        if has_border:
            db_m = pl_triton.load(dborder_ref.at[0, 0, :], mask=mask, other=0.0)
            z_tot = z_m + db_m
        else:
            z_tot = z_m

        mean_m = jnp.sum(z_tot) * inv_n
        diff_m = (z_tot - mean_m) * mask_f
        var_m = jnp.sum(diff_m * diff_m) * inv_n
        rstd_m = jax.lax.rsqrt(var_m + 1e-5)

        mean_ref[0, 0] = mean_m
        rstd_ref[0, 0] = rstd_m

    return _kernel


def _make_sparse_port_bwd_kernel(
    n_pix: int,
    n_pad: int,
    port_flat_indices: tuple[int, ...] = (),
    has_border: bool = True,
):
    fuse_port = len(port_flat_indices) > 0

    def _kernel_fused(
        z_ref,          # (1, 1, n_pix)
        dborder_ref,    # (1, 1, n_pix)
        mean_ref,       # (1, 1)
        rstd_ref,       # (1, 1)
        c1_ref,         # (1, 1)
        c2_ref,         # (1, 1)
        gz_out_ref,     # (1, 1, out_pts, 1)
        gz_base_ref,    # (1, 1, n_pix)
    ):
        idx = jnp.arange(n_pad)
        mask = idx < n_pix
        mask_f = mask.astype(jnp.float32)

        z_m = pl_triton.load(z_ref.at[0, 0, :], mask=mask, other=0.0)
        if has_border:
            db_m = pl_triton.load(dborder_ref.at[0, 0, :], mask=mask, other=0.0)
            z_tot = z_m + db_m
        else:
            z_tot = z_m

        mean_m = mean_ref[0, 0]
        rstd_m = rstd_ref[0, 0]
        c1 = c1_ref[0, 0]
        c2 = c2_ref[0, 0]

        z_norm = (z_tot - mean_m) * rstd_m * mask_f
        gz_base = -rstd_m * (c1 + z_norm * c2) * mask_f

        for k, flat_idx in enumerate(port_flat_indices):
            gz_k = rstd_m * gz_out_ref[0, 0, k, 0]
            gz_base = jnp.where(idx == flat_idx, gz_base + gz_k, gz_base)

        pl_triton.store(gz_base_ref.at[0, 0, :], gz_base, mask=mask)

    def _kernel_unfused(
        z_ref,          # (1, 1, n_pix)
        dborder_ref,    # (1, 1, n_pix)
        mean_ref,       # (1, 1)
        rstd_ref,       # (1, 1)
        c1_ref,         # (1, 1)
        c2_ref,         # (1, 1)
        gz_base_ref,    # (1, 1, n_pix)
    ):
        idx = jnp.arange(n_pad)
        mask = idx < n_pix
        mask_f = mask.astype(jnp.float32)

        z_m = pl_triton.load(z_ref.at[0, 0, :], mask=mask, other=0.0)
        if has_border:
            db_m = pl_triton.load(dborder_ref.at[0, 0, :], mask=mask, other=0.0)
            z_tot = z_m + db_m
        else:
            z_tot = z_m

        mean_m = mean_ref[0, 0]
        rstd_m = rstd_ref[0, 0]
        c1 = c1_ref[0, 0]
        c2 = c2_ref[0, 0]

        z_norm = (z_tot - mean_m) * rstd_m * mask_f
        gz_base = -rstd_m * (c1 + z_norm * c2) * mask_f
        pl_triton.store(gz_base_ref.at[0, 0, :], gz_base, mask=mask)

    return _kernel_fused if fuse_port else _kernel_unfused


def _make_tiled_sparse_port_bwd_kernel(
    block_n: int,
    port_flat_indices: tuple[int, ...] = (),
    has_border: bool = True,
):
    """
    Parallel `(B, M, num_tiles)` sparse output-port backward kernel.
    When `port_flat_indices` is provided (`<= 64` output pixels), fuses the output-port
    gradient scatter directly in registers before writing each tile to HBM.
    """
    fuse_port = len(port_flat_indices) > 0

    def _kernel_fused(
        z_ref,          # (1, 1, 1, block_n)
        dborder_ref,    # (1, 1, 1, block_n)
        mean_ref,       # (1, 1)
        rstd_ref,       # (1, 1)
        c1_ref,         # (1, 1)
        c2_ref,         # (1, 1)
        gz_out_ref,     # (1, 1, out_pts, 1)
        gz_base_ref,    # (1, 1, 1, block_n)
    ):
        t_idx = pl.program_id(2)
        idx = t_idx * block_n + jnp.arange(block_n)

        z_m = pl_triton.load(z_ref.at[0, 0, 0, :])
        if has_border:
            db_m = pl_triton.load(dborder_ref.at[0, 0, 0, :])
            z_tot = z_m + db_m
        else:
            z_tot = z_m

        mean_m = mean_ref[0, 0]
        rstd_m = rstd_ref[0, 0]
        c1 = c1_ref[0, 0]
        c2 = c2_ref[0, 0]

        z_norm = (z_tot - mean_m) * rstd_m
        gz_base = -rstd_m * (c1 + z_norm * c2)

        for k, flat_idx in enumerate(port_flat_indices):
            gz_k = rstd_m * gz_out_ref[0, 0, k, 0]
            gz_base = jnp.where(idx == flat_idx, gz_base + gz_k, gz_base)

        pl_triton.store(gz_base_ref.at[0, 0, 0, :], gz_base)

    def _kernel_unfused(
        z_ref,          # (1, 1, 1, block_n)
        dborder_ref,    # (1, 1, 1, block_n)
        mean_ref,       # (1, 1)
        rstd_ref,       # (1, 1)
        c1_ref,         # (1, 1)
        c2_ref,         # (1, 1)
        gz_base_ref,    # (1, 1, 1, block_n)
    ):
        z_m = pl_triton.load(z_ref.at[0, 0, 0, :])
        if has_border:
            db_m = pl_triton.load(dborder_ref.at[0, 0, 0, :])
            z_tot = z_m + db_m
        else:
            z_tot = z_m

        mean_m = mean_ref[0, 0]
        rstd_m = rstd_ref[0, 0]
        c1 = c1_ref[0, 0]
        c2 = c2_ref[0, 0]

        z_norm = (z_tot - mean_m) * rstd_m
        gz_base = -rstd_m * (c1 + z_norm * c2)
        pl_triton.store(gz_base_ref.at[0, 0, 0, :], gz_base)

    return _kernel_fused if fuse_port else _kernel_unfused


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def pallas_standardize_output_port(
    z_bmhw: jax.Array,
    delta_border_1mhw: jax.Array,
    roi_out_offset: tuple = (16, 35),
    out_size: tuple[int, int] = (8, 1),
    has_border: bool = True,
) -> jax.Array:
    """
    Computes the standardized output port slice or coordinate gather
    `standardize(z_bmhw + (delta_border if has_border else 0))` at the output port
    by reducing `(mean, rstd)` in a Pallas kernel (`16 KB` output) and applying the affine
    standardization ONLY to the output port pixels.
    """
    z_out_norm, _ = _pallas_std_out_fwd(
        z_bmhw, delta_border_1mhw, roi_out_offset, out_size, has_border
    )
    return z_out_norm


def _is_coord_offset(roi_out_offset: tuple) -> bool:
    return (
        isinstance(roi_out_offset, tuple)
        and len(roi_out_offset) == 2
        and isinstance(roi_out_offset[0], tuple)
    )


def _pallas_std_out_fwd(
    z_bmhw: jax.Array,
    delta_border_1mhw: jax.Array,
    roi_out_offset: tuple,
    out_size: tuple[int, int],
    has_border: bool = True,
):
    bsz, m_dirs, h_v, w_v = z_bmhw.shape
    n_pix = h_v * w_v

    if n_pix <= MEAN_RSTD_SINGLE_PLANE_MAX_PIX:
        n_pad = _next_pow2(n_pix)
        z_flat = jnp.reshape(z_bmhw, (bsz, m_dirs, n_pix))
        db_flat = jnp.reshape(delta_border_1mhw, (1, m_dirs, n_pix))

        mean_bm, rstd_bm = pl.pallas_call(
            _make_mean_rstd_fwd_kernel(n_pix, n_pad, has_border=has_border),
            out_shape=(
                jax.ShapeDtypeStruct((bsz, m_dirs), jnp.float32),
                jax.ShapeDtypeStruct((bsz, m_dirs), jnp.float32),
            ),
            in_specs=[
                pl.BlockSpec((1, 1, n_pad), lambda b, m: (b, m, 0)),
                pl.BlockSpec((1, 1, n_pad), lambda b, m: (0, m, 0)),
            ],
            out_specs=(
                pl.BlockSpec((1, 1), lambda b, m: (b, m)),
                pl.BlockSpec((1, 1), lambda b, m: (b, m)),
            ),
            grid=(bsz, m_dirs),
        )(z_flat, db_flat)
    else:
        block_n = _select_tile_block_n(n_pix)
        if n_pix % block_n == 0:
            num_tiles = n_pix // block_n
            z_tiled = jnp.reshape(z_bmhw, (bsz, m_dirs, num_tiles, block_n))
            db_tiled = jnp.reshape(delta_border_1mhw, (1, m_dirs, num_tiles, block_n))
            mean_bm, rstd_bm = _compute_tiled_mean_rstd(
                z_tiled, db_tiled, n_pix=n_pix, num_tiles=num_tiles, block_n=block_n, has_border=has_border
            )
        else:
            z_tot = z_bmhw + delta_border_1mhw if has_border else z_bmhw
            mean_bm = jnp.mean(z_tot, axis=(-2, -1))
            var_bm = jnp.mean(jnp.square(z_tot - mean_bm[:, :, jnp.newaxis, jnp.newaxis]), axis=(-2, -1))
            rstd_bm = jax.lax.rsqrt(var_bm + 1e-5)

    out_h, out_w = out_size
    if _is_coord_offset(roi_out_offset):
        rows_idx = jnp.asarray(roi_out_offset[0], dtype=jnp.int32)
        cols_idx = jnp.asarray(roi_out_offset[1], dtype=jnp.int32)
        z_slice = z_bmhw[:, :, rows_idx, cols_idx]
        if has_border:
            z_slice = z_slice + delta_border_1mhw[:, :, rows_idx, cols_idx]
        z_slice = jnp.reshape(z_slice, (bsz, m_dirs, out_h, out_w))
    else:
        ro_r, ro_c = roi_out_offset
        z_slice = z_bmhw[:, :, ro_r : ro_r + out_h, ro_c : ro_c + out_w]
        if has_border:
            z_slice = z_slice + delta_border_1mhw[:, :, ro_r : ro_r + out_h, ro_c : ro_c + out_w]

    z_out_norm = (
        z_slice - mean_bm[:, :, jnp.newaxis, jnp.newaxis]
    ) * rstd_bm[:, :, jnp.newaxis, jnp.newaxis]
    return z_out_norm, (z_bmhw, delta_border_1mhw, mean_bm, rstd_bm, z_out_norm)


def _pallas_std_out_bwd(
    roi_out_offset: tuple,
    out_size: tuple[int, int],
    has_border: bool,
    res,
    gz_out_norm: jax.Array,
):
    z_bmhw, delta_border_1mhw, mean_bm, rstd_bm, z_out_norm = res
    bsz, m_dirs, h_v, w_v = z_bmhw.shape
    n_pix = h_v * w_v
    inv_n = 1.0 / float(n_pix)

    c1_bm = jnp.sum(gz_out_norm, axis=(-2, -1)) * inv_n
    c2_bm = jnp.sum(gz_out_norm * z_out_norm, axis=(-2, -1)) * inv_n

    out_h, out_w = out_size
    is_coords = _is_coord_offset(roi_out_offset)

    # Fast path when output port covers the entire ROI (e.g. full-grid Game of Life)
    if not is_coords and roi_out_offset == (0, 0) and (out_h, out_w) == (h_v, w_v):
        gz_bmhw = rstd_bm[:, :, jnp.newaxis, jnp.newaxis] * (
            gz_out_norm
            - c1_bm[:, :, jnp.newaxis, jnp.newaxis]
            - z_out_norm * c2_bm[:, :, jnp.newaxis, jnp.newaxis]
        )
        gdb = jnp.sum(gz_bmhw, axis=0, keepdims=True) if has_border else jnp.zeros_like(delta_border_1mhw)
        return gz_bmhw, gdb

    # Compute static 1D flat indices for in-register Pallas port gradient fusion (when <= 64 pts)
    if is_coords:
        out_pts = len(roi_out_offset[0])
        port_flat_indices = (
            tuple(int(r) * w_v + int(c) for r, c in zip(roi_out_offset[0], roi_out_offset[1]))
            if out_pts <= 64
            else ()
        )
    else:
        ro_r, ro_c = roi_out_offset
        out_pts = out_h * out_w
        port_flat_indices = (
            tuple((ro_r + r) * w_v + (ro_c + c) for r in range(out_h) for c in range(out_w))
            if out_pts <= 64
            else ()
        )
    fuse_port_in_kernel = len(port_flat_indices) > 0

    if n_pix <= MEAN_RSTD_SINGLE_PLANE_MAX_PIX:
        n_pad = _next_pow2(n_pix)
        z_flat = jnp.reshape(z_bmhw, (bsz, m_dirs, n_pix))
        db_flat = jnp.reshape(delta_border_1mhw, (1, m_dirs, n_pix))

        if fuse_port_in_kernel:
            gz_out_4d = jnp.reshape(gz_out_norm, (bsz, m_dirs, out_pts, 1))
            gz_base_flat = pl.pallas_call(
                _make_sparse_port_bwd_kernel(
                    n_pix, n_pad, port_flat_indices=port_flat_indices, has_border=has_border
                ),
                out_shape=jax.ShapeDtypeStruct((bsz, m_dirs, n_pix), jnp.float32),
                in_specs=[
                    pl.BlockSpec((1, 1, n_pad), lambda b, m: (b, m, 0)),
                    pl.BlockSpec((1, 1, n_pad), lambda b, m: (0, m, 0)),
                    pl.BlockSpec((1, 1), lambda b, m: (b, m)),
                    pl.BlockSpec((1, 1), lambda b, m: (b, m)),
                    pl.BlockSpec((1, 1), lambda b, m: (b, m)),
                    pl.BlockSpec((1, 1), lambda b, m: (b, m)),
                    pl.BlockSpec((1, 1, out_pts, 1), lambda b, m: (b, m, 0, 0)),
                ],
                out_specs=pl.BlockSpec((1, 1, n_pad), lambda b, m: (b, m, 0)),
                grid=(bsz, m_dirs),
            )(z_flat, db_flat, mean_bm, rstd_bm, c1_bm, c2_bm, gz_out_4d)
            gz_bmhw = jnp.reshape(gz_base_flat, (bsz, m_dirs, h_v, w_v))
            gdb = jnp.sum(gz_bmhw, axis=0, keepdims=True) if has_border else jnp.zeros_like(delta_border_1mhw)
            return gz_bmhw, gdb

        gz_base_flat = pl.pallas_call(
            _make_sparse_port_bwd_kernel(n_pix, n_pad, has_border=has_border),
            out_shape=jax.ShapeDtypeStruct((bsz, m_dirs, n_pix), jnp.float32),
            in_specs=[
                pl.BlockSpec((1, 1, n_pad), lambda b, m: (b, m, 0)),
                pl.BlockSpec((1, 1, n_pad), lambda b, m: (0, m, 0)),
                pl.BlockSpec((1, 1), lambda b, m: (b, m)),
                pl.BlockSpec((1, 1), lambda b, m: (b, m)),
                pl.BlockSpec((1, 1), lambda b, m: (b, m)),
                pl.BlockSpec((1, 1), lambda b, m: (b, m)),
            ],
            out_specs=pl.BlockSpec((1, 1, n_pad), lambda b, m: (b, m, 0)),
            grid=(bsz, m_dirs),
        )(z_flat, db_flat, mean_bm, rstd_bm, c1_bm, c2_bm)
        gz_bmhw = jnp.reshape(gz_base_flat, (bsz, m_dirs, h_v, w_v))
    else:
        block_n = _select_tile_block_n(n_pix)
        if n_pix % block_n == 0:
            num_tiles = n_pix // block_n
            z_tiled = jnp.reshape(z_bmhw, (bsz, m_dirs, num_tiles, block_n))
            db_tiled = jnp.reshape(delta_border_1mhw, (1, m_dirs, num_tiles, block_n))
            if fuse_port_in_kernel:
                gz_out_4d = jnp.reshape(gz_out_norm, (bsz, m_dirs, out_pts, 1))
                gz_tiled = pl.pallas_call(
                    _make_tiled_sparse_port_bwd_kernel(
                        block_n, port_flat_indices=port_flat_indices, has_border=has_border
                    ),
                    out_shape=jax.ShapeDtypeStruct((bsz, m_dirs, num_tiles, block_n), jnp.float32),
                    in_specs=[
                        pl.BlockSpec((1, 1, 1, block_n), lambda b, m, t: (b, m, t, 0)),
                        pl.BlockSpec((1, 1, 1, block_n), lambda b, m, t: (0, m, t, 0)),
                        pl.BlockSpec((1, 1), lambda b, m, t: (b, m)),
                        pl.BlockSpec((1, 1), lambda b, m, t: (b, m)),
                        pl.BlockSpec((1, 1), lambda b, m, t: (b, m)),
                        pl.BlockSpec((1, 1), lambda b, m, t: (b, m)),
                        pl.BlockSpec((1, 1, out_pts, 1), lambda b, m, t: (b, m, 0, 0)),
                    ],
                    out_specs=pl.BlockSpec((1, 1, 1, block_n), lambda b, m, t: (b, m, t, 0)),
                    grid=(bsz, m_dirs, num_tiles),
                )(z_tiled, db_tiled, mean_bm, rstd_bm, c1_bm, c2_bm, gz_out_4d)
                gz_bmhw = jnp.reshape(gz_tiled, (bsz, m_dirs, h_v, w_v))
                gdb = jnp.sum(gz_bmhw, axis=0, keepdims=True) if has_border else jnp.zeros_like(delta_border_1mhw)
                return gz_bmhw, gdb

            gz_tiled = pl.pallas_call(
                _make_tiled_sparse_port_bwd_kernel(block_n, has_border=has_border),
                out_shape=jax.ShapeDtypeStruct((bsz, m_dirs, num_tiles, block_n), jnp.float32),
                in_specs=[
                    pl.BlockSpec((1, 1, 1, block_n), lambda b, m, t: (b, m, t, 0)),
                    pl.BlockSpec((1, 1, 1, block_n), lambda b, m, t: (0, m, t, 0)),
                    pl.BlockSpec((1, 1), lambda b, m, t: (b, m)),
                    pl.BlockSpec((1, 1), lambda b, m, t: (b, m)),
                    pl.BlockSpec((1, 1), lambda b, m, t: (b, m)),
                    pl.BlockSpec((1, 1), lambda b, m, t: (b, m)),
                ],
                out_specs=pl.BlockSpec((1, 1, 1, block_n), lambda b, m, t: (b, m, t, 0)),
                grid=(bsz, m_dirs, num_tiles),
            )(z_tiled, db_tiled, mean_bm, rstd_bm, c1_bm, c2_bm)
            gz_bmhw = jnp.reshape(gz_tiled, (bsz, m_dirs, h_v, w_v))
        else:
            z_tot = z_bmhw + delta_border_1mhw if has_border else z_bmhw
            z_norm = (
                z_tot - mean_bm[:, :, jnp.newaxis, jnp.newaxis]
            ) * rstd_bm[:, :, jnp.newaxis, jnp.newaxis]
            gz_bmhw = -rstd_bm[:, :, jnp.newaxis, jnp.newaxis] * (
                c1_bm[:, :, jnp.newaxis, jnp.newaxis]
                + z_norm * c2_bm[:, :, jnp.newaxis, jnp.newaxis]
            )

    gz_port = rstd_bm[:, :, jnp.newaxis, jnp.newaxis] * gz_out_norm
    if is_coords:
        rows_idx = jnp.asarray(roi_out_offset[0], dtype=jnp.int32)
        cols_idx = jnp.asarray(roi_out_offset[1], dtype=jnp.int32)
        gz_port_flat = jnp.reshape(gz_port, (bsz, m_dirs, rows_idx.shape[0]))
        gz_bmhw = gz_bmhw.at[:, :, rows_idx, cols_idx].add(gz_port_flat)
    else:
        ro_r, ro_c = roi_out_offset
        gz_bmhw = gz_bmhw.at[:, :, ro_r : ro_r + out_h, ro_c : ro_c + out_w].add(gz_port)

    gdb = jnp.sum(gz_bmhw, axis=0, keepdims=True) if has_border else jnp.zeros_like(delta_border_1mhw)
    return gz_bmhw, gdb


pallas_standardize_output_port.defvjp(_pallas_std_out_fwd, _pallas_std_out_bwd)


def pallas_fused_output_port_epilogue(
    z_roi_bmhw: jax.Array,
    delta_border_1mhw: jax.Array,
    step_params: PrecomputedStepParams,
    topology: ChipTopology2D,
    final_activate=None,
    has_border: bool = True,
) -> jax.Array:
    """
    Final-step (`t = depth`) Pallas output port epilogue that avoids materializing
    the standardized `(B, M, H_v, W_v)` tensor in HBM. Supports rectangular `PortSpec`,
    full-grid outputs, and scattered `CoordinatePortSpec`.
    """
    if isinstance(topology.out_port, CoordinatePortSpec):
        roi_out_offset = (
            tuple(int(r) for r in topology.roi_out_offset[0]),
            tuple(int(c) for c in topology.roi_out_offset[1]),
        )
    else:
        roi_out_offset = topology.roi_out_offset

    z_out = pallas_standardize_output_port(
        z_bmhw=z_roi_bmhw,
        delta_border_1mhw=delta_border_1mhw,
        roi_out_offset=roi_out_offset,
        out_size=topology.out_port.size,
        has_border=has_border,
    )
    sum_lin_out = jnp.sum(z_out, axis=1)
    if final_activate is None:
        m_dirs = z_out.shape[1]
        return (
            (step_params.w_eff_out + step_params.s_eff_out)[jnp.newaxis, :, :] * sum_lin_out
            + (float(m_dirs) * step_params.s_eff_out * step_params.b_out)[jnp.newaxis, :, :]
        )
    b_out_bcast = step_params.b_out[jnp.newaxis, jnp.newaxis, :, :]
    sum_act_out = jnp.sum(final_activate(z_out + b_out_bcast), axis=1)
    outputs = (
        step_params.w_eff_out[jnp.newaxis, :, :] * sum_lin_out
        + step_params.s_eff_out[jnp.newaxis, :, :] * sum_act_out
    )
    return final_activate(outputs)


# ============================================================================
# 3. Zero-Pad Fourier Shift, Pruned Separable RFFT, & Small-Kernel Operators
# ============================================================================

def rfft2_unshifted_filter_grad_to_oihw(
    grad_ker_freq_unshifted_mhw: jax.Array,
    kernel_size: tuple[int, int] = (23, 13),
    pad_hw: tuple[int, int] = (11, 6),
    fft_shape: tuple[int, int] = (64, 64),
) -> jax.Array:
    """
    Recovers spatial `OIHW` filter gradients `(M, 1, Kh, Kw)` from unshifted frequency
    cotangents using a circular `jnp.roll` on the `M=16` `irfft2` output, eliminating
    the `33.55 MB` per-step `jnp.pad` in RFFT backward passes.
    """
    kh, kw = kernel_size
    rh, rw = pad_hw
    grad_flipped_unshifted = jnp.fft.irfft2(
        grad_ker_freq_unshifted_mhw, s=fft_shape, axes=(-2, -1)
    )
    grad_flipped_full = jnp.roll(
        grad_flipped_unshifted, shift=(rh, rw), axis=(-2, -1)
    )
    grad_flipped = grad_flipped_full[:, :kh, :kw]
    grad_spatial = jnp.flip(grad_flipped, axis=(-2, -1))
    return grad_spatial[:, jnp.newaxis, :, :]


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def hoist_filter_rfft2_nopad(
    filters_oihw: jax.Array,
    kernel_size: tuple[int, int] = (23, 13),
    pad_hw: tuple[int, int] = (11, 6),
    fft_shape: tuple[int, int] = (64, 64),
) -> jax.Array:
    return precompute_filter_rfft2(filters_oihw, fft_shape=fft_shape)


def _hoist_nopad_fwd(filters_oihw, kernel_size, pad_hw, fft_shape):
    return precompute_filter_rfft2(filters_oihw, fft_shape=fft_shape), None


def _hoist_nopad_bwd(kernel_size, pad_hw, fft_shape, _, g_ker_freq_mhw):
    return (
        rfft2_unshifted_filter_grad_to_oihw(
            g_ker_freq_mhw, kernel_size=kernel_size, pad_hw=pad_hw, fft_shape=fft_shape
        ),
    )


hoist_filter_rfft2_nopad.defvjp(_hoist_nopad_fwd, _hoist_nopad_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def hoist_filter_rfft2_zero_nopad(
    filters_oihw: jax.Array,
    kernel_size: tuple[int, int] = (23, 13),
    pad_hw: tuple[int, int] = (11, 6),
    fft_shape: tuple[int, int] = (64, 64),
) -> jax.Array:
    fh, fw = fft_shape
    m = filters_oihw.shape[0]
    return jnp.zeros((m, fh, fw // 2 + 1), dtype=jnp.complex64)


def _hoist_zero_nopad_fwd(filters_oihw, kernel_size, pad_hw, fft_shape):
    return hoist_filter_rfft2_zero_nopad(filters_oihw, kernel_size, pad_hw, fft_shape), None


def _hoist_zero_nopad_bwd(kernel_size, pad_hw, fft_shape, _, g_ker_freq_mhw):
    return (
        rfft2_unshifted_filter_grad_to_oihw(
            g_ker_freq_mhw, kernel_size=kernel_size, pad_hw=pad_hw, fft_shape=fft_shape
        ),
    )


hoist_filter_rfft2_zero_nopad.defvjp(_hoist_zero_nopad_fwd, _hoist_zero_nopad_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def conv2d_hybrid_nopad_roi_step(
    roi_cells_bhw: jax.Array,
    filters_oihw: jax.Array,
    ker_freq_token: jax.Array,
    pad_hw: tuple[int, int] = (11, 6),
    fft_shape: tuple[int, int] = (64, 64),
    need_dx: bool = True,
) -> jax.Array:
    """
    Zero-pad hybrid ROI convolution step used by `backend="pallas_fused"` on medium kernels:
    - Forward: Direct batched NCHW convolution.
    - Backward `dX`: Explicit forward NCHW convolution with flipped filter `(1, M, Kh, Kw)`.
    - Backward `dG`: Pruned zero-pad RFFT Plancherel inner product (no `jnp.pad` allocation!).
    """
    r_h, r_w = pad_hw
    lhs = roi_cells_bhw[:, jnp.newaxis, :, :]
    return jax.lax.conv_general_dilated(
        lhs=lhs,
        rhs=filters_oihw,
        window_strides=(1, 1),
        padding=((r_h, r_h), (r_w, r_w)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )


def _conv2d_hybrid_nopad_fwd(
    roi_cells_bhw, filters_oihw, ker_freq_token, pad_hw, fft_shape, need_dx
):
    z_roi = conv2d_hybrid_nopad_roi_step(
        roi_cells_bhw, filters_oihw, ker_freq_token, pad_hw, fft_shape, need_dx
    )
    return z_roi, (roi_cells_bhw, filters_oihw)


def _conv2d_hybrid_nopad_bwd(pad_hw, fft_shape, need_dx, res, g_z_roi):
    roi_cells_bhw, filters_oihw = res
    r_h, r_w = pad_hw
    if need_dx:
        g_flip_iohw = jnp.transpose(
            jnp.flip(filters_oihw, axis=(-2, -1)), (1, 0, 2, 3)
        )
        g_roi_nchw = jax.lax.conv_general_dilated(
            lhs=g_z_roi,
            rhs=g_flip_iohw,
            window_strides=(1, 1),
            padding=((r_h, r_h), (r_w, r_w)),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        g_roi_cells = g_roi_nchw[:, 0, :, :]
    else:
        g_roi_cells = jnp.zeros_like(roi_cells_bhw)

    cell_freq = rfft2_pruned(roi_cells_bhw, fft_shape=fft_shape)
    g_freq_unshifted = rfft2_pruned(g_z_roi, fft_shape=fft_shape)
    g_ker_freq_unshifted = jnp.sum(
        g_freq_unshifted * jnp.conj(cell_freq)[:, jnp.newaxis, :, :], axis=0
    )
    return g_roi_cells, jnp.zeros_like(filters_oihw), g_ker_freq_unshifted


conv2d_hybrid_nopad_roi_step.defvjp(
    _conv2d_hybrid_nopad_fwd, _conv2d_hybrid_nopad_bwd
)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def conv2d_small_kernel_roi_step(
    roi_cells_bhw: jax.Array,
    filters_oihw: jax.Array,
    pad_hw: tuple[int, int] = (1, 1),
    need_dx: bool = True,
) -> jax.Array:
    """
    Spatial ROI convolution step with grouped-batch filter gradient (`conv2d_grouped_filter_grad`)
    used by `backend="pallas_fused"` / `"auto"` when `Kh * Kw <= 64` (e.g. 3x3 GoL/CLP) or
    when the grid exceeds `fft_shape`:
    - Forward: Direct batched NCHW convolution.
    - Backward `dX`: Explicit forward NCHW convolution with flipped filter `(1, M, Kh, Kw)`.
    - Backward `dG`: Grouped-batch (`feature_group_count = G_b`) spatial cross-correlation,
      avoiding both cuDNN split-K atomic contention and 64x64 2D FFT overhead.
    """
    r_h, r_w = pad_hw
    lhs = roi_cells_bhw[:, jnp.newaxis, :, :]
    return jax.lax.conv_general_dilated(
        lhs=lhs,
        rhs=filters_oihw,
        window_strides=(1, 1),
        padding=((r_h, r_h), (r_w, r_w)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )


def _conv2d_small_kernel_fwd(roi_cells_bhw, filters_oihw, pad_hw, need_dx):
    z_roi = conv2d_small_kernel_roi_step(roi_cells_bhw, filters_oihw, pad_hw, need_dx)
    return z_roi, (roi_cells_bhw, filters_oihw)


def _conv2d_small_kernel_bwd(pad_hw, need_dx, res, g_z_roi):
    roi_cells_bhw, filters_oihw = res
    r_h, r_w = pad_hw
    if need_dx:
        g_flip_iohw = jnp.transpose(
            jnp.flip(filters_oihw, axis=(-2, -1)), (1, 0, 2, 3)
        )
        g_roi_nchw = jax.lax.conv_general_dilated(
            lhs=g_z_roi,
            rhs=g_flip_iohw,
            window_strides=(1, 1),
            padding=((r_h, r_h), (r_w, r_w)),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        g_roi_cells = g_roi_nchw[:, 0, :, :]
    else:
        g_roi_cells = jnp.zeros_like(roi_cells_bhw)

    g_filters = conv2d_grouped_filter_grad(roi_cells_bhw, g_z_roi, pad_hw=pad_hw)
    return g_roi_cells, g_filters


conv2d_small_kernel_roi_step.defvjp(
    _conv2d_small_kernel_fwd, _conv2d_small_kernel_bwd
)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def conv2d_pruned_rfft_roi_step(
    roi_cells_bhw: jax.Array,
    ker_freq_mhw: jax.Array,
    roi_shape: tuple[int, int] = (40, 50),
    pad_hw: tuple[int, int] = (11, 6),
    fft_shape: tuple[int, int] = (64, 64),
    need_dx: bool = True,
) -> jax.Array:
    """
    Pruned separable 2D RFFT/IRFFT step with zero-pad Plancherel backward reduction
    used by `backend="pallas_sram"`. Prunes dead border rows prior to both forward 1D `rfft`
    and backward 1D `irfft`, reducing 1D transform counts by `1 - H_v / Fh`.
    """
    z_roi, _ = _conv2d_pruned_rfft_fwd(
        roi_cells_bhw, ker_freq_mhw, roi_shape, pad_hw, fft_shape, need_dx
    )
    return z_roi


def _conv2d_pruned_rfft_fwd(
    roi_cells_bhw, ker_freq_mhw, roi_shape, pad_hw, fft_shape, need_dx
):
    r_h, r_w = pad_hw
    roi_h, roi_w = roi_shape
    in_h, in_w = roi_cells_bhw.shape[-2], roi_cells_bhw.shape[-1]

    cell_freq = rfft2_pruned(roi_cells_bhw, fft_shape=fft_shape)
    out_freq = cell_freq[:, jnp.newaxis, :, :] * ker_freq_mhw[jnp.newaxis, :, :, :]
    z_roi = irfft2_pruned(
        out_freq, roi_shape=(roi_h, roi_w), pad_hw=(r_h, r_w), fft_shape=fft_shape
    )
    return z_roi, (cell_freq, ker_freq_mhw, in_h, in_w)


def _conv2d_pruned_rfft_bwd(roi_shape, pad_hw, fft_shape, need_dx, res, g_z_roi):
    cell_freq, ker_freq_mhw, in_h, in_w = res
    r_h, r_w = pad_hw
    fh, fw = fft_shape
    g_freq_unshifted = rfft2_pruned(g_z_roi, fft_shape=fft_shape)
    if need_dx:
        g_cell_freq = jnp.sum(
            g_freq_unshifted * jnp.conj(ker_freq_mhw)[jnp.newaxis, :, :, :], axis=1
        )
        if in_h < fh:
            g_ifft_h = jnp.fft.ifft(g_cell_freq, n=fh, axis=-2)
            g_ifft_h_cropped = jnp.roll(g_ifft_h, shift=r_h, axis=-2)[:, :in_h, :]
            g_spatial_w = jnp.fft.irfft(g_ifft_h_cropped, n=fw, axis=-1)
            g_roi_cells = jnp.roll(g_spatial_w, shift=r_w, axis=-1)[:, :, :in_w]
        else:
            g_roi_full = jnp.fft.irfft2(g_cell_freq, s=fft_shape, axes=(-2, -1))
            g_roi_cells = jnp.roll(
                g_roi_full, shift=(r_h, r_w), axis=(-2, -1)
            )[:, :in_h, :in_w]
    else:
        g_roi_cells = jnp.zeros(
            (g_z_roi.shape[0], in_h, in_w), dtype=g_z_roi.dtype
        )

    g_ker_freq_unshifted = jnp.sum(
        g_freq_unshifted * jnp.conj(cell_freq)[:, jnp.newaxis, :, :], axis=0
    )
    return g_roi_cells, g_ker_freq_unshifted


conv2d_pruned_rfft_roi_step.defvjp(
    _conv2d_pruned_rfft_fwd, _conv2d_pruned_rfft_bwd
)
