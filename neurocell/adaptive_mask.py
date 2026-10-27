'''
Epoch-adaptive spatial extent masking for Von Neumann Networks (Milestone 7).

At the start of training any VNN, the final self-assembled circuit extent is unknown
a priori. However, at the end of Epoch `e`, the learned cellular parameters
`(W^(e), B^(e), S^(e))`, directional Green's filters `G^(e)`, I/O port anchors,
and optional empirical activation variance reveal the currently active region.

This module provides a cheap (`< 0.15 ms`) pre-epoch mask estimator
(`compute_adaptive_spatial_mask`) that:
1. Identifies active/salient cells and I/O port anchors on the active ROI `(H_v, W_v)`.
2. Estimates the effective spatial support radius `(r_eff_h, r_eff_w)` of the learned
   Green's filters `G^(e)`.
3. Computes the active bounding region and dilates it by a configurable safety margin
   `margin` (e.g. `+4` cells in all directions) so frontier cells receive backward
   gradients and can grow outward during Epoch `e + 1`.
4. Quantizes the active sub-grid dimensions `(H_act, W_act)` to multiples of
   `bucket_multiple` (default `8`) so XLA caches a small, fixed set of compiled shapes
   with zero recompilation churn across epochs.

Copyright (c) 2026 Shekhar S. Chandra. All rights reserved.
Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

@author Shakes
'''
from dataclasses import dataclass, replace
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from neurocell.topology import ChipTopology2D, CoordinatePortSpec, PortSpec, _next_power_of_two
from neurocell.params import VNNParams
from neurocell.kernels_pallas import TILED_BLOCK_N, _select_tile_block_n


@dataclass(frozen=True)
class AdaptiveMaskSpec:
    """
    Describes the active sub-ROI bounding box, 2D boolean mask, tile activity mask,
    and cropped `ChipTopology2D` view for an epoch.
    """
    roi_bbox: tuple[int, int, int, int]
    grid_bbox: tuple[int, int, int, int]
    active_shape: tuple[int, int]
    active_mask_roi: jax.Array
    cropped_mask_roi: jax.Array
    tile_mask: jax.Array
    eff_kernel_radii: tuple[int, int]
    active_fraction: float
    salient_cell_count: int
    is_full_roi: bool
    cropped_topology: ChipTopology2D
    fft_shape: tuple[int, int]
    raw_roi_bbox: tuple[int, int, int, int] | None = None


def estimate_effective_filter_radii(
    greens_filters: jax.Array,
    threshold: float = 0.05,
) -> tuple[int, int]:
    """
    Estimates the effective spatial half-radii `(r_eff_h, r_eff_w)` of the learned
    Green's function filters `G` of shape `(M, 1, Kh, Kw)`.

    Any filter tap `(u, v)` with `max_m |G_{m,0,u,v}| >= threshold * max |G|` is
    considered active.
    """
    kh, kw = greens_filters.shape[-2], greens_filters.shape[-1]
    center_h, center_w = (kh - 1) // 2, (kw - 1) // 2
    max_per_tap = jnp.max(jnp.abs(greens_filters[:, 0, :, :]), axis=0)
    peak = jnp.max(max_per_tap)
    active_taps = max_per_tap >= (threshold * peak + 1e-12)

    rows = jnp.arange(kh, dtype=jnp.int32)[:, jnp.newaxis]
    cols = jnp.arange(kw, dtype=jnp.int32)[jnp.newaxis, :]
    dist_h = jnp.where(active_taps, jnp.abs(rows - center_h), 0)
    dist_w = jnp.where(active_taps, jnp.abs(cols - center_w), 0)
    r_eff_h = max(1, int(jnp.max(dist_h)))
    r_eff_w = max(1, int(jnp.max(dist_w)))
    return r_eff_h, r_eff_w


def _dilate_mask_2d(mask_hw: jax.Array, margin_h: int, margin_w: int) -> jax.Array:
    """
    Morphologically dilates a 2D boolean mask `(H_v, W_v)` by a rectangular structuring
    element of half-radii `(margin_h, margin_w)` on GPU using separable max-pooling.
    """
    if margin_h <= 0 and margin_w <= 0:
        return mask_hw
    win_h = 2 * max(0, int(margin_h)) + 1
    win_w = 2 * max(0, int(margin_w)) + 1
    x = mask_hw.astype(jnp.float32)[jnp.newaxis, :, :, jnp.newaxis]
    pooled = jax.lax.reduce_window(
        x,
        init_value=0.0,
        computation=jax.lax.max,
        window_dimensions=(1, win_h, win_w, 1),
        window_strides=(1, 1, 1, 1),
        padding="SAME",
    )
    return pooled[0, :, :, 0] > 0.5


def _quantize_interval(
    lo: int,
    hi: int,
    max_len: int,
    bucket_multiple: int,
) -> tuple[int, int]:
    """
    Expands interval `[lo, hi)` so its length `hi - lo` is a multiple of `bucket_multiple`
    (or `max_len` when within one `bucket_multiple` of `max_len`), centered on `[lo, hi)`
    and snapped to a canonical grid multiple to prevent 1-pixel sliding recompilations.
    """
    lo = max(0, min(lo, max_len - 1))
    hi = max(lo + 1, min(hi, max_len))
    raw_len = hi - lo
    if bucket_multiple <= 1 or raw_len >= max_len:
        return 0, max_len
    q_len = min(max_len, ((raw_len + bucket_multiple - 1) // bucket_multiple) * bucket_multiple)
    if q_len >= max_len or (max_len - q_len) < bucket_multiple:
        return 0, max_len

    ideal_lo = lo - 0.5 * (q_len - raw_len)
    step = max(1, bucket_multiple // 2)
    min_valid_lo = max(0, hi - q_len)
    max_valid_lo = min(lo, max_len - q_len)

    # Search for a step-aligned start index in [min_valid_lo, max_valid_lo] closest to ideal_lo
    first_aligned = ((min_valid_lo + step - 1) // step) * step
    if first_aligned <= max_valid_lo:
        best_lo = first_aligned
        best_dist = abs(first_aligned - ideal_lo)
        cand = first_aligned + step
        while cand <= max_valid_lo:
            d = abs(cand - ideal_lo)
            if d < best_dist:
                best_dist = d
                best_lo = cand
            cand += step
        new_lo = best_lo
    else:
        # Fall back to quarter-step alignment or exact centered start
        q_step = max(1, bucket_multiple // 4)
        first_q = ((min_valid_lo + q_step - 1) // q_step) * q_step
        if first_q <= max_valid_lo:
            best_lo = first_q
            best_dist = abs(first_q - ideal_lo)
            cand = first_q + q_step
            while cand <= max_valid_lo:
                d = abs(cand - ideal_lo)
                if d < best_dist:
                    best_dist = d
                    best_lo = cand
                cand += q_step
            new_lo = best_lo
        else:
            new_lo = max(min_valid_lo, min(int(round(ideal_lo)), max_valid_lo))

    return int(new_lo), int(new_lo + q_len)


def _build_cropped_topology(
    topology: ChipTopology2D,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
) -> tuple[ChipTopology2D, tuple[int, int]]:
    """
    Constructs a cropped `ChipTopology2D` view for active ROI slice `[r0:r1, c0:c1]`.
    """
    h_act = r1 - r0
    w_act = c1 - c0
    kh, kw = topology.kernel_size
    fft_shape = (
        _next_power_of_two(h_act + kh - 1),
        _next_power_of_two(w_act + kw - 1),
    )
    if isinstance(topology.out_port, CoordinatePortSpec):
        ro_rows, ro_cols = topology.roi_out_offset
        cropped_out_offset = (
            tuple(int(r - r0) for r in ro_rows),
            tuple(int(c - c0) for c in ro_cols),
        )
    else:
        ro_r, ro_c = topology.roi_out_offset
        cropped_out_offset = (int(ro_r - r0), int(ro_c - c0))

    cropped_topo = replace(
        topology,
        roi_shape=(h_act, w_act),
        roi_out_offset=cropped_out_offset,
        fft_shape=fft_shape,
    )
    return cropped_topo, fft_shape


@partial(
    jax.jit,
    static_argnames=(
        "r_h",
        "r_w",
        "roi_h",
        "roi_w",
        "margin_h",
        "margin_w",
        "block_n",
        "num_tiles",
        "has_activity_map",
    ),
)
def _compute_adaptive_mask_tensors(
    cell_params: jax.Array,
    greens_filters: jax.Array,
    io_mask_roi: jax.Array,
    act_roi: jax.Array,
    r_h: int,
    r_w: int,
    roi_h: int,
    roi_w: int,
    margin_h: int,
    margin_w: int,
    block_n: int,
    num_tiles: int,
    salience_threshold: float,
    bias_threshold: float,
    filter_threshold: float,
    activity_threshold: float,
    has_activity_map: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Fused XLA kernel computing filter radii, cell salience, morphological dilation,
    bounding-box coordinates, and tile masks in a single GPU dispatch.
    """
    kh, kw = greens_filters.shape[-2], greens_filters.shape[-1]
    center_h, center_w = (kh - 1) // 2, (kw - 1) // 2
    max_per_tap = jnp.max(jnp.abs(greens_filters[:, 0, :, :]), axis=0)
    peak = jnp.max(max_per_tap)
    active_taps = max_per_tap >= (filter_threshold * peak + 1e-12)
    rows_k = jnp.arange(kh, dtype=jnp.int32)[:, jnp.newaxis]
    cols_k = jnp.arange(kw, dtype=jnp.int32)[jnp.newaxis, :]
    r_eff_h = jnp.maximum(1, jnp.max(jnp.where(active_taps, jnp.abs(rows_k - center_h), 0)))
    r_eff_w = jnp.maximum(1, jnp.max(jnp.where(active_taps, jnp.abs(cols_k - center_w), 0)))

    w_roi = jax.lax.dynamic_slice(cell_params[0], (r_h, r_w), (roi_h, roi_w))
    b_roi = jax.lax.dynamic_slice(cell_params[1], (r_h, r_w), (roi_h, roi_w))
    s_roi = jax.lax.dynamic_slice(cell_params[2], (r_h, r_w), (roi_h, roi_w))
    s_prob = jax.nn.sigmoid(s_roi)

    non_io = ~io_mask_roi
    non_io_count = jnp.maximum(1.0, jnp.sum(non_io.astype(jnp.float32)))

    # 1. Bias-driven salience above background LayerNorm drift floor
    bias_drive = jnp.abs(b_roi) * s_prob
    updated_non_io = non_io & (jnp.abs(b_roi) > 1e-5)
    updated_count = jnp.maximum(1.0, jnp.sum(updated_non_io.astype(jnp.float32)))
    b_mean_act = jnp.sum(jnp.where(updated_non_io, bias_drive, 0.0)) / updated_count
    b_std_act = jnp.sqrt(
        jnp.sum(jnp.where(updated_non_io, jnp.square(bias_drive - b_mean_act), 0.0))
        / updated_count
    )
    z_bias_act = (bias_drive - b_mean_act) / (b_std_act + 1e-6)
    salient_bias = (
        non_io
        & (s_prob > 0.5)
        & (bias_drive > bias_threshold)
        & ((z_bias_act > salience_threshold) | (bias_drive > 4.0 * bias_threshold))
    )

    # 2. Differentiated high-gain relay salience
    w_eff_mag = jnp.abs(w_roi) * s_prob
    w_eff_mean = jnp.sum(jnp.where(non_io, w_eff_mag, 0.0)) / non_io_count
    w_eff_std = jnp.sqrt(
        jnp.sum(jnp.where(non_io, jnp.square(w_eff_mag - w_eff_mean), 0.0)) / non_io_count
    )
    z_w_eff = (w_eff_mag - w_eff_mean) / (w_eff_std + 1e-6)
    salient_relay = (
        non_io
        & (jnp.abs(b_roi) > 0.5 * bias_threshold)
        & (z_w_eff > salience_threshold)
        & (s_prob > 0.85)
    )

    seed_mask = io_mask_roi | salient_bias | salient_relay
    if has_activity_map:
        seed_mask = seed_mask | (act_roi > activity_threshold)

    salient_count = jnp.sum(seed_mask.astype(jnp.int32))
    dilated_mask = _dilate_mask_2d(seed_mask, margin_h=margin_h, margin_w=margin_w)

    active_rows = jnp.any(dilated_mask, axis=1)
    active_cols = jnp.any(dilated_mask, axis=0)
    any_active = jnp.any(active_rows).astype(jnp.int32)
    row_indices = jnp.arange(roi_h, dtype=jnp.int32)
    col_indices = jnp.arange(roi_w, dtype=jnp.int32)
    r0_raw = jnp.min(jnp.where(active_rows, row_indices, roi_h))
    r1_raw = jnp.max(jnp.where(active_rows, row_indices, -1)) + 1
    c0_raw = jnp.min(jnp.where(active_cols, col_indices, roi_w))
    c1_raw = jnp.max(jnp.where(active_cols, col_indices, -1)) + 1

    n_pix = roi_h * roi_w
    flat_mask = jnp.reshape(dilated_mask, (n_pix,))
    if num_tiles * block_n > n_pix:
        flat_mask = jnp.pad(
            flat_mask, ((0, num_tiles * block_n - n_pix),), constant_values=False
        )
    tile_mask = jnp.any(jnp.reshape(flat_mask, (num_tiles, block_n)), axis=1)

    stats_i32 = jnp.stack(
        [r_eff_h, r_eff_w, salient_count, any_active, r0_raw, r1_raw, c0_raw, c1_raw],
        axis=0,
    )
    return stats_i32, dilated_mask, tile_mask


def compute_adaptive_spatial_mask(
    params: VNNParams,
    topology: ChipTopology2D,
    margin: int | tuple[int, int] = 4,
    bucket_multiple: int = 8,
    salience_threshold: float = 2.5,
    bias_threshold: float = 0.05,
    filter_threshold: float = 0.05,
    activity_map: jax.Array | None = None,
    activity_threshold: float = 1e-3,
    prev_mask: AdaptiveMaskSpec | None = None,
) -> AdaptiveMaskSpec:
    """
    Computes the cheap pre-epoch active spatial mask and bucket-quantized sub-ROI bounding
    box for the next epoch without assuming any a priori final circuit extent.

    Args:
        params: Current epoch's learned `VNNParams`.
        topology: Full `ChipTopology2D` specification.
        margin: Safety padding (in cells) added around the active circuit envelope to
            allow outward circuit growth during the next epoch.
        bucket_multiple: Quantization step for `(H_act, W_act)` to prevent XLA recompilation.
        salience_threshold: Z-score threshold for detecting non-IO cells whose learned
            `(W, B, S)` parameters have diverged from the quiescent background.
        bias_threshold: Absolute bias threshold `|B * sigmoid(S)| > bias_threshold`
            identifying active self-sustaining bias cells.
        filter_threshold: Relative threshold for effective Green's filter support radii.
        activity_map: Optional `(H, W)` or `(H_v, W_v)` empirical activation variance or
            magnitude map from the current epoch.
        activity_threshold: Threshold applied to `activity_map` when provided.

    Returns:
        `AdaptiveMaskSpec` containing the quantized sub-ROI bounding box, masks, and
        cropped topology view.
    """
    if isinstance(margin, int):
        margin_h, margin_w = int(margin), int(margin)
    else:
        margin_h, margin_w = int(margin[0]), int(margin[1])

    H, W = topology.grid_shape
    r_h, r_w = topology.pad_h, topology.pad_w
    roi_h, roi_w = topology.roi_shape

    # 1. I/O port anchor mask on the ROI (roi_h, roi_w)
    io_mask_roi = topology.io_state_mask[r_h : H - r_h, r_w : W - r_w]

    n_pix = roi_h * roi_w
    block_n = _select_tile_block_n(n_pix, TILED_BLOCK_N)
    num_tiles = (n_pix + block_n - 1) // block_n

    # Check if output port covers the entire grid/ROI and no localized activity_map is given
    is_full_grid_out = (
        isinstance(topology.out_port, PortSpec)
        and topology.out_port.size == topology.grid_shape
    )
    if is_full_grid_out and activity_map is None:
        r_eff_h, r_eff_w = estimate_effective_filter_radii(
            params.greens_filters, threshold=filter_threshold
        )
        full_mask = jnp.ones((roi_h, roi_w), dtype=jnp.bool_)
        tile_mask = jnp.ones((num_tiles,), dtype=jnp.bool_)
        return AdaptiveMaskSpec(
            roi_bbox=(0, roi_h, 0, roi_w),
            grid_bbox=(r_h, H - r_h, r_w, W - r_w),
            active_shape=(roi_h, roi_w),
            active_mask_roi=full_mask,
            cropped_mask_roi=full_mask,
            tile_mask=tile_mask,
            eff_kernel_radii=(r_eff_h, r_eff_w),
            active_fraction=1.0,
            salient_cell_count=int(roi_h * roi_w),
            is_full_roi=True,
            cropped_topology=topology,
            fft_shape=topology.fft_shape,
        )

    if activity_map is not None:
        if activity_map.shape == (H, W):
            act_roi = activity_map[r_h : H - r_h, r_w : W - r_w]
        else:
            act_roi = activity_map
        has_act = True
    else:
        act_roi = io_mask_roi.astype(jnp.float32)
        has_act = False

    stats_i32, dilated_mask, tile_mask = _compute_adaptive_mask_tensors(
        params.cell_params,
        params.greens_filters,
        io_mask_roi,
        act_roi,
        r_h=r_h,
        r_w=r_w,
        roi_h=roi_h,
        roi_w=roi_w,
        margin_h=margin_h,
        margin_w=margin_w,
        block_n=block_n,
        num_tiles=num_tiles,
        salience_threshold=salience_threshold,
        bias_threshold=bias_threshold,
        filter_threshold=filter_threshold,
        activity_threshold=activity_threshold,
        has_activity_map=has_act,
    )
    stats_host = np.asarray(stats_i32)
    r_eff_h = int(stats_host[0])
    r_eff_w = int(stats_host[1])
    salient_count = int(stats_host[2])
    any_active = bool(stats_host[3])

    if not any_active:
        r0_raw, r1_raw = 0, roi_h
        c0_raw, c1_raw = 0, roi_w
    else:
        r0_raw = int(stats_host[4])
        r1_raw = int(stats_host[5])
        c0_raw = int(stats_host[6])
        c1_raw = int(stats_host[7])

    if prev_mask is not None:
        raw_prev = prev_mask.raw_roi_bbox if prev_mask.raw_roi_bbox is not None else prev_mask.roi_bbox
        r0_raw = min(r0_raw, raw_prev[0])
        r1_raw = max(r1_raw, raw_prev[1])
        c0_raw = min(c0_raw, raw_prev[2])
        c1_raw = max(c1_raw, raw_prev[3])

    raw_bbox = (r0_raw, r1_raw, c0_raw, c1_raw)

    # Hysteresis containment check: if the previous quantized box already covers raw_bbox,
    # reuse it directly without shifting or cascading quantization padding.
    if prev_mask is not None and prev_mask.roi_bbox[0] <= r0_raw and r1_raw <= prev_mask.roi_bbox[1]:
        r0, r1 = prev_mask.roi_bbox[0], prev_mask.roi_bbox[1]
    else:
        r0, r1 = _quantize_interval(r0_raw, r1_raw, roi_h, bucket_multiple)

    if prev_mask is not None and prev_mask.roi_bbox[2] <= c0_raw and c1_raw <= prev_mask.roi_bbox[3]:
        c0, c1 = prev_mask.roi_bbox[2], prev_mask.roi_bbox[3]
    else:
        c0, c1 = _quantize_interval(c0_raw, c1_raw, roi_w, bucket_multiple)

    h_act = r1 - r0
    w_act = c1 - c0
    is_full = (h_act == roi_h) and (w_act == roi_w)

    cropped_topo, cropped_fft_shape = _build_cropped_topology(topology, r0, r1, c0, c1)
    cropped_mask = dilated_mask[r0:r1, c0:c1]

    return AdaptiveMaskSpec(
        roi_bbox=(r0, r1, c0, c1),
        grid_bbox=(r_h + r0, r_h + r1, r_w + c0, r_w + c1),
        active_shape=(h_act, w_act),
        active_mask_roi=dilated_mask,
        cropped_mask_roi=cropped_mask,
        tile_mask=tile_mask,
        eff_kernel_radii=(r_eff_h, r_eff_w),
        active_fraction=float(h_act * w_act) / float(roi_h * roi_w),
        salient_cell_count=salient_count,
        is_full_roi=is_full,
        cropped_topology=cropped_topo,
        fft_shape=cropped_fft_shape,
        raw_roi_bbox=raw_bbox,
    )
