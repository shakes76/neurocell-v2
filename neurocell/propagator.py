'''
Stateless, pure-ROI propagation engines for Von Neumann Networks.

Exploits linearity of the Green's function convolution to decompose each step into:
    `conv_VALID(Embed(C_roi, C_border), G) == conv_SAME(C_roi, G) + Delta_border`
where `Delta_border = conv_VALID(C_border_only, G)` has batch size `B = 1` and is
hoisted outside the depth loop. Even when step `t = 0` inputs touch the outer border
(e.g. full-grid 64x64 Game of Life), step `t = 0` maps `(B, H, W) -> (B, H_v, W_v)`
and all subsequent steps `t >= 1` execute strictly on the `(B, H_v, W_v)` interior ROI.

Supports six Green's function & epilogue execution backends:
- `"auto"`: Analytical shape-driven cost model selecting `"pallas_fused"` or `"pallas_sram"`.
- `"nchw_direct"`: Batched NCHW cuDNN spatial convolution + XLA ROI epilogue.
- `"rfft_hoisted"`: Hoisted Overlap-Save 2D RFFT with pruned Plancherel `custom_vjp`.
- `"nchw_rfft_bwd"`: Hybrid NCHW direct forward + forward-conv `dX` + Plancherel RFFT `dG`.
- `"pallas_fused"`: Shape-adaptive NCHW/RFFT/grouped-grad conv + custom Pallas Triton fused
  register/SRAM epilogues (`pallas_fused_roi_step_epilogue` & `pallas_fused_output_port_epilogue`).
- `"pallas_sram"`: Pruned separable 2D RFFT + zero-pad Plancherel backward + custom
  Pallas Triton fused SRAM epilogues for O(1) kernel-size scaling.

Copyright (c) 2026 Shekhar S. Chandra. All rights reserved.
Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

@author Shakes
'''
import math
from typing import Sequence
import jax
import jax.numpy as jnp

from neurocell.topology import ChipTopology2D, CoordinatePortSpec
from neurocell.params import VNNParams, precompute_step_params
from neurocell.kernels_greens import (
    conv2d_batched_nchw,
    apply_spatial_filter_2d,
    hoist_filter_rfft2,
    conv2d_roi_rfft_step,
    conv2d_valid_rfft_precomputed,
    hoist_filter_rfft2_zero_fwd,
    conv2d_hybrid_roi_step,
)
from neurocell.kernels_pallas import (
    is_supported_pallas_activation,
    pallas_fused_roi_step_epilogue,
    pallas_fused_output_port_epilogue,
    hoist_filter_rfft2_nopad,
    hoist_filter_rfft2_zero_nopad,
    conv2d_hybrid_nopad_roi_step,
    conv2d_small_kernel_roi_step,
    conv2d_pruned_rfft_roi_step,
)
from neurocell.epilogue import (
    fused_roi_step_epilogue,
    fused_roi_epilogue,
    fused_output_port_epilogue,
)

SUPPORTED_BACKENDS = (
    "auto",
    "nchw_direct",
    "rfft_hoisted",
    "nchw_rfft_bwd",
    "pallas_fused",
    "pallas_sram",
)


def select_optimal_backend(topology: ChipTopology2D) -> str:
    """
    Analytical cost-model dispatcher (`backend="auto"`):
    - For tiny grids (`H_v * W_v < 256`, e.g. 8x8 CLP), selects `"nchw_direct"` where XLA
      fuses the sub-300KB tensors in L1/L2 cache without Triton launch overhead.
    - For large kernels (`Kh * Kw > 512`) whose linear convolution fits inside `fft_shape`,
      selects `"pallas_sram"` (pruned separable 2D RFFT forward/backward + Pallas epilogue).
    - For all other topologies, selects `"pallas_fused"`.
    """
    kh, kw = topology.kernel_size
    roi_h, roi_w = topology.roi_shape
    if roi_h * roi_w < 256:
        return "nchw_direct"
    fh, fw = topology.fft_shape
    rfft_fits = (roi_h + kh - 1 <= fh) and (roi_w + kw - 1 <= fw)
    if topology.padding_mode != "CIRCULAR" and (kh * kw > 512) and rfft_fits:
        return "pallas_sram"
    return "pallas_fused"


def _normalize_port_inputs(
    inputs: jax.Array | Sequence[jax.Array],
    topology: ChipTopology2D,
) -> tuple[int, list[jax.Array]]:
    """Normalizes single-port or multi-port batched inputs into a list of per-port arrays."""
    if isinstance(inputs, (list, tuple)):
        return inputs[0].shape[0], list(inputs)
    bsz = inputs.shape[0]
    if len(topology.in_ports) == 1:
        port0 = topology.in_ports[0]
        if inputs.ndim == 2 or (inputs.ndim == 3 and inputs.shape[1:] == port0.size):
            return bsz, [inputs]
    return bsz, [inputs[:, idx] for idx in range(len(topology.in_ports))]


def inject_inputs_2d(
    inputs: jax.Array | Sequence[jax.Array],
    topology: ChipTopology2D,
    dtype=jnp.float32,
) -> jax.Array:
    """
    Places batched input tensors into an initialized zero cellular field `(B, H, W)`.
    Supports both rectangular `PortSpec` and scattered `CoordinatePortSpec`.
    """
    H, W = topology.grid_shape
    B, port_arrays = _normalize_port_inputs(inputs, topology)

    cells = jnp.zeros((B, H, W), dtype=dtype)
    for inp, port in zip(port_arrays, topology.in_ports):
        if isinstance(port, CoordinatePortSpec):
            rows_idx = jnp.asarray(port.rows, dtype=jnp.int32)
            cols_idx = jnp.asarray(port.cols, dtype=jnp.int32)
            inp_flat = jnp.reshape(inp, (B, port.num_points)).astype(dtype)
            cells = cells.at[:, rows_idx, cols_idx].set(inp_flat)
        else:
            r0, c0 = port.offset
            if inp.ndim == 3:
                ih, iw = inp.shape[1], inp.shape[2]
                cells = cells.at[:, r0 : r0 + ih, c0 : c0 + iw].set(inp.astype(dtype))
            else:
                ph, pw = port.size
                inp_shaped = jnp.reshape(inp, (B, ph, pw)).astype(dtype)
                cells = cells.at[:, r0 : r0 + ph, c0 : c0 + pw].set(inp_shaped)

    return cells


def inject_inputs_roi_2d(
    inputs: jax.Array | Sequence[jax.Array],
    topology: ChipTopology2D,
    dtype=jnp.float32,
    roi_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[jax.Array | None, bool]:
    """
    Attempts to inject batched inputs directly into the `(B, H_v, W_v)` active ROI buffer
    (or the cropped `(B, r1 - r0, c1 - c0)` sub-ROI when `roi_bbox` is provided).
    Returns `(roi_cells, True)` if all input ports fit strictly within the target ROI, or
    `(None, False)` if any input port touches the outer dead border.
    """
    if not topology.in_in_roi:
        return None, False

    H, W = topology.grid_shape
    r_h, r_w = (topology.pad_h, topology.pad_w) if topology.padding_mode == "VALID" else (0, 0)
    roi_h, roi_w = topology.roi_shape
    if roi_bbox is not None:
        b_r0, b_r1, b_c0, b_c1 = roi_bbox
    else:
        b_r0, b_r1, b_c0, b_c1 = 0, roi_h, 0, roi_w

    min_r, max_r = r_h + b_r0, r_h + b_r1
    min_c, max_c = r_w + b_c0, r_w + b_c1
    B, port_arrays = _normalize_port_inputs(inputs, topology)

    for inp, port in zip(port_arrays, topology.in_ports):
        if isinstance(port, CoordinatePortSpec):
            if not all(min_r <= r < max_r and min_c <= c < max_c for r, c in port.coords):
                return None, False
        else:
            r0, c0 = port.offset
            ih, iw = (inp.shape[1], inp.shape[2]) if inp.ndim == 3 else port.size
            if r0 < min_r or c0 < min_c or (r0 + ih) > max_r or (c0 + iw) > max_c:
                return None, False

    sub_h, sub_w = b_r1 - b_r0, b_c1 - b_c0
    roi_cells = jnp.zeros((B, sub_h, sub_w), dtype=dtype)
    for inp, port in zip(port_arrays, topology.in_ports):
        if isinstance(port, CoordinatePortSpec):
            rows_idx = jnp.asarray([r - min_r for r in port.rows], dtype=jnp.int32)
            cols_idx = jnp.asarray([c - min_c for c in port.cols], dtype=jnp.int32)
            inp_flat = jnp.reshape(inp, (B, port.num_points)).astype(dtype)
            roi_cells = roi_cells.at[:, rows_idx, cols_idx].set(inp_flat)
        else:
            r0 = port.offset[0] - min_r
            c0 = port.offset[1] - min_c
            if inp.ndim == 3:
                ih, iw = inp.shape[1], inp.shape[2]
                roi_cells = roi_cells.at[:, r0 : r0 + ih, c0 : c0 + iw].set(inp.astype(dtype))
            else:
                ph, pw = port.size
                inp_shaped = jnp.reshape(inp, (B, ph, pw)).astype(dtype)
                roi_cells = roi_cells.at[:, r0 : r0 + ph, c0 : c0 + pw].set(inp_shaped)

    return roi_cells, True


def _extract_output_from_full_grid(
    final_cells: jax.Array,
    topology: ChipTopology2D,
) -> jax.Array:
    """Slices or gathers the output port predictions from a full `(B, H, W)` universe."""
    out_h, out_w = topology.out_port.size
    if isinstance(topology.out_port, CoordinatePortSpec):
        rows_idx = jnp.asarray(topology.out_port.rows, dtype=jnp.int32)
        cols_idx = jnp.asarray(topology.out_port.cols, dtype=jnp.int32)
        gathered = final_cells[:, rows_idx, cols_idx]
        return jnp.reshape(gathered, (final_cells.shape[0], out_h, out_w))
    out_r, out_c = topology.out_port.offset
    return final_cells[:, out_r : out_r + out_h, out_c : out_c + out_w]


def propagate_vnn_2d(
    params: VNNParams,
    inputs: jax.Array | Sequence[jax.Array],
    topology: ChipTopology2D,
    activate=jax.nn.tanh,
    final_activate=None,
    norm: bool = True,
    sharpen_filter: jax.Array | None = None,
    gated_filters: bool = False,
    hard_pin_io: bool = False,
    backend: str = "nchw_direct",
    return_universe: bool = False,
    spatial_mask=None,
    codd_mode: str = "continuous",
    state_temperature: float | jax.Array = 1.0,
    active_threshold: float = 0.6,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """
    Executes batched forward propagation of a 2D Von Neumann Network over `topology.depth` steps.
    When `spatial_mask` (`AdaptiveMaskSpec`) is provided and `not spatial_mask.is_full_roi`,
    executes strictly on the active sub-ROI `[r0:r1, c0:c1]`.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend}'; choose from {SUPPORTED_BACKENDS}."
        )
    use_cropped_mask = (
        spatial_mask is not None
        and not spatial_mask.is_full_roi
        and topology.padding_mode != "CIRCULAR"
    )
    active_topology = spatial_mask.cropped_topology if use_cropped_mask else topology
    roi_bbox = spatial_mask.roi_bbox if use_cropped_mask else None

    if backend == "auto":
        backend = select_optimal_backend(active_topology)

    sharpen_fn = (
        (lambda w: apply_spatial_filter_2d(w, sharpen_filter, padding="SAME"))
        if sharpen_filter is not None
        else None
    )

    # 1. Pre-step parameter fusion (executed ONCE per batch on (H, W))
    step_params = precompute_step_params(
        params=params,
        topology=topology,
        activate=activate,
        sharpen_weights_fn=sharpen_fn,
        hard_pin_io=hard_pin_io,
        roi_bbox=roi_bbox,
        codd_mode=codd_mode,
        state_temperature=state_temperature,
        active_threshold=active_threshold,
    )

    filters_oihw = params.greens_filters
    if gated_filters:
        filters_oihw = jax.nn.sigmoid(filters_oihw)

    H, W = topology.grid_shape
    r_h, r_w = topology.pad_h, topology.pad_w
    kh, kw = topology.kernel_size
    conv_pad_hw = ((kh - 1) // 2, (kw - 1) // 2)
    roi_h, roi_w = active_topology.roi_shape
    fh, fw = active_topology.fft_shape

    has_valid_border = topology.padding_mode == "VALID"
    use_pallas_epilogue = (
        backend in ("pallas_fused", "pallas_sram")
        and norm
        and (roi_h * roi_w >= 256)
        and is_supported_pallas_activation(activate)
    )

    # Circular boundary mode path
    if topology.padding_mode == "CIRCULAR":
        roi_cells = inject_inputs_2d(inputs, topology, dtype=params.cell_params.dtype)
        B = roi_cells.shape[0]
        zero_border = jnp.zeros((1, topology.directions, H, W), dtype=params.cell_params.dtype)
        cp_h, cp_w = conv_pad_hw
        for _ in range(topology.depth - 1):
            padded = jnp.pad(roi_cells, ((0, 0), (cp_h, cp_h), (cp_w, cp_w)), mode="wrap")
            z_roi = conv2d_batched_nchw(
                padded[:, jnp.newaxis, :, :], filters_oihw, padding="VALID"
            )
            if use_pallas_epilogue:
                roi_cells = pallas_fused_roi_step_epilogue(
                    z_roi,
                    zero_border,
                    step_params.w_eff_roi,
                    step_params.s_eff_roi,
                    step_params.b_roi,
                    activate,
                    False,
                )
            else:
                roi_cells = fused_roi_step_epilogue(
                    z_roi_bmhw=z_roi,
                    step_params=step_params,
                    activate=activate,
                    norm=norm,
                )
        padded_final = jnp.pad(roi_cells, ((0, 0), (cp_h, cp_h), (cp_w, cp_w)), mode="wrap")
        z_final = conv2d_batched_nchw(
            padded_final[:, jnp.newaxis, :, :], filters_oihw, padding="VALID"
        )
        if not return_universe:
            if use_pallas_epilogue:
                return pallas_fused_output_port_epilogue(
                    z_roi_bmhw=z_final,
                    delta_border_1mhw=zero_border,
                    step_params=step_params,
                    topology=topology,
                    final_activate=final_activate,
                    has_border=False,
                )
            return fused_output_port_epilogue(
                z_roi_bmhw=z_final,
                step_params=step_params,
                topology=topology,
                final_activate=final_activate,
                norm=norm,
            )
        final_cells = fused_roi_step_epilogue(
            z_roi_bmhw=z_final,
            step_params=step_params,
            activate=final_activate,
            norm=norm,
        )
        if final_activate is not None:
            final_cells = final_activate(final_cells)
        return _extract_output_from_full_grid(final_cells, topology), final_cells

    # 2. Check if step t=0 inputs fit inside the active ROI (or sub-ROI bbox)
    roi_cells, fits_in_roi = inject_inputs_roi_2d(
        inputs, topology, dtype=params.cell_params.dtype, roi_bbox=roi_bbox
    )
    if not fits_in_roi:
        cells_t0 = inject_inputs_2d(inputs, topology, dtype=params.cell_params.dtype)
        B = cells_t0.shape[0]
    else:
        cells_t0 = None
        B = roi_cells.shape[0]

    # 3. Hoist B=1 border convolution ONCE outside the depth loop (for VALID mode t >= 1)
    if has_valid_border:
        border_only_2d = step_params.border_2d.at[r_h : H - r_h, r_w : W - r_w].set(0.0)
        if roi_bbox is not None:
            b_r0, b_r1, b_c0, b_c1 = roi_bbox
            border_only_2d = border_only_2d[
                b_r0 : b_r1 + kh - 1, b_c0 : b_c1 + kw - 1
            ]
        delta_border = conv2d_batched_nchw(
            border_only_2d[jnp.newaxis, jnp.newaxis, :, :],
            filters_oihw,
            padding="VALID",
        )  # (1, M, roi_h, roi_w)
    else:
        delta_border = jnp.zeros(
            (1, topology.directions, roi_h, roi_w), dtype=params.cell_params.dtype
        )
    zero_border = jnp.zeros_like(delta_border)

    # Determine convolution strategy in pallas_fused:
    # - Tiny kernels (Kh * Kw <= 64) or oversized grids: direct cuDNN NCHW conv.
    # - Medium kernels (64 < Kh * Kw <= 512) on compact SRAM-resident sub-ROIs
    #   (roi_h * roi_w <= (fh * fw) // 3, i.e., <= 80 KB/group in 100 KB/SM SRAM):
    #   spatial grouped-batch filter gradient (`conv2d_small_kernel_roi_step`), avoiding
    #   34.60 MB 64x64 complex64 FFT padding overhead.
    # - Otherwise: zero-pad hybrid NCHW forward + Plancherel RFFT backward.
    rfft_fits_roi = (roi_h + kh - 1 <= fh) and (roi_w + kw - 1 <= fw)
    use_small_kernel_conv = (kh * kw <= 64) or (not rfft_fits_roi)
    use_sram_grouped_conv = (
        (not use_small_kernel_conv)
        and (kh * kw <= 512)
        and (roi_h * roi_w <= (fh * fw) // 3)
    )

    def _apply_hidden_epilogue(z_roi: jax.Array, step_idx: int) -> jax.Array:
        if roi_bbox is not None and z_roi.shape[-2:] != (roi_h, roi_w):
            b_r0, b_r1, b_c0, b_c1 = roi_bbox
            z_roi = z_roi[:, :, b_r0:b_r1, b_c0:b_c1]
        step_has_border = (step_idx > 0) and has_valid_border
        if use_pallas_epilogue:
            db_step = delta_border if step_has_border else zero_border
            return pallas_fused_roi_step_epilogue(
                z_roi,
                db_step,
                step_params.w_eff_roi,
                step_params.s_eff_roi,
                step_params.b_roi,
                activate,
                step_has_border,
            )
        if step_has_border:
            z_roi = z_roi + delta_border
        return fused_roi_step_epilogue(
            z_roi_bmhw=z_roi,
            step_params=step_params,
            activate=activate,
            norm=norm,
        )

    if backend == "nchw_direct":
        for t in range(topology.depth - 1):
            if t == 0 and not fits_in_roi:
                z_roi = conv2d_batched_nchw(
                    cells_t0[:, jnp.newaxis, :, :], filters_oihw, padding="VALID"
                )
            else:
                z_roi = conv2d_batched_nchw(
                    roi_cells[:, jnp.newaxis, :, :], filters_oihw, padding="SAME"
                )
            roi_cells = _apply_hidden_epilogue(z_roi, t)

        if topology.depth == 1 and not fits_in_roi:
            z_final = conv2d_batched_nchw(
                cells_t0[:, jnp.newaxis, :, :], filters_oihw, padding="VALID"
            )
        else:
            z_final = conv2d_batched_nchw(
                roi_cells[:, jnp.newaxis, :, :], filters_oihw, padding="SAME"
            )

    elif backend == "rfft_hoisted":
        ker_freq_mhw = hoist_filter_rfft2(
            filters_oihw,
            kernel_size=topology.kernel_size,
            fft_shape=(fh, fw),
        )
        for t in range(topology.depth - 1):
            if t == 0 and not fits_in_roi:
                if H <= fh and W <= fw:
                    z_roi = conv2d_valid_rfft_precomputed(
                        cells_bhw=cells_t0,
                        ker_freq_mhw=ker_freq_mhw,
                        roi_shape=(roi_h, roi_w),
                        kernel_size=topology.kernel_size,
                        fft_shape=(fh, fw),
                    )
                else:
                    z_roi = conv2d_batched_nchw(
                        cells_t0[:, jnp.newaxis, :, :], filters_oihw, padding="VALID"
                    )
            else:
                z_roi = conv2d_roi_rfft_step(
                    roi_cells_bhw=roi_cells,
                    ker_freq_mhw=ker_freq_mhw,
                    roi_shape=(roi_h, roi_w),
                    pad_hw=conv_pad_hw,
                    fft_shape=(fh, fw),
                    need_dx=(t > 0),
                )
            roi_cells = _apply_hidden_epilogue(z_roi, t)

        if topology.depth == 1 and not fits_in_roi:
            z_final = conv2d_valid_rfft_precomputed(
                cells_bhw=cells_t0,
                ker_freq_mhw=ker_freq_mhw,
                roi_shape=(roi_h, roi_w),
                kernel_size=topology.kernel_size,
                fft_shape=(fh, fw),
            )
        else:
            z_final = conv2d_roi_rfft_step(
                roi_cells_bhw=roi_cells,
                ker_freq_mhw=ker_freq_mhw,
                roi_shape=(roi_h, roi_w),
                pad_hw=conv_pad_hw,
                fft_shape=(fh, fw),
                need_dx=(topology.depth > 1),
            )

    elif backend == "nchw_rfft_bwd":
        ker_freq_token = hoist_filter_rfft2_zero_fwd(
            filters_oihw,
            kernel_size=topology.kernel_size,
            fft_shape=(fh, fw),
        )
        filters_sg = jax.lax.stop_gradient(filters_oihw)
        for t in range(topology.depth - 1):
            if t == 0 and not fits_in_roi:
                z_roi = conv2d_batched_nchw(
                    cells_t0[:, jnp.newaxis, :, :], filters_oihw, padding="VALID"
                )
            else:
                z_roi = conv2d_hybrid_roi_step(
                    roi_cells_bhw=roi_cells,
                    filters_oihw=filters_sg,
                    ker_freq_token=ker_freq_token,
                    pad_hw=conv_pad_hw,
                    fft_shape=(fh, fw),
                    need_dx=(t > 0),
                )
            roi_cells = _apply_hidden_epilogue(z_roi, t)

        if topology.depth == 1 and not fits_in_roi:
            z_final = conv2d_batched_nchw(
                cells_t0[:, jnp.newaxis, :, :], filters_oihw, padding="VALID"
            )
        else:
            z_final = conv2d_hybrid_roi_step(
                roi_cells_bhw=roi_cells,
                filters_oihw=filters_sg,
                ker_freq_token=ker_freq_token,
                pad_hw=conv_pad_hw,
                fft_shape=(fh, fw),
                need_dx=(topology.depth > 1),
            )

    elif backend == "pallas_fused":
        if use_small_kernel_conv:
            for t in range(topology.depth - 1):
                if t == 0 and not fits_in_roi:
                    z_roi = conv2d_batched_nchw(
                        cells_t0[:, jnp.newaxis, :, :], filters_oihw, padding="VALID"
                    )
                else:
                    z_roi = conv2d_batched_nchw(
                        roi_cells[:, jnp.newaxis, :, :], filters_oihw, padding="SAME"
                    )
                roi_cells = _apply_hidden_epilogue(z_roi, t)

            if topology.depth == 1 and not fits_in_roi:
                z_final = conv2d_batched_nchw(
                    cells_t0[:, jnp.newaxis, :, :], filters_oihw, padding="VALID"
                )
            else:
                z_final = conv2d_batched_nchw(
                    roi_cells[:, jnp.newaxis, :, :], filters_oihw, padding="SAME"
                )
        elif use_sram_grouped_conv:
            for t in range(topology.depth - 1):
                if t == 0 and not fits_in_roi:
                    z_roi = conv2d_batched_nchw(
                        cells_t0[:, jnp.newaxis, :, :], filters_oihw, padding="VALID"
                    )
                else:
                    z_roi = conv2d_small_kernel_roi_step(
                        roi_cells_bhw=roi_cells,
                        filters_oihw=filters_oihw,
                        pad_hw=conv_pad_hw,
                        need_dx=(t > 0),
                    )
                roi_cells = _apply_hidden_epilogue(z_roi, t)

            if topology.depth == 1 and not fits_in_roi:
                z_final = conv2d_batched_nchw(
                    cells_t0[:, jnp.newaxis, :, :], filters_oihw, padding="VALID"
                )
            else:
                z_final = conv2d_small_kernel_roi_step(
                    roi_cells_bhw=roi_cells,
                    filters_oihw=filters_oihw,
                    pad_hw=conv_pad_hw,
                    need_dx=(topology.depth > 1),
                )
        else:
            ker_freq_token = hoist_filter_rfft2_zero_nopad(
                filters_oihw,
                kernel_size=topology.kernel_size,
                pad_hw=conv_pad_hw,
                fft_shape=(fh, fw),
            )
            filters_sg = jax.lax.stop_gradient(filters_oihw)
            for t in range(topology.depth - 1):
                if t == 0 and not fits_in_roi:
                    z_roi = conv2d_batched_nchw(
                        cells_t0[:, jnp.newaxis, :, :], filters_oihw, padding="VALID"
                    )
                else:
                    z_roi = conv2d_hybrid_nopad_roi_step(
                        roi_cells_bhw=roi_cells,
                        filters_oihw=filters_sg,
                        ker_freq_token=ker_freq_token,
                        pad_hw=conv_pad_hw,
                        fft_shape=(fh, fw),
                        need_dx=(t > 0),
                    )
                roi_cells = _apply_hidden_epilogue(z_roi, t)

            if topology.depth == 1 and not fits_in_roi:
                z_final = conv2d_batched_nchw(
                    cells_t0[:, jnp.newaxis, :, :], filters_oihw, padding="VALID"
                )
            else:
                z_final = conv2d_hybrid_nopad_roi_step(
                    roi_cells_bhw=roi_cells,
                    filters_oihw=filters_sg,
                    ker_freq_token=ker_freq_token,
                    pad_hw=conv_pad_hw,
                    fft_shape=(fh, fw),
                    need_dx=(topology.depth > 1),
                )

    else:  # backend == "pallas_sram"
        ker_freq_mhw = hoist_filter_rfft2_nopad(
            filters_oihw,
            kernel_size=topology.kernel_size,
            pad_hw=(r_h, r_w),
            fft_shape=(fh, fw),
        )
        for t in range(topology.depth - 1):
            if t == 0 and not fits_in_roi:
                if H <= fh and W <= fw:
                    z_roi = conv2d_pruned_rfft_roi_step(
                        roi_cells_bhw=cells_t0,
                        ker_freq_mhw=ker_freq_mhw,
                        roi_shape=(roi_h, roi_w),
                        pad_hw=(kh - 1, kw - 1),
                        fft_shape=(fh, fw),
                        need_dx=False,
                    )
                else:
                    z_roi = conv2d_batched_nchw(
                        cells_t0[:, jnp.newaxis, :, :], filters_oihw, padding="VALID"
                    )
            else:
                z_roi = conv2d_pruned_rfft_roi_step(
                    roi_cells_bhw=roi_cells,
                    ker_freq_mhw=ker_freq_mhw,
                    roi_shape=(roi_h, roi_w),
                    pad_hw=conv_pad_hw,
                    fft_shape=(fh, fw),
                    need_dx=(t > 0),
                )
            roi_cells = _apply_hidden_epilogue(z_roi, t)

        if topology.depth == 1 and not fits_in_roi:
            z_final = conv2d_pruned_rfft_roi_step(
                roi_cells_bhw=cells_t0,
                ker_freq_mhw=ker_freq_mhw,
                roi_shape=(roi_h, roi_w),
                pad_hw=(kh - 1, kw - 1),
                fft_shape=(fh, fw),
                need_dx=False,
            )
        else:
            z_final = conv2d_pruned_rfft_roi_step(
                roi_cells_bhw=roi_cells,
                ker_freq_mhw=ker_freq_mhw,
                roi_shape=(roi_h, roi_w),
                pad_hw=conv_pad_hw,
                fft_shape=(fh, fw),
                need_dx=(topology.depth > 1),
            )

    final_has_border = (topology.depth > 1) and has_valid_border
    if roi_bbox is not None and z_final.shape[-2:] != (roi_h, roi_w):
        b_r0, b_r1, b_c0, b_c1 = roi_bbox
        z_final = z_final[:, :, b_r0:b_r1, b_c0:b_c1]

    # Fast output-port path when return_universe=False and output port lies inside ROI
    if not return_universe and topology.out_in_roi:
        if use_pallas_epilogue:
            db_final = delta_border if final_has_border else zero_border
            return pallas_fused_output_port_epilogue(
                z_roi_bmhw=z_final,
                delta_border_1mhw=db_final,
                step_params=step_params,
                topology=active_topology,
                final_activate=final_activate,
                has_border=final_has_border,
            )
        if final_has_border:
            z_final = z_final + delta_border
        return fused_output_port_epilogue(
            z_roi_bmhw=z_final,
            step_params=step_params,
            topology=active_topology,
            final_activate=final_activate,
            norm=norm,
        )

    # Full-grid output or return_universe=True path
    if use_pallas_epilogue and is_supported_pallas_activation(final_activate):
        db_final = delta_border if final_has_border else zero_border
        roi_final = pallas_fused_roi_step_epilogue(
            z_final,
            db_final,
            step_params.w_eff_roi,
            step_params.s_eff_roi,
            step_params.b_roi,
            final_activate,
            final_has_border,
        )
    else:
        if final_has_border:
            z_final = z_final + delta_border
        roi_final = fused_roi_step_epilogue(
            z_roi_bmhw=z_final,
            step_params=step_params,
            activate=final_activate,
            norm=norm,
        )

    if has_valid_border:
        final_border_act = (
            step_params.b_full if final_activate is None else final_activate(step_params.b_full)
        )
        final_border_2d = final_border_act * step_params.s_prob
        final_c_border = jnp.broadcast_to(final_border_2d[jnp.newaxis, :, :], (B, H, W))
        if roi_bbox is not None:
            b_r0, b_r1, b_c0, b_c1 = roi_bbox
            final_cells = final_c_border.at[
                :, r_h + b_r0 : r_h + b_r1, r_w + b_c0 : r_w + b_c1
            ].set(roi_final)
        else:
            final_cells = final_c_border.at[:, r_h : H - r_h, r_w : W - r_w].set(roi_final)
    else:
        if roi_bbox is not None:
            b_r0, b_r1, b_c0, b_c1 = roi_bbox
            final_cells = jnp.zeros((B, H, W), dtype=roi_final.dtype).at[
                :, b_r0:b_r1, b_c0:b_c1
            ].set(roi_final)
        else:
            final_cells = roi_final

    if final_activate is not None:
        final_cells = final_activate(final_cells)

    outputs = _extract_output_from_full_grid(final_cells, topology)
    if return_universe:
        return outputs, final_cells
    return outputs
