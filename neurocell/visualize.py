'''
Visualization, temporal wave-propagation unrolling, GIF animation, and interactive HTML
exploration tools for trained Von Neumann Networks (VNNs).

Provides:
- `unroll_vnn_propagation_2d`: Step-by-step cellular state trajectory `C^(0..depth)` and
  directional wave energy `E^(1..depth)` across arbitrary 2D topologies.
- `load_checkpoint_with_topology`: Reconstructs `(VNNParams, ChipTopology2D, metadata)`
  automatically from `.npz` checkpoints.
- `compute_vnn_diagnostics`: Computes Codd role distributions (`{0: Off, 1: Wire, 2: Gate}`),
  effective linear routing maps, nonlinear gate salience maps, adaptive mask bounds, and
  per-direction Green's filter norms / center-of-mass vectors.
- `plot_vnn_overview` & `plot_vnn_parameter_suite`: High-DPI static Matplotlib visualizations.
- `animate_vnn_propagation`: Multi-panel animated GIF of signal propagation `t = 0..depth`.
- `export_interactive_explorer_html`: Self-contained interactive HTML5/Canvas inspector for
  exploring weights, states, Green's filters, and live wave propagation in any browser.

Copyright (c) 2026 Shekhar S. Chandra. All rights reserved.
Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

@author Shakes
'''
from pathlib import Path
from typing import Sequence
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
import jax
import jax.numpy as jnp

from neurocell.topology import ChipTopology2D, CoordinatePortSpec, PortSpec
from neurocell.params import VNNParams, precompute_step_params
from neurocell.adaptive_mask import compute_adaptive_spatial_mask
from neurocell.kernels_greens import apply_spatial_filter_2d, conv2d_batched_nchw
from neurocell.epilogue import fused_roi_epilogue
from neurocell.propagator import _extract_output_from_full_grid, inject_inputs_2d
from neurocell.model import VonNeumannNetwork


def unroll_vnn_propagation_2d(
    params: VNNParams,
    inputs: jax.Array | Sequence[jax.Array],
    topology: ChipTopology2D,
    activate=jax.nn.tanh,
    final_activate=None,
    norm: bool = True,
    sharpen_filter: jax.Array | None = None,
    gated_filters: bool = False,
    hard_pin_io: bool = False,
    codd_mode: str = "continuous",
    state_temperature: float = 1.0,
    active_threshold: float = 0.6,
) -> dict[str, jax.Array]:
    """
    Executes an unrolled forward pass across `t = 0, 1, ..., topology.depth`, recording
    the full 2D cellular universe state `C^(t)` and directional wave energy at every step.

    Returns a dictionary with:
    - `"frames"`: `jax.Array` of shape `(depth + 1, B, H, W)` (`t = 0` is the input-injected grid).
    - `"dir_energy"`: `jax.Array` of shape `(depth + 1, B, H, W)` measuring mean absolute
      directional response `(1/M) sum_m |Z_m^(t)|` inside the active ROI at each step.
    - `"outputs"`: Final output port tensor of shape `(B, out_h, out_w)`.
    """
    sharpen_fn = (
        (lambda w: apply_spatial_filter_2d(w, sharpen_filter, padding="SAME"))
        if sharpen_filter is not None
        else None
    )
    step_params = precompute_step_params(
        params=params,
        topology=topology,
        activate=activate,
        sharpen_weights_fn=sharpen_fn,
        hard_pin_io=hard_pin_io,
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
    pad_h, pad_w = (kh - 1) // 2, (kw - 1) // 2

    cells_t = inject_inputs_2d(inputs, topology, dtype=params.cell_params.dtype)
    B = cells_t.shape[0]
    c_border_hidden = jnp.broadcast_to(step_params.border_2d[jnp.newaxis, :, :], (B, H, W))
    final_border_act = (
        step_params.b_full if final_activate is None else final_activate(step_params.b_full)
    )
    c_border_final = jnp.broadcast_to(
        (final_border_act * step_params.s_prob)[jnp.newaxis, :, :], (B, H, W)
    )

    frames = [cells_t]
    dir_energies = [jnp.abs(cells_t)]

    for t in range(topology.depth):
        if topology.padding_mode == "CIRCULAR":
            padded = jnp.pad(
                cells_t[:, jnp.newaxis, :, :],
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode="wrap",
            )
            z_roi = conv2d_batched_nchw(padded, filters_oihw, padding="VALID")
        else:
            z_roi = conv2d_batched_nchw(
                cells_t[:, jnp.newaxis, :, :],
                filters_oihw,
                padding=topology.padding_mode,
            )

        # Record directional wave energy inside ROI before Codd reduction
        energy_roi = jnp.mean(jnp.abs(z_roi), axis=1)
        energy_full = jnp.zeros((B, H, W), dtype=cells_t.dtype)
        if topology.padding_mode == "VALID":
            energy_full = energy_full.at[:, r_h : H - r_h, r_w : W - r_w].set(energy_roi)
        else:
            energy_full = energy_roi
        dir_energies.append(energy_full)

        act_t = activate if t < topology.depth - 1 else None
        border_t = c_border_hidden if t < topology.depth - 1 else c_border_final
        cells_t = fused_roi_epilogue(
            z_roi_bmhw=z_roi,
            step_params=step_params,
            c_border_bhw=border_t,
            topology=topology,
            activate=act_t,
            norm=norm,
        )
        if t == topology.depth - 1 and final_activate is not None:
            cells_t = final_activate(cells_t)
        frames.append(cells_t)

    outputs = _extract_output_from_full_grid(cells_t, topology)
    return {
        "frames": jnp.stack(frames, axis=0),
        "dir_energy": jnp.stack(dir_energies, axis=0),
        "outputs": outputs,
    }


def load_checkpoint_with_topology(
    filename: str | Path,
    topology: ChipTopology2D | None = None,
) -> tuple[VNNParams, ChipTopology2D, dict]:
    """
    Loads a saved `.npz` checkpoint and automatically reconstructs both `VNNParams`
    and `ChipTopology2D` from the saved arrays and metadata.
    """
    params = VonNeumannNetwork.load_checkpoint(str(filename))
    meta = {}
    with np.load(str(filename), allow_pickle=False) as data:
        for k in data.files:
            if k not in ("weights", "biases", "states", "filters"):
                val = data[k]
                meta[k] = val.tolist() if isinstance(val, np.ndarray) else val

    if topology is not None:
        return params, topology, meta

    grid_shape = params.weights.shape
    directions, _, kh, kw = params.greens_filters.shape
    kernel_size = (int(kh), int(kw))
    depth = int(meta.get("depth", 4))
    padding_mode = str(meta.get("padding_mode", "VALID"))

    # Case 1: 3-port ALU checkpoint metadata
    if "input_offset_x1" in meta and "input_offset_x2" in meta and "ctrl_offset" in meta:
        in_size = tuple(int(v) for v in meta.get("input_size", [8, 1]))
        ctrl_size = tuple(int(v) for v in meta.get("ctrl_size", [1, 8]))
        out_size = tuple(int(v) for v in meta.get("output_size", [8, 1]))
        in_ports = (
            PortSpec(offset=tuple(int(v) for v in meta["input_offset_x1"]), size=in_size, name="x1"),
            PortSpec(offset=tuple(int(v) for v in meta["input_offset_x2"]), size=in_size, name="x2"),
            PortSpec(offset=tuple(int(v) for v in meta["ctrl_offset"]), size=ctrl_size, name="ctrl"),
        )
        out_port = PortSpec(
            offset=tuple(int(v) for v in meta["output_offset"]), size=out_size, name="out"
        )
        topology = ChipTopology2D.create(
            grid_shape=grid_shape,
            kernel_size=kernel_size,
            directions=directions,
            depth=depth,
            in_ports=in_ports,
            out_port=out_port,
            padding_mode=padding_mode,
        )
    # Case 2: Coordinate-port CLP checkpoint metadata
    elif "input_coords" in meta and "output_coords" in meta:
        topology = ChipTopology2D.clp_chip(
            grid_shape=grid_shape,
            in_coords=meta["input_coords"],
            out_coords=meta["output_coords"],
            kernel_size=kernel_size,
            directions=directions,
            depth=depth,
            padding_mode=padding_mode,
        )
    # Case 3: Single rectangular input/output port metadata
    elif "input_offset" in meta and "output_offset" in meta:
        in_port = PortSpec(
            offset=tuple(int(v) for v in meta["input_offset"]),
            size=tuple(int(v) for v in meta["input_size"]),
            name="in",
        )
        out_port = PortSpec(
            offset=tuple(int(v) for v in meta["output_offset"]),
            size=tuple(int(v) for v in meta["output_size"]),
            name="out",
        )
        topology = ChipTopology2D.create(
            grid_shape=grid_shape,
            kernel_size=kernel_size,
            directions=directions,
            depth=depth,
            in_ports=(in_port,),
            out_port=out_port,
            padding_mode=padding_mode,
            allow_border_output=True,
        )
    # Case 4: Canonical 62x62 ALU chip dimensions
    elif grid_shape == (62, 62) and kernel_size == (23, 13):
        topology = ChipTopology2D.alu_chip(
            n=int(meta.get("n", 8)),
            kernel_size=kernel_size,
            directions=directions,
            depth=depth,
        )
    else:
        # Generic fallback: full-grid or interior ROI topology
        topology = ChipTopology2D.gol_chip(
            grid_shape=grid_shape,
            kernel_size=kernel_size,
            directions=directions,
            depth=depth,
            padding_mode=padding_mode,
        )

    return params, topology, meta


def compute_vnn_diagnostics(
    params: VNNParams,
    topology: ChipTopology2D,
    activate=jax.nn.tanh,
    codd_mode: str = "continuous",
    state_temperature: float = 1.0,
    active_threshold: float = 0.6,
    adaptive_margin: int = 4,
) -> dict:
    """
    Computes structural, routing, and spectral diagnostics for a trained VNN model.
    """
    step_params = precompute_step_params(
        params=params,
        topology=topology,
        activate=activate,
        codd_mode=codd_mode,
        state_temperature=state_temperature,
        active_threshold=active_threshold,
    )
    mask_spec = compute_adaptive_spatial_mask(
        params=params,
        topology=topology,
        margin=adaptive_margin,
        bucket_multiple=8,
    )

    s_raw = np.asarray(params.states)
    s_prob = np.asarray(step_params.s_prob)
    s_roles_raw = np.asarray(
        VonNeumannNetwork.threshold_states(params.states, active_threshold=active_threshold)
    )
    # Cleanly quantize roles to {0.0: Off, 1.0: Passive Wire, 2.0: Active Gate}
    s_roles = np.where(s_roles_raw <= 0.0, 0.0, s_roles_raw)
    w_full = np.asarray(params.weights)
    b_full = np.asarray(params.biases)

    r_h, r_w = topology.pad_h, topology.pad_w
    H, W = topology.grid_shape
    if topology.padding_mode == "VALID" and (r_h > 0 or r_w > 0):
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[r_h : H - r_h, r_w : W - r_w] = True
        s_roles = np.where(roi_mask, s_roles, 0.0)

    # Effective linear routing weight W * (1 - s_prob) and nonlinear gate salience |act(B) * s_prob|
    w_wire = w_full * (1.0 - s_prob)
    b_act = np.asarray(b_full if activate is None else activate(params.biases))
    gate_salience = np.abs(b_act * s_prob)

    roles_roi = s_roles[r_h : H - r_h, r_w : W - r_w] if topology.padding_mode == "VALID" else s_roles
    total_roi = roles_roi.size
    n_off = int(np.sum(roles_roi <= 0.0))
    n_wire = int(np.sum(roles_roi == 1.0))
    n_gate = int(np.sum(roles_roi == 2.0))

    # Green's filter norms and directional center-of-mass vectors
    filters_np = np.asarray(params.greens_filters[:, 0, :, :])
    m_dirs, kh, kw = filters_np.shape
    r_grid, c_grid = np.meshgrid(
        np.arange(kh, dtype=np.float32) - (kh - 1) / 2.0,
        np.arange(kw, dtype=np.float32) - (kw - 1) / 2.0,
        indexing="ij",
    )
    filter_norms = []
    filter_com = []
    for m in range(m_dirs):
        fm = filters_np[m]
        abs_fm = np.abs(fm)
        mass = float(np.sum(abs_fm)) + 1e-12
        norm_l2 = float(np.linalg.norm(fm))
        com_r = float(np.sum(abs_fm * r_grid) / mass)
        com_c = float(np.sum(abs_fm * c_grid) / mass)
        filter_norms.append(round(norm_l2, 4))
        filter_com.append((round(com_r, 3), round(com_c, 3)))

    return {
        "s_raw": s_raw,
        "s_prob": s_prob,
        "s_roles": s_roles,
        "w_full": w_full,
        "b_full": b_full,
        "w_wire": w_wire,
        "gate_salience": gate_salience,
        "mask_spec": mask_spec,
        "role_counts_roi": {
            "total_roi_cells": int(total_roi),
            "off_cells": n_off,
            "wire_cells": n_wire,
            "gate_cells": n_gate,
            "off_pct": round(100.0 * n_off / max(1, total_roi), 2),
            "wire_pct": round(100.0 * n_wire / max(1, total_roi), 2),
            "gate_pct": round(100.0 * n_gate / max(1, total_roi), 2),
        },
        "filter_norms": filter_norms,
        "filter_com": filter_com,
    }


def _draw_topology_overlays(
    ax,
    topology: ChipTopology2D,
    mask_spec=None,
    show_roi: bool = True,
    show_adaptive_bbox: bool = True,
):
    """Draws ROI dashed bounds, adaptive sub-ROI box, and Input/Output port boxes on `ax`."""
    r_h, r_w = topology.pad_h, topology.pad_w
    roi_h, roi_w = topology.roi_shape
    if show_roi and topology.padding_mode == "VALID" and (r_h > 0 or r_w > 0):
        ax.add_patch(
            patches.Rectangle(
                (r_w - 0.5, r_h - 0.5),
                roi_w,
                roi_h,
                linewidth=1.2,
                edgecolor="#38bdf8",
                facecolor="none",
                linestyle="--",
                label="Valid ROI",
            )
        )

    if (
        show_adaptive_bbox
        and mask_spec is not None
        and not mask_spec.is_full_roi
    ):
        b_r0, b_r1, b_c0, b_c1 = mask_spec.roi_bbox
        ax.add_patch(
            patches.Rectangle(
                (r_w + b_c0 - 0.5, r_h + b_r0 - 0.5),
                b_c1 - b_c0,
                b_r1 - b_r0,
                linewidth=1.5,
                edgecolor="#a855f7",
                facecolor="none",
                linestyle=":",
                label="Adaptive Sub-ROI",
            )
        )

    in_colors = ["#22c55e", "#10b981", "#f59e0b", "#06b6d4"]
    for idx, port in enumerate(topology.in_ports):
        col = in_colors[idx % len(in_colors)]
        if isinstance(port, CoordinatePortSpec):
            for r, c in port.coords:
                ax.add_patch(
                    patches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1, linewidth=1.5, edgecolor=col, facecolor="none"
                    )
                )
        elif port.size != topology.grid_shape:
            r0, c0 = port.offset
            ph, pw = port.size
            ax.add_patch(
                patches.Rectangle(
                    (c0 - 0.5, r0 - 0.5), pw, ph, linewidth=1.6, edgecolor=col, facecolor="none"
                )
            )

    out_port = topology.out_port
    if isinstance(out_port, CoordinatePortSpec):
        for r, c in out_port.coords:
            ax.add_patch(
                patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, linewidth=1.8, edgecolor="#ef4444", facecolor="none"
                )
            )
    elif out_port.size != topology.grid_shape:
        or0, oc0 = out_port.offset
        oph, opw = out_port.size
        ax.add_patch(
            patches.Rectangle(
                (oc0 - 0.5, or0 - 0.5), opw, oph, linewidth=1.8, edgecolor="#ef4444", facecolor="none"
            )
        )


def plot_vnn_parameter_suite(
    params: VNNParams,
    topology: ChipTopology2D,
    output_dir: str | Path = ".",
    prefix: str = "vnn_model",
    adaptive_margin: int = 4,
    codd_mode: str = "continuous",
    active_threshold: float = 0.6,
) -> dict[str, Path]:
    """
    Generates a complete suite of high-resolution static PNG visualizations exploring the
    trained VNN's states, discrete Codd roles, weights, biases, effective wire/gate maps,
    adaptive spatial mask, and directional Green's functions.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = compute_vnn_diagnostics(
        params=params,
        topology=topology,
        codd_mode=codd_mode,
        active_threshold=active_threshold,
        adaptive_margin=adaptive_margin,
    )
    mask_spec = diag["mask_spec"]
    rc = diag["role_counts_roi"]
    saved_files = {}

    # 1. Six-Panel Master Overview Figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=150)
    fig.subplots_adjust(hspace=0.30, wspace=0.26)

    # Panel (0, 0): Continuous Codd States S
    ax = axes[0, 0]
    im = ax.imshow(diag["s_raw"], cmap="viridis")
    _draw_topology_overlays(ax, topology, mask_spec)
    ax.set_title(
        f"1. Continuous Codd States S ({topology.H}x{topology.W})\n"
        "[Green=In, Amber=Ctrl, Red=Out, Cyan=ROI, Purple=Adaptive]",
        fontsize=9.5,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel (0, 1): Discrete Codd Circuit Roles {0, 1, 2}
    ax = axes[0, 1]
    im = ax.imshow(diag["s_roles"], cmap="plasma", vmin=-0.5, vmax=2.0)
    _draw_topology_overlays(ax, topology, mask_spec)
    ax.set_title(
        f"2. Discrete Codd Roles {{0:Off, 1:Wire, 2:Gate}}\n"
        f"[ROI: {rc['gate_pct']}% Gate ({rc['gate_cells']}), "
        f"{rc['wire_pct']}% Wire ({rc['wire_cells']}), {rc['off_pct']}% Off]",
        fontsize=9.5,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel (0, 2): Learned Cellular Weights W
    ax = axes[0, 2]
    w_lim = max(0.5, float(np.percentile(np.abs(diag["w_full"]), 99)))
    im = ax.imshow(diag["w_full"], cmap="coolwarm", vmin=-w_lim, vmax=w_lim)
    _draw_topology_overlays(ax, topology, mask_spec)
    ax.set_title("3. Learned Cellular Weights W\n[Linear Signal Amplification / Attenuation]", fontsize=9.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel (1, 0): Learned Cellular Biases B
    ax = axes[1, 0]
    b_lim = max(0.5, float(np.percentile(np.abs(diag["b_full"]), 99)))
    im = ax.imshow(diag["b_full"], cmap="RdBu_r", vmin=-b_lim, vmax=b_lim)
    _draw_topology_overlays(ax, topology, mask_spec)
    ax.set_title("4. Learned Cellular Biases B\n[Nonlinear Decision Thresholds]", fontsize=9.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel (1, 1): Effective Linear Wire Routing W * (1 - sigma(S))
    ax = axes[1, 1]
    ww_lim = max(0.2, float(np.percentile(np.abs(diag["w_wire"]), 99)))
    im = ax.imshow(diag["w_wire"], cmap="coolwarm", vmin=-ww_lim, vmax=ww_lim)
    _draw_topology_overlays(ax, topology, mask_spec)
    ax.set_title(
        "5. Effective Passive Routing W_eff = W(1 - σ(S))\n"
        "[Highlights Pure Linear Signal Corridors]",
        fontsize=9.5,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel (1, 2): Effective Nonlinear Gate Salience |tanh(B) * sigma(S)| + Adaptive Mask
    ax = axes[1, 2]
    im = ax.imshow(diag["gate_salience"], cmap="magma")
    _draw_topology_overlays(ax, topology, mask_spec)
    ax.set_title(
        f"6. Active Gate Salience |tanh(B)σ(S)| & Adaptive Mask\n"
        f"[Active Sub-ROI: {mask_spec.active_shape} ({mask_spec.active_fraction*100:.1f}% of ROI)]",
        fontsize=9.5,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Neurocell v2 Model Inspector — Grid {topology.grid_shape}, Kernel {topology.kernel_size}, "
        f"Directions M={topology.directions}, Depth d={topology.depth}",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    overview_path = out_dir / f"{prefix}_overview.png"
    plt.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files["overview"] = overview_path

    # 2. Multi-Direction Green's Filter Gallery with Center-of-Mass Arrows
    m_dirs = topology.directions
    cols = int(math.ceil(math.sqrt(m_dirs)))
    rows = int(math.ceil(m_dirs / cols))
    fig_f, axes_f = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.8 * rows), dpi=150)
    axes_flat = np.atleast_1d(axes_f).flatten()
    filters_np = np.asarray(params.greens_filters[:, 0, :, :])
    kh, kw = topology.kernel_size
    cy, cx = (kh - 1) / 2.0, (kw - 1) / 2.0

    for m in range(len(axes_flat)):
        ax_m = axes_flat[m]
        if m < m_dirs:
            fm = filters_np[m]
            vmax = max(1e-4, float(np.max(np.abs(fm))))
            ax_m.imshow(fm, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            com_r, com_c = diag["filter_com"][m]
            ax_m.plot(cx, cy, "k+", markersize=5)
            ax_m.arrow(
                cx,
                cy,
                com_c * 2.0,
                com_r * 2.0,
                head_width=0.6,
                head_length=0.6,
                fc="#10b981",
                ec="#10b981",
                linewidth=1.3,
            )
            ax_m.set_title(f"G_{m} (‖G‖={diag['filter_norms'][m]:.2f})", fontsize=8.5)
            ax_m.axis("off")
        else:
            ax_m.axis("off")

    fig_f.suptitle(
        f"Learned Directional Green's Functions ({m_dirs} Filters of Shape {kh}x{kw})\n"
        "[Green Arrow = Receptive Field Center-of-Mass Vector]",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    filters_path = out_dir / f"{prefix}_greens_filters.png"
    plt.savefig(filters_path, dpi=150, bbox_inches="tight")
    plt.close(fig_f)
    saved_files["greens_filters"] = filters_path

    return saved_files


def animate_vnn_propagation(
    params: VNNParams,
    topology: ChipTopology2D,
    inputs: jax.Array | Sequence[jax.Array],
    output_gif: str | Path = "vnn_signal_propagation.gif",
    target_bits: Sequence[int] | None = None,
    sample_idx: int = 0,
    fps: float = 1.5,
    title_prefix: str = "VNN Cellular Wave Propagation",
    codd_mode: str = "continuous",
    active_threshold: float = 0.6,
) -> Path:
    """
    Creates a 3-panel animated GIF showing:
    1. Learned Codd Circuit Roles `{0: Off, 1: Wire, 2: Gate}` + I/O port boxes,
    2. Live cellular activation wavefront `C^(t)` across `t = 0 .. depth`,
    3. Directional wave energy `E^(t)` and live Output Port readout bars at step `t`.
    """
    out_gif_path = Path(output_gif)
    out_gif_path.parent.mkdir(parents=True, exist_ok=True)

    unrolled = unroll_vnn_propagation_2d(
        params=params,
        inputs=inputs,
        topology=topology,
        codd_mode=codd_mode,
        active_threshold=active_threshold,
    )
    frames_np = np.asarray(unrolled["frames"][:, sample_idx])      # (T, H, W)
    energy_np = np.asarray(unrolled["dir_energy"][:, sample_idx])  # (T, H, W)
    num_frames = frames_np.shape[0]

    diag = compute_vnn_diagnostics(
        params=params,
        topology=topology,
        codd_mode=codd_mode,
        active_threshold=active_threshold,
    )
    mask_spec = diag["mask_spec"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), dpi=130)
    fig.subplots_adjust(wspace=0.28, top=0.84, bottom=0.12)

    # Panel 1 (Static Reference): Discrete Codd Circuit Topology
    ax_topo = axes[0]
    im_topo = ax_topo.imshow(diag["s_roles"], cmap="plasma", vmin=-0.5, vmax=2.0)
    _draw_topology_overlays(ax_topo, topology, mask_spec)
    ax_topo.set_title("Learned Codd Circuit Roles\n[2=Gate, 1=Wire, <=0=Off]", fontsize=10)
    plt.colorbar(im_topo, ax=ax_topo, fraction=0.046, pad=0.04)

    # Panel 2 (Animated): Cellular Wavefront C^(t)
    ax_wave = axes[1]
    vmax_c = max(0.8, float(np.percentile(np.abs(frames_np), 99.5)))
    im_wave = ax_wave.imshow(
        frames_np[0], cmap="RdBu_r", vmin=-vmax_c, vmax=vmax_c, animated=True
    )
    _draw_topology_overlays(ax_wave, topology, mask_spec)
    wave_title = ax_wave.set_title(
        f"Cellular Field C^(t=0 / {num_frames - 1})\n[Input Injection]", fontsize=10
    )
    plt.colorbar(im_wave, ax=ax_wave, fraction=0.046, pad=0.04)

    # Panel 3 (Animated): Output Port Readout Bars at Step t
    ax_out = axes[2]
    out_port = topology.out_port
    out_h, out_w = out_port.size
    num_out_cells = out_h * out_w

    def _extract_step_out(frame_2d: np.ndarray) -> np.ndarray:
        if isinstance(out_port, CoordinatePortSpec):
            vals = [frame_2d[r, c] for r, c in out_port.coords]
            return np.asarray(vals, dtype=np.float32)
        or0, oc0 = out_port.offset
        return frame_2d[or0 : or0 + out_h, oc0 : oc0 + out_w].flatten()

    if num_out_cells <= 32:
        y_pos = np.arange(num_out_cells)
        init_vals = _extract_step_out(frames_np[0])
        bars = ax_out.barh(y_pos, init_vals, color="#3b82f6", alpha=0.85, height=0.65)
        ax_out.axvline(0.5, color="#10b981", linestyle="--", linewidth=1.2, label="Threshold (0.5)")
        ax_out.axvline(0.0, color="#64748b", linestyle=":", linewidth=1.0)
        if target_bits is not None and len(target_bits) == num_out_cells:
            ax_out.scatter(
                target_bits,
                y_pos,
                color="#ef4444",
                zorder=5,
                s=45,
                 marker="D",
                label="Target Bit",
            )
        ax_out.set_yticks(y_pos)
        ax_out.set_yticklabels([f"Out[{i}]" for i in range(num_out_cells)], fontsize=8)
        ax_out.invert_yaxis()
        ax_out.set_xlim(-0.6, 1.4)
        ax_out.set_xlabel("Cell Activation")
        ax_out.legend(loc="lower right", fontsize=8)
        out_title = ax_out.set_title("Output Port Readout (t=0)", fontsize=10)
        use_bar_chart = True
    else:
        vmax_e = max(0.2, float(np.percentile(energy_np, 99.5)))
        im_energy = ax_out.imshow(
            energy_np[0], cmap="magma", vmin=0.0, vmax=vmax_e, animated=True
        )
        _draw_topology_overlays(ax_out, topology, mask_spec)
        out_title = ax_out.set_title("Directional Wave Energy E^(t=0)", fontsize=10)
        plt.colorbar(im_energy, ax=ax_out, fraction=0.046, pad=0.04)
        use_bar_chart = False

    suptitle = fig.suptitle(
        f"{title_prefix} — Step t = 0 / {num_frames - 1}",
        fontsize=12,
        fontweight="bold",
    )

    def _update(frame_idx: int):
        im_wave.set_array(frames_np[frame_idx])
        stage_lbl = (
            "Input Injection"
            if frame_idx == 0
            else (
                "Final Output Readout"
                if frame_idx == num_frames - 1
                else f"Green's Propagation Step {frame_idx}"
            )
        )
        wave_title.set_text(
            f"Cellular Field C^(t={frame_idx} / {num_frames - 1})\n[{stage_lbl}]"
        )
        suptitle.set_text(f"{title_prefix} — Step t = {frame_idx} / {num_frames - 1}")

        if use_bar_chart:
            vals = _extract_step_out(frames_np[frame_idx])
            preds = (vals >= 0.5).astype(int)
            for idx, (bar, v) in enumerate(zip(bars, vals)):
                bar.set_width(float(v))
                if target_bits is not None and len(target_bits) == num_out_cells:
                    match = int(preds[idx]) == int(target_bits[idx])
                    bar.set_color("#10b981" if match else "#f59e0b")
            out_title.set_text(
                f"Output Port Readout (t={frame_idx})\n"
                f"Pred Bits: {''.join(str(p) for p in preds[:16])}"
            )
        else:
            im_energy.set_array(energy_np[frame_idx])
            out_title.set_text(f"Directional Wave Energy E^(t={frame_idx})")

        return [im_wave]

    interval_ms = int(1000.0 / max(0.2, fps))
    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=num_frames,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    ani.save(str(out_gif_path), writer="pillow", fps=fps)
    plt.close(fig)
    return out_gif_path


EXPLORER_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Neurocell v2 — Interactive VNN Model & Wave Propagation Explorer</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    :root {
      --bg: #0b0f19;
      --card: #111827;
      --fg: #f8fafc;
      --muted: #94a3b8;
      --border: #1e293b;
      --primary: #3b82f6;
    }
    body {
      background-color: var(--bg);
      color: var(--fg);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0;
      padding: 0;
    }
    .card {
      background-color: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
    }
    .btn-mode {
      padding: 0.38rem 0.75rem;
      font-size: 0.78rem;
      font-weight: 600;
      border-radius: 0.4rem;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--fg);
      cursor: pointer;
      transition: all 0.12s ease;
    }
    .btn-mode.active {
      background: var(--primary);
      border-color: var(--primary);
      color: #ffffff;
    }
    canvas {
      image-rendering: pixelated;
    }
  </style>
</head>
<body class="p-4 md:p-6">
  <div class="max-w-7xl mx-auto space-y-5">
    <!-- Header -->
    <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-3 border-b pb-4" style="border-color: var(--border);">
      <div>
        <div class="flex items-center gap-2">
          <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full" style="background: rgba(16,185,129,0.15); color: #10b981;" id="badge-grid"></span>
          <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full" style="background: rgba(59,130,246,0.15); color: #3b82f6;" id="badge-roi"></span>
          <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full" style="background: rgba(168,85,247,0.15); color: #a855f7;" id="badge-adaptive"></span>
        </div>
        <h1 class="text-xl md:text-2xl font-bold mt-1" id="page-title">Neurocell v2 — Trained VNN Interactive Explorer</h1>
        <p class="text-xs md:text-sm" style="color: var(--muted);">Hover over any cell (r, c) to inspect its learned Codd state, discrete role {0, 1, 2}, weight, bias, and temporal wave trajectory C^(0..d).</p>
      </div>
    </div>

    <!-- Top Row: Spatial Parameter Explorer + Cell Inspector -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">
      <div class="lg:col-span-7 card p-4 space-y-3">
        <div class="flex flex-wrap gap-1.5" id="param-mode-btns">
          <button class="btn-mode active" onclick="setParamView('s_raw', this)">States S</button>
          <button class="btn-mode" onclick="setParamView('s_roles', this)">Codd Roles {0,1,2}</button>
          <button class="btn-mode" onclick="setParamView('w_full', this)">Weights W</button>
          <button class="btn-mode" onclick="setParamView('b_full', this)">Biases B</button>
          <button class="btn-mode" onclick="setParamView('w_wire', this)">Wire Routing W(1-σ(S))</button>
          <button class="btn-mode" onclick="setParamView('gate_salience', this)">Gate Salience |tanh(B)σ(S)|</button>
        </div>
        <div class="flex justify-center bg-slate-950 p-3 rounded-lg border" style="border-color: var(--border);">
          <canvas id="canvas-param" class="cursor-crosshair rounded"></canvas>
        </div>
        <div class="flex flex-wrap items-center justify-between text-xs" style="color: var(--muted);">
          <span>Cyan Dashed = Valid ROI • Purple Dotted = Adaptive Sub-ROI (margin=4)</span>
          <span>Green = Input Ports • Red = Output Port</span>
        </div>
      </div>

      <!-- Cell Inspector + Role Breakdown -->
      <div class="lg:col-span-5 card p-4 space-y-4">
        <div class="flex items-center justify-between border-b pb-2" style="border-color: var(--border);">
          <h2 class="text-sm font-bold">Cell Inspector</h2>
          <span class="font-mono text-xs px-2 py-0.5 rounded bg-blue-500/20 text-blue-400" id="insp-coords">(r=0, c=0)</span>
        </div>
        <div class="grid grid-cols-2 gap-2 text-xs font-mono">
          <div class="p-2 rounded bg-slate-900 border" style="border-color: var(--border);">
            <div style="color: var(--muted);">Codd State S / Prob σ(S)</div>
            <div class="text-sm font-bold mt-0.5" id="insp-state">-</div>
          </div>
          <div class="p-2 rounded bg-slate-900 border" style="border-color: var(--border);">
            <div style="color: var(--muted);">Discrete Codd Role</div>
            <div class="text-sm font-bold mt-0.5" id="insp-role">-</div>
          </div>
          <div class="p-2 rounded bg-slate-900 border" style="border-color: var(--border);">
            <div style="color: var(--muted);">Weight W / Wire W_eff</div>
            <div class="text-sm font-bold mt-0.5" id="insp-weight">-</div>
          </div>
          <div class="p-2 rounded bg-slate-900 border" style="border-color: var(--border);">
            <div style="color: var(--muted);">Bias B / Gate Salience</div>
            <div class="text-sm font-bold mt-0.5" id="insp-bias">-</div>
          </div>
        </div>

        <div>
          <div class="text-xs font-semibold mb-1" style="color: var(--muted);">Cell Temporal Trajectory C^(t) for t = 0..d</div>
          <div class="grid grid-cols-5 gap-1 text-center text-xs font-mono" id="insp-trajectory"></div>
        </div>

        <div class="border-t pt-3 space-y-1.5 text-xs" style="border-color: var(--border);">
          <div class="font-semibold">Active ROI Codd Role Breakdown</div>
          <div class="flex justify-between"><span style="color: var(--muted);">Role 2 (Active Logic Gate):</span><span class="font-mono font-bold text-amber-400" id="stat-gate"></span></div>
          <div class="flex justify-between"><span style="color: var(--muted);">Role 1 (Passive Signal Wire):</span><span class="font-mono font-bold text-emerald-400" id="stat-wire"></span></div>
          <div class="flex justify-between"><span style="color: var(--muted);">Role 0 (Confluent / Off):</span><span class="font-mono font-bold text-slate-400" id="stat-off"></span></div>
        </div>
      </div>
    </div>

    <!-- Second Row: Signal Propagation Animator + Green's Filters -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">
      <div class="lg:col-span-7 card p-4 space-y-3">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <div class="flex items-center gap-2">
            <button class="btn-mode active" id="btn-play" onclick="togglePlay()">▶ Play Wave</button>
            <div class="flex gap-1" id="step-btns"></div>
          </div>
          <div class="flex gap-1">
            <button class="btn-mode active" id="btn-wave-c" onclick="setWaveMode('frames')">Activation C^(t)</button>
            <button class="btn-mode" id="btn-wave-e" onclick="setWaveMode('dir_energy')">Wave Energy E^(t)</button>
          </div>
        </div>
        <div class="flex justify-center bg-slate-950 p-3 rounded-lg border" style="border-color: var(--border);">
          <canvas id="canvas-wave" class="cursor-crosshair rounded"></canvas>
        </div>
        <div class="text-xs font-mono" id="out-readout-bar" style="color: var(--muted);"></div>
      </div>

      <div class="lg:col-span-5 card p-4 space-y-3">
        <div class="flex items-center justify-between">
          <h2 class="text-sm font-bold">Directional Green's Filters (G_m)</h2>
          <span class="text-xs font-mono" style="color: var(--muted);" id="filter-detail-label">Select a filter</span>
        </div>
        <div class="grid grid-cols-4 gap-2" id="filters-gallery"></div>
      </div>
    </div>
  </div>

  <script>
    const DATA = __EXPLORER_DATA_JSON__;
    let currentParamView = 's_raw';
    let currentWaveMode = 'frames';
    let currentStep = DATA.frames.length - 1;
    let playTimer = null;
    const SCALE = Math.max(4, Math.min(8, Math.floor(380 / Math.max(DATA.H, DATA.W))));

    function bwrCmap(v, vmin, vmax) {
      const t = Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin + 1e-12)));
      if (t < 0.5) {
        const s = t * 2;
        return [Math.round(30 + 225 * s), Math.round(64 + 191 * s), 255];
      } else {
        const s = (t - 0.5) * 2;
        return [255, Math.round(255 * (1 - s * 0.85)), Math.round(255 * (1 - s * 0.85))];
      }
    }

    function viridisCmap(v, vmin, vmax) {
      const t = Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin + 1e-12)));
      const r = Math.round(68 + t * (253 - 68));
      const g = Math.round(1 + t * (231 - 1));
      const b = Math.round(84 + (1 - Math.abs(t - 0.5) * 2) * 100);
      return [r, g, b];
    }

    function drawMatrix(canvas, mat, cmapFn) {
      canvas.width = DATA.W * SCALE;
      canvas.height = DATA.H * SCALE;
      const ctx = canvas.getContext('2d');
      for (let r = 0; r < DATA.H; r++) {
        for (let c = 0; c < DATA.W; c++) {
          const rgb = cmapFn(mat[r][c]);
          ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
          ctx.fillRect(c * SCALE, r * SCALE, SCALE, SCALE);
        }
      }
      // ROI box
      if (DATA.pad_h > 0 || DATA.pad_w > 0) {
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(DATA.pad_w * SCALE, DATA.pad_h * SCALE, DATA.roi_w * SCALE, DATA.roi_h * SCALE);
      }
      // Adaptive bbox
      if (DATA.adaptive_bbox) {
        const [r0, r1, c0, c1] = DATA.adaptive_bbox;
        ctx.strokeStyle = '#a855f7';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([2, 2]);
        ctx.strokeRect((DATA.pad_w + c0) * SCALE, (DATA.pad_h + r0) * SCALE, (c1 - c0) * SCALE, (r1 - r0) * SCALE);
      }
      ctx.setLineDash([]);
      // Input ports
      DATA.in_ports.forEach(p => {
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 1.8;
        ctx.strokeRect(p.offset[1] * SCALE, p.offset[0] * SCALE, p.size[1] * SCALE, p.size[0] * SCALE);
      });
      // Output port
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2.0;
      ctx.strokeRect(DATA.out_port.offset[1] * SCALE, DATA.out_port.offset[0] * SCALE, DATA.out_port.size[1] * SCALE, DATA.out_port.size[0] * SCALE);
    }

    function renderParamCanvas() {
      const mat = DATA[currentParamView];
      let flat = mat.flat();
      let vmin = Math.min(...flat), vmax = Math.max(...flat);
      const canvas = document.getElementById('canvas-param');
      if (currentParamView === 'w_full' || currentParamView === 'b_full' || currentParamView === 'w_wire') {
        const lim = Math.max(0.5, Math.max(Math.abs(vmin), Math.abs(vmax)));
        drawMatrix(canvas, mat, v => bwrCmap(v, -lim, lim));
      } else {
        drawMatrix(canvas, mat, v => viridisCmap(v, vmin, vmax));
      }
    }

    function setParamView(view, btn) {
      currentParamView = view;
      document.querySelectorAll('#param-mode-btns .btn-mode').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderParamCanvas();
    }

    function setStep(t) {
      currentStep = t;
      document.querySelectorAll('#step-btns .btn-mode').forEach((b, idx) => {
        b.classList.toggle('active', idx === t);
      });
      const mat = DATA[currentWaveMode][t];
      const flat = mat.flat();
      const lim = Math.max(0.5, Math.max(...flat.map(Math.abs)));
      const canvas = document.getElementById('canvas-wave');
      if (currentWaveMode === 'frames') {
        drawMatrix(canvas, mat, v => bwrCmap(v, -lim, lim));
      } else {
        drawMatrix(canvas, mat, v => viridisCmap(v, 0, lim));
      }
      const outVals = DATA.step_outputs[t];
      const bits = outVals.map(v => (v >= 0.5 ? 1 : 0)).join('');
      document.getElementById('out-readout-bar').textContent =
        `Step t=${t} Output Readout: [${outVals.map(v => v.toFixed(2)).join(', ')}] → Rounded Bits: ${bits}`;
    }

    function setWaveMode(mode) {
      currentWaveMode = mode;
      document.getElementById('btn-wave-c').classList.toggle('active', mode === 'frames');
      document.getElementById('btn-wave-e').classList.toggle('active', mode === 'dir_energy');
      setStep(currentStep);
    }

    function togglePlay() {
      const btn = document.getElementById('btn-play');
      if (playTimer) {
        clearInterval(playTimer);
        playTimer = null;
        btn.textContent = '▶ Play Wave';
      } else {
        btn.textContent = '⏸ Pause';
        playTimer = setInterval(() => {
          setStep((currentStep + 1) % DATA.frames.length);
        }, 750);
      }
    }

    function inspectCell(r, c) {
      if (r < 0 || r >= DATA.H || c < 0 || c >= DATA.W) return;
      document.getElementById('insp-coords').textContent = `(r=${r}, c=${c})`;
      const sRaw = DATA.s_raw[r][c];
      const sProb = DATA.s_prob[r][c];
      const role = DATA.s_roles[r][c];
      const roleStr = role <= 0 ? `${role.toFixed(2)} (0: Off)` : (role === 1 ? '1 (Passive Wire)' : '2 (Active Gate)');
      document.getElementById('insp-state').textContent = `${sRaw.toFixed(3)} / σ=${sProb.toFixed(3)}`;
      document.getElementById('insp-role').textContent = roleStr;
      document.getElementById('insp-weight').textContent = `${DATA.w_full[r][c].toFixed(3)} / ${DATA.w_wire[r][c].toFixed(3)}`;
      document.getElementById('insp-bias').textContent = `${DATA.b_full[r][c].toFixed(3)} / ${DATA.gate_salience[r][c].toFixed(3)}`;

      const traj = document.getElementById('insp-trajectory');
      traj.innerHTML = DATA.frames.map((f, t) => `
        <div class="p-1.5 rounded bg-slate-900 border" style="border-color: var(--border);">
          <div style="color: var(--muted);">t=${t}</div>
          <div class="font-bold">${f[r][c].toFixed(2)}</div>
        </div>
      `).join('');
    }

    function renderFilters() {
      const gal = document.getElementById('filters-gallery');
      gal.innerHTML = '';
      DATA.filters.forEach((fMat, m) => {
        const wrap = document.createElement('div');
        wrap.className = 'flex flex-col items-center p-1.5 rounded border bg-slate-950 cursor-pointer';
        wrap.style.borderColor = 'var(--border)';
        const lbl = document.createElement('div');
        lbl.className = 'text-xs font-mono mb-1';
        lbl.textContent = `G_${m} (${DATA.filter_norms[m]})`;
        const c = document.createElement('canvas');
        const kh = fMat.length, kw = fMat[0].length;
        const sc = Math.max(3, Math.floor(64 / Math.max(kh, kw)));
        c.width = kw * sc;
        c.height = kh * sc;
        const ctx = c.getContext('2d');
        const lim = Math.max(0.01, ...fMat.flat().map(Math.abs));
        for (let r = 0; r < kh; r++) {
          for (let col = 0; col < kw; col++) {
            const rgb = bwrCmap(fMat[r][col], -lim, lim);
            ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
            ctx.fillRect(col * sc, r * sc, sc, sc);
          }
        }
        wrap.onclick = () => {
          const [cr, cc] = DATA.filter_com[m];
          document.getElementById('filter-detail-label').textContent =
            `G_${m}: L2=${DATA.filter_norms[m]}, CoM=(Δr=${cr}, Δc=${cc})`;
        };
        wrap.appendChild(lbl);
        wrap.appendChild(c);
        gal.appendChild(wrap);
      });
    }

    window.addEventListener('DOMContentLoaded', () => {
      document.getElementById('badge-grid').textContent = `Grid: ${DATA.H}×${DATA.W} (K=${DATA.kh}×${DATA.kw}, M=${DATA.M}, d=${DATA.depth})`;
      document.getElementById('badge-roi').textContent = `Valid ROI: ${DATA.roi_h}×${DATA.roi_w}`;
      document.getElementById('badge-adaptive').textContent = `Adaptive Sub-ROI: ${DATA.adaptive_shape[0]}×${DATA.adaptive_shape[1]} (${DATA.adaptive_pct}%)`;

      const rc = DATA.role_counts_roi;
      document.getElementById('stat-gate').textContent = `${rc.gate_cells} cells (${rc.gate_pct}%)`;
      document.getElementById('stat-wire').textContent = `${rc.wire_cells} cells (${rc.wire_pct}%)`;
      document.getElementById('stat-off').textContent = `${rc.off_cells} cells (${rc.off_pct}%)`;

      const stepBtns = document.getElementById('step-btns');
      DATA.frames.forEach((_, t) => {
        const b = document.createElement('button');
        b.className = 'btn-mode';
        b.textContent = `t=${t}`;
        b.onclick = () => setStep(t);
        stepBtns.appendChild(b);
      });

      renderParamCanvas();
      setStep(DATA.frames.length - 1);
      inspectCell(DATA.out_port.offset[0], DATA.out_port.offset[1]);
      renderFilters();

      ['canvas-param', 'canvas-wave'].forEach(id => {
        const el = document.getElementById(id);
        el.addEventListener('mousemove', e => {
          const rect = el.getBoundingClientRect();
          const c = Math.floor(((e.clientX - rect.left) * (el.width / rect.width)) / SCALE);
          const r = Math.floor(((e.clientY - rect.top) * (el.height / rect.height)) / SCALE);
          inspectCell(r, c);
        });
      });
    });
  </script>
</body>
</html>
"""


def export_interactive_explorer_html(
    params: VNNParams,
    topology: ChipTopology2D,
    inputs: jax.Array | Sequence[jax.Array],
    output_html: str | Path = "vnn_model_explorer.html",
    sample_idx: int = 0,
    adaptive_margin: int = 4,
    codd_mode: str = "continuous",
    active_threshold: float = 0.6,
) -> Path:
    """
    Exports a standalone, zero-dependency Interactive HTML5 Explorer (`vnn_model_explorer.html`)
    for inspecting any trained VNN's states, Codd roles `{0, 1, 2}`, weights, biases,
    Green's filters, and step-by-step signal propagation `t = 0 .. depth`.
    """
    out_path = Path(output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    diag = compute_vnn_diagnostics(
        params=params,
        topology=topology,
        codd_mode=codd_mode,
        active_threshold=active_threshold,
        adaptive_margin=adaptive_margin,
    )
    unrolled = unroll_vnn_propagation_2d(
        params=params,
        inputs=inputs,
        topology=topology,
        codd_mode=codd_mode,
        active_threshold=active_threshold,
    )
    frames_np = np.asarray(unrolled["frames"][:, sample_idx])
    energy_np = np.asarray(unrolled["dir_energy"][:, sample_idx])

    out_port = topology.out_port
    out_h, out_w = out_port.size
    step_outputs = []
    for t in range(frames_np.shape[0]):
        if isinstance(out_port, CoordinatePortSpec):
            vals = [float(frames_np[t, r, c]) for r, c in out_port.coords]
        else:
            or0, oc0 = out_port.offset
            vals = frames_np[t, or0 : or0 + out_h, oc0 : oc0 + out_w].flatten().tolist()
        step_outputs.append([round(float(v), 4) for v in vals[:16]])

    mask_spec = diag["mask_spec"]
    payload = {
        "H": topology.H,
        "W": topology.W,
        "kh": topology.kernel_size[0],
        "kw": topology.kernel_size[1],
        "M": topology.directions,
        "depth": topology.depth,
        "pad_h": topology.pad_h,
        "pad_w": topology.pad_w,
        "roi_h": topology.roi_shape[0],
        "roi_w": topology.roi_shape[1],
        "adaptive_bbox": list(mask_spec.roi_bbox) if not mask_spec.is_full_roi else None,
        "adaptive_shape": list(mask_spec.active_shape),
        "adaptive_pct": round(mask_spec.active_fraction * 100.0, 1),
        "in_ports": [
            {"offset": list(p.offset), "size": list(p.size), "name": p.name}
            for p in topology.in_ports
        ],
        "out_port": {
            "offset": list(out_port.offset),
            "size": list(out_port.size),
            "name": out_port.name,
        },
        "s_raw": np.round(diag["s_raw"], 4).tolist(),
        "s_prob": np.round(diag["s_prob"], 4).tolist(),
        "s_roles": np.round(diag["s_roles"], 2).tolist(),
        "w_full": np.round(diag["w_full"], 4).tolist(),
        "b_full": np.round(diag["b_full"], 4).tolist(),
        "w_wire": np.round(diag["w_wire"], 4).tolist(),
        "gate_salience": np.round(diag["gate_salience"], 4).tolist(),
        "role_counts_roi": diag["role_counts_roi"],
        "filters": np.round(np.asarray(params.greens_filters[:, 0, :, :]), 4).tolist(),
        "filter_norms": diag["filter_norms"],
        "filter_com": diag["filter_com"],
        "frames": np.round(frames_np, 4).tolist(),
        "dir_energy": np.round(energy_np, 4).tolist(),
        "step_outputs": step_outputs,
    }

    html_content = EXPLORER_HTML_TEMPLATE.replace("__EXPLORER_DATA_JSON__", json.dumps(payload))
    out_path.write_text(html_content, encoding="utf-8")
    return out_path
