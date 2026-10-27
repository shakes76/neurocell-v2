'''
CLI driver for visualizing and animating trained Von Neumann Network (VNN) models.

Loads any saved `.npz` VNN checkpoint (defaulting to `alu_arithmetic_cells.npz`) or
initializes a specified topology preset (`alu`, `gol`, `clp`, `mlp_chip`, `tm_chip`)
and generates:
1. Terminal structural & Codd circuit role summary (`{0: Off, 1: Wire, 2: Gate}`),
2. Six-panel high-DPI overview figure (`<prefix>_overview.png`) & Green's filter gallery
   (`<prefix>_greens_filters.png`),
3. Multi-panel animated GIF of cellular wave propagation `C^(0..depth)`
   (`<prefix>_propagation.gif`),
4. Self-contained Interactive HTML5 Model & Wave Explorer (`<prefix>_explorer.html`).

Copyright (c) 2026 Shekhar S. Chandra. All rights reserved.
Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

@author Shakes
'''
import argparse
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

from neurocell import (
    SUPPORTED_CODD_MODES,
    ChipTopology2D,
    animate_vnn_propagation,
    compute_vnn_diagnostics,
    export_interactive_explorer_html,
    init_vnn_params,
    load_checkpoint_with_topology,
    plot_vnn_parameter_suite,
    unroll_vnn_propagation_2d,
)


ALU_OPCODES = {
    "pass": (4, lambda a, b: np.uint8(a)),
    "inc": (1, lambda a, b: np.uint8(a + 1)),
    "add": (2, lambda a, b: np.uint8(a + b)),
    "adc": (3, lambda a, b: np.uint8(a + b + 1)),
    "dec": (5, lambda a, b: np.uint8(a - 1)),
    "sub": (6, lambda a, b: np.uint8(a - b)),
    "sbc": (7, lambda a, b: np.uint8(a - b - 1)),
}


def _build_alu_sample(x1_val: int, x2_val: int, op_name: str, n: int = 8):
    """Constructs a single `(1, 3, n, 1)` ALU input tensor and target bit vector."""
    ctrl_code, op_fn = ALU_OPCODES[op_name]
    y_val = int(op_fn(np.uint8(x1_val), np.uint8(x2_val)))
    x1_bits = np.unpackbits(np.array([x1_val], dtype=np.uint8), count=n).astype(np.float32)
    x2_bits = np.unpackbits(np.array([x2_val], dtype=np.uint8), count=n).astype(np.float32)
    c_bits = np.unpackbits(np.array([ctrl_code], dtype=np.uint8), count=n).astype(np.float32)
    target_bits = np.unpackbits(np.array([y_val], dtype=np.uint8), count=n).astype(int).tolist()

    sample_in = jnp.stack(
        [
            jnp.asarray(x1_bits).reshape(1, n, 1),
            jnp.asarray(x2_bits).reshape(1, n, 1),
            jnp.asarray(c_bits).reshape(1, n, 1),
        ],
        axis=1,
    )
    title = (
        f"8-Bit ALU Wave Propagation: {x1_val} ({op_name.upper()}, ctrl={ctrl_code}) {x2_val} = {y_val}"
    )
    return sample_in, target_bits, y_val, title


def _build_generic_sample(topology: ChipTopology2D, seed: int = 0):
    """Constructs a synthetic input batch `(1, ...)` for any `ChipTopology2D`."""
    rng = jax.random.PRNGKey(seed)
    if len(topology.in_ports) == 1 and topology.in_ports[0].size == topology.grid_shape:
        # Full-grid cellular automaton (e.g. Game of Life): inject a glider + blinker pattern
        H, W = topology.grid_shape
        board = np.zeros((1, 1, H, W), dtype=np.float32)
        cr, cc = H // 2, W // 2
        glider = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32)
        board[0, 0, cr - 1 : cr + 2, cc - 1 : cc + 2] = glider
        return jnp.asarray(board), None, f"Full-Grid VNN ({H}x{W}) Wave Propagation"

    port_tensors = []
    for idx, p in enumerate(topology.in_ports):
        subkey = jax.random.fold_in(rng, idx)
        ph, pw = p.size
        bits = jax.random.bernoulli(subkey, p=0.5, shape=(1, ph, pw)).astype(jnp.float32)
        port_tensors.append(bits)
    if len(port_tensors) == 1:
        return port_tensors[0], None, f"VNN ({topology.H}x{topology.W}) Wave Propagation"
    return port_tensors, None, f"VNN ({topology.H}x{topology.W}) Wave Propagation"


def main():
    parser = argparse.ArgumentParser(
        description="Explore, visualize, and animate trained Neurocell v2 VNN checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="alu_arithmetic_cells.npz",
        help="Path to trained `.npz` checkpoint file.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default=None,
        choices=("alu", "gol", "clp", "mlp_chip", "tm_chip"),
        help="Optional topology preset override (if exploring an un-checkpointed or custom preset).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/dashboard",
        help="Directory to save generated PNG plots, GIF animation, and HTML explorer.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="vnn_model",
        help="Filename prefix for saved artifacts.",
    )
    parser.add_argument(
        "--x1",
        type=int,
        default=42,
        help="First 8-bit integer operand (0..127) for ALU wave propagation.",
    )
    parser.add_argument(
        "--x2",
        type=int,
        default=21,
        help="Second 8-bit integer operand (0..127) for ALU wave propagation.",
    )
    parser.add_argument(
        "--op",
        type=str,
        default="add",
        choices=tuple(ALU_OPCODES.keys()),
        help="ALU operation to animate ('add', 'sub', 'inc', 'dec', 'adc', 'sbc', 'pass').",
    )
    parser.add_argument(
        "--codd-mode",
        type=str,
        default="continuous",
        choices=SUPPORTED_CODD_MODES,
        help="Codd state gating mode used during visualization.",
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=0.6,
        help="Threshold for classifying Codd states into discrete roles {0, 1, 2}.",
    )
    parser.add_argument(
        "--adaptive-margin",
        type=int,
        default=4,
        help="Safety margin (in cells) for visualizing the adaptive active sub-ROI.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.5,
        help="Frames per second for the saved `.gif` animation.",
    )
    args = parser.parse_args()

    preset_map = {
        "alu": ChipTopology2D.alu_chip,
        "gol": ChipTopology2D.gol_chip,
        "clp": ChipTopology2D.clp_chip,
        "mlp_chip": ChipTopology2D.mlp_chip,
        "tm_chip": ChipTopology2D.tm_chip,
    }
    explicit_topo = preset_map[args.topology]() if args.topology is not None else None

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists() and args.topology is None:
        print(f"Loading trained VNN checkpoint: {ckpt_path}")
        params, topology, meta = load_checkpoint_with_topology(ckpt_path, topology=explicit_topo)
    else:
        topology = explicit_topo or ChipTopology2D.alu_chip()
        print(
            f"Checkpoint '{ckpt_path}' not found or topology preset '{args.topology}' requested; "
            f"initializing {topology.grid_shape} topology."
        )
        params = init_vnn_params(jax.random.PRNGKey(42), topology, legacy_compat=True)
        meta = {}

    # Prepare sample input for wave propagation
    is_alu_topo = (
        len(topology.in_ports) == 3
        and topology.in_ports[0].size == (8, 1)
        and topology.out_port.size == (8, 1)
    )
    if is_alu_topo:
        sample_in, target_bits, y_val, anim_title = _build_alu_sample(
            args.x1, args.x2, args.op, n=8
        )
    else:
        sample_in, target_bits, anim_title = _build_generic_sample(topology)
        y_val = None

    # 1. Compute structural diagnostics & unrolled propagation
    diag = compute_vnn_diagnostics(
        params=params,
        topology=topology,
        codd_mode=args.codd_mode,
        active_threshold=args.active_threshold,
        adaptive_margin=args.adaptive_margin,
    )
    unrolled = unroll_vnn_propagation_2d(
        params=params,
        inputs=sample_in,
        topology=topology,
        codd_mode=args.codd_mode,
        active_threshold=args.active_threshold,
    )

    rc = diag["role_counts_roi"]
    mask_spec = diag["mask_spec"]
    print("\n=== Neurocell v2 Model & Codd Topology Summary ===")
    print(
        f"Grid Shape: {topology.grid_shape} | Kernel: {topology.kernel_size} | "
        f"Directions: {topology.directions} | Depth: {topology.depth}"
    )
    print(
        f"Valid ROI:  {topology.roi_shape} ({rc['total_roi_cells']} cells) | "
        f"Adaptive Sub-ROI (margin={args.adaptive_margin}): {mask_spec.active_shape} "
        f"({mask_spec.active_fraction * 100.0:.1f}% of ROI, bbox={mask_spec.roi_bbox})"
    )
    print(
        f"Discrete Codd Roles in ROI (threshold={args.active_threshold}):\n"
        f"  - Role 2 (Active Logic Gate):  {rc['gate_cells']:4d} cells ({rc['gate_pct']:.2f}%)\n"
        f"  - Role 1 (Passive Signal Wire): {rc['wire_cells']:4d} cells ({rc['wire_pct']:.2f}%)\n"
        f"  - Role 0 (Confluent / Off):     {rc['off_cells']:4d} cells ({rc['off_pct']:.2f}%)"
    )

    out_final = np.asarray(unrolled["outputs"][0]).flatten()
    pred_bits = (np.round(out_final).astype(int)).tolist()
    if is_alu_topo and target_bits is not None:
        pred_u8 = int(np.packbits(np.array(pred_bits, dtype=np.uint8))[0])
        print(
            f"\nAnimated Sample: {args.x1} ({args.op.upper()}) {args.x2}\n"
            f"  - Target Output:    {y_val:3d} (bits: {''.join(str(b) for b in target_bits)})\n"
            f"  - Predicted Output: {pred_u8:3d} (bits: {''.join(str(b) for b in pred_bits)}, "
            f"raw={np.round(out_final, 3).tolist()})"
        )

    # 2. Save static overview & Green's filter plots
    saved_plots = plot_vnn_parameter_suite(
        params=params,
        topology=topology,
        output_dir=args.output_dir,
        prefix=args.prefix,
        adaptive_margin=args.adaptive_margin,
        codd_mode=args.codd_mode,
        active_threshold=args.active_threshold,
    )

    # 3. Save animated GIF of wave propagation
    gif_path = Path(args.output_dir) / f"{args.prefix}_propagation.gif"
    animate_vnn_propagation(
        params=params,
        topology=topology,
        inputs=sample_in,
        output_gif=gif_path,
        target_bits=target_bits,
        fps=args.fps,
        title_prefix=anim_title,
        codd_mode=args.codd_mode,
        active_threshold=args.active_threshold,
    )

    # 4. Save standalone Interactive HTML Explorer
    html_path = Path(args.output_dir) / f"{args.prefix}_explorer.html"
    export_interactive_explorer_html(
        params=params,
        topology=topology,
        inputs=sample_in,
        output_html=html_path,
        adaptive_margin=args.adaptive_margin,
        codd_mode=args.codd_mode,
        active_threshold=args.active_threshold,
    )

    print("\n=== Saved Visualization Artifacts ===")
    print(f"  1. Master 6-Panel Overview PNG: {saved_plots['overview']}")
    print(f"  2. Green's Filter Gallery PNG:  {saved_plots['greens_filters']}")
    print(f"  3. Wave Propagation Animated GIF: {gif_path}")
    print(f"  4. Interactive HTML5 Explorer:  {html_path}")


if __name__ == "__main__":
    main()
