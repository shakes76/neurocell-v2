'''
Unified GPU parameter PyTrees, initializers, and pre-step algebraic fusion for VNNs.

Packs per-cell parameters (W, B, S_raw) into a single contiguous `(3, H, W)` tensor
and stores Green's function filters in native `OIHW` `(M, 1, Kh, Kw)` layout.

Copyright (c) 2026 Shekhar S. Chandra. All rights reserved.
Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

@author Shakes
'''
from typing import NamedTuple
import jax
import jax.numpy as jnp

from neurocell.topology import ChipTopology2D, CoordinatePortSpec

STATE_ACTIVE = 0.7
STATE_SIGNAL = 0.2
STATE_ACTIVE_VALUE = 1.0
STATE_SIGNAL_VALUE = 0.0

SUPPORTED_CODD_MODES = (
    "continuous",
    "annealed",
    "ste_binary",
    "ste_ternary",
)


def compute_codd_state_probs(
    s_raw: jax.Array,
    io_state_mask: jax.Array,
    hard_pin_io: bool = False,
    codd_mode: str = "continuous",
    state_temperature: float | jax.Array = 1.0,
    active_threshold: float = 0.6,
) -> jax.Array:
    """
    Computes effective Codd gating probabilities `s_prob in [0, 1]` on `(H, W)`
    from raw state parameters `s_raw` under one of four Codd regimes:
    - `"continuous"`: Standard unit-temperature sigmoid `sigmoid(s)`.
    - `"annealed"`: Temperature-scaled continuous sigmoid `sigmoid(s / tau)`.
    - `"ste_binary"`: Straight-Through Estimator (STE) hardening `s_prob` to `{0.0, 1.0}`
      in the forward pass (`s >= 0.0 -> 1.0`, else `0.0`) while propagating smooth
      `sigmoid(s / tau)` gradients in the backward pass.
    - `"ste_ternary"`: Straight-Through Estimator (STE) hardening `s_prob` into the three
      discrete Codd roles `{0.0, 0.5, 1.0}` corresponding to `threshold_states`:
      - Role 2 (`s >= active_threshold`): `1.0` (Active Nonlinear Logic Gate)
      - Role 1 (`0.0 < s < active_threshold`): `0.5` (Passive Signal Wire)
      - Role 0 (`s <= 0.0`): `0.0` (Inactive / Pure Linear Cell)
      while propagating smooth `sigmoid(s / tau)` gradients in the backward pass.
    """
    if codd_mode not in SUPPORTED_CODD_MODES:
        raise ValueError(
            f"Unsupported codd_mode '{codd_mode}'; choose from {SUPPORTED_CODD_MODES}."
        )
    s_base = s_raw if hard_pin_io else jnp.where(io_state_mask, STATE_ACTIVE_VALUE, s_raw)

    if codd_mode == "continuous":
        s_prob = jax.nn.sigmoid(s_base)
    else:
        inv_tau = 1.0 / jnp.maximum(jnp.asarray(state_temperature, dtype=s_base.dtype), 1e-4)
        s_soft = jax.nn.sigmoid(s_base * inv_tau)
        if codd_mode == "annealed":
            s_prob = s_soft
        elif codd_mode == "ste_binary":
            s_hard = jnp.where(s_base >= 0.0, 1.0, 0.0).astype(s_base.dtype)
            s_prob = s_soft + jax.lax.stop_gradient(s_hard - s_soft)
        else:  # codd_mode == "ste_ternary"
            s_hard = jnp.where(
                s_base >= active_threshold,
                1.0,
                jnp.where(s_base > 0.0, 0.5, 0.0),
            ).astype(s_base.dtype)
            s_prob = s_soft + jax.lax.stop_gradient(s_hard - s_soft)

    if hard_pin_io:
        s_prob = jnp.where(io_state_mask, STATE_ACTIVE_VALUE, s_prob)
    return s_prob


class VNNParams(NamedTuple):
    """
    Learnable parameters of a Von Neumann Network as a flat JAX PyTree.

    Attributes:
        cell_params: Array of shape `(3, H, W)` packing `[weights, biases, states_raw]`
            contiguously in device memory.
        greens_filters: Array of shape `(M, 1, Kh, Kw)` in native `OIHW` layout.
    """
    cell_params: jax.Array
    greens_filters: jax.Array

    @property
    def weights(self) -> jax.Array:
        """Returns the 2D weights field W of shape (H, W)."""
        return self.cell_params[0]

    @property
    def biases(self) -> jax.Array:
        """Returns the 2D biases field B of shape (H, W)."""
        return self.cell_params[1]

    @property
    def states(self) -> jax.Array:
        """Returns the 2D raw Codd states field S of shape (H, W)."""
        return self.cell_params[2]

    @property
    def filters_ihwo(self) -> jax.Array:
        """Returns the Green's filters in legacy `IHWO` `(1, Kh, Kw, M)` layout."""
        return jnp.transpose(self.greens_filters, (1, 2, 3, 0))


class PrecomputedStepParams(NamedTuple):
    """
    Pre-step fused cell parameters computed ONCE per forward pass outside `jax.lax.scan`.

    By absorbing I/O pinning, `sigmoid(S)`, ROI slicing, and the `1 / M` direction
    averaging factor before the temporal depth loop, this eliminates:
    - 5 `dynamic_update_slice` calls per forward pass,
    - `depth` redundant `jax.nn.sigmoid` evaluations on `(H, W)`, and
    - `M`-way redundant gating multiplications inside the direction dimension.
    """
    w_eff_roi: jax.Array    # (H_v, W_v): (1/M) * W_pinned * (1 - sigmoid(S_pinned))
    s_eff_roi: jax.Array    # (H_v, W_v): (1/M) * sigmoid(S_pinned)
    b_roi: jax.Array        # (H_v, W_v): B inside active ROI
    border_2d: jax.Array    # (H, W): static border template act(B) * sigmoid(S_pinned)
    w_eff_out: jax.Array    # (out_h, out_w): sliced/gathered for output-only final step t = depth
    s_eff_out: jax.Array    # (out_h, out_w): sliced/gathered for output-only final step t = depth
    b_out: jax.Array        # (out_h, out_w): sliced/gathered for output-only final step t = depth
    w_pinned: jax.Array     # (H, W): full pinned weights (used when return_universe=True)
    b_full: jax.Array       # (H, W): full biases
    s_prob: jax.Array       # (H, W): full post-sigmoid pinned Codd states in [0, 1]


def _next_haiku_key(rng: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Replicates `hk.PRNGSequence.next()` key splitting (`jax.random.split(rng)`).
    """
    return tuple(jax.random.split(rng))


def init_vnn_params(
    rng: jax.Array,
    topology: ChipTopology2D,
    dtype=jnp.float32,
    clear_outside_io: bool = False,
    legacy_compat: bool = True,
) -> VNNParams:
    """
    Initializes `VNNParams` for a given `ChipTopology2D`.

    When `legacy_compat=True` (default), uses the exact same `hk.PRNGSequence` key
    splitting order and `variance_scaling` initializers as `contrib/universe.py`
    (`UniverseLP.initialize`), producing bit-for-bit identical initial weights,
    biases, states, and Green's filters.
    """
    H, W = topology.grid_shape
    kh, kw = topology.kernel_size
    M = topology.directions

    init_scale = 1.0 / jnp.sqrt(topology.depth * topology.N)
    init_filter_scale = 1.0 / jnp.sqrt(kh * kw * M)

    cell_initializer = jax.nn.initializers.variance_scaling(
        init_scale, mode="fan_in", distribution="truncated_normal"
    )
    filter_initializer = jax.nn.initializers.variance_scaling(
        init_filter_scale, mode="fan_in", distribution="truncated_normal"
    )

    if legacy_compat:
        # Match Haiku's PRNGSequence(rng) key consumption order in UniverseLP.initialize:
        # 1st key -> states 's'
        # 2nd key -> weights 'w'
        # ('b' uses jnp.zeros and consumes no RNG key)
        # 3rd key -> filters 'f' in shape (1, Kh, Kw, M)
        rng, key_s = _next_haiku_key(rng)
        rng, key_w = _next_haiku_key(rng)
        rng, key_f = _next_haiku_key(rng)
    else:
        key_s, key_w, key_f = jax.random.split(rng, 3)

    # 1. Initialize Codd states S
    states = cell_initializer(key_s, (H, W), jnp.float32)
    states = jax.nn.sigmoid(states)

    # 2. Initialize weights W
    weights = cell_initializer(key_w, (H, W), jnp.float32).astype(dtype)

    if clear_outside_io:
        for port in topology.in_ports:
            c0 = port.offset[1]
            states = states.at[:, : c0 + 1].set(0.0)
            weights = weights.at[:, : c0 + 1].set(0.0)
        out_c = topology.out_port.offset[1]
        states = states.at[:, out_c:].set(0.0)
        weights = weights.at[:, out_c:].set(0.0)

    # Pin I/O ports to 1.0 in states and weights (matching VectorCells.__call__)
    states = jnp.where(topology.io_state_mask, STATE_ACTIVE_VALUE, states)
    weights = jnp.where(topology.io_state_mask, 1.0, weights)

    # 3. Initialize biases B to zeros
    biases = jnp.zeros((H, W), dtype=dtype)

    # Pack [W, B, S] into contiguous (3, H, W) tensor
    cell_params = jnp.stack([weights, biases, states.astype(dtype)], axis=0)

    # 4. Initialize Green's filters in (1, Kh, Kw, M) so fan_in matches FiltersCells,
    # then transpose to native OIHW (M, 1, Kh, Kw) layout
    filters_ihwo = filter_initializer(key_f, (1, kh, kw, M), jnp.float32).astype(dtype)
    greens_filters = jnp.transpose(filters_ihwo, (3, 0, 1, 2))

    return VNNParams(cell_params=cell_params, greens_filters=greens_filters)


def precompute_step_params(
    params: VNNParams,
    topology: ChipTopology2D,
    activate=jax.nn.tanh,
    sharpen_weights_fn=None,
    hard_pin_io: bool = False,
    roi_bbox: tuple[int, int, int, int] | None = None,
    codd_mode: str = "continuous",
    state_temperature: float | jax.Array = 1.0,
    active_threshold: float = 0.6,
) -> PrecomputedStepParams:
    """
    Computes fused ROI and output-port parameters ONCE per batch before `jax.lax.scan`.

    Supports rectangular sub-ROI `PortSpec`, full-grid `PortSpec`, and scattered
    `CoordinatePortSpec` outputs, as well as continuous, temperature-annealed, and
    Straight-Through Estimator (STE) discrete Codd state hardening (`codd_mode`).
    """
    H, W = topology.grid_shape
    r_h, r_w = topology.pad_h, topology.pad_w
    inv_M = 1.0 / float(topology.directions)

    w_raw = params.cell_params[0]
    b_full = params.cell_params[1]
    s_raw = params.cell_params[2]

    if sharpen_weights_fn is not None:
        w_raw = sharpen_weights_fn(w_raw)

    w_pinned = jnp.where(topology.out_weight_mask, 1.0, w_raw)
    s_prob = compute_codd_state_probs(
        s_raw=s_raw,
        io_state_mask=topology.io_state_mask,
        hard_pin_io=hard_pin_io,
        codd_mode=codd_mode,
        state_temperature=state_temperature,
        active_threshold=active_threshold,
    )

    # Slice active ROI [r_h : H - r_h, r_w : W - r_w]
    w_roi = w_pinned[r_h : H - r_h, r_w : W - r_w]
    b_roi = b_full[r_h : H - r_h, r_w : W - r_w]
    s_roi = s_prob[r_h : H - r_h, r_w : W - r_w]

    w_eff_roi = inv_M * w_roi * (1.0 - s_roi)
    s_eff_roi = inv_M * s_roi

    if roi_bbox is not None:
        r0, r1, c0, c1 = roi_bbox
        w_eff_roi_sub = w_eff_roi[r0:r1, c0:c1]
        s_eff_roi_sub = s_eff_roi[r0:r1, c0:c1]
        b_roi_sub = b_roi[r0:r1, c0:c1]
    else:
        w_eff_roi_sub = w_eff_roi
        s_eff_roi_sub = s_eff_roi
        b_roi_sub = b_roi

    # Precompute static 2D border values where VALID conv output is identically zero
    border_act = b_full if activate is None else activate(b_full)
    border_2d = border_act * s_prob

    # Slice or gather output port coordinates for final step t = depth
    out_port = topology.out_port
    if isinstance(out_port, CoordinatePortSpec):
        if topology.out_in_roi:
            rows_roi = out_port.rows - r_h
            cols_roi = out_port.cols - r_w
            w_eff_out = w_eff_roi[rows_roi, cols_roi][:, jnp.newaxis]
            s_eff_out = s_eff_roi[rows_roi, cols_roi][:, jnp.newaxis]
            b_out = b_roi[rows_roi, cols_roi][:, jnp.newaxis]
        else:
            w_eff_full = inv_M * w_pinned * (1.0 - s_prob)
            s_eff_full = inv_M * s_prob
            w_eff_out = w_eff_full[out_port.rows, out_port.cols][:, jnp.newaxis]
            s_eff_out = s_eff_full[out_port.rows, out_port.cols][:, jnp.newaxis]
            b_out = b_full[out_port.rows, out_port.cols][:, jnp.newaxis]
    elif topology.out_in_roi:
        ro_r, ro_c = topology.roi_out_offset
        out_h, out_w = out_port.size
        w_eff_out = w_eff_roi[ro_r : ro_r + out_h, ro_c : ro_c + out_w]
        s_eff_out = s_eff_roi[ro_r : ro_r + out_h, ro_c : ro_c + out_w]
        b_out = b_roi[ro_r : ro_r + out_h, ro_c : ro_c + out_w]
    else:
        out_r, out_c = out_port.offset
        out_h, out_w = out_port.size
        w_eff_full = inv_M * w_pinned * (1.0 - s_prob)
        s_eff_full = inv_M * s_prob
        w_eff_out = w_eff_full[out_r : out_r + out_h, out_c : out_c + out_w]
        s_eff_out = s_eff_full[out_r : out_r + out_h, out_c : out_c + out_w]
        b_out = b_full[out_r : out_r + out_h, out_c : out_c + out_w]

    return PrecomputedStepParams(
        w_eff_roi=w_eff_roi_sub,
        s_eff_roi=s_eff_roi_sub,
        b_roi=b_roi_sub,
        border_2d=border_2d,
        w_eff_out=w_eff_out,
        s_eff_out=s_eff_out,
        b_out=b_out,
        w_pinned=w_pinned,
        b_full=b_full,
        s_prob=s_prob,
    )


def export_legacy_haiku_params(params: VNNParams, module_name: str = "universe_lp") -> dict:
    """
    Exports `VNNParams` into a legacy Haiku parameter dictionary `{'universe_lp': {'w', 'b', 's', 'f'}}`.
    """
    return {
        module_name: {
            "w": params.weights,
            "b": params.biases,
            "s": params.states,
            "f": params.filters_ihwo,
        }
    }


def import_legacy_haiku_params(hk_params: dict, module_name: str = "universe_lp") -> VNNParams:
    """
    Imports a legacy Haiku parameter dictionary into a packed `VNNParams` PyTree.
    """
    sub = hk_params[module_name]
    cell_params = jnp.stack([sub["w"], sub["b"], sub["s"]], axis=0)
    greens_filters = jnp.transpose(sub["f"], (3, 0, 1, 2))
    return VNNParams(cell_params=cell_params, greens_filters=greens_filters)
