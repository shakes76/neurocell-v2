'''
Fused spatial standardization, factored Codd gating, and direction reduction epilogues.

Operates strictly on the active `VALID` convolution Region-of-Interest (ROI)
`(H_v, W_v)` for hidden steps and slices directly to the output port `(out_h, out_w)`
on the final step `t = depth`.

Copyright (c) 2026 Shekhar S. Chandra. All rights reserved.
Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

@author Shakes
'''
import jax
import jax.numpy as jnp

from neurocell.topology import ChipTopology2D, CoordinatePortSpec
from neurocell.params import PrecomputedStepParams


def standardize_spatial_nchw(z_bmhw: jax.Array) -> jax.Array:
    """
    Standardizes each `(b, m)` spatial plane `(H_v, W_v)` to zero mean and unit variance.
    Uses `jax.nn.standardize` along contiguous innermost axes `(-2, -1)`.
    """
    return jax.nn.standardize(z_bmhw, axis=(-2, -1))


def fused_roi_step_epilogue(
    z_roi_bmhw: jax.Array,
    step_params: PrecomputedStepParams,
    activate=jax.nn.tanh,
    norm: bool = True,
) -> jax.Array:
    """
    Pure-ROI Von Neumann neuron update and direction reduction returning `roi_next`
    of shape `(B, H_v, W_v)` without padding or copying into a full `(B, H, W)` buffer.

    Mathematical identity on ROI `(H_v, W_v)`:
        `roi_next = W_eff * sum_m(Z_m) + S_eff * sum_m(activate(Z_m + B_roi))`
    where `W_eff = (1/M) * W * (1 - sigmoid(S))` and `S_eff = (1/M) * sigmoid(S)`.
    """
    if norm:
        z_roi_bmhw = standardize_spatial_nchw(z_roi_bmhw)

    sum_lin = jnp.sum(z_roi_bmhw, axis=1)
    b_roi_bcast = step_params.b_roi[jnp.newaxis, jnp.newaxis, :, :]
    if activate is None:
        M = z_roi_bmhw.shape[1]
        sum_act = sum_lin + float(M) * step_params.b_roi[jnp.newaxis, :, :]
    else:
        sum_act = jnp.sum(activate(z_roi_bmhw + b_roi_bcast), axis=1)

    return (
        step_params.w_eff_roi[jnp.newaxis, :, :] * sum_lin
        + step_params.s_eff_roi[jnp.newaxis, :, :] * sum_act
    )


def fused_roi_epilogue(
    z_roi_bmhw: jax.Array,
    step_params: PrecomputedStepParams,
    c_border_bhw: jax.Array,
    topology: ChipTopology2D,
    activate=jax.nn.tanh,
    norm: bool = True,
) -> jax.Array:
    """
    Computes the Von Neumann neuron update on the active ROI `(H_v, W_v)` and embeds
    the result into a full-grid border template `(B, H, W)`.
    """
    roi_next = fused_roi_step_epilogue(
        z_roi_bmhw=z_roi_bmhw,
        step_params=step_params,
        activate=activate,
        norm=norm,
    )
    r_h, r_w = topology.pad_h, topology.pad_w
    H, W = topology.grid_shape
    return c_border_bhw.at[:, r_h : H - r_h, r_w : W - r_w].set(roi_next)


def fused_output_port_epilogue(
    z_roi_bmhw: jax.Array,
    step_params: PrecomputedStepParams,
    topology: ChipTopology2D,
    final_activate=None,
    norm: bool = True,
) -> jax.Array:
    """
    Final-step (`t = depth`) epilogue that standardizes over the active ROI `(H_v, W_v)`
    and immediately slices to the output port coordinates `(out_h, out_w)` BEFORE
    applying Codd gating and direction reduction.

    For the 62x62 ALU chip (`out_size = (8, 1)`), this reduces final-step Codd work
    from 3,844 cells to 8 cells (a 480x reduction).
    """
    if norm:
        z_roi_bmhw = standardize_spatial_nchw(z_roi_bmhw)

    out_h, out_w = topology.out_port.size
    if isinstance(topology.out_port, CoordinatePortSpec):
        rows_idx, cols_idx = topology.roi_out_offset
        z_gathered = z_roi_bmhw[:, :, rows_idx, cols_idx]
        z_out = jnp.reshape(
            z_gathered, (z_roi_bmhw.shape[0], z_roi_bmhw.shape[1], out_h, out_w)
        )
    else:
        ro_r, ro_c = topology.roi_out_offset
        # Slice strictly to the output port inside ROI: (B, M, out_h, out_w)
        z_out = z_roi_bmhw[:, :, ro_r : ro_r + out_h, ro_c : ro_c + out_w]

    sum_lin_out = jnp.sum(z_out, axis=1)
    if final_activate is None:
        M = z_out.shape[1]
        outputs = (
            (step_params.w_eff_out + step_params.s_eff_out)[jnp.newaxis, :, :] * sum_lin_out
            + (float(M) * step_params.s_eff_out * step_params.b_out)[jnp.newaxis, :, :]
        )
    else:
        b_out_bcast = step_params.b_out[jnp.newaxis, jnp.newaxis, :, :]
        sum_act_out = jnp.sum(final_activate(z_out + b_out_bcast), axis=1)
        outputs = (
            step_params.w_eff_out[jnp.newaxis, :, :] * sum_lin_out
            + step_params.s_eff_out[jnp.newaxis, :, :] * sum_act_out
        )
        outputs = final_activate(outputs)

    return outputs
