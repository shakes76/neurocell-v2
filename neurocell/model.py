'''
High-level stateless `VonNeumannNetwork` model interface.

Provides a clean functional API (`init`, `apply`, `effective_states`, `threshold_states`,
`save_checkpoint`, `load_checkpoint`) backed by `ChipTopology2D` and `VNNParams`.

Copyright (c) 2026 Shekhar S. Chandra. All rights reserved.
Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

@author Shakes
'''
from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp

from neurocell.topology import ChipTopology2D
from neurocell.params import (
    STATE_ACTIVE_VALUE,
    VNNParams,
    compute_codd_state_probs,
    init_vnn_params,
)
from neurocell import filters as nc_filters
from neurocell.propagator import propagate_vnn_2d
from neurocell.adaptive_mask import AdaptiveMaskSpec, compute_adaptive_spatial_mask


class VonNeumannNetwork:
    """
    Stateless, GPU-optimal Von Neumann Network (VNN) model.

    By default, `apply(params, inputs)` runs in stateless mode (`return_universe=False`),
    returning only the output port predictions `(B, out_h, out_w)` so XLA dead-code-eliminates
    all non-output cellular updates on the final propagation step.
    """

    def __init__(
        self,
        topology: ChipTopology2D,
        activate=jax.nn.tanh,
        final_activate=None,
        norm: bool = True,
        sharpen: bool = False,
        gated_filters: bool = False,
        hard_pin_io: bool = False,
        backend: str = "nchw_direct",
        dtype=jnp.float32,
        codd_mode: str = "continuous",
        state_temperature: float = 1.0,
        active_threshold: float = 0.6,
    ):
        self.topology = topology
        self.activate = activate
        self.final_activate = final_activate
        self.norm = norm
        self.sharpen = sharpen
        self.gated_filters = gated_filters
        self.hard_pin_io = hard_pin_io
        self.backend = backend
        self.dtype = dtype
        self.codd_mode = codd_mode
        self.state_temperature = state_temperature
        self.active_threshold = active_threshold
        self.sharpen_filter = (
            nc_filters.sharpen(dtype=jnp.float32, norm=True) if sharpen else None
        )

    def init(
        self,
        rng: jax.Array,
        clear_outside_io: bool = False,
        legacy_compat: bool = True,
    ) -> VNNParams:
        """Initializes packed `VNNParams` for this network's topology."""
        return init_vnn_params(
            rng=rng,
            topology=self.topology,
            dtype=self.dtype,
            clear_outside_io=clear_outside_io,
            legacy_compat=legacy_compat,
        )

    def compute_spatial_mask(
        self,
        params: VNNParams,
        margin: int | tuple[int, int] = 4,
        bucket_multiple: int = 8,
        prev_mask: AdaptiveMaskSpec | None = None,
        **kwargs,
    ) -> AdaptiveMaskSpec:
        """
        Computes the cheap pre-epoch active spatial mask (`AdaptiveMaskSpec`) for the
        next epoch from the current epoch's learned parameters and a safety `margin`.
        """
        return compute_adaptive_spatial_mask(
            params=params,
            topology=self.topology,
            margin=margin,
            bucket_multiple=bucket_multiple,
            prev_mask=prev_mask,
            **kwargs,
        )

    def apply(
        self,
        params: VNNParams,
        inputs: jax.Array | Sequence[jax.Array],
        return_universe: bool = False,
        spatial_mask: AdaptiveMaskSpec | None = None,
        codd_mode: str | None = None,
        state_temperature: float | jax.Array | None = None,
        active_threshold: float | None = None,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """
        Executes the forward pass of the Von Neumann Network.
        """
        return propagate_vnn_2d(
            params=params,
            inputs=inputs,
            topology=self.topology,
            activate=self.activate,
            final_activate=self.final_activate,
            norm=self.norm,
            sharpen_filter=self.sharpen_filter,
            gated_filters=self.gated_filters,
            hard_pin_io=self.hard_pin_io,
            backend=self.backend,
            return_universe=return_universe,
            spatial_mask=spatial_mask,
            codd_mode=self.codd_mode if codd_mode is None else codd_mode,
            state_temperature=(
                self.state_temperature if state_temperature is None else state_temperature
            ),
            active_threshold=(
                self.active_threshold if active_threshold is None else active_threshold
            ),
        )

    def __call__(
        self,
        params: VNNParams,
        inputs: jax.Array | Sequence[jax.Array],
        return_universe: bool = False,
        spatial_mask: AdaptiveMaskSpec | None = None,
        codd_mode: str | None = None,
        state_temperature: float | jax.Array | None = None,
        active_threshold: float | None = None,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Alias for `apply`."""
        return self.apply(
            params,
            inputs,
            return_universe=return_universe,
            spatial_mask=spatial_mask,
            codd_mode=codd_mode,
            state_temperature=state_temperature,
            active_threshold=active_threshold,
        )

    def effective_states(
        self,
        params: VNNParams,
        codd_mode: str | None = None,
        state_temperature: float | jax.Array | None = None,
        active_threshold: float | None = None,
    ) -> jax.Array:
        """
        Returns the effective Codd state probabilities `s in [0, 1]` on `(H, W)`
        after applying the I/O port pinning mask and Codd gating (`codd_mode`).
        """
        return compute_codd_state_probs(
            s_raw=params.states,
            io_state_mask=self.topology.io_state_mask,
            hard_pin_io=self.hard_pin_io,
            codd_mode=self.codd_mode if codd_mode is None else codd_mode,
            state_temperature=(
                self.state_temperature if state_temperature is None else state_temperature
            ),
            active_threshold=(
                self.active_threshold if active_threshold is None else active_threshold
            ),
        )

    @staticmethod
    def threshold_states(states_raw: jax.Array, active_threshold: float = 0.6) -> jax.Array:
        """
        Thresholds raw Codd states into discrete roles matching `contrib/` visualization:
        - `2` for Active decision neurons (`s >= active_threshold`)
        - `1` for Passive signal wires (`0 < s < active_threshold`)
        - `0` (or `<= 0`) for Inactive / isolated cells
        """
        s = jnp.where(states_raw >= active_threshold, 2.0, states_raw)
        s = jnp.where(jnp.logical_and(s > 0.0, s < active_threshold), 1.0, s)
        return s

    def save_checkpoint(self, filename: str, params: VNNParams, **extra_metadata):
        """
        Saves `VNNParams` to an `.npz` file using the legacy key names
        (`weights`, `biases`, `states`, `filters`) for 100% compatibility with
        existing analysis and visualization scripts.
        """
        jnp.savez(
            filename,
            weights=params.weights,
            biases=params.biases,
            states=params.states,
            filters=params.filters_ihwo,
            N=self.topology.N,
            depth=self.topology.depth,
            kernel_size=np.asarray(self.topology.kernel_size),
            directions=self.topology.directions,
            allow_pickle=False,
            **extra_metadata,
        )

    @staticmethod
    def load_checkpoint(filename: str) -> VNNParams:
        """Loads a saved `.npz` checkpoint into a packed `VNNParams` PyTree."""
        with np.load(filename, allow_pickle=False) as data:
            w = jnp.asarray(data["weights"])
            b = jnp.asarray(data["biases"])
            s = jnp.asarray(data["states"])
            f_ihwo = jnp.asarray(data["filters"])
        cell_params = jnp.stack([w, b, s], axis=0)
        greens_filters = jnp.transpose(f_ihwo, (3, 0, 1, 2))
        return VNNParams(cell_params=cell_params, greens_filters=greens_filters)
