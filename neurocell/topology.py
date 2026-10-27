'''
Cellular universe topologies, port specifications, and static ROI masks for VNNs.

Decouples compile-time spatial geometry, input/output port coordinates (rectangular
slices, full-grid fields, or scattered 2D coordinates), active Region-of-Interest (ROI)
bounds, and static pinning masks from learnable parameters.

Copyright (c) 2026 Shekhar S. Chandra. All rights reserved.
Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

@author Shakes
'''
from dataclasses import dataclass
from typing import Sequence
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PortSpec:
    """
    Defines a rectangular input, control, or output port on the cellular grid.
    """
    offset: tuple[int, ...]
    size: tuple[int, ...]
    name: str = ""

    def __post_init__(self):
        if len(self.offset) != len(self.size):
            raise ValueError(
                f"PortSpec offset {self.offset} and size {self.size} must have matching rank."
            )
        if any(s <= 0 for s in self.size):
            raise ValueError(f"PortSpec size dimensions must be positive, got {self.size}.")


@dataclass(frozen=True)
class CoordinatePortSpec:
    """
    Defines a scattered coordinate input or output port (e.g. for Circular Layered
    Perceptrons / `CoordinateCells` where cells are addressed by explicit `(row, col)` pairs).
    """
    coords: tuple[tuple[int, int], ...]
    name: str = ""

    def __init__(self, coords: Sequence[Sequence[int]] | jax.Array, name: str = ""):
        coords_tuple = tuple((int(c[0]), int(c[1])) for c in coords)
        if len(coords_tuple) == 0:
            raise ValueError("CoordinatePortSpec coords must be non-empty.")
        object.__setattr__(self, "coords", coords_tuple)
        object.__setattr__(self, "name", str(name))

    @property
    def num_points(self) -> int:
        """Number of coordinate points in this port."""
        return len(self.coords)

    @property
    def size(self) -> tuple[int, int]:
        """Logical shape `(K, 1)` of values injected/extracted at these coordinates."""
        return (len(self.coords), 1)

    @property
    def offset(self) -> tuple[int, int]:
        """Top-left bounding-box corner `(min_r, min_c)` of the coordinate set."""
        min_r = min(r for r, _ in self.coords)
        min_c = min(c for _, c in self.coords)
        return (min_r, min_c)

    @property
    def rows(self) -> jax.Array:
        """1D int32 array of row indices."""
        return jnp.array([r for r, _ in self.coords], dtype=jnp.int32)

    @property
    def cols(self) -> jax.Array:
        """1D int32 array of column indices."""
        return jnp.array([c for _, c in self.coords], dtype=jnp.int32)

    def fits_in_roi(self, grid_shape: tuple[int, int], r_h: int, r_w: int) -> bool:
        """Returns `True` if all coordinates lie strictly within `[r_h:H-r_h, r_w:W-r_w]`."""
        H, W = grid_shape
        return all(r_h <= r < (H - r_h) and r_w <= c < (W - r_w) for r, c in self.coords)


AnyPortSpec = PortSpec | CoordinatePortSpec


def _next_power_of_two(x: int) -> int:
    """Returns the smallest power of two >= x."""
    return 1 << (x - 1).bit_length()


def _port_within_bounds(
    port: AnyPortSpec,
    r_min: int,
    r_max: int,
    c_min: int,
    c_max: int,
) -> bool:
    """Checks whether a `PortSpec` or `CoordinatePortSpec` lies strictly within `[r_min:r_max, c_min:c_max]`."""
    if isinstance(port, CoordinatePortSpec):
        return all(r_min <= r < r_max and c_min <= c < c_max for r, c in port.coords)
    r0, c0 = port.offset
    rh, cw = port.size
    return r0 >= r_min and c0 >= c_min and (r0 + rh) <= r_max and (c0 + cw) <= c_max


@dataclass(frozen=True)
class ChipTopology2D:
    """
    Immutable compile-time specification of a 2D Von Neumann Network cellular topology.

    Supports arbitrary 2D VNN architectures:
    - Rectangular sub-grid I/O ports (`PortSpec`),
    - Full-grid cellular automata (`in_port.size == out_port.size == grid_shape`, e.g. Game of Life),
    - Scattered 2D coordinate ports (`CoordinatePortSpec`, e.g. Circular Layered Perceptrons),
    - `"VALID"`, `"SAME"`, and `"CIRCULAR"` spatial boundary modes.
    """
    grid_shape: tuple[int, int]
    kernel_size: tuple[int, int]
    directions: int
    depth: int
    in_ports: tuple[AnyPortSpec, ...]
    out_port: AnyPortSpec
    pad_h: int
    pad_w: int
    roi_shape: tuple[int, int]
    roi_out_offset: tuple[int, int]
    io_state_mask: jax.Array
    out_weight_mask: jax.Array
    fft_shape: tuple[int, int]
    padding_mode: str = "VALID"
    in_in_roi: bool = True
    out_in_roi: bool = True

    @property
    def H(self) -> int:
        """Grid height (rows)."""
        return self.grid_shape[0]

    @property
    def W(self) -> int:
        """Grid width (columns)."""
        return self.grid_shape[1]

    @property
    def N(self) -> int:
        """Maximum grid dimension (legacy compatibility)."""
        return max(self.grid_shape)

    @classmethod
    def create(
        cls,
        grid_shape: int | tuple[int, int],
        kernel_size: tuple[int, int],
        directions: int,
        depth: int,
        in_ports: Sequence[AnyPortSpec],
        out_port: AnyPortSpec,
        fft_shape: tuple[int, int] | None = None,
        padding_mode: str = "VALID",
        allow_border_output: bool = False,
    ) -> "ChipTopology2D":
        """
        Constructs a validated 2D cellular topology with precomputed ROI bounds and static masks.
        """
        if isinstance(grid_shape, int):
            grid_shape = (grid_shape, grid_shape)
        else:
            grid_shape = (int(grid_shape[0]), int(grid_shape[1]))

        kh, kw = int(kernel_size[0]), int(kernel_size[1])
        if kh % 2 == 0 or kw % 2 == 0:
            raise ValueError(f"Green's function kernel_size must be odd, got {kernel_size}.")

        padding_mode = padding_mode.upper()
        if padding_mode not in ("VALID", "SAME", "CIRCULAR"):
            raise ValueError(
                f"Unsupported padding_mode '{padding_mode}'; choose from ('VALID', 'SAME', 'CIRCULAR')."
            )

        H, W = grid_shape
        if padding_mode == "VALID":
            pad_h = (kh - 1) // 2
            pad_w = (kw - 1) // 2
        else:
            pad_h = 0
            pad_w = 0

        roi_h = H - 2 * pad_h
        roi_w = W - 2 * pad_w
        if roi_h <= 0 or roi_w <= 0:
            raise ValueError(
                f"Grid {grid_shape} is too small for kernel {kernel_size}; valid ROI is ({roi_h}, {roi_w})."
            )

        in_ports_tuple = tuple(in_ports)

        # Validate port containment within the universe grid [0:H, 0:W]
        for port in in_ports_tuple + (out_port,):
            if not _port_within_bounds(port, 0, H, 0, W):
                raise ValueError(
                    f"Port '{port.name}' at offset={port.offset}, size={port.size} "
                    f"exceeds grid_shape={grid_shape}."
                )

        # Check whether inputs and outputs lie strictly inside the active ROI [pad_h:H-pad_h, pad_w:W-pad_w]
        in_in_roi = all(
            _port_within_bounds(p, pad_h, H - pad_h, pad_w, W - pad_w)
            for p in in_ports_tuple
        )
        out_in_roi = _port_within_bounds(out_port, pad_h, H - pad_h, pad_w, W - pad_w)

        # Full-grid output port (e.g. Game of Life N x N -> N x N) is automatically allowed
        is_full_grid_out = (
            isinstance(out_port, PortSpec)
            and out_port.offset == (0, 0)
            and out_port.size == grid_shape
        )

        if padding_mode == "VALID" and not out_in_roi and not allow_border_output and not is_full_grid_out:
            raise ValueError(
                f"Output port {out_port} lies outside the active VALID convolution ROI "
                f"[{pad_h}:{H - pad_h}, {pad_w}:{W - pad_w}]."
            )

        out_r, out_c = out_port.offset
        if isinstance(out_port, CoordinatePortSpec):
            roi_out_offset = (
                tuple(int(r - pad_h) for r, _ in out_port.coords),
                tuple(int(c - pad_w) for _, c in out_port.coords),
            )
        else:
            roi_out_offset = (max(0, out_r - pad_h), max(0, out_c - pad_w))

        # Build static boolean masks for pinning I/O states and output weights
        io_state_mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
        for port in in_ports_tuple:
            if isinstance(port, CoordinatePortSpec):
                io_state_mask = io_state_mask.at[port.rows, port.cols].set(True)
            else:
                r0, c0 = port.offset
                rh, cw = port.size
                io_state_mask = io_state_mask.at[r0 : r0 + rh, c0 : c0 + cw].set(True)

        out_weight_mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
        if isinstance(out_port, CoordinatePortSpec):
            io_state_mask = io_state_mask.at[out_port.rows, out_port.cols].set(True)
            out_weight_mask = out_weight_mask.at[out_port.rows, out_port.cols].set(True)
        else:
            out_h, out_w = out_port.size
            io_state_mask = io_state_mask.at[out_r : out_r + out_h, out_c : out_c + out_w].set(True)
            out_weight_mask = out_weight_mask.at[out_r : out_r + out_h, out_c : out_c + out_w].set(True)

        if fft_shape is None:
            if padding_mode == "CIRCULAR":
                fft_shape = (H, W)
            else:
                # Linear convolution of active ROI (roi_h, roi_w) with (kh, kw) has support
                # (roi_h + kh - 1, roi_w + kw - 1).
                fft_shape = (
                    _next_power_of_two(roi_h + kh - 1),
                    _next_power_of_two(roi_w + kw - 1),
                )

        return cls(
            grid_shape=grid_shape,
            kernel_size=(kh, kw),
            directions=int(directions),
            depth=int(depth),
            in_ports=in_ports_tuple,
            out_port=out_port,
            pad_h=pad_h,
            pad_w=pad_w,
            roi_shape=(roi_h, roi_w),
            roi_out_offset=roi_out_offset,
            io_state_mask=io_state_mask,
            out_weight_mask=out_weight_mask,
            fft_shape=fft_shape,
            padding_mode=padding_mode,
            in_in_roi=in_in_roi,
            out_in_roi=out_in_roi,
        )

    @classmethod
    def alu_chip(
        cls,
        n: int = 8,
        kernel_size: tuple[int, int] = (23, 13),
        directions: int = 16,
        depth: int = 4,
        gap: int = 5,
        ctrl_gap: int = 5,
    ) -> "ChipTopology2D":
        """
        Constructs the canonical 2D ALU Chip topology from the VNN paper.
        """
        padding = 2 * max(kernel_size[0], kernel_size[1])
        N = 2 * n + padding
        center = N // 2
        input_top = center - (2 * n) // 2
        input_left = center - n
        output_mid = center - n // 2
        output_left = input_left + n + 2 * gap

        in_ports = (
            PortSpec(offset=(input_top, input_left), size=(n, 1), name="x1"),
            PortSpec(offset=(input_top + n + 1, input_left), size=(n, 1), name="x2"),
            PortSpec(offset=(input_top - ctrl_gap, input_left + gap), size=(1, n), name="ctrl"),
        )
        out_port = PortSpec(offset=(output_mid, output_left), size=(n, 1), name="out")

        return cls.create(
            grid_shape=(N, N),
            kernel_size=kernel_size,
            directions=directions,
            depth=depth,
            in_ports=in_ports,
            out_port=out_port,
        )

    @classmethod
    def mlp_chip(
        cls,
        n: int = 8,
        layers: int = 5,
        ctrl_gap: int = 1,
        kernel_size: tuple[int, int] = (17, 3),
        directions: int = 8,
    ) -> "ChipTopology2D":
        """
        Constructs the 2D Vertical MLP topology for ALU/XOR tasks.
        """
        padding = 2 * max(kernel_size[0], kernel_size[1])
        N = 2 * n + padding
        center = N // 2
        gap = 2 * layers + 1
        input_top = center - (3 * n + ctrl_gap) // 2
        input_left = center - gap // 2
        output_mid = center - n // 2
        output_left = input_left + gap
        depth = output_left - input_left

        in_ports = (
            PortSpec(offset=(input_top, input_left), size=(3 * n + ctrl_gap, 1), name="x_in"),
        )
        out_port = PortSpec(offset=(output_mid, output_left), size=(n, 1), name="out")

        return cls.create(
            grid_shape=(N, N),
            kernel_size=kernel_size,
            directions=directions,
            depth=depth,
            in_ports=in_ports,
            out_port=out_port,
        )

    @classmethod
    def gol_chip(
        cls,
        grid_shape: int | tuple[int, int] = (64, 64),
        kernel_size: tuple[int, int] = (3, 3),
        directions: int = 64,
        depth: int = 2,
        padding_mode: str = "VALID",
        n: int | None = None,
    ) -> "ChipTopology2D":
        """
        Constructs a full-grid Cellular Automaton / Game of Life VNN topology
        (`input_size = output_size = grid_shape`, matching `contrib/test_jax_vnn_full_gol.py`).
        """
        if n is not None:
            grid_shape = (int(n), int(n))
        elif isinstance(grid_shape, int):
            grid_shape = (grid_shape, grid_shape)
        in_ports = (PortSpec(offset=(0, 0), size=grid_shape, name="universe_in"),)
        out_port = PortSpec(offset=(0, 0), size=grid_shape, name="universe_out")
        return cls.create(
            grid_shape=grid_shape,
            kernel_size=kernel_size,
            directions=directions,
            depth=depth,
            in_ports=in_ports,
            out_port=out_port,
            padding_mode=padding_mode,
            allow_border_output=True,
        )

    @classmethod
    def clp_chip(
        cls,
        grid_shape: int | tuple[int, int] = (8, 8),
        in_coords: Sequence[Sequence[int]] | jax.Array | None = None,
        out_coords: Sequence[Sequence[int]] | jax.Array | None = None,
        kernel_size: tuple[int, int] = (3, 3),
        directions: int = 1,
        depth: int = 4,
        padding_mode: str = "VALID",
        n: int | None = None,
    ) -> "ChipTopology2D":
        """
        Constructs a Coordinate-addressed Circular Layered Perceptron (CLP) topology
        matching `contrib/test_jax_vnn_clp_classify_xor.py`.
        """
        if n is not None:
            grid_shape = (int(n), int(n))
        elif isinstance(grid_shape, int):
            grid_shape = (grid_shape, grid_shape)
        H, W = grid_shape
        r_h = (kernel_size[0] - 1) // 2 if padding_mode.upper() == "VALID" else 0
        r_w = (kernel_size[1] - 1) // 2 if padding_mode.upper() == "VALID" else 0
        if in_coords is None:
            in_coords = ((r_h + 1, r_w + 1), (H - r_h - 2, r_w + 1))
        if out_coords is None:
            out_coords = ((r_h + 1, W - r_w - 2), (H - r_h - 2, W - r_w - 2))
        in_port = CoordinatePortSpec(coords=in_coords, name="clp_in")
        out_port = CoordinatePortSpec(coords=out_coords, name="clp_out")
        return cls.create(
            grid_shape=grid_shape,
            kernel_size=kernel_size,
            directions=directions,
            depth=depth,
            in_ports=(in_port,),
            out_port=out_port,
            padding_mode=padding_mode,
            allow_border_output=True,
        )

    @classmethod
    def tm_chip(
        cls,
        n: int = 8,
        kernel_size: tuple[int, int] = (13, 13),
        directions: int = 16,
        gap: int = 1,
    ) -> "ChipTopology2D":
        """
        Constructs a 2D Turing Machine tape topology matching `contrib/test_jax_vnn_tm_arithmetic.py`.
        """
        length = 2 * n
        padding = 2 * max(kernel_size[0], kernel_size[1])
        N = length + 2 * padding
        center = N // 2
        input_left = padding
        output_left = input_left + length + gap
        depth = output_left - input_left

        in_ports = (PortSpec(offset=(center, input_left), size=(1, length), name="tape_in"),)
        out_port = PortSpec(offset=(center, output_left), size=(1, n), name="tape_out")
        return cls.create(
            grid_shape=(N, N),
            kernel_size=kernel_size,
            directions=directions,
            depth=depth,
            in_ports=in_ports,
            out_port=out_port,
        )
