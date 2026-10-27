'''
Green's function convolution operators for Von Neumann Networks (VNNs).

Provides three GPU-optimized Green's function propagation backends:
1. `nchw_direct` (`conv2d_batched_nchw`): Native batched `NCHW` direct spatial convolution.
2. `rfft_hoisted` (`hoist_filter_rfft2` + `conv2d_roi_rfft_step`): Hoisted 64x64 Overlap-Save
   2D RFFT convolution with Plancherel frequency-domain backward reduction (`custom_vjp`),
   achieving O(1) kernel-size complexity and up to 8.85x speedup on large receptive fields.
3. `nchw_rfft_bwd` (`hoist_filter_rfft2_zero_fwd` + `conv2d_hybrid_roi_step`): Hybrid backend
   combining direct NCHW forward convolution, explicit forward-conv input gradients (`dX`),
   and hoisted 64x64 RFFT Plancherel filter gradients (`dG`) to eliminate cuDNN split-K atomic
   contention on medium receptive fields like `(23, 13)`.

Copyright (c) 2026 Shekhar S. Chandra. All rights reserved.
Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

@author Shakes
'''
from functools import partial
import math
import jax
import jax.numpy as jnp


def rfft2_pruned(
    x: jax.Array,
    fft_shape: tuple[int, int] = (64, 64),
) -> jax.Array:
    """
    Computes the 2D Real FFT on `fft_shape = (Fh, Fw)` while pruning dead zero-padded rows.

    When `x.shape[-2] < Fh`, standard `jnp.fft.rfft2(x, s=(Fh, Fw))` zero-pads both axes
    to `(Fh, Fw)` first and runs horizontal 1D RFFTs on `Fh - H_in` rows of zeros.
    Decomposing into a 1D `rfft` along `axis=-1` on the `H_in` active rows followed by
    a 1D `fft` along `axis=-2` on the `(Fw // 2 + 1)` frequency columns is algebraically
    identical for any `(H_in, W_in)` and `(Fh, Fw)`, eliminates the `(..., Fh, Fw)` real
    padding buffer, and reduces horizontal 1D RFFTs by `1 - H_in / Fh`.
    """
    fh, fw = fft_shape
    in_h = x.shape[-2]
    if in_h < fh:
        x_rfft_w = jnp.fft.rfft(x, n=fw, axis=-1)
        return jnp.fft.fft(x_rfft_w, n=fh, axis=-2)
    return jnp.fft.rfft2(x, s=fft_shape, axes=(-2, -1))


def irfft2_pruned(
    x_freq: jax.Array,
    roi_shape: tuple[int, int],
    pad_hw: tuple[int, int],
    fft_shape: tuple[int, int] = (64, 64),
) -> jax.Array:
    """
    Computes the 2D Inverse Real FFT on `fft_shape = (Fh, Fw)` and slices to
    `[..., r_h : r_h + roi_h, r_w : r_w + roi_w]` while pruning dead border rows
    before the horizontal 1D `irfft`.
    """
    r_h, r_w = pad_hw
    roi_h, roi_w = roi_shape
    fh, fw = fft_shape
    if roi_h < fh:
        x_ifft_h = jnp.fft.ifft(x_freq, n=fh, axis=-2)[..., r_h : r_h + roi_h, :]
        x_spatial_w = jnp.fft.irfft(x_ifft_h, n=fw, axis=-1)
        return x_spatial_w[..., :, r_w : r_w + roi_w]
    conv_full = jnp.fft.irfft2(x_freq, s=fft_shape, axes=(-2, -1))
    return conv_full[..., r_h : r_h + roi_h, r_w : r_w + roi_w]


def _select_batch_groups(batch_size: int, max_groups: int = 8) -> int:
    """
    Selects the largest divisor of `batch_size` up to `max_groups` while keeping at least
    16 channels per group (`b_per_group >= 16`) when `batch_size >= 16` for 64-byte SIMD /
    Tensor-Core vectorization on SM 8.9.
    """
    effective_max = min(batch_size, max_groups)
    if batch_size >= 16:
        effective_max = min(effective_max, max(1, batch_size // 16))
    for g in range(effective_max, 0, -1):
        if batch_size % g == 0:
            return g
    return 1


def conv2d_grouped_filter_grad(
    roi_cells_bhw: jax.Array,
    g_z_roi: jax.Array,
    pad_hw: tuple[int, int],
    max_groups: int = 8,
) -> jax.Array:
    """
    Computes the exact spatial filter gradient `dG` of shape `(M, 1, Kh, Kw)` for
    `conv2d(roi_cells_bhw[:, None], filters_oihw, padding=((r_h, r_h), (r_w, r_w)))`
    by partitioning the batch dimension `B` into `G_b` independent groups via
    `feature_group_count = G_b` followed by a fast register/L1 sum over `G_b`.

    For small kernels (`Kh * Kw <= 64`, e.g. 3x3 GoL/CLP or 17x3 MLP), this avoids
    both cuDNN split-K atomic accumulation contention (`G_b = 1`) and 64x64 2D FFT
    padding overhead.
    """
    bsz, h_in, w_in = roi_cells_bhw.shape
    _, m_dirs, h_out, w_out = g_z_roi.shape
    r_h, r_w = pad_hw
    g_b = _select_batch_groups(bsz, max_groups=max_groups)
    b_per_group = bsz // g_b

    lhs = jnp.reshape(roi_cells_bhw, (1, bsz, h_in, w_in))
    rhs = jnp.reshape(
        jnp.transpose(
            jnp.reshape(g_z_roi, (g_b, b_per_group, m_dirs, h_out, w_out)),
            (0, 2, 1, 3, 4),
        ),
        (g_b * m_dirs, b_per_group, h_out, w_out),
    )
    dg_grouped = jax.lax.conv_general_dilated(
        lhs=lhs,
        rhs=rhs,
        window_strides=(1, 1),
        padding=((r_h, r_h), (r_w, r_w)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=g_b,
    )
    dg_gm = jnp.reshape(dg_grouped, (g_b, m_dirs, 1, dg_grouped.shape[-2], dg_grouped.shape[-1]))
    return jnp.sum(dg_gm, axis=0)


def install_lean_rfft_transpose() -> bool:
    """
    Installs the lean 1-IRFFT `_rfft_transpose` rule from `contrib/upstream/jax-rfft-transpose.patch`
    into `jax._src.lax.fft` so any raw `jnp.fft.rfft2` autodiff calls (e.g. `contrib/convs.py`)
    use a single `FftType.IRFFT` instead of two complex `FftType.FFT` calls.
    """
    try:
        from jax._src.lax import fft as lax_fft
        from jax._src.lax import lax
        from jax._src.lax import slicing
    except ImportError:
        return False

    @partial(jax.jit, static_argnums=1)
    def _lean_rfft_transpose(t, fft_lengths):
        n = fft_lengths[-1]
        m = n // 2 + 1
        is_odd = n % 2
        full = partial(lax.full_like, t, dtype=t.dtype)
        reflected_bins = lax.concatenate(
            [
                full(1.0, shape=(1,)),
                full(0.0, shape=(m - 2 + is_odd,)),
                full(1.0, shape=(1 - is_odd,)),
            ],
            dimension=0,
        )
        reflected_bins = lax.expand_dims(reflected_bins, range(t.ndim - 1))

        z = lax.conj(t)
        reflected = z
        for axis in range(t.ndim - len(fft_lengths), t.ndim - 1):
            d = t.shape[axis]
            rev = lax.rev(reflected, dimensions=(axis,))
            reflected = lax.concatenate(
                [
                    slicing.slice_in_dim(rev, d - 1, d, axis=axis),
                    slicing.slice_in_dim(rev, 0, d - 1, axis=axis),
                ],
                dimension=axis,
            )

        half = lax.full_like(t, 0.5)
        h = half * (z + reflected_bins * lax.conj(reflected))
        out = lax_fft.fft(h, lax_fft.FftType.IRFFT, fft_lengths)
        return out * lax.full_like(out, math.prod(fft_lengths))

    lax_fft._rfft_transpose = _lean_rfft_transpose
    return True


def conv2d_batched_nchw(
    cells_nchw: jax.Array,
    filters_oihw: jax.Array,
    padding: str = "VALID",
    window_strides: tuple[int, int] = (1, 1),
    lhs_dilation: tuple[int, int] | None = None,
    rhs_dilation: tuple[int, int] | None = None,
    feature_group_count: int = 1,
    precision=None,
) -> jax.Array:
    """
    Executes native batched 2D Green's function convolution in `NCHW` layout.

    Because the spatial dimensions `(H_out, W_out)` are the innermost axes in `NCHW`,
    each `(batch, direction)` feature map is contiguous in HBM/L2 cache, enabling
    coalesced warp reductions during downstream spatial standardization.

    Args:
        cells_nchw: Batched cellular field of shape `(B, C_in, H, W)`.
        filters_oihw: Green's function kernels of shape `(M, C_in, Kh, Kw)`.
        padding: Spatial padding mode (`"VALID"` or `"SAME"`).
        window_strides: Convolution window strides, defaults to `(1, 1)`.
        lhs_dilation: Optional input dilation.
        rhs_dilation: Optional kernel dilation.
        feature_group_count: Optional grouped convolution count.
        precision: Optional `jax.lax.Precision` override.

    Returns:
        Propagated multi-direction cellular field of shape `(B, M, H_out, W_out)`.
    """
    return jax.lax.conv_general_dilated(
        lhs=cells_nchw,
        rhs=filters_oihw,
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=feature_group_count,
        precision=precision,
    )


def apply_spatial_filter_2d(
    field_hw: jax.Array,
    kernel_hw: jax.Array,
    padding: str = "SAME",
) -> jax.Array:
    """
    Applies a fixed 2D spatial stencil (e.g. `sharpen` or `smooth`) to a 2D array `(H, W)`.
    """
    lhs = field_hw[jnp.newaxis, jnp.newaxis, :, :]
    rhs = kernel_hw[jnp.newaxis, jnp.newaxis, :, :]
    out = jax.lax.conv_general_dilated(
        lhs=lhs,
        rhs=rhs,
        window_strides=(1, 1),
        padding=padding,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return out[0, 0]


def precompute_filter_rfft2(
    filters_oihw: jax.Array,
    fft_shape: tuple[int, int] = (64, 64),
) -> jax.Array:
    """
    Precomputes the 2D Real FFT of spatially flipped Green's function filters ONCE per batch.

    In `jax.lax.conv_general_dilated`, 2D convolution is defined as cross-correlation.
    To match `conv_general_dilated` via the convolution theorem, the spatial filter axes
    `(-2, -1)` are flipped prior to zero-padded `rfft2`.

    Args:
        filters_oihw: Green's function filters of shape `(M, 1, Kh, Kw)`.
        fft_shape: 2D power-of-two FFT grid `(Fh, Fw)`, defaults to `(64, 64)`.

    Returns:
        Complex frequency-domain filters of shape `(M, Fh, Fw // 2 + 1)`.
    """
    filters_flipped = jnp.flip(filters_oihw[:, 0, :, :], axis=(-2, -1))
    return jnp.fft.rfft2(filters_flipped, s=fft_shape, axes=(-2, -1))


def rfft2_filter_grad_to_oihw(
    grad_ker_freq_mhw: jax.Array,
    kernel_size: tuple[int, int] = (23, 13),
    fft_shape: tuple[int, int] = (64, 64),
) -> jax.Array:
    """
    Converts an accumulated frequency-domain filter cotangent `(M, Fh, Fw // 2 + 1)`
    back to spatial `OIHW` `(M, 1, Kh, Kw)` using only `M` 2D IRFFTs.
    """
    kh, kw = kernel_size
    grad_flipped_full = jnp.fft.irfft2(grad_ker_freq_mhw, s=fft_shape, axes=(-2, -1))
    grad_flipped = grad_flipped_full[:, :kh, :kw]
    grad_spatial = jnp.flip(grad_flipped, axis=(-2, -1))
    return grad_spatial[:, jnp.newaxis, :, :]


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def hoist_filter_rfft2(
    filters_oihw: jax.Array,
    kernel_size: tuple[int, int] = (23, 13),
    fft_shape: tuple[int, int] = (64, 64),
) -> jax.Array:
    """
    Differentiable wrapper around `precompute_filter_rfft2` whose custom VJP receives
    the accumulated Plancherel frequency-domain cotangent `sum_t sum_b g_freq * conj(cell_freq)`
    from `conv2d_roi_rfft_step` and executes a single `M`-channel `irfft2` per batch.
    """
    return precompute_filter_rfft2(filters_oihw, fft_shape=fft_shape)


def _hoist_filter_rfft2_fwd(
    filters_oihw: jax.Array,
    kernel_size: tuple[int, int],
    fft_shape: tuple[int, int],
):
    return precompute_filter_rfft2(filters_oihw, fft_shape=fft_shape), None


def _hoist_filter_rfft2_bwd(
    kernel_size: tuple[int, int],
    fft_shape: tuple[int, int],
    _,
    g_ker_freq_mhw: jax.Array,
):
    return (
        rfft2_filter_grad_to_oihw(
            g_ker_freq_mhw, kernel_size=kernel_size, fft_shape=fft_shape
        ),
    )


hoist_filter_rfft2.defvjp(_hoist_filter_rfft2_fwd, _hoist_filter_rfft2_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def conv2d_roi_rfft_step(
    roi_cells_bhw: jax.Array,
    ker_freq_mhw: jax.Array,
    roi_shape: tuple[int, int] = (40, 50),
    pad_hw: tuple[int, int] = (11, 6),
    fft_shape: tuple[int, int] = (64, 64),
    need_dx: bool = True,
) -> jax.Array:
    """
    Executes a single 2D `SAME` ROI Green's function convolution step via 64x64 Overlap-Save RFFT.

    Because `H_v + Kh - 1 = 40 + 23 - 1 = 62 <= 64` and `W_v + Kw - 1 = 50 + 13 - 1 = 62 <= 64`,
    zero-padded `rfft2` on `roi_cells_bhw` `(B, 40, 50)` followed by multiplication with
    `ker_freq_mhw` `(M, 64, 33)` and `irfft2` yields the exact linear `SAME` convolution on
    slice `[:, :, r_h : r_h + roi_h, r_w : r_w + roi_w]` with zero circular wrap-around.

    In the backward pass (`custom_vjp`), Plancherel's theorem is used to:
    1. Reduce `g_freq * conj(ker_freq_mhw)` over `M = 16` directions in the frequency domain
       before calling `irfft2` (requiring only `B = 128` inverse transforms instead of `2,048`).
    2. Reduce `g_freq * conj(cell_freq)` over `B = 128` samples in the frequency domain and pass
       the `(M, 64, 33)` cotangent directly to `hoist_filter_rfft2`.
    """
    r_h, r_w = pad_hw
    roi_h, roi_w = roi_shape
    cell_freq = rfft2_pruned(roi_cells_bhw, fft_shape=fft_shape)
    out_freq = cell_freq[:, jnp.newaxis, :, :] * ker_freq_mhw[jnp.newaxis, :, :, :]
    return irfft2_pruned(out_freq, roi_shape=(roi_h, roi_w), pad_hw=(r_h, r_w), fft_shape=fft_shape)


def _conv2d_roi_rfft_step_fwd(
    roi_cells_bhw: jax.Array,
    ker_freq_mhw: jax.Array,
    roi_shape: tuple[int, int],
    pad_hw: tuple[int, int],
    fft_shape: tuple[int, int],
    need_dx: bool,
):
    r_h, r_w = pad_hw
    roi_h, roi_w = roi_shape
    in_h, in_w = roi_cells_bhw.shape[-2], roi_cells_bhw.shape[-1]
    cell_freq = rfft2_pruned(roi_cells_bhw, fft_shape=fft_shape)
    out_freq = cell_freq[:, jnp.newaxis, :, :] * ker_freq_mhw[jnp.newaxis, :, :, :]
    z_roi = irfft2_pruned(out_freq, roi_shape=(roi_h, roi_w), pad_hw=(r_h, r_w), fft_shape=fft_shape)
    return z_roi, (cell_freq, ker_freq_mhw, in_h, in_w)


def _conv2d_roi_rfft_step_bwd(
    roi_shape: tuple[int, int],
    pad_hw: tuple[int, int],
    fft_shape: tuple[int, int],
    need_dx: bool,
    res,
    g_z_roi: jax.Array,
):
    cell_freq, ker_freq_mhw, in_h, in_w = res
    r_h, r_w = pad_hw
    roi_h, roi_w = roi_shape
    fh, fw = fft_shape
    pad_h_after = fh - (r_h + roi_h)
    pad_w_after = fw - (r_w + roi_w)

    g_padded = jnp.pad(
        g_z_roi, ((0, 0), (0, 0), (r_h, pad_h_after), (r_w, pad_w_after))
    )
    g_freq = jnp.fft.rfft2(g_padded, s=fft_shape, axes=(-2, -1))

    if need_dx:
        g_cell_freq = jnp.sum(
            g_freq * jnp.conj(ker_freq_mhw)[jnp.newaxis, :, :, :], axis=1
        )
        g_roi_full = jnp.fft.irfft2(g_cell_freq, s=fft_shape, axes=(-2, -1))
        g_roi_cells = g_roi_full[:, :in_h, :in_w]
    else:
        g_roi_cells = jnp.zeros(
            (g_z_roi.shape[0], in_h, in_w), dtype=g_z_roi.dtype
        )

    g_ker_freq = jnp.sum(
        g_freq * jnp.conj(cell_freq)[:, jnp.newaxis, :, :], axis=0
    )
    return g_roi_cells, g_ker_freq


conv2d_roi_rfft_step.defvjp(_conv2d_roi_rfft_step_fwd, _conv2d_roi_rfft_step_bwd)


def conv2d_valid_rfft_precomputed(
    cells_bhw: jax.Array,
    ker_freq_mhw: jax.Array,
    roi_shape: tuple[int, int] = (40, 50),
    kernel_size: tuple[int, int] = (23, 13),
    fft_shape: tuple[int, int] = (64, 64),
) -> jax.Array:
    """
    Computes exact `VALID` 2D cross-correlation on a full `(B, H, W)` cellular grid using
    precomputed `ker_freq_mhw` `(M, Fh, Fw // 2 + 1)`.

    In overlap-save 2D RFFT, the valid non-aliased region of `conv(H x W, Kh x Kw)` starts at
    offset `(Kh - 1, Kw - 1)` and has shape `(H_v, W_v) = (H - Kh + 1, W - Kw + 1)`.
    """
    kh, kw = kernel_size
    return conv2d_roi_rfft_step(
        roi_cells_bhw=cells_bhw,
        ker_freq_mhw=ker_freq_mhw,
        roi_shape=roi_shape,
        pad_hw=(kh - 1, kw - 1),
        fft_shape=fft_shape,
        need_dx=True,
    )


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def hoist_filter_rfft2_zero_fwd(
    filters_oihw: jax.Array,
    kernel_size: tuple[int, int] = (23, 13),
    fft_shape: tuple[int, int] = (64, 64),
) -> jax.Array:
    """
    Zero-cost forward token whose custom VJP converts the accumulated Plancherel
    frequency-domain filter gradient `(M, Fh, Fw // 2 + 1)` into spatial `OIHW` `(M, 1, Kh, Kw)`
    via a single `M`-channel `irfft2` per batch for the `"nchw_rfft_bwd"` hybrid backend.
    """
    fh, fw = fft_shape
    m = filters_oihw.shape[0]
    return jnp.zeros((m, fh, fw // 2 + 1), dtype=jnp.complex64)


def _hoist_zero_fwd(
    filters_oihw: jax.Array,
    kernel_size: tuple[int, int],
    fft_shape: tuple[int, int],
):
    return hoist_filter_rfft2_zero_fwd(filters_oihw, kernel_size, fft_shape), None


def _hoist_zero_bwd(
    kernel_size: tuple[int, int],
    fft_shape: tuple[int, int],
    _,
    g_ker_freq_mhw: jax.Array,
):
    return (
        rfft2_filter_grad_to_oihw(
            g_ker_freq_mhw, kernel_size=kernel_size, fft_shape=fft_shape
        ),
    )


hoist_filter_rfft2_zero_fwd.defvjp(_hoist_zero_fwd, _hoist_zero_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def conv2d_hybrid_roi_step(
    roi_cells_bhw: jax.Array,
    filters_oihw: jax.Array,
    ker_freq_token: jax.Array,
    pad_hw: tuple[int, int] = (11, 6),
    fft_shape: tuple[int, int] = (64, 64),
    need_dx: bool = True,
) -> jax.Array:
    """
    Hybrid ROI Green's function step (`backend="nchw_rfft_bwd"`):
    - Forward: Direct batched `NCHW` cuDNN convolution (`0.35 ms/step` on `(23, 13)`).
    - Backward `dX`: Explicit forward `NCHW` convolution with spatially flipped & transposed
      filter `(1, M, Kh, Kw)` (`0.58 ms/step`, skipped at `t = 0` when `need_dx=False`).
    - Backward `dG`: 64x64 RFFT Plancherel inner product (`0.93 ms/step`), avoiding cuDNN's
      atomic split-K reduction contention and accumulating across steps before a single `irfft2`.
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


def _conv2d_hybrid_roi_step_fwd(
    roi_cells_bhw: jax.Array,
    filters_oihw: jax.Array,
    ker_freq_token: jax.Array,
    pad_hw: tuple[int, int],
    fft_shape: tuple[int, int],
    need_dx: bool,
):
    z_roi = conv2d_hybrid_roi_step(
        roi_cells_bhw, filters_oihw, ker_freq_token, pad_hw, fft_shape, need_dx
    )
    return z_roi, (roi_cells_bhw, filters_oihw)


def _conv2d_hybrid_roi_step_bwd(
    pad_hw: tuple[int, int],
    fft_shape: tuple[int, int],
    need_dx: bool,
    res,
    g_z_roi: jax.Array,
):
    roi_cells_bhw, filters_oihw = res
    r_h, r_w = pad_hw
    in_h, in_w = roi_cells_bhw.shape[-2], roi_cells_bhw.shape[-1]
    fh, fw = fft_shape

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
    pad_h_after = fh - (r_h + in_h)
    pad_w_after = fw - (r_w + in_w)
    g_padded = jnp.pad(
        g_z_roi, ((0, 0), (0, 0), (r_h, pad_h_after), (r_w, pad_w_after))
    )
    g_freq = jnp.fft.rfft2(g_padded, s=fft_shape, axes=(-2, -1))
    g_ker_freq = jnp.sum(
        g_freq * jnp.conj(cell_freq)[:, jnp.newaxis, :, :], axis=0
    )

    return g_roi_cells, jnp.zeros_like(filters_oihw), g_ker_freq


conv2d_hybrid_roi_step.defvjp(_conv2d_hybrid_roi_step_fwd, _conv2d_hybrid_roi_step_bwd)
