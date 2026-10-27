'''
Conv module for all convolution algorithms
'''
import jax
import jax.numpy as jnp

#JIT compiled members for performance
from functools import partial

@partial(jax.jit, static_argnums=(2,3,4,5))
def _filter(cells, nethood, dn, window_strides, lhs_dilation=None, rhs_dilation=None):
    """
    Filter the array provided by filter provided.
    """
    #get weighted sum of neighbors using convolution
    cells = jax.lax.conv_general_dilated(cells[jnp.newaxis,...,jnp.newaxis], # lhs = NHWC image tensor
                                        nethood,  # rhs = OHWI conv kernel tensor
                                        window_strides,  # window strides
                                        'SAME', # padding mode
                                        lhs_dilation,  # lhs/image dilation
                                        rhs_dilation,  # rhs/kernel dilation
                                        dn)     # dimension_numbers = lhs, rhs, out dimension permutation
    return cells[0,...,0] # NHWC

# conv operators
@partial(jax.jit, static_argnums=(2,3,4,5,6,7))
def _fast_conv(sub_cells, filters, dn, window_strides, padding_mode="SAME", lhs_dilation=None, rhs_dilation=None, feature_group_count=1):
    """
    nD conv done with filters with JAX builtin general conv func. Supports multiple filters in channel dim.
    Assumes sub_cells has matching channel dim to filters
    The dims are defined by dn and strides

    Cells is assumed to be the relevant sub-region to process
    """
    #Conv via 2D brute force
    #get weighted sum of neighbors using convolution
    sub_cells_conv = jax.lax.conv_general_dilated(sub_cells[jnp.newaxis,...], # lhs = NHWC image tensor
                                        filters,  # rhs = OHWI conv kernel tensor
                                        window_strides,  # window strides
                                        padding_mode, # padding mode
                                        lhs_dilation,  # lhs/image dilation
                                        rhs_dilation,  # rhs/kernel dilation
                                        dn,  # dimension_numbers = lhs, rhs, out dimension permutation
                                        feature_group_count, #used for grouped convolutions
                                        )
    return sub_cells_conv[0] # NHWC, remove extra batch dim, return multi-filters result

@partial(jax.jit, static_argnums=(2,3,4,5))
def _fast_conv_2D(image, kernel, epsilon=1e-7, apply_antialias=False, aa_cutoff=0.2, aa_order=3):
    """
    Performs a 2D convolution using FFT with proper phase alignment,
    numerical noise suppression, and optional anti-aliasing.

    Args:
        image: 2D JAX array representing the image.
        kernel: 2D JAX array representing the filter.
        epsilon: Threshold to zero-out microscopic floating-point noise.
        apply_antialias: If True, applies a Butterworth low-pass filter in the
                         frequency domain to prevent frequency aliasing.
        aa_cutoff: Cutoff frequency for the Butterworth filter (0.0 to 0.5).
        aa_order: Order of the Butterworth filter. Higher means steeper cutoff
                  but more potential for spatial ringing.

    Returns:
        2D JAX array of the convolved image, same shape as input image.

    @author Gemini Pro 3.1
    """
    img_shape = image.shape
    ker_shape = kernel.shape

    # 1. Pad for linear convolution to prevent spatial aliasing (circular wraparound)
    fft_shape = (img_shape[0] + ker_shape[0] - 1, img_shape[1] + ker_shape[1] - 1)

    padded_image = jnp.pad(
        image,
        ((0, fft_shape[0] - img_shape[0]), (0, fft_shape[1] - img_shape[1])),
        mode='constant'
    )

    # 2. Pad the kernel to match the fft_shape, placing it dead center.
    pad_h_before = (fft_shape[0] - ker_shape[0]) // 2
    pad_h_after = fft_shape[0] - ker_shape[0] - pad_h_before

    pad_w_before = (fft_shape[1] - ker_shape[1]) // 2
    pad_w_after = fft_shape[1] - ker_shape[1] - pad_w_before

    padded_kernel = jnp.pad(
        kernel,
        ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
        mode='constant'
    )

    # 3. Apply ifftshift to the padded spatial kernel to zero the phase.
    shifted_kernel = jnp.fft.ifftshift(padded_kernel)

    # 4. Transform to frequency domain using Real FFT
    image_freq = jnp.fft.rfft2(padded_image)
    kernel_freq = jnp.fft.rfft2(shifted_kernel)

    # 5. Point-wise multiplication in the frequency domain
    result_freq = image_freq * kernel_freq

    # 6. Apply optional frequency-domain anti-aliasing (Butterworth low-pass)
    # This prevents high-frequency data from folding over if subsampling later
    if apply_antialias:
        fy = jnp.fft.fftfreq(fft_shape[0])
        fx = jnp.fft.rfftfreq(fft_shape[1])
        FY, FX = jnp.meshgrid(fy, fx, indexing='ij')

        # Squared normalized frequency
        F_sq = FY**2 + FX**2

        # Butterworth low-pass filter
        # Maximally flat passband with controllable roll-off
        aa_filter = 1.0 / (1.0 + (F_sq / (aa_cutoff**2))**aa_order)
        result_freq = result_freq * aa_filter

    # 7. Inverse Real FFT to return to the spatial domain
    conv_result = jnp.fft.irfft2(result_freq, s=fft_shape)

    # 8. Crop back to the original image shape (handles the 'same' mode cropping safely)
    cropped_result = conv_result[:img_shape[0], :img_shape[1]]

    # 9. Handle Epsilon: Suppress microscopic floating-point numerical noise
    cleaned_result = jnp.where(jnp.abs(cropped_result) < epsilon, 0.0, cropped_result)

    return cleaned_result

# Define the complete list of static argument indices so JAX knows not to trace them as dynamic arrays
# Indices: epsilon (2), suppress_epsilons (3), pad_volume (4),
# normalize (5), apply_antialias (6), aa_cutoff (7), aa_order (8)
_static_argnums = (2, 3, 4, 5, 6, 7, 8)
@partial(jax.jit, static_argnums=_static_argnums)
def _fast_conv_3D(volume, kernel, epsilon=1e-7, suppress_epsilons=False, padding_mode="SAME", normalize=False, apply_antialias=False, aa_cutoff=0.5, aa_order=3):
    """
    Performs a 3D convolution using FFT with proper phase alignment,
    numerical noise suppression, and optional anti-aliasing.

    Args:
        volume: 3D or 4D JAX array representing the volume. If 4D, expects (Z, Y, X, C).
        kernel: 3D or 4D JAX array. If 4D, expects (Z, Y, X, C) for multi-channel CNN-like output.
        epsilon: Threshold to zero-out microscopic floating-point noise.
        suppress_epsilons: If True, applies the epsilon threshold to zero-out noise.
        padding_mode: String indicating the spatial padding and cropping strategy:
                      - "SAME": Output shape matches input. Pads volume to prevent aliasing.
                      - "VALID": Only returns regions of full overlap. Output shape shrinks.
                      - "CIRCULAR": Fastest mode. No volume padding (circular wrap-around).
        normalize: If True, normalizes the output feature maps to have a mean of zero
                   and a variance of 1. Variance normalization is computed in the
                   frequency domain for circular convolution (`padding_mode='CIRCULAR'`)
                   and in the spatial domain for linear convolutions (`'SAME'` or `'VALID'`).
        apply_antialias: If True, applies a Butterworth low-pass filter in the
                         frequency domain to prevent frequency aliasing.
        aa_cutoff: Cutoff frequency for the Butterworth filter (0.0 to 0.5).
        aa_order: Order of the Butterworth filter.

    Returns:
        JAX array of the convolved volume. Shape matches volume (for SAME/CIRCULAR)
        or shrinks (for VALID). Includes channel dim if volume or kernel is 4D.
    """
    # Track original dimensions for backward compatibility
    vol_was_3d = volume.ndim == 3
    if vol_was_3d:
        volume = volume[..., None] # Convert to (Z, Y, X, 1)

    ker_was_3d = kernel.ndim == 3
    if ker_was_3d:
        kernel = kernel[..., None] # Convert to (Kz, Ky, Kx, 1)

    vol_shape = volume.shape[:3]
    ker_shape = kernel.shape[:3]
    padding_mode = padding_mode.upper()

    # 1. Calculate the target shape
    if padding_mode in ["SAME", "VALID"]:
        # Linear convolution prevents spatial aliasing
        fft_shape = (
            vol_shape[0] + ker_shape[0] - 1,
            vol_shape[1] + ker_shape[1] - 1,
            vol_shape[2] + ker_shape[2] - 1
        )
    elif padding_mode == "CIRCULAR":
        fft_shape = vol_shape
    else:
        raise ValueError(f"Unsupported padding_mode: {padding_mode}. Use 'SAME', 'VALID', or 'CIRCULAR'.")

    # 2. Transform to frequency domain using N-Dimensional Real FFT
    # The `s` parameter automatically pads the arrays with zeros at the end,
    # entirely replacing the need for manual `jnp.pad` and `ifftshift`.
    # axes=(0, 1, 2) ensures the channel dimension is not transformed.
    vol_freq = jnp.fft.rfftn(volume, s=fft_shape, axes=(0, 1, 2))
    ker_freq = jnp.fft.rfftn(kernel, s=fft_shape, axes=(0, 1, 2))

    # 3. Point-wise multiplication in the frequency domain
    # vol_freq is (Z, Y, X_freq, Cv) and ker_freq is (Kz, Ky, X_freq, Ck).
    # We rely on standard JAX broadcasting to align the channels correctly.
    result_freq = vol_freq * ker_freq

    # 4. Apply optional 3D frequency-domain anti-aliasing (Butterworth low-pass)
    if apply_antialias:
        fz = jnp.fft.fftfreq(fft_shape[0])
        fy = jnp.fft.fftfreq(fft_shape[1])
        fx = jnp.fft.rfftfreq(fft_shape[2])  # Real FFT only goes up to Nyquist on the last axis
        FZ, FY, FX = jnp.meshgrid(fz, fy, fx, indexing='ij')

        # 3D Squared normalized frequency
        F_sq = FZ**2 + FY**2 + FX**2

        # 3D Butterworth low-pass filter (expanded to broadcast across channels)
        aa_filter = 1.0 / (1.0 + (F_sq / (aa_cutoff**2))**aa_order)
        result_freq = result_freq * aa_filter[..., None]

    # 5. Optional Zero-Mean Normalization
    # The DC component (0,0,0) of the frequency spectrum represents the sum of the spatial values.
    # Zeroing it out perfectly subtracts the mean from the resulting spatial feature maps.
    if normalize:
        result_freq = result_freq.at[0, 0, 0, :].set(0.0)

    # 6. Optional Unit-Variance Normalization (Frequency Domain for unpadded volume)
    # Parseval's theorem allows us to compute spatial variance directly from the frequency spectrum.
    # Because rfftn only keeps half the X frequencies, we apply a weight mask to recover full energy.
    if normalize and padding_mode == "CIRCULAR":
        Nx = fft_shape[2]
        Nx_rfft = result_freq.shape[2]

        # Weight mask: DC is 1, conjugate symmetric pairs are 2, Nyquist (if even) is 1.
        weights = jnp.full((Nx_rfft,), 2.0)
        weights = weights.at[0].set(1.0)
        if Nx % 2 == 0:
            weights = weights.at[-1].set(1.0)

        weights = weights.reshape(1, 1, -1, 1) # Reshape for broadcasting

        # Sum of squared magnitudes in the full theoretical frequency spectrum
        sum_sq_X = jnp.sum(jnp.abs(result_freq)**2 * weights, axis=(0, 1, 2), keepdims=True)

        # Spatial standard deviation derived via Parseval's theorem
        N_total = fft_shape[0] * fft_shape[1] * fft_shape[2]
        sigma = jnp.sqrt(sum_sq_X) / N_total

        result_freq = result_freq / (sigma + 1e-8)

    # 7. Inverse N-Dimensional Real FFT to return to the spatial domain
    conv_result = jnp.fft.irfftn(result_freq, s=fft_shape, axes=(0, 1, 2))

    # 8. Crop or roll back to the desired spatial mode
    sz, sy, sx = ker_shape[0] // 2, ker_shape[1] // 2, ker_shape[2] // 2

    if padding_mode == "SAME":
        # Shift the crop window by the kernel's center to get 'same' convolution
        cropped_result = conv_result[
            sz : sz + vol_shape[0],
            sy : sy + vol_shape[1],
            sx : sx + vol_shape[2],
            :
        ]
    elif padding_mode == "VALID":
        # Only return the interior region where the kernel completely overlaps the volume
        cropped_result = conv_result[
            ker_shape[0] - 1 : vol_shape[0],
            ker_shape[1] - 1 : vol_shape[1],
            ker_shape[2] - 1 : vol_shape[2],
            :
        ]
    elif padding_mode == "CIRCULAR":
        # For circular convolution, roll the array to center the kernel response
        cropped_result = jnp.roll(conv_result, shift=(-sz, -sy, -sx), axis=(0, 1, 2))

    # 9. Optional Unit-Variance Normalization (Spatial Domain for padded volume)
    # Frequency variance computation is inaccurate when padding because the frequency
    # spectrum represents the padded space, not the cropped map. Thus, we apply it spatially.
    if normalize and padding_mode in ["SAME", "VALID"]:
        std_dev = jnp.std(cropped_result, axis=(0, 1, 2), keepdims=True)
        cropped_result = cropped_result / (std_dev + 1e-8)  # Add small epsilon to prevent div by zero

    # 10. Handle Epsilon: Optionally suppress microscopic floating-point numerical noise
    if suppress_epsilons:
        cleaned_result = jnp.where(jnp.abs(cropped_result) < epsilon, 0.0, cropped_result)
    else:
        cleaned_result = cropped_result

    if vol_was_3d and ker_was_3d:
        return cleaned_result[..., 0]

    return cleaned_result

# Compiled Function Variants

# Memory-efficient version for Deep Learning training (Backward pass memory savings)
# Applying jax.checkpoint (remat) inside jax.jit ensures intermediate FFT arrays are discarded
# and recomputed during backpropagation, saving massive amounts of VRAM.
_fast_conv_3D_compact_jit = jax.jit(
    jax.checkpoint(
        _fast_conv_3D,
        static_argnums=_static_argnums
    ),
    static_argnums=_static_argnums
)

# --- User-Facing Wrappers ---
# These wrappers allow you to call the function using kwargs (e.g., epsilon=1e-6)
# while safely translating them into pure positional arguments for the compiled JAX inner functions.
# This entirely prevents the "static_argnums out of bounds" ValueError!
# def _fast_conv_3D(volume, kernel, epsilon=1e-7, suppress_epsilons=True, pad_volume=True, normalize=False, apply_antialias=False, aa_cutoff=0.2, aa_order=3):
#     """Standard fast 3D FFT convolution."""
#     return _fast_conv_3D_jit(volume, kernel, epsilon, suppress_epsilons, pad_volume, normalize, apply_antialias, aa_cutoff, aa_order)
def _fast_conv_3D_compact(volume, kernel, epsilon=1e-7, suppress_epsilons=False, padding_mode="SAME", normalize=False, apply_antialias=False, aa_cutoff=0.5, aa_order=3):
    """Memory-efficient 3D FFT convolution for neural network training."""
    return _fast_conv_3D_compact_jit(volume, kernel, epsilon, suppress_epsilons, padding_mode, normalize, apply_antialias, aa_cutoff, aa_order)
