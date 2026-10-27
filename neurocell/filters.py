'''
Discrete filter stencils and spatial kernels for Von Neumann Networks.

Copyright (c) 2026 Shekhar S. Chandra. All rights reserved.
Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

@author Shakes
'''
import jax.numpy as jnp


def smooth(dtype=jnp.float32, norm: bool = False):
    """Returns the 3x3 window or smoothing kernel."""
    center = 8 if norm else 1
    kernel = [
        [1, 1, 1],
        [1, center, 1],
        [1, 1, 1],
    ]
    return jnp.array(kernel, dtype=dtype)


def backward(dtype=jnp.float32, norm: bool = False):
    """Returns the 3x3 backward propagation kernel."""
    center = 3 if norm else 0
    kernel = [
        [1, 0, 0],
        [1, center, 0],
        [1, 0, 0],
    ]
    return jnp.array(kernel, dtype=dtype)


def forward(dtype=jnp.float32, norm: bool = False):
    """Returns the 3x3 forward propagation kernel."""
    center = 3 if norm else 0
    kernel = [
        [0, 0, 1],
        [0, center, 1],
        [0, 0, 1],
    ]
    return jnp.array(kernel, dtype=dtype)


def von_neumann(dtype=jnp.float32, norm: bool = False):
    """Returns the 3x3 bidirectional von Neumann propagation kernel."""
    center = 6 if norm else 1
    kernel = [
        [1, 0, 1],
        [1, center, 1],
        [1, 0, 1],
    ]
    return jnp.array(kernel, dtype=dtype)


def sharpen(dtype=jnp.float32, norm: bool = True):
    """Returns the 3x3 spatial sharpen kernel."""
    kernel = jnp.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=dtype,
    )
    if norm:
        kernel = kernel / jnp.array(5.0, dtype=dtype)
    return kernel


def sharpen_3d(dtype=jnp.float32, norm: bool = True):
    """Returns the 3x3x3 spatial sharpen kernel."""
    kernel = jnp.array(
        [
            [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
            [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
        ],
        dtype=dtype,
    )
    if norm:
        kernel = kernel / jnp.array(5.0, dtype=dtype)
    return kernel
