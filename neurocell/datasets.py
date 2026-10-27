'''
Vectorized data pipelines and contiguous GPU dataset loaders for Neurocell VNNs.

Matches `contrib/data.py` bit-for-bit on the same NumPy random seed while
vectorizing label generation (`np.select` instead of nested Python loops) and
preparing contiguous device-resident batch tensors for `jax.lax.scan`.

Copyright (c) 2026 Shekhar S. Chandra. All rights reserved.
Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

@author Shakes
'''
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split


class ALUDeviceDataset(NamedTuple):
    """
    Contiguous GPU-resident dataset for 8-bit ALU Arithmetic & Logic training and evaluation.

    Shapes (for samples=128000, batch_size=128, test_size=0.2, n=8):
        x_train:        (800, 128, 3, 8, 1)  float32
        y_train:        (800, 128, 8)        float32
        y_train_packed: (800, 128, 1)        uint8
        x_test:         (200, 128, 3, 8, 1)  float32
        y_test:         (200, 128, 8)        float32
        y_test_packed:  (200, 128, 1)        uint8
    """
    x_train: jax.Array
    y_train: jax.Array
    y_train_packed: jax.Array
    x_test: jax.Array
    y_test: jax.Array
    y_test_packed: jax.Array


def generate_arithmetic(
    samples: int,
    operation: str = "add",
    n: int = 8,
    batch_size: int = 64,
    noise_level: float = 0.2,
):
    """
    Generates binary arithmetic dataset (`add`, `minus`, or `xor`) with uniform input noise.
    """
    bit_depth = 2 ** (n - 1)
    num_batches = samples // batch_size
    delta = noise_level

    numbers = np.random.randint(1, bit_depth, size=(num_batches, batch_size, 1), dtype=np.uint8)
    operands = np.random.randint(1, bit_depth, size=(num_batches, batch_size, 1), dtype=np.uint8)

    op = operation.lower()
    if op == "minus":
        results = numbers - operands
    elif op == "xor":
        results = numbers ^ operands
    else:
        results = numbers + operands

    X1 = np.unpackbits(numbers, axis=2, count=n)
    X2 = np.unpackbits(operands, axis=2, count=n)
    Y_raw = np.unpackbits(results, axis=2, count=n)

    rand1 = np.random.uniform(-delta, +delta, size=X1.shape)
    rand2 = np.random.uniform(-delta, +delta, size=X2.shape)
    X1f = X1.astype(np.float32) + rand1
    X2f = X2.astype(np.float32) + rand2

    return X1f, X2f, Y_raw


def generate_arithmetic_logic(
    samples: int,
    n: int = 8,
    ctrl_bits: int = 4,
    batch_size: int = 64,
    noise_level: float = 0.2,
):
    """
    Vectorized ALU truth-table generator matching `contrib/data.py` bit-for-bit on the same RNG seed.
    """
    bit_depth = 2 ** (n - 1)
    ctrl_bit_depth = 2 ** ctrl_bits
    num_batches = samples // batch_size
    delta = noise_level

    numbers = np.random.randint(1, bit_depth, size=(num_batches, batch_size, 1), dtype=np.uint8)
    operands = np.random.randint(1, bit_depth, size=(num_batches, batch_size, 1), dtype=np.uint8)
    controls = np.random.randint(1, ctrl_bit_depth, size=(num_batches, batch_size, 1), dtype=np.uint8)

    # Vectorized uint8 ALU truth table matching contrib/data.py exactly
    one = np.uint8(1)
    conds = [
        (controls == 0) | (controls == 4),
        controls == 1,
        (controls == 2) | (controls == 8),
        controls == 3,
        controls == 5,
        controls == 6,
        controls == 7,
    ]
    choices = [
        numbers,
        (numbers + one).astype(np.uint8),
        (numbers + operands).astype(np.uint8),
        (numbers + operands + one).astype(np.uint8),
        (numbers - one).astype(np.uint8),
        (numbers - operands).astype(np.uint8),
        (numbers - operands - one).astype(np.uint8),
    ]
    y_packed_u8 = np.select(conds, choices, default=numbers).astype(np.uint8)

    X1 = np.unpackbits(numbers, axis=2, count=n)
    X2 = np.unpackbits(operands, axis=2, count=n)
    C = np.unpackbits(controls, axis=2, count=n)
    Y_raw = np.unpackbits(y_packed_u8, axis=2, count=n)

    rand1 = np.random.uniform(-delta, +delta, size=X1.shape)
    rand2 = np.random.uniform(-delta, +delta, size=X2.shape)
    rand3 = np.random.uniform(-delta, +delta, size=C.shape)
    X1f = X1.astype(np.float32) + rand1
    X2f = X2.astype(np.float32) + rand2
    Cf = C.astype(np.float32) + rand3

    return X1f, X2f, Cf, Y_raw


def prepare_alu_device_dataset(
    samples: int = 128000,
    n: int = 8,
    ctrl_bits: int = 4,
    batch_size: int = 128,
    noise_level: float = 0.2,
    test_size: float = 0.2,
    split_seed: int = 42,
) -> ALUDeviceDataset:
    """
    Generates the ALU dataset, performs the exact 80/20 train/test split used in
    `contrib/test_jax_vnn_alu_arithmetic.py`, and stages contiguous FP32/uint8
    tensors directly into GPU HBM for zero-overhead `jax.lax.scan` epochs.
    """
    X1f, X2f, Cf, Y_raw = generate_arithmetic_logic(
        samples=samples,
        n=n,
        ctrl_bits=ctrl_bits,
        batch_size=batch_size,
        noise_level=noise_level,
    )
    X_preproc = np.concatenate([X1f, X2f, Cf], axis=-1)
    X, X_test, Y, Y_test = train_test_split(
        X_preproc, Y_raw, test_size=test_size, random_state=split_seed
    )

    X_splits = np.dsplit(X, 3)
    X_test_splits = np.dsplit(X_test, 3)

    # Stack into contiguous (num_batches, B, 3, n, 1) float32 arrays
    x_train_np = np.stack(X_splits, axis=2).astype(np.float32)[..., np.newaxis]
    x_test_np = np.stack(X_test_splits, axis=2).astype(np.float32)[..., np.newaxis]

    y_train_np = Y.astype(np.float32)
    y_test_np = Y_test.astype(np.float32)

    y_train_packed_np = np.packbits(Y, axis=2)
    y_test_packed_np = np.packbits(Y_test, axis=2)

    return ALUDeviceDataset(
        x_train=jax.device_put(jnp.asarray(x_train_np)),
        y_train=jax.device_put(jnp.asarray(y_train_np)),
        y_train_packed=jax.device_put(jnp.asarray(y_train_packed_np)),
        x_test=jax.device_put(jnp.asarray(x_test_np)),
        y_test=jax.device_put(jnp.asarray(y_test_np)),
        y_test_packed=jax.device_put(jnp.asarray(y_test_packed_np)),
    )
