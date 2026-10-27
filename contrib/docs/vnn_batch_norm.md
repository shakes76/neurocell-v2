# VNN 3D Batch Normalization

This document details the implementation of Batch Normalization in the 3D Vector Neural Network (VNN) architecture (specifically `Universe3D`), and how it maintains JAX JIT compilation compatibility and high-performance memory checkpointing.

## Overview
Unlike standard Multi-Layer Perceptrons (MLPs) or Convolutional Neural Networks (CNNs) where layers are distinct objects, the VNN `Universe3D` propagates states dynamically through a cellular grid using iterative `for` loops inside memory-efficient `jax.checkpoint` blocks.

Because the entire spatial propagation step (`forward.py`) is compiled using `jax.vmap` over the batch dimension, the batch dimension itself is completely hidden from the standard inner operations. This prevents the usage of standard object-oriented normalization layers like `haiku.BatchNorm`.

## Native JAX Implementation
To implement true Batch Normalization while retaining VRAM-saving JIT structures, we utilize JAX collective operations:

1. **Batch Axis Mapping:** We explicitly declare the batch dimension to JAX's `vmap` inside `Universe3D` using `axis_name='batch'`. This exposes the batched data to distributed collective operations even inside the traced single-sample inner loop.
   ```python
   self.forward_op = jax.vmap(forward._propagate_mlp_blocks_fast, in_axes=[...], axis_name='batch')
   ```

2. **Cross-Batch Statistics (`pmean`):** Inside the `_forward_column_fast` and `_forward_full_fast` operators, we compute spatial statistics for the single sample, and then average them across the mapped batch dimension using `jax.lax.pmean`.
   ```python
   spatial_mean = jnp.mean(result_block, axis=mean_axes, keepdims=True)
   batch_mean = jax.lax.pmean(spatial_mean, axis_name='batch')
   ```

3. **Exponential Moving Average (EMA) Threading:** The running mean and running variance (EMA) required for inference are initialized as `haiku.get_state` buffers inside `Universe3D`. They are passed down into the deep propagation loops as immutable arrays, updated sequentially during training, and returned back to the outer `Universe3D` layer where `hk.set_state` stores the final updated values.

## Architecture: Per-Block Normalization
To mimic the spatial weight-sharing of a standard 3D Convolutional Neural Network (CNN) while preserving VRAM efficiency, the VNN network depth is divided into a discrete number of structural **blocks**:
- **Shared Filters:** Each block shares the exact same convolutional filter kernel across all of its sequential hyperplanes (Z-slices).
- **Block-Level Statistics:** Instead of independent normalization for every single hyperplane (which destroys spatial continuity), the Batch Normalization affine parameters (`scale`, `shift`) and EMA states are aggregated and **shared per block**.
- **Smooth Momentum:** Because the cellular VNN processes each block's hyperplanes sequentially, true 3D spatial aggregation across the entire block would require buffering the entire block in VRAM. Instead, each 2D Z-slice is normalized using its own batch statistics, but the shared block EMA is updated continuously with a dynamically scaled momentum (`0.1 / depth_block`). This elegant approach ensures the EMA smoothly captures the statistics of the entire block without the final slice disproportionately dominating the moving average, all without the memory penalty of true 3D batch normalisation.

## Usage
Batch Normalization is automatically triggered in `Universe3D` when `norm=True` is provided to the network call.

During training, pass `is_training=True` to compute batch statistics and update the EMAs. During inference/testing, pass `is_training=False` to use the stored EMAs.

```python
# Training
logits, state = network.apply(params, state, batch, is_training=True)

# Evaluation
logits, state_test = network.apply(params, state, batch, is_training=False)
```
