# Hyperplane Block-Based Propagation in 3D VNNs

This document details the hyperplane block-based propagation algorithm used to evaluate 3D Von Neumann Networks (VNNs). This propagation scheme is specifically designed to be memory efficient and robust while supporting advanced features like the `expand` mode.

## Overview

Unlike standard multi-layer perceptrons (MLPs), VNNs propagate their state dynamically across a cellular grid. In a 3D Universe (`Universe3D`), running full 3D convolutions over the entire grid for every propagation step is highly inefficient.

Instead, the `_propagate_mlp_blocks_fast` algorithm simulates a feedforward CNN-like architecture by slicing the universe into hyperplanes (columns or 2D slices depending on the orientation) and propagating signals block-by-block.

### Block Propagation and Slicing (`_forward_column_fast_compact`)
The grid is virtually divided into a sequence of `blocks`. Each block contains a specific number of propagation steps (`depth_block`). For each step within a block, the network applies `_forward_column_fast_compact`. 

A critical optimization in `_forward_column_fast_compact` is its inherent array slicing. Rather than operating on the entire universe array, it uses the global spatial `index` of the current step to slice out only a narrow sub-array corresponding to the filter's local receptive field (e.g., from `index - width_pad` to `index + width_pad`). It applies the 3D convolution strictly to this active zone. Because this slicing is tightly bound to absolute grid coordinates, attempting to externally subset the `cells` array (e.g., isolating only the active propagation region ahead of time) would break the global index mapping. Thus, this internal slicing is not redundant—it is precisely what prevents full-grid convolutions and makes training memory-tractable without needing outer array fragmentation.

## Modes of Operation

### Compress Mode (Default)
By default, after every single propagation block, the network operates in `COMPRESS` mode. The convolution operation generates a multi-channel intermediate state (where channels represent `directions`), but before placing the result back into the `cells` array, it compresses these channels down to a single scalar value. 

While compressing immediately minimizes VRAM requirements—ensuring the grid remains a single-channel entity throughout the network—it creates an informational bottleneck. Valuable directional features learned within one block are collapsed before they can be leveraged by the next block.

### Expand Mode
The **Expand Mode** (`expand=True` flag in `Universe3D`) bypasses the compression bottleneck.

#### How it Works
When `expand=True` and `blocks > 1`:
1. **Initial Channel Expansion**: Before the block propagation loop begins, the current 1-channel `cells` array is expanded to `directions` channels and stored in `cells_expanded`. This initialization occurs outside the loop using `jnp.repeat` so that it doesn't overwrite states generated sequentially by the intermediate blocks.
2. **Intermediate Blocks (`EXPAND` Mode)**: During the intermediate blocks (except the final one), the `EXPAND` mode is used. The `_forward_column_fast_compact` operator reads the localized slice from the multi-channel `cells_expanded`, applies the filters without compressing the channel axis, and writes the multi-channel results back to `cells_expanded`. Concurrently, it updates the persistent 1-channel `cells` array with the mean value to preserve visual consistency.
3. **Final Block Compression**: After all intermediate blocks have run in `EXPAND` mode, the final propagation block automatically switches back to `COMPRESS` mode. It takes the uncompressed output from `cells_expanded` and compresses it down to the final 1-channel output `result_block`.

This hybrid approach allows VNNs to maintain rich, multi-channel representations between sequential internal blocks. Because `_forward_column_fast_compact` relies on its own narrow local slicing against the global array shape, `cells_expanded` maintains the same spatial shape as the full universe, satisfying the index mapping while safely carrying the expanded channels.
