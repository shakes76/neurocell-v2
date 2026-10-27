# Stage 05 — Milestone 6: Model-, Input-, and Output-Agnostic Core Primitives (FP32)

## 1. Objective

Milestones 1–5 (`v2.0`) delivered a $2.94\times$ training speedup (`27.79 s/epoch` $\to$ `9.44 s/epoch`) on the canonical $62\times 62$ 8-bit ALU chip, but several core primitives in [`neurocell/kernels_pallas.py`](file:///home/shakes/Work/neurocell-release/neurocell/kernels_pallas.py), [`neurocell/propagator.py`](file:///home/shakes/Work/neurocell-release/neurocell/propagator.py), and [`neurocell/topology.py`](file:///home/shakes/Work/neurocell-release/neurocell/topology.py) remained coupled to the ALU chip's geometry and activation function:

1. **Spatial Tile Limit (`N_PIX <= 2048`)**: The Pallas epilogue originally required the entire 2D ROI to fit in a single `<= 2048` power-of-two tile, preventing execution on larger grids such as $64\times 64$ Game of Life (`3,844`–`4,096` pixels), $80\times 80$ (`6,400` pixels), or $800\times 17$ MNIST (`13,600` pixels).
2. **Hardcoded `tanh` Activation**: `use_pallas_epilogue` required `activate is jax.nn.tanh`, forcing models using `leaky_relu` (Game of Life), `sigmoid`, `relu`, `gelu`, or linear activations back to unfused XLA epilogues.
3. **Rectangular Sub-ROI Port Assumption**: `ChipTopology2D`, `precompute_step_params`, and `pallas_fused_output_port_epilogue` only supported rectangular `PortSpec` slices strictly inside the `VALID` interior, excluding full-grid $N\times N \to N\times N$ cellular automata and coordinate-addressed Circular Layered Perceptrons (`CoordinatePortSpec`).
4. **Redundant $t = 0$ / Borderless HBM Reductions & Unpruned Forward 2D RFFTs**: At step $t = 0$ (and in `"SAME"`/`"CIRCULAR"` boundary modes), `delta_border` is identically zero, yet the Pallas kernel loaded `dborder_ref` in forward/backward and summed the `16.38 MB` `gz` tensor over the batch in backward (`gdb = jnp.sum(gz, axis=0)`). Similarly, `rfft2` padded dead rows to `Fh` before running horizontal 1D RFFTs.

---

## 2. Key Architectural Innovations

### 2.1 Single-Pass (`N_PIX <= 8192`) & Two-Stage Hardware-Tiled (`N_PIX > 8192`) Pallas Epilogues
- Because `_make_roi_epilogue_fwd_kernel` and `_make_roi_epilogue_bwd_kernel` stream sequentially across directions $m \in [0, M-1]$ while keeping only `acc_lin` and `acc_act` live in registers, the single-pass kernel (`grid = (B,)`) scales cleanly up to `SINGLE_TILE_MAX_PIX = 8192` (`62x62 = 3,844`, `64x64 = 4,096`, `80x80 = 6,400`) in **1 forward launch** and **1 backward launch** with **zero `jnp.pad` allocations**.
- For very large grids (`N_PIX > 8192`, e.g., $96\times 96 = 9,216$ or $800\times 17 = 13,600$), [`pallas_fused_roi_step_epilogue`](file:///home/shakes/Work/neurocell-release/neurocell/kernels_pallas.py) automatically dispatches to a two-stage hardware-tiled (`BLOCK_N <= 1024`, `grid = (B, num_tiles)`) pipeline using Chan's exact parallel variance combination formula:
  $$\mu = \frac{1}{N_{\text{pix}}} \sum_{t=0}^{T-1} S_{1,t}, \qquad \sigma^2 = \frac{1}{N_{\text{pix}}} \sum_{t=0}^{T-1} \left(M_{2,t} + n_t(\mu_t - \mu)^2\right)$$

### 2.2 Pluggable Activation Functions & `has_border: bool` Fast Path
- `_resolve_activation_spec(activate)` compiles `tanh`, `leaky_relu` (including `functools.partial(jax.nn.leaky_relu, negative_slope=...)`), `relu`, `sigmoid`, `gelu`, and `None` (`identity`) directly into both the single-pass and tiled Pallas Triton forward and backward kernels.
- Passing `has_border=(step_idx > 0 and has_valid_border)` skips loading `dborder_ref` in forward/backward and replaces `gdb = jnp.sum(gz, axis=0, keepdims=True)` with `jnp.zeros_like(delta_border_1mhw)` whenever no border contribution is present, eliminating a `16.38 MB` HBM read at step $t = 0$ and on every step of `"SAME"` / `"CIRCULAR"` topologies.

### 2.3 Universal I/O Port Topologies & Pure-ROI Hoisting for $t \ge 1$ on Full-Grid Models
- Added [`CoordinatePortSpec`](file:///home/shakes/Work/neurocell-release/neurocell/topology.py) for scattered 2D coordinate ports (CLP) and preset constructors `ChipTopology2D.gol_chip`, `ChipTopology2D.clp_chip`, and `ChipTopology2D.tm_chip`, alongside `"VALID"`, `"SAME"`, and `"CIRCULAR"` boundary modes.
- Even when step $t = 0$ inputs touch the outer border (e.g., full-grid $64\times 64 \to 64\times 64$ Game of Life in `"VALID"` mode), [`propagate_vnn_2d`](file:///home/shakes/Work/neurocell-release/neurocell/propagator.py) executes `VALID` convolution on `(B, H, W)` only at $t = 0$ to produce `(B, M, H_v, W_v)` and executes **all subsequent steps $t \ge 1$** using Pure-ROI `(B, H_v, W_v)` convolutions + hoisted $B=1$ `delta_border` + Pallas fused epilogues.

### 2.4 Shape-Driven Separable `rfft2_pruned`, `conv2d_grouped_filter_grad`, & `backend="auto"`
- [`rfft2_pruned`](file:///home/shakes/Work/neurocell-release/neurocell/kernels_greens.py) computes horizontal 1D `rfft` strictly on the $H_{\text{in}} < F_h$ active rows prior to vertical 1D `fft`, eliminating the `(B, M, Fh, Fw)` real padding buffer and cutting horizontal 1D RFFTs by $1 - H_{\text{in}} / F_h$ (`37.5%` on $40 / 64$).
- [`select_optimal_backend`](file:///home/shakes/Work/neurocell-release/neurocell/propagator.py) (`backend="auto"`) dispatches across `nchw_direct`, `pallas_fused`, and `pallas_sram` purely from `(grid_shape, roi_shape, kernel_size, fft_shape)`.

---

## 3. Verified GPU Benchmark Results (NVIDIA RTX 4070 Laptop GPU, FP32)

| VNN Architecture | Topology & Shapes | `nchw_direct` (ms/step) | `pallas_fused` v2.1 (ms/step) | `auto` v2.1 (ms/step) | Notes / Speedup |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **8-bit ALU Chip (`depth = 4`)** | `62x62` grid (`40x50` ROI), $K=23\times 13, M=16, B=128$, `tanh` | `11.49 ms` (`9.19 s/ep`) | **`10.17 ms` (`8.14 s/ep`)** | `10.46 ms` (`8.36 s/ep`) | **$3.42\times$ vs `contrib/` (`27.79 s/ep`)**, **$13.8\%$ faster than v2.0 (`9.44 s/ep`)** |
| **8-bit ALU Chip (`depth = 8`)** | `62x62` grid (`40x50` ROI), $K=23\times 13, M=16, B=128$, `tanh` | `23.87 ms` (`19.09 s/ep`) | `20.83 ms` (`16.66 s/ep`) | **`20.57 ms` (`16.46 s/ep`)** | **$1.16\times$ faster** than `nchw_direct` |
| **Full-Grid Game of Life** | `64x64 -> 64x64`, $K=3\times 3, M=64, B=64, \text{depth}=10$, `leaky_relu` | `42.86 ms` | `37.92 ms` | **`37.88 ms`** | **$1.13\times$ faster** than `nchw_direct` (`~1.45x` vs unhoisted full-grid) |
| **Large-Grid VNN** | `82x82` grid (`80x80 = 6,400` ROI), $K=3\times 3, M=16, B=64, \text{depth}=6$ | `11.91 ms` | **`10.96 ms`** | `11.38 ms` | Previously overflowed `N_PIX <= 2048` |
| **Coordinate-Port CLP** | `8x8` grid (`CoordinatePortSpec`), $K=3\times 3, M=16, B=128, \text{depth}=8$ | `0.76 ms` | **`0.73 ms`** | `0.74 ms` | Native scattered coordinate I/O |

---

## 4. Head-to-Head Accuracy & Convergence Parity (`v1` vs Milestone 6 `v2.1`)

Training both Legacy `v1` ([`contrib/universe.py`](file:///home/shakes/Work/neurocell-release/contrib/universe.py)) and Milestone 6 `v2.1` (`backend="auto"` / `"pallas_fused"`) from the exact same `PRNGKey(42)` initialization on identical mini-batches produces **6-decimal-place identical losses and 100% identical test accuracies** at every epoch:

| Epoch | Legacy `v1` Loss | Milestone 6 `v2.1` Loss | Legacy `v1` Test Accuracy | Milestone 6 `v2.1` Test Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| **1** | `0.161258` | `0.161258` | `0.90%` | `0.90%` |
| **2** | `0.141024` | `0.141024` | `8.28%` | `8.28%` |
| **3** | `0.113796` | `0.113796` | `51.60%` | `51.60%` |
| **4** | `0.099253` | `0.099253` | `52.85%` | `52.85%` |
| **5** | `0.094330` | `0.094330` | `53.05%` | `53.05%` |
| **30 (Checkpoint)** | `0.150637` | `0.150637` | `98.35%` (`98.12%` 2.5k split) | `98.35%` (`98.12%` 2.5k split) |

- **5-Epoch (`400`-step) Parameter Relative $L_2$ Error (`v1` vs `v2.1`)**:
  - Weights $W$: `6.823e-07`
  - Biases $B$: `2.897e-06`
  - States $S$: `3.756e-07`
  - Green's Filters $G$: `2.389e-06`

---

## 5. Verification Suite

All 24 unit tests across 6 test modules pass (`Ran 24 tests — OK`):
- [`tests/unit_topology.py`](file:///home/shakes/Work/neurocell-release/tests/unit_topology.py)
- [`tests/unit_params.py`](file:///home/shakes/Work/neurocell-release/tests/unit_params.py)
- [`tests/unit_epilogue.py`](file:///home/shakes/Work/neurocell-release/tests/unit_epilogue.py)
- [`tests/unit_kernels_greens.py`](file:///home/shakes/Work/neurocell-release/tests/unit_kernels_greens.py)
- [`tests/unit_kernels_pallas.py`](file:///home/shakes/Work/neurocell-release/tests/unit_kernels_pallas.py)
- [`tests/unit_vnn_alu_parity.py`](file:///home/shakes/Work/neurocell-release/tests/unit_vnn_alu_parity.py)
- [`tests/unit_milestone6_general_vnn.py`](file:///home/shakes/Work/neurocell-release/tests/unit_milestone6_general_vnn.py)
