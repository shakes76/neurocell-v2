# Stage 07: Parallel `(B, M, Tiles)` Pallas Kernels, SRAM-Resident Grouped Gradients & STE Discrete Codd Hardening (`v2.3` — Milestone 8)

## 1. Executive Summary
Milestone 8 (`v2.3`) advances Neurocell along three orthogonal VNN-specific axes in pure `float32`:
1. **Parallel `(B, M, num_tiles)` Pallas Epilogues & In-Register Sparse Output-Port Scatter**:
   - Parallelizes tiled Chan/Welford forward statistics (`_make_tiled_chan_stats_fwd_kernel`), tiled sparse output-port backward standardization (`_make_tiled_sparse_port_bwd_kernel`), and low-batch tiled backward Stage 2 (`_make_parallel_tiled_bwd_stage2_kernel`) across all three dimensions `grid = (B, M, num_tiles)`, eliminating sequential per-direction `for m in range(m_dirs)` loops inside tile kernels.
   - Fuses the sparse output-port backward gradient scatter (`gz_base + rstd_m * gz_out`) directly into registers inside both `_make_sparse_port_bwd_kernel` (`grid = (B, M)`) and `_make_tiled_sparse_port_bwd_kernel` (`grid = (B, M, num_tiles)`) before the single coalesced `pl_triton.store`, completely eliminating the post-kernel `.at[...].add(gz_port)` `16.38 MB`–`52.4 MB` HBM buffer copy for both rectangular `PortSpec` and scattered `CoordinatePortSpec` output ports (**1.51x faster** output-port backward pass: `0.483 ms` $\to$ `0.320 ms`, `max_diff = 2.38e-07`).
2. **SRAM-Resident Grouped-Batch Filter Gradients (`<= 100 KB/SM`) & Coalesced Plancherel Reductions**:
   - Optimizes `_select_batch_groups` in `neurocell/kernels_greens.py` so each batch group maintains `b_per_group >= 16` channels (`64-byte` vectorized SIMD/Tensor-Core alignment on Ada Lovelace `SM 8.9`), and automatically dispatches `conv2d_small_kernel_roi_step` in `pallas_fused` whenever a medium kernel's active ROI fits inside the `<= 100 KB/SM` SRAM budget (`roi_h * roi_w <= (Fh * Fw) // 3`, e.g., `24x32 = 768` or `32x32 = 1,024` cells where `16 * 1,024 * 4 B = 64 KB <= 100 KB/SM`). This avoids materializing the `34.60 MB` `(B, M, 64, 33)` `complex64` FFT tensor in HBM on compact ROIs (**1.80x faster** per-step backward `dX + dG` on `24x32`: `1.302 ms` $\to$ `0.725 ms`).
   - Replaces `jnp.einsum("bmhw,bhw->mhw", ...)` and `jnp.einsum("bmhw,mhw->bhw", ...)` in all RFFT backward passes with coalesced broadcast-multiply-`jnp.sum` reductions (eliminating XLA's `2,112`-batch `cgemmStridedBatched` cuBLAS dispatch and layout transposes: `1.452 ms` $\to$ `1.214 ms`, **1.20x faster** with `0.00e+00` bitwise difference), and prunes dead vertical rows prior to the backward horizontal 1D `irfft` in `_conv2d_pruned_rfft_bwd`.
3. **Temperature-Annealed & Straight-Through Estimator (STE) Discrete Codd State Hardening (`{0, 1, 2}`)**:
   - Introduces `SUPPORTED_CODD_MODES = ("continuous", "annealed", "ste_binary", "ste_ternary")`, `compute_codd_state_probs`, and `compute_annealed_temperature` in `neurocell/params.py` and `neurocell/trainer.py`.
   - Under `"ste_ternary"`, forward Codd gating probabilities $s_{\text{prob}}$ are quantized strictly to $\{0.0, 0.5, 1.0\}$ (mapping 1-to-1 to Codd roles `{0: Inactive, 1: Passive Signal Wire, 2: Active Nonlinear Logic Gate}` from `VonNeumannNetwork.threshold_states`), achieving **zero (`0.0`) post-training thresholding discrepancy** while propagating smooth temperature-scaled gradients $\nabla_S \mathcal{L} = \frac{1}{\tau} s_{\text{soft}}(1 - s_{\text{soft}}) \frac{\partial \mathcal{L}}{\partial s_{\text{prob}}}$ in the backward pass.
   - Because Codd gating is fused into `precompute_step_params` on the 2D `(H, W)` grid (`15 KB`) once per batch outside the depth loop, dynamic temperature annealing and STE hardening add **zero per-step Pallas kernel overhead** and require **zero XLA recompilation** when `state_temperature` changes across epochs.

---

## 2. Mathematical & Hardware Formulation

### 2.1 In-Register Fused Sparse Output-Port Backward Scatter
In `_pallas_std_out_bwd`, the gradient of the spatially standardized output port $\hat{Z}^{\text{out}}_{b,m} \in \mathbb{R}^{H_{\text{out}} \times W_{\text{out}}}$ with respect to the full ROI field $Z_{b,m} \in \mathbb{R}^{H_v \times W_v}$ is the sum of a dense affine projection term and a sparse port impulse:
$$\frac{\partial \mathcal{L}}{\partial Z_{b,m,r,c}} = -\sigma^{-1}_{b,m}\left(c_{1,b,m} + \hat{Z}_{b,m,r,c}\, c_{2,b,m}\right) + \sigma^{-1}_{b,m} \sum_{k=0}^{P-1} \mathbb{I}\left[(r, c) = (r_k, c_k)\right] G^{\text{out}}_{b,m,k}$$
Previously, `_make_sparse_port_bwd_kernel` wrote the `16.38 MB` dense base tensor to HBM and then invoked a separate XLA `.at[:, :, ro_r:ro_r+out_h, ro_c:ro_c+out_w].add(gz_port)` kernel, triggering a second read/write copy of `gz_bmhw`.

In Milestone 8 (`v2.3`), both `_make_sparse_port_bwd_kernel` (`grid = (B, M)`) and `_make_tiled_sparse_port_bwd_kernel` (`grid = (B, M, num_tiles)`) accept the static tuple `port_flat_indices` ($f_k = r_k W_v + c_k$ for either rectangular `PortSpec` or scattered `CoordinatePortSpec`) and accumulate the sparse port impulse directly in registers via `selp.f32` (`jnp.where`) before the single coalesced HBM store:
```python
for k, flat_idx in enumerate(port_flat_indices):
    gz_k = rstd_m * gz_out_ref[0, 0, k, 0]
    gz_base = jnp.where(idx == flat_idx, gz_base + gz_k, gz_base)
pl_triton.store(gz_base_ref.at[0, 0, :], gz_base, mask=mask)
```

### 2.2 SRAM-Resident Grouped-Batch Filter Gradients (`<= 100 KB/SM`)
When adaptive spatial masking (or a compact topology) reduces the active ROI area $H_v \times W_v$ such that:
$$H_v \times W_v \le \frac{F_h \times F_w}{3} \quad \text{and} \quad 64 < K_h \times K_w \le 512$$
padding $(H_v, W_v)$ up to the $(F_h, F_w) = (64, 64)$ FFT grid in `conv2d_hybrid_nopad_roi_step` materializes a `34.60 MB` `(B, M, 64, 33)` `complex64` buffer in HBM where $> 67\%$ of spatial pixels are zero padding.

By partitioning the batch $B = 128$ into $G_b = 8$ groups of $B_{\text{grp}} = 16$ channels (`64-byte` aligned per group), the spatial input tile per group has size:
$$16 \times H_v \times W_v \times 4\text{ bytes} \le 16 \times 1,024 \times 4\text{ bytes} = 64\text{ KB} \le 100\text{ KB/SM}$$
Together with the $M = 16$ directional filters (`19.1 KB`), the entire group working set (`83.1 KB <= 100 KB/SM`) stays resident inside the Ada Lovelace SM's `100 KB` Shared Memory / L1 cache, reducing per-step backward `dX + dG` time from `1.302 ms` to **`0.725 ms`** (**1.80x faster**) on `24x32` and reducing the full 4-step ALU training step from `6.83 ms` (`v2.2`) to **`5.34 ms`** (`v2.3`).

### 2.3 Straight-Through Estimator (STE) Discrete Codd State Hardening
Let $s \in \mathbb{R}^{H \times W}$ denote the raw Codd state parameters (after I/O port pinning) and $\tau > 0$ denote `state_temperature`. Define the temperature-scaled soft surrogate:
$$s_{\text{soft}}(s; \tau) = \sigma\!\left(\frac{s}{\tau}\right) = \frac{1}{1 + \exp(-s / \tau)}$$
Neurocell `v2.3` supports four Codd gating modes (`codd_mode`):
1. **`"continuous"`**: $s_{\text{prob}} = \sigma(s)$ ($\tau = 1$).
2. **`"annealed"`**: $s_{\text{prob}} = s_{\text{soft}}(s; \tau)$.
3. **`"ste_binary"`**:
   $$s_{\text{hard}} = \mathbb{I}(s \ge 0) \in \{0.0, 1.0\}, \qquad s_{\text{prob}} = s_{\text{soft}} + \text{stop\_gradient}(s_{\text{hard}} - s_{\text{soft}})$$
4. **`"ste_ternary"`** (3-state Codd role hardening matching `threshold_states` with `active_threshold` $\theta_{\text{act}} = 0.6$):
   $$s_{\text{hard}} = \begin{cases} 1.0 & \text{if } s \ge \theta_{\text{act}} \quad \text{(Role 2: Active Nonlinear Logic Gate)} \\ 0.5 & \text{if } 0 < s < \theta_{\text{act}} \quad \text{(Role 1: Passive Signal Wire)} \\ 0.0 & \text{if } s \le 0 \quad \text{(Role 0: Inactive / Pure Linear Cell)} \end{cases}$$
   $$s_{\text{prob}} = s_{\text{soft}} + \text{stop\_gradient}(s_{\text{hard}} - s_{\text{soft}})$$

---

## 3. RTX 4060 Laptop GPU Benchmark Results

### 3.1 Micro-Kernel Benchmarks
| Kernel / Operator | Milestone 7 (`v2.2`) | Milestone 8 (`v2.3`) | Speedup | Max Numerical Diff |
| :--- | :---: | :---: | :---: | :---: |
| **Sparse Output-Port Backward (`40x50`, `B=128, M=16`)** | `0.483 ms` | **`0.320 ms`** | **1.51x** | `2.38e-07` |
| **Plancherel `dG` Reduction (`40x50`, `B=128, M=16`)** | `1.452 ms` (`einsum`) | **`1.214 ms`** (`sum`) | **1.20x** | `0.00e+00` |
| **Pruned RFFT Backward Step (`40x50`, `B=128, M=16`)** | `1.998 ms` | **`1.814 ms`** | **1.10x** | `1.91e-06` |
| **Compact-ROI Backward `dX + dG` (`24x32`, `B=128, M=16`)** | `1.302 ms` (RFFT) | **`0.725 ms`** (SRAM Grouped) | **1.80x** | `< 1e-5` |

### 3.2 End-to-End Training Step & Epoch Benchmarks
| Model & Configuration | Legacy `v1` | `v2.1` (M6 Full ROI) | `v2.2` (M7 Adaptive) | `v2.3` (M8 Adaptive + SRAM) | Speedup vs `v2.2` | Speedup vs Legacy `v1` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **8-bit ALU (`depth=4, B=128, pallas_fused`)** | `34.78 ms` (`27.82 s/ep`) | `9.52 ms` (`7.62 s/ep`) | `6.83 ms` (`5.46 s/ep`) | **`5.34 ms` (`4.27 s/ep`)** | **1.28x** | **6.51x** |
| **8-bit ALU (`depth=8, B=128, pallas_fused`)** | — | `20.23 ms` (`16.18 s/ep`) | `14.54 ms` (`11.63 s/ep`) | **`11.70 ms` (`9.36 s/ep`)** | **1.24x** | — |
| **8-bit ALU (`depth=4, B=128, pallas_sram`)** | — | `12.61 ms` (`10.09 s/ep`) | — | **`9.61 ms` (`7.69 s/ep`)** | **1.31x** | — |
| **Multi-Tile `98x98` Grid (`96x96` ROI, `9,216` pix)** | — | `3.53 ms` (Full) | — | **`0.39 ms`** (Adaptive `16x32`) | **9.17x** (vs Full) | — |

### 3.3 Codd Hardening Mode Step Latency (`ALU depth=4, Adaptive 32x32, B=128`)
| Codd Mode (`codd_mode`) | Forward State Domain | Post-Training Thresholding Loss | Step Time (`ms/step`) | Epoch Time (`s/epoch`) |
| :--- | :---: | :---: | :---: | :---: |
| **`"continuous"`** | $(0, 1)$ | Non-zero if hard-thresholded post-hoc | `7.25 ms` | `5.80 s` |
| **`"annealed"` ($\tau = 0.3$)** | $(0, 1)$ sharpened | Vanishes as $\tau \to 0^+$ | `6.80 ms` | `5.44 s` |
| **`"ste_binary"` ($\tau = 0.3$)** | $\{0.0, 1.0\}$ | **`0.0` (exact binary)** | `6.70 ms` | `5.36 s` |
| **`"ste_ternary"` ($\tau = 0.3$)** | $\{0.0, 0.5, 1.0\} \leftrightarrow \{0, 1, 2\}$ | **`0.0` (exact 3-state Codd roles)** | **`6.08 ms`** | **`4.87 s`** |

---

## 4. Verification Summary
- All 33 unit tests in `tests/unit_*.py` pass (`Ran 33 tests in 404.132s — OK`), including the 5 new Milestone 8 tests in [`tests/unit_milestone8_pallas_and_codd.py`](file:///home/shakes/Work/neurocell-release/tests/unit_milestone8_pallas_and_codd.py).
