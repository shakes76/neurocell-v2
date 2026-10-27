# Stage 06 — Milestone 7: Epoch-Adaptive Active-Region Pre-Compute Masking

## 1. Objective

At the start of training any Von Neumann Network (VNN), the final spatial extent of the self-assembled cellular circuit is **unknown a priori**. However, at the end of Epoch $e$, the learned cellular parameters $(W^{(e)}, B^{(e)}, S^{(e)})$, directional Green's function filters $G^{(e)}$, I/O port anchors, and optional empirical activation maps reveal the **currently active circuit region** for Epoch $e + 1$.

Milestone 7 introduces a cheap ($\sim 0.50\text{ ms}$), model-agnostic pre-epoch spatial mask estimator ([`compute_adaptive_spatial_mask`](file:///home/shakes/Work/neurocell-release/neurocell/adaptive_mask.py#L312-L466) in [`neurocell/adaptive_mask.py`](file:///home/shakes/Work/neurocell-release/neurocell/adaptive_mask.py)) that:
1. Detects salient Codd cells and I/O port anchors on the active ROI $(H_v, W_v)$ without assuming any a priori final circuit geometry.
2. Dilates the active envelope by a configurable **safety margin** (`margin`, default `4` cells in all directions) so frontier cells immediately outside the current circuit receive backward gradients and can be recruited into the circuit during Epoch $e + 1$.
3. Quantizes the active sub-grid dimensions $(H_{\text{act}}, W_{\text{act}})$ to multiples of `bucket_multiple = 8` with symmetric-slack alignment, tail-bucket absorption, and containment hysteresis (`prev_mask`) so XLA caches a small, fixed set of bucket shapes with zero recompilation churn.
4. Keeps full `VNNParams` (`(3, H, W)` and `(M, 1, Kh, Kw)`) and Optax optimizer states (`opt_state`) **shape-invariant** across epochs while slicing per-step convolutions and Pallas epilogues to `roi_bbox = (r0, r1, c0, c1)`.

---

## 2. Key Architectural Innovations

### 2.1 Fused Single-Dispatch Pre-Epoch Mask Estimator (`_compute_adaptive_mask_tensors`)
- [`_compute_adaptive_mask_tensors`](file:///home/shakes/Work/neurocell-release/neurocell/adaptive_mask.py#L169-L309) compiles effective Green's filter radius estimation (`estimate_effective_filter_radii`), Codd parameter salience detection, 2D morphological dilation (`_dilate_mask_2d` via `jax.lax.reduce_window`), bounding-box extraction, and hardware tile mask generation into a **single JIT-compiled GPU kernel** (`0.50 ms` execution latency, `< 0.01%` of an epoch).
- **Codd-Aware Salience above Background LayerNorm Drift**:
  - Because spatial standardization (`layernorm_2d`) couples all cells inside the active ROI via the spatial mean and variance, Adam/AdamW normalization ($\hat{m}_t / \sqrt{\hat{v}_t}$) induces a small uniform background drift across updated non-I/O cells.
  - `_compute_adaptive_mask_tensors` normalizes `bias_drive = |B| * sigmoid(S)` against the mean and standard deviation of updated non-I/O cells (`z_bias_act > salience_threshold`) and requires Codd-active states (`sigmoid(S) > 0.5`), preventing background LayerNorm drift from falsely triggering boundary expansion while immediately detecting true frontier circuit recruitment.

### 2.2 Safety Margin Dilation, Bucket Quantization, & Containment Hysteresis
- **Configurable Growth Margin (`margin = 4`)**: Morphological dilation adds a `margin`-cell border around all I/O port anchors and salient Codd cells so the circuit can grow outward by up to `margin` cells per epoch.
- **Canonical Bucket Alignment & Tail Absorption ([`_quantize_interval`](file:///home/shakes/Work/neurocell-release/neurocell/adaptive_mask.py#L101-L157))**:
  - Quantizes `(H_act, W_act)` to multiples of `bucket_multiple = 8` and aligns `r0, c0` to `bucket_multiple // 2` closest to the symmetric center so 1-pixel frontier growth stays inside the existing slack without sliding the bounding box.
  - Absorbs tail buckets when `max_len - q_len < bucket_multiple` (e.g., snapping `48` directly to `50` on a `40x50` ROI), avoiding a separate compilation for a `96%`-area sub-grid.
- **Monotonic Containment Hysteresis (`prev_mask`)**:
  - `AdaptiveMaskSpec` tracks both the quantized `roi_bbox` and the unquantized `raw_roi_bbox`. When `prev_mask` is supplied, `raw_roi_bbox` grows monotonically (`union(raw_bbox, prev_mask.raw_roi_bbox)`) and reuses `prev_mask.roi_bbox` whenever `raw_bbox` remains contained within `prev_mask.roi_bbox`, preventing quantization slack from compounding across epochs and ensuring already-trained frontier cells are never amputated.

### 2.3 Full-Grid Parameter Invariance & Pre-Sliced Hoisted Border Convolutions
- [`precompute_step_params`](file:///home/shakes/Work/neurocell-release/neurocell/params.py#L180-L246) computes weight sharpening and I/O pinning on the full `(H, W)` grid and slices `w_eff_roi[r0:r1, c0:c1]`, `s_eff_roi[r0:r1, c0:c1]`, and `b_roi[r0:r1, c0:c1]` to `(H_act, W_act)`. In the backward pass, JAX autodiff scatters cotangents directly back into the `[r_h+r0 : r_h+r1, r_w+c0 : r_w+c1]` slice of the full `(3, H, W)` `VNNParams` tensor.
- In [`propagate_vnn_2d`](file:///home/shakes/Work/neurocell-release/neurocell/propagator.py#L345-L357), `border_only_2d` is pre-sliced to `[b_r0 : b_r1 + kh - 1, b_c0 : b_c1 + kw - 1]` *before* the hoisted `B = 1` `VALID` convolution, producing `(1, M, H_act, W_act)` directly and eliminating XLA backward-convolution negative-padding legalization warnings.
- [`VNNTrainer`](file:///home/shakes/Work/neurocell-release/neurocell/trainer.py#L132-L237) provides `train_step_adaptive`, `train_epoch_adaptive`, `evaluate_packed_adaptive`, and `evaluate_metric_adaptive` with automatic per-`(roi_bbox, is_full_roi)` JIT caching and zero-overhead delegation to `self.train_epoch` when `mask_spec.is_full_roi` is `True`.

---

## 3. Verified GPU Benchmark Results (NVIDIA RTX 4070 Laptop GPU, FP32)

| Benchmark / Workload | Full ROI (`v2.1`) | Adaptive Mask (`v2.2`, `margin=4`) | Active Sub-ROI & Area Fraction | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **Pre-Epoch Mask Estimator (`compute_adaptive_spatial_mask`)** | — | **`0.50 ms / epoch`** | `(32, 32)` on `(40, 50)` (`51.2%`) | `< 0.01%` of epoch time |
| **8-bit ALU Chip (`depth = 4, B = 128`)** | `9.52 ms/step` (`7.62 s/ep`) | **`6.83 ms/step` (`5.46 s/ep`)** | `(32, 32) = 1,024 / 2,000` (`51.2%`) | **$1.39\times$ vs `v2.1`** (**$5.09\times$ vs `contrib/`**) |
| **8-bit ALU Chip (`depth = 8, B = 128`)** | `20.23 ms/step` (`16.18 s/ep`) | **`14.54 ms/step` (`11.63 s/ep`)** | `(32, 32) = 1,024 / 2,000` (`51.2%`) | **$1.39\times$ vs `v2.1`** |
| **Large-Grid VNN (`82x82` grid, `80x80` ROI, `K = 3x3, depth = 4, B = 128`)** | `14.56 ms/step` | **`0.87 ms/step`** | `(24, 32) = 768 / 6,400` (`12.0%`) | **$16.71\times$ vs `v2.1`** |

---

## 4. Multi-Epoch Circuit Growth & Accuracy Verification

### 4.1 10-Epoch Early-Stage Trajectory (`12,800` samples, `optax.adamw(0.001)`, `margin = 4`)
During early epochs, the 8-bit ALU circuit operates inside the `(32, 32)` bucket (`51.2%` of the `40x50` ROI, running **$1.37\times$ faster** per epoch: `0.566s` vs `0.786s`) and automatically expands to `(40, 32)` (`64.0%`) in Epoch 5 and `(40, 40)` (`80.0%`, `0.685s/ep`) in Epoch 6 as frontier cells are recruited:

| Epoch | Full ROI (`40x50`) Loss | Adaptive (`margin=4`) Loss | Full ROI Test Acc | Adaptive Test Acc | Adaptive Active Shape (% of ROI) | Full ROI Epoch Time | Adaptive Epoch Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `0.159135` | `0.152498` | `1.64%` | `4.73%` | `(32, 32)` (`51.2%`) | `0.779s` | **`0.566s`** |
| **2** | `0.136425` | `0.124553` | `19.49%` | `43.32%` | `(32, 32)` (`51.2%`) | `0.788s` | **`0.589s`** |
| **3** | `0.112922` | `0.102053` | `52.34%` | `53.28%` | `(32, 32)` (`51.2%`) | `0.787s` | **`0.577s`** |
| **4** | `0.099366` | `0.096975` | `53.28%` | `53.20%` | `(32, 32)` (`51.2%`) | `0.787s` | **`0.587s`** |
| **5** | `0.094381` | `0.094815` | `53.09%` | `53.36%` | `(40, 32)` (`64.0%`) | `0.784s` | `0.630s` (cached) |
| **8** | `0.079506` | `0.084876` | `53.52%` | `53.12%` | `(40, 40)` (`80.0%`) | `0.787s` | **`0.689s`** |
| **10** | `0.071228` | `0.078978` | `54.02%` | `53.87%` | `(40, 40)` (`80.0%`) | `0.787s` | **`0.685s`** |

### 4.2 Backend Parity under Spatial Masking (`< 1e-5` Relative $L_2$ Error)
[`tests/unit_milestone7_adaptive_mask.py`](file:///home/shakes/Work/neurocell-release/tests/unit_milestone7_adaptive_mask.py) verifies that executing `VonNeumannNetwork.apply(..., spatial_mask=mask_spec)` produces identical forward predictions, full `(B, H, W)` universe states, and backward parameter gradients (`< 1e-5` relative $L_2$ error) across all 6 backends (`nchw_direct`, `rfft_hoisted`, `nchw_rfft_bwd`, `pallas_fused`, `pallas_sram`, and `auto`) while keeping `VNNParams` shapes `(62, 62)` and `(16, 1, 23, 13)` invariant.

---

## 5. Verification Suite

All 28 unit tests across 7 test modules pass (`Ran 28 tests — OK`):
- [`tests/unit_topology.py`](file:///home/shakes/Work/neurocell-release/tests/unit_topology.py)
- [`tests/unit_params.py`](file:///home/shakes/Work/neurocell-release/tests/unit_params.py)
- [`tests/unit_epilogue.py`](file:///home/shakes/Work/neurocell-release/tests/unit_epilogue.py)
- [`tests/unit_kernels_greens.py`](file:///home/shakes/Work/neurocell-release/tests/unit_kernels_greens.py)
- [`tests/unit_kernels_pallas.py`](file:///home/shakes/Work/neurocell-release/tests/unit_kernels_pallas.py)
- [`tests/unit_vnn_alu_parity.py`](file:///home/shakes/Work/neurocell-release/tests/unit_vnn_alu_parity.py)
- [`tests/unit_milestone6_general_vnn.py`](file:///home/shakes/Work/neurocell-release/tests/unit_milestone6_general_vnn.py)
- [`tests/unit_milestone7_adaptive_mask.py`](file:///home/shakes/Work/neurocell-release/tests/unit_milestone7_adaptive_mask.py)
