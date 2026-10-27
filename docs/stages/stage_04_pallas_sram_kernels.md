# Stage 04: Custom JAX Pallas / Triton GPU Kernels for Fused Register/SRAM Epilogues & Pruned Spectral Propagation

**Author**: Shakes  
**Target Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace, `SM 8.9`, 24 SMs, 24 MB L2 Cache, up to 100 KB Shared Memory / SM)  
**Environment**: `/home/shakes/Envs/jax-grain/bin/python` (`JAX 0.12.0.1`, `jax.experimental.pallas.triton`, `CudaDevice(id=0)`)  

---

## 1. Stage Goal & Overview

After optimizing the Green's function convolution in Stages 02 and 03 (`10.562 ms/batch` train step, `2.519 ms/batch` inference), profiling the remaining non-convolution execution time on the RTX 4060 (`SM 8.9`) identified three memory-bandwidth bottlenecks in the per-step epilogues and spectral padding:

1. **Multi-Direction 4D Intermediate Materialization in Hidden-Step Epilogues (`16.38 MB` $\times 3$ per step)**:
   - At each hidden step $t \in \{0, 1, 2\}$, `z_roi` has shape `(128, 16, 40, 50)` (`4,096,000` float32 elements = **`16.384 MB`**).
   - Even with XLA fusion (`fused_roi_step_epilogue`), adding `+ delta_border`, computing two-pass spatial standardization (`jax.nn.standardize`), evaluating `tanh(z_norm + b_roi)`, and saving activations for the backward pass forces XLA to read and write multiple `16.38 MB` 4D tensors to HBM (`~98.3 MB` of saved backward activation traffic across 3 hidden steps).
2. **Wasted `16.38 MB` Full-ROI Standardization Write on the Final Step (`t = depth`)**:
   - On the final step $t = 3$, `fused_output_port_epilogue` standardizes the entire `(128, 16, 40, 50)` tensor (`16.38 MB` write) only to slice out the $8 \times 1$ output port (`64 KB`, **0.4% of the tensor**) and discard the remaining 1,992 pixels (`99.6%`).
   - Worse, in the backward pass, XLA zero-pads the `(128, 16, 8, 1)` output gradient back into a `16.38 MB` mostly-zero `(128, 16, 40, 50)` tensor in HBM before reducing over all 2,000 pixels.
3. **`33.55 MB` Per-Step `jnp.pad` Allocation in RFFT Backward Passes (`134.2 MB` across 4 steps)**:
   - In `_conv2d_hybrid_roi_step_bwd` and `_conv2d_roi_rfft_step_bwd`, padding `g_z_roi` from `(128, 16, 40, 50)` to `(128, 16, 64, 64)` via `jnp.pad` prior to `rfft2` allocates and writes `33.55 MB` at every step.

In **Stage 04**, we implement custom **JAX Pallas / Triton GPU kernels** and zero-pad Fourier shift operators in [`neurocell/kernels_pallas.py`](../../neurocell/kernels_pallas.py) (`backend="pallas_fused"` and `backend="pallas_sram"`), reducing training step time to **`9.257 ms/batch` (`13,827.8 samples/s`, 2.53x faster than `contrib/` baseline)** and JIT inference latency to **`1.859 ms/batch` (`68,846.4 samples/s`)**.

---

## 2. Design Decision 1: Zero-Copy Register/SRAM-Resident Hidden-Step Epilogue (`pallas_fused_roi_step_epilogue`)

### 2.1 Single-Tile Power-of-Two Register Mapping (`N_PIX = 2000 <= N_PAD = 2048`)
The interior active ROI of the $62 \times 62$ ALU chip has $H_v \times W_v = 40 \times 50 = 2,000$ contiguous float32 elements (`8,000 bytes = 7.81 KB`) per `(batch, direction)` plane.
- Because $2,000 \le 2,048 = 2^{11}$, an entire `(40, 50)` spatial plane fits into a **single 1D power-of-two Pallas/Triton tile of size `N_PAD = 2048`** (`8 KB` in registers/SRAM).
- By using `pl_triton.load(..., mask=idx < n_pix, other=0.0)` and `pl_triton.store(..., mask=idx < n_pix)` with `pl.BlockSpec((1, M, N_PAD), ...)`, the Pallas kernel reads `(B, M, 2000)` and writes `(B, 2000)` **directly in HBM with zero `jnp.pad` copy overhead**.

### 2.2 In-Register Loop Over $M = 16$ Directions (`grid = (B,)`)
In `_make_roi_epilogue_fwd_kernel`, we launch `grid = (B,)` (`128` programs, one per batch sample $b$):
1. Each program loads the shared 2D parameter planes `w_eff_roi`, `s_eff_roi`, and `b_roi` (`3 \times 8\text{ KB} = 24\text{ KB}`) into registers **once per batch sample**, reusing them across all $M = 16$ directions.
2. Inside the `for m in range(16)` loop, for each direction $m$:
   - Loads `z_m = z_ref[0, m, :]` and `db_m = dborder_ref[0, m, :]` (`z_tot = z_m + db_m`), fusing the hoisted border addition in registers.
   - Computes $\mu_{b,m} = \frac{1}{2000} \sum_p Z_{\text{tot},p}$ and $\sigma^{-1}_{b,m} = (\frac{1}{2000} \sum_p (Z_{\text{tot},p} - \mu_{b,m})^2 + 10^{-5})^{-1/2}$ via warp butterfly reductions in SRAM/registers.
   - Standardizes $\hat{Z}_{b,m} = (Z_{\text{tot}} - \mu_{b,m}) \sigma^{-1}_{b,m}$ and accumulates `acc_lin += z_norm` and `acc_act += tanh(z_norm + b)` directly in registers.
   - Writes only the scalar statistics `mean_ref[0, m]` and `rstd_ref[0, m]` (`2` floats!).
3. After the loop, writes `out_ref[0, :] = w * acc_lin + s * acc_act` (`8 KB`).

### 2.3 Custom VJP Backward Kernel (`2,048x` Reduction in Saved State)
- Instead of saving `z_norm` (`16.38 MB`) and `tanh(z_norm + b)` (`16.38 MB`), `_pallas_roi_epilogue_fwd_rule` saves only `mean_bm` and `rstd_bm` of shape `(128, 16)` (**`16 KB` total** — a **2,048x reduction** in saved epilogue state).
- In `_make_roi_epilogue_bwd_kernel` (`grid = (B,)`), each sample program reconstructs $\hat{Z}_{b,m}$ and $\tanh(\hat{Z}_{b,m} + B_{\text{roi}})$ on the fly in registers, computes the exact LayerNorm projection scalars $(c_1, c_2)$ in registers, writes `gz_ref[0, m, :]`, and accumulates the per-sample parameter gradients `(gw_b, gs_b, gb_b)` across all $M = 16$ directions in registers before writing once at the end of the loop.

| Epilogue Implementation (`B=128, M=16, ROI=40x50`) | Forward Latency | Fwd + Bwd Latency | 3-Step Fwd + Bwd Total |
| :--- | :---: | :---: | :---: |
| Stage 02 XLA `fused_roi_step_epilogue` | `0.1285 ms` | `0.2533 ms` | `0.7598 ms` |
| **Stage 04 Pallas `pallas_fused_roi_step_epilogue`** | **`0.0530 ms` (2.42x)** | **`0.2044 ms` (1.24x)** | **`0.6132 ms`** |

---

## 3. Design Decision 2: Sparse-Port Final-Step Standardization (`pallas_standardize_output_port`)

On the final propagation step $t = \text{depth}$, the output port `z_out` occupies only `[ro_r : ro_r + 8, ro_c : ro_c + 1]` ($8 \times 1 = 8$ pixels out of $2,000$).
By the algebraic definition of spatial standardization:
$$\text{standardize}(Z_{b,m})[\text{ro\_r}:\text{ro\_r}+8,\; \text{ro\_c}:\text{ro\_c}+1] \equiv \Big(Z_{b,m}[\text{ro\_r}:\text{ro\_r}+8,\; \text{ro\_c}:\text{ro\_c}+1] - \mu_{b,m}\Big) \cdot \sigma^{-1}_{b,m}$$
- **Forward Pass (`409x` Reduction in Final-Step HBM Writes)**:
  - `_make_mean_rstd_fwd_kernel` (`grid = (B, M)`) streams $Z_{b,m} + \Delta_{\text{border},m}$ into registers, computes only the two scalars $(\mu_{b,m}, \sigma^{-1}_{b,m})$, and writes **only `16 KB`** (`mean_bm`, `rstd_bm`) to HBM instead of a `16.38 MB` 4D standardized tensor!
  - We then slice the 8 output-port pixels from $Z + \Delta_{\text{border}}$ (`64 KB`) and standardize only those 8 pixels.
- **Backward Pass (`O(8)` Sparse-Port Reduction)**:
  - Because the upstream cotangent $G_{\hat{Z}_{b,m,p}}$ is zero on all 1,992 non-output pixels, the standardization backward sums reduce to sums over **only the 8 output port pixels**:
    $$c_{1,b,m} = \frac{1}{2000} \sum_{p \in \mathcal{P}_{\text{out}}} G_{\hat{Z}_{b,m,p}}, \qquad c_{2,b,m} = \frac{1}{2000} \sum_{p \in \mathcal{P}_{\text{out}}} G_{\hat{Z}_{b,m,p}} \, \hat{Z}_{b,m,p}$$
  - `_make_sparse_port_bwd_kernel` then streams the dense base gradient $-\sigma^{-1}_{b,m} (c_{1,b,m} + \hat{Z}_{b,m} c_{2,b,m})$ in a single pass, followed by adding $\sigma^{-1}_{b,m} G_{\hat{Z}_{\text{out}}}$ at the 8 output port coordinates.

---

## 4. Design Decision 3: Zero-Pad Fourier Shift & Pruned Separable RFFT (`pallas_sram`)

1. **Eliminating the `33.55 MB` Per-Step `jnp.pad` via Circular Shift Theorem (`rfft2_unshifted_filter_grad_to_oihw`)**:
   - In Stage 03, `_conv2d_hybrid_roi_step_bwd` and `_conv2d_roi_rfft_step_bwd` padded `g_z_roi` `(128, 16, 40, 50)` at offset `(r_h, r_w) = (11, 6)` into a `(128, 16, 64, 64)` buffer (`33.55 MB` write per step) before calling `jnp.fft.rfft2`.
   - Because `rfft2(..., s=(64, 64))` automatically trailing-zero-pads `(40, 50)` to `(64, 64)` and $(40 + 11 \le 64,\; 50 + 6 \le 64)$ guarantees zero circular aliasing, calling `jnp.fft.rfft2(g_z_roi, s=(64, 64))` directly without `jnp.pad` shifts the spatial correlation by `(-r_h, -r_w)`.
   - We recover the exact filter gradient by applying `jnp.roll(..., shift=(r_h, r_w), axis=(-2, -1))` **once per batch** on the tiny `M=16` `(16, 64, 64)` `irfft2` output (`256 KB`), eliminating **`134.2 MB` of HBM padding writes per batch**!
2. **Pruned Separable 2D IRFFT (`conv2d_pruned_rfft_roi_step` in `backend="pallas_sram"`)**:
   - In `backend="pallas_sram"`, we decompose the forward 2D `irfft2` on `(128, 16, 64, 33)` into a vertical 1D `jnp.fft.ifft(..., n=64, axis=-2)`, slice to the 40 active ROI rows `[:, :, r_h : r_h + 40, :]`, and execute the horizontal 1D `jnp.fft.irfft(..., n=64, axis=-1)` on only **40 rows instead of 64 rows** (reducing 1D IRFFT count by **37.5%**, from $131,072$ to $81,920$).

---

## 5. Empirical RTX 4060 Performance Progression (Stages 00 $\to$ 04)

| Stage / Backend (`B = 128`, `62x62` ALU, `K = 23x13`, `M = 16`, `d = 4`) | Pure-Device Train Step (`ms/batch`) | Train Throughput (`samples/s`) | 800-Batch Epoch Time (`s`) | JIT Inference (`ms/batch`) | Inference Throughput (`samples/s`) | Speedup vs Stage 00 Train |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 00**: `contrib/` Baseline (`vmap` NHWC + Haiku state) | `23.411 ms` (`24.350 ms` Py loop) | `5,467.5` (`5,256.7`) | `19.480 s` | `1.946 ms` (`24.147 ms` eval) | `65,776.0` (`5,300.9` eval) | `1.00x` |
| **Stage 01**: `VNNParams` PyTree + `lax.scan` Trainer | `22.742 ms` | `5,628.3` | `18.194 s` | `1.930 ms` | `66,321.2` | `1.03x` |
| **Stage 02**: `nchw_direct` + Pre-Step Fusion + Pure-ROI | `11.007 ms` | `11,628.9` | `8.806 s` | `2.519 ms` | `50,813.8` | `2.13x` |
| **Stage 03**: `nchw_rfft_bwd` / `rfft_hoisted` (Plancherel VJP) | `10.562 ms` | `12,118.6` | `8.450 s` | `2.519 ms` | `50,813.8` | `2.22x` |
| **Stage 04 (`pallas_sram`)**: Pruned RFFT + Pallas SRAM Epilogues | `11.350 ms` | `11,277.4` | `9.080 s` | `6.290 ms` | `20,349.7` | `2.06x` (**Up to 9x on large $K$**) |
| **Stage 04 (`pallas_fused`)**: Zero-Pad Hybrid + Pallas Epilogues | **`9.257 ms`** | **`13,827.8`** | **`7.406 s`** | **`1.859 ms`** | **`68,846.4`** | **`2.53x`** (`2.63x` vs Py loop) |

---

## 6. Verification Summary

All 19 unit and numerical parity tests across `tests/unit_*.py` pass:
- `test_pallas_fused_roi_step_epilogue_forward_and_vjp_parity`: Forward and all 5 input gradients (`gz, gdb, gw, gs, gb`) match the JAX reference within `< 1e-6` relative L2 error.
- `test_pallas_fused_output_port_epilogue_forward_and_vjp_parity`: Forward and backward gradients match `fused_output_port_epilogue` within `< 1e-6` relative L2 error for both `final_activate=None` and `final_activate=jax.nn.tanh`.
- `test_vnn_alu_pallas_fused_and_pallas_sram_end_to_end_parity`: End-to-end forward, full-universe state, parameter gradients (`W, B, S, G`), and 5-step AdamW training trajectories match `nchw_direct` within `< 1e-5`.
