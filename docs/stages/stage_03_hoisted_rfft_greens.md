# Stage 03: Hoisted $64\times 64$ Overlap-Save 2D RFFT & Hybrid Plancherel Green's Propagators

**Author**: Shakes  
**Target Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace, `SM 8.9`, 24 SMs, 24 MB L2 Cache)  
**Environment**: `/home/shakes/Envs/jax-grain/bin/python` (`JAX 0.12.0.1`, `CudaDevice(id=0)`)  

---

## 1. Stage Goal & Overview

In Stage 02, converting the Green's function convolution from `jax.vmap(B=1)` `NHWC` to batched `NCHW` on the pure $(40, 50)$ active ROI (`backend="nchw_direct"`) reduced the pure-device training step latency from `23.411 ms/batch` to `11.007 ms/batch` (`2.519 ms` forward + `8.488 ms` backward).

Profiling the `11.007 ms` training step revealed two fundamental hardware properties of Green's function propagation on Ada Lovelace (`SM 8.9`):
1. **In the Backward Pass, Filter Gradient Reduction ($\nabla_G \mathcal{L}$) Dominates**:
   - A single forward `NCHW` convolution on `(128, 1, 40, 50)` with `(16, 1, 23, 13)` takes only **`0.354 ms`** (`1.416 ms` across 4 steps).
   - However, XLA's default backward convolution (`dX` + `dG`) takes **`2.173 ms/step`** (`8.693 ms` across 4 steps): `0.719 ms/step` for `cudnnConvolutionBackwardData` ($\nabla X$) and **`1.312 ms/step`** for `cudnnConvolutionBackwardFilter` ($\nabla G$), where 256,000 spatial positions per step contend via split-K/atomic reductions into a tiny $16 \times 1 \times 23 \times 13$ (`19.1 KB`) filter tensor.
2. **Spatial Convolution Scales as $O(K_h K_w)$, Whereas 2D RFFT Scales as $O(1)$ in Kernel Size**:
   - As VNN receptive fields grow beyond $(23, 13)$ (e.g., $(27, 27)$, $(31, 31)$, $(37, 37)$ for long-range cellular routing), cuDNN spatial convolution latency explodes from `10.48 ms` to **`88.11 ms`** per 4-step pass.

In **Stage 03**, we implement two spectral Green's function propagation backends in [`neurocell/kernels_greens.py`](../../neurocell/kernels_greens.py) and [`neurocell/propagator.py`](../../neurocell/propagator.py):
1. **`"rfft_hoisted"`**: Hoisted $64\times 64$ Overlap-Save 2D Real FFT (`rfft2`) forward and Plancherel `custom_vjp` backward propagation, achieving **$O(1)$ complexity in kernel size** and up to **8.85x training speedup** on large Green's function receptive fields.
2. **`"nchw_rfft_bwd"`**: A hybrid `custom_vjp` backend that pairs direct `NCHW` forward convolution (`0.354 ms/step`) and explicit forward-convolution input gradients (`0.582 ms/step`, skipped at $t=0$) with hoisted $64\times 64$ RFFT Plancherel filter gradients (`0.938 ms/step`), reducing the $(23, 13)$ ALU training step further to **`10.562 ms/batch` (`12,118.6 samples/s`, 2.22x faster than `contrib/` baseline)**.

---

## 2. Design Decision 1: Alias-Free $64\times 64$ Overlap-Save 2D RFFT on the $(40, 50)$ ROI

### 2.1 Zero Circular Wrap-Around Proof
By the Pure-ROI decomposition proved in Stage 02, each step requires the `SAME`-padded cross-correlation of the interior active ROI $\mathring{C}^{(t)}_b \in \mathbb{R}^{40 \times 50}$ with $G \in \mathbb{R}^{16 \times 1 \times 23 \times 13}$.
- The linear convolution of a $(40, 50)$ signal with a spatially flipped $(23, 13)$ filter has support:
  $$(H_{\text{lin}}, W_{\text{lin}}) = (40 + 23 - 1,\; 50 + 13 - 1) = (62, 62)$$
- Because $62 \le 64$ along both axes, embedding $\mathring{C}^{(t)}_b$ into a power-of-two $64 \times 64$ FFT grid (`s=(64, 64)`) produces **zero circular aliasing** anywhere in the $(62, 62)$ support.
- The exact `SAME`-padded $(40, 50)$ output sits at slice `[r_h : r_h + 40, r_w : r_w + 50] = [11:51, 6:56]`:
  $$Z^{(t)}_{b,m} = \text{irfft2}\!\Big(\text{rfft2}\big(\mathring{C}^{(t)}_b,\, s=(64,64)\big) \odot \hat{G}_m,\, s=(64,64)\Big)[11:51,\; 6:56]$$
  where $\hat{G}_m = \text{rfft2}(\text{flip}(G_{m,0}, \text{axes}=(-2,-1)), s=(64,64)) \in \mathbb{C}^{64 \times 33}$ is **hoisted once per batch** outside the depth loop (`hoist_filter_rfft2`).

---

## 3. Design Decision 2: Plancherel Frequency-Domain Backward Reduction (`@jax.custom_vjp`)

### 3.1 The Naive Autodiff Bottleneck on `irfft2`
If JAX differentiates naively through `conv2d_roi_rfft_step`:
1. Forward saves `out_freq` of shape `(128, 16, 64, 33)` (`34.60 MB` per step) for the backward pass.
2. Backward runs `rfft2` on `g_padded` `(128, 16, 64, 64)` (`2,048` 2D RFFTs) and then runs `irfft2` on all $M=16$ channels before summing over $M$ (`2,048` 2D IRFFTs).

### 3.2 Plancherel Reduction Before `irfft2`
By linearity of the inverse Fourier transform and Plancherel's theorem, we attach custom VJPs to `conv2d_roi_rfft_step` and `hoist_filter_rfft2`:
1. **Activation Checkpointing Reduction (`34.60 MB` $\to$ `1.08 MB`)**:
   - `_conv2d_roi_rfft_step_fwd` saves only `cell_freq` $\in \mathbb{C}^{128 \times 64 \times 33}$ (`1.08 MB`) and `ker_freq_mhw` $\in \mathbb{C}^{16 \times 64 \times 33}$ (`135 KB`), reducing saved backward state by **28x**.
2. **16x Fewer Inverse FFTs for $\nabla_{\mathring{C}^{(t)}} \mathcal{L}$**:
   - Given the padded upstream cotangent $\hat{g}^{(t)} = \text{rfft2}(\text{pad}(g_{Z^{(t)}}), s=(64,64)) \in \mathbb{C}^{B \times M \times 64 \times 33}$, we reduce over $M=16$ directions **in the frequency domain** via a complex inner product:
     $$\widehat{\nabla \mathring{C}^{(t)}}_b[u, v] = \sum_{m=0}^{M-1} \hat{g}^{(t)}_{b,m}[u, v] \cdot \overline{\hat{G}_m[u, v]}$$
   - We then execute only **$B = 128$** `irfft2` transforms (instead of $B \times M = 2,048$) to recover $\nabla \mathring{C}^{(t)} \in \mathbb{R}^{B \times 40 \times 50}$.
   - Furthermore, at $t = 0$, $\mathring{C}^{(0)}$ is the constant input tensor (`inputs`), which requires no gradient; passing `need_dx=(t > 0)` skips the $t=0$ input gradient completely.
3. **128x Fewer Inverse FFTs for $\nabla_G \mathcal{L}$ (Zero Atomic Contention)**:
   - Each step reduces over the batch dimension $B = 128$ in the frequency domain via `jnp.einsum("bmhw,bhw->mhw", g_freq, jnp.conj(cell_freq))`.
   - JAX's autodiff sums these `(16, 64, 33)` frequency cotangents across all $d=4$ steps and passes their sum into `_hoist_filter_rfft2_bwd`, which executes **only $M = 16$ `irfft2` transforms once per batch** (`rfft2_filter_grad_to_oihw`)!
   - Furthermore, because `rfft2`/`irfft2` executes in IEEE FP32 (`23` mantissa bits) rather than cuDNN's default TF32 Tensor Core reduction (`10` mantissa bits), `rfft_hoisted` achieves `< 1e-5` relative L2 error against an IEEE FP32 reference (`jax.lax.Precision.HIGHEST`).

---

## 4. Design Decision 3: Hybrid NCHW-Forward / Plancherel-RFFT-Backward (`nchw_rfft_bwd`)

Micro-benchmarking the individual forward, backward-data ($\nabla X$), and backward-filter ($\nabla G$) passes on `(23, 13)` (`B=128, M=16, ROI=40x50`) on the RTX 4060 revealed:

| Single-Step Operation (`B=128, M=16, ROI=40x50, K=23x13`) | Latency / Step (`ms`) | 4-Step Total (`ms`) | Why |
| :--- | :---: | :---: | :--- |
| **Forward: Direct NCHW Conv (`conv_general_dilated`)** | **`0.3541 ms`** | **`1.416 ms`** | Reads `1 MB`, writes `16 MB` in a single cuDNN kernel |
| Forward: $64\times 64$ RFFT (`rfft2` + `irfft2`) | `1.4828 ms` | `5.931 ms` | `irfft2` transforms $2,048$ slices (`34.6 MB` complex $\to$ `33.6 MB` real) |
| **Bwd $\nabla X$: Explicit Forward Conv (flipped $G_{\text{IOHW}}$)** | **`0.5820 ms`** | **`1.746 ms`** (3 steps) | Uses cuDNN forward implicit-GEMM ($C_{\text{in}}=16 \to C_{\text{out}}=1$), skipped at $t=0$ |
| Bwd $\nabla X$: XLA Default `cudnnConvolutionBackwardData` | `0.7189 ms` | `2.157 ms` (3 steps) | Slower backward-data kernel |
| **Bwd $\nabla G$: $64\times 64$ RFFT Plancherel Reduction** | **`0.9385 ms`** | **`3.754 ms`** | Coalesced cuBLAS reduction over $B=128$ in freq domain + 16 `irfft2` once |
| Bwd $\nabla G$: XLA Default `cudnnConvolutionBackwardFilter` | `1.3121 ms` | `5.249 ms` | Split-K atomic contention into `19.1 KB` filter tensor |

By combining the fastest algorithm for each of the three phases into `conv2d_hybrid_roi_step` (`backend="nchw_rfft_bwd"`):
- **Forward**: Direct `NCHW` cuDNN convolution (`0.354 ms/step`).
- **Backward $\nabla X$**: Explicit forward `NCHW` convolution with spatially flipped and channel-transposed filter `G_flip_iohw` (`0.582 ms/step` for $t \in \{1, 2, 3\}$, skipped at $t=0$).
- **Backward $\nabla G$**: $64\times 64$ RFFT Plancherel inner product accumulated across all 4 steps into `hoist_filter_rfft2_zero_fwd` (`0.938 ms/step`).

This hybrid backend achieves **`10.562 ms/batch`** (`12,118.6 samples/s`) on the `(23, 13)` ALU chip with **bit-exact `0.0` forward loss and `W/B/S` gradient parity** against `nchw_direct` and `3.33e-7` relative `G` gradient parity.

---

## 5. Empirical RTX 4060 Crossover Benchmarks Across Kernel Sizes

Below is the measured 4-step convolution benchmark (`B=128, M=16, ROI=40x50`) comparing `nchw_direct` against `rfft_hoisted` as the Green's function kernel size increases from $(15, 15)$ to $(45, 45)$:

| Kernel Size $(K_h \times K_w)$ | MACs / Pixel | `nchw_direct` Forward (4 steps) | `rfft_hoisted` Forward (4 steps) | `nchw_direct` Fwd+Bwd (4 steps) | `rfft_hoisted` Fwd+Bwd (4 steps) | `rfft_hoisted` Speedup (Fwd+Bwd) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`15 x 15`** | $225$ | **`1.442 ms`** | `5.972 ms` | **`8.049 ms`** | `10.144 ms` | `0.79x` |
| **`23 x 13` (ALU)** | $299$ | **`1.448 ms`** | `5.931 ms` | **`10.476 ms`** (`10.562 ms` full hybrid train) | `10.827 ms` (`11.862 ms` full RFFT train) | `0.97x` (**`1.04x` hybrid**) |
| **`27 x 27`** | $729$ | **`4.859 ms`** | `6.065 ms` | `32.596 ms` | **`10.425 ms`** | **3.13x** |
| **`31 x 31`** | $961$ | `6.307 ms` | **`5.970 ms`** | `64.847 ms` | **`10.011 ms`** | **6.48x** |
| **`37 x 37`** | $1,369$ | `8.598 ms` | **`6.033 ms`** | `88.107 ms` | **`9.953 ms`** | **8.85x** |
| **`45 x 45`** | $2,025$ | `16.860 ms` | **`6.379 ms`** | `35.838 ms` | **`9.969 ms`** | **3.60x** |

### Key Takeaways from the Crossover Table
1. **Exact Crossover at $K_h K_w \approx 300$ (`23x13`)**:
   - The ALU chip's $(23, 13)$ kernel ($299$ taps) sits right at the hardware crossover point between spatial convolution (`10.48 ms`) and $64\times 64$ RFFT convolution (`10.83 ms`).
   - On $(23, 13)$, our **Hybrid `nchw_rfft_bwd`** backend achieves the best overall training throughput (**`10.562 ms/batch`**, down from `23.411 ms/batch` in Stage 00 = **2.22x speedup**), while `nchw_direct` achieves the best inference throughput (**`2.519 ms/batch`**).
2. **Flat $O(1)$ Scaling of `rfft_hoisted` (`~10.0 ms` from `15x15` to `45x45`)**:
   - Because `rfft2` and `irfft2` always operate on the fixed $(64, 64)$ grid regardless of $(K_h, K_w)$, `rfft_hoisted` takes `~6.0 ms` in forward and `~10.0 ms` in forward+backward across all kernel sizes up to $H_v + K_h - 1 \le 64$, achieving up to **8.85x speedup** over `nchw_direct` on large receptive fields.

---

## 6. Verification Summary

All 16 unit and numerical parity tests across `tests/unit_topology.py`, `tests/unit_params.py`, `tests/unit_epilogue.py`, `tests/unit_vnn_alu_parity.py`, and `tests/unit_kernels_greens.py` pass:
- `test_valid_rfft_matches_legacy_fast_conv_and_nchw`: Forward relative L2 error `< 1e-6`, backward `dX` and `dG` relative L2 error `< 1e-5` against IEEE FP32 reference.
- `test_roi_rfft_step_forward_and_custom_vjp_parity`: Forward and `custom_vjp` backward parity verified for both `need_dx=True` and `need_dx=False`.
- `test_hybrid_roi_step_forward_and_custom_vjp_parity`: Bit-exact `0.0` forward parity and `< 1e-5` backward gradient parity.
- `test_vnn_alu_rfft_hoisted_and_hybrid_end_to_end_parity`: Full 4-step ALU VNN forward, universe-state, and parameter gradient parity (`W, B, S, G`) verified with non-zero biases.
