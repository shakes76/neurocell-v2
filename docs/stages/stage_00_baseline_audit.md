# Stage 00 Design Decision Record: Baseline Roofline Audit & Bottleneck Quantification

## 1. Context & Objective
Before writing any code for **Neurocell v2**, we audited and benchmarked the legacy paper-mapped implementation in `contrib/` ([`universe.py`](file:///home/shakes/Work/neurocell-release/contrib/universe.py), [`convs.py`](file:///home/shakes/Work/neurocell-release/contrib/convs.py), [`forward.py`](file:///home/shakes/Work/neurocell-release/contrib/forward.py), [`test_jax_vnn_alu_arithmetic.py`](file:///home/shakes/Work/neurocell-release/contrib/test_jax_vnn_alu_arithmetic.py), and [`test_jax_mlp_alu_arithmetic.py`](file:///home/shakes/Work/neurocell-release/contrib/test_jax_mlp_alu_arithmetic.py)) on the target hardware using the `~/Envs/jax-grain` environment.

- **Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (AD107, Ada Lovelace SM 8.9, 24 SMs, 8,188 MiB GDDR6 VRAM, 24 MB L2 Cache, 100 KB configurable Shared Memory/SM).
- **Software**: Linux, Python in `~/Envs/jax-grain`, `JAX 0.12.0.1` (`CudaDevice(id=0)`), `dm-haiku`, `optax`.
- **Target Task**: 8-bit ALU Arithmetic & Logic (`n = 8`, `ctrl_bits = 4`, `batch_size = 128`, `samples = 128,000` $\implies$ `800` train batches, `200` test batches).
  - **VNN Full Configuration**: Grid $N = 62 \times 62$, kernel size $K = 23 \times 13$, directions $M = 16$, depth $d = 4$, total parameters = **$16,316$** ($65.26\text{ KB}$ FP32).
  - **MLP Baseline Configuration**: 6 linear layers ($24 \to 24 \to 20 \to 16 \to 12 \to 10 \to 8$) with LayerNorm and ReLU, total parameters = **$2,022$**.

---

## 2. Empirical Baseline Measurements on RTX 4060 Laptop GPU

We measured four distinct execution modes for both `contrib/` VNN Full and `contrib/` MLP on the RTX 4060 Laptop GPU to separate **host/Python dispatch overhead** from **device/XLA kernel execution overhead**:

| Metric (Batch Size $B = 128$) | Legacy `contrib/` VNN Full ($16,316$ params) | Legacy `contrib/` MLP ($2,022$ params) | VNN vs. MLP Gap |
| :--- | :---: | :---: | :---: |
| **1. Pure Device JIT Train Step (`update`)** | **23.411 ms / batch**<br/>(`5,467.5 samples/s`) | **0.241 ms / batch**<br/>(`530,762.0 samples/s`) | **97.1x slower on GPU** |
| **2. Python Batch Loop Train Step (Driver)** | **24.350 ms / batch**<br/>(`19.48 s` / 800-batch epoch) | **17.863 ms / batch**<br/>(`14.29 s` / 800-batch epoch) | 1.36x (MLP is 98.6% Python-bound!) |
| **3. Un-JITted Evaluation Step (Driver)** | **24.147 ms / batch**<br/>(`5,301.0 samples/s` / `41.4 batches/s`) | **13.140 ms / batch**<br/>(`9,741.4 samples/s` / `76.1 batches/s`) | **12.4x slower than JIT VNN!** |
| **4. Pure Device JIT Inference Step** | **1.946 ms / batch**<br/>(`65,786.0 samples/s`) | **0.032 ms / batch**<br/>(`4,053,096.2 samples/s`) | **60.8x slower on GPU** |

### Key Insights from Empirical Profiling
1. **Why the MLP Driver Appeared to Take ~300s in Table 1**:
   - On the GPU itself, one training step of the 2,022-parameter MLP takes only **`0.241 ms`** (`0.193 s` per 800-batch epoch!).
   - However, [`contrib/test_jax_mlp_alu_arithmetic.py`](file:///home/shakes/Work/neurocell-release/contrib/test_jax_mlp_alu_arithmetic.py#L168-L192) runs an 800-step Python loop with per-batch array slicing (`x[..., jnp.newaxis]`) and un-JITted `network.apply` evaluation, inflating each step by **74x** from `0.241 ms` to `17.863 ms`!
2. **True Device-to-Device Gap**:
   - Comparing pure `@jax.jit` device execution, the legacy VNN Full training step (`23.411 ms/batch`) is **97.1x slower** than the MLP (`0.241 ms/batch`), and JITted VNN inference (`1.946 ms/batch`) is **60.8x slower** than the MLP (`0.032 ms/batch`).
3. **Note on `Throughput` Bug in Legacy Driver Scripts**:
   - In [`contrib/test_jax_vnn_alu_arithmetic.py:286`](file:///home/shakes/Work/neurocell-release/contrib/test_jax_vnn_alu_arithmetic.py#L286), `throughput = len(X_test) / eval_elapsed` divided by `len(X_test) = 200` (the number of **batches**) instead of `len(X_test) * batch_size = 25,600` (the number of **samples**), and timed an **un-JITted** `network.apply` loop (`24.147 ms/batch` $\implies 41.4\text{ batches/s}$). In Neurocell v2, we report both `samples/sec` and `batches/sec` using `@jax.jit`-compiled evaluation.

---

## 3. Quantitative Roofline & Bottleneck Analysis of `contrib/` VNN Full

### 3.1 Theoretical Compute Floor on RTX 4060 (15.1 TFLOPS FP32)
For $B = 128$, grid $(62, 62)$, `VALID` kernel $(23, 13)$, valid interior $(H_v, W_v) = (40, 50)$, $M = 16$ directions, and $d = 4$ depth steps:
- **Direct Convolution FLOPs per step**:
  $$\text{FLOPs}_{\text{conv, step}} = 128 \times 40 \times 50 \times 16 \times (2 \times 23 \times 13 \times 1) = \mathbf{2.449\text{ GFLOPs}}$$
  - Forward ($d = 4$): **$9.80\text{ GFLOPs}$** $\implies$ theoretical compute floor at $15.1\text{ TFLOPS}$ = **`0.65 ms`**.
  - Forward + Backward ($3 \times 9.80 = \mathbf{29.39\text{ GFLOPs}}$) $\implies$ theoretical compute floor = **`1.95 ms`** (vs. **`23.411 ms`** actual = **8.3% compute efficiency**).
- **Hoisted $64 \times 64$ 2D RFFT Convolution FLOPs**:
  - Forward ($d = 4$): **$1.17\text{ GFLOPs}$** (**8.35x fewer FLOPs** than direct convolution) $\implies$ theoretical compute floor = **`0.08 ms`** (Forward) / **`0.23 ms`** (Forward + Backward).

### 3.2 Six Root-Cause Architectural Bottlenecks Documented for Elimination
1. **Pathology 1 — Stateful Haiku `UniverseLP` & Full-Batch `hk.set_state("universe", cells)`**:
   - `_forward` calls `mlp.initialize(...)` on every step, and `mlp.__call__` writes the entire `(128, 62, 62)` batch universe (`1.97 MB`) into Haiku state.
   - Returning `state` from `loss` and `update` forces XLA to treat all $3,844$ cells at the final depth step $t=4$ as live outputs, preventing Dead-Code Elimination (DCE) of the $3,836$ non-output cells.
2. **Pathology 2 — Outer `jax.vmap` Over Single-Sample `B=1` `NHWC` ($C_{\text{in}}=1$) Convolutions**:
   - `jax.vmap` wraps `_fast_conv(sub_cells[jnp.newaxis, ...])` of shape `(1, 62, 62, 1)`.
   - In `NHWC`, the $M = 16$ direction axis has `stride = 1`, making the subsequent spatial standardization `jax.nn.standardize(result, axis=(0, 1))` reduce across non-contiguous strided memory (`stride = 16`).
3. **Pathology 3 — `jnp.pad` *Before* Codd Update & Direction Reduction (48% Dead Border)**:
   - `VALID` convolution outputs `(128, 40, 50, 16)` (`16.38 MB`), which fits inside the RTX 4060's **24 MB L2 cache**.
   - Calling `jnp.pad` *before* the Codd update inflates the tensor to `(128, 62, 62, 16)` (**31.49 MB** — exceeding L2 cache capacity!) and evaluates `tanh` + FMA across $1,844$ zero-padded border cells (**48.0% of the grid**) across all 128 samples and 16 directions.
4. **Pathology 4 — Separated `w, b, s` Tensors & Unfactored Per-Direction Gating**:
   - Executes 5 `dynamic_update_slice` calls per forward pass, recomputes `jax.nn.sigmoid(states)` at every depth step $t = 1 \dots 4$, and multiplies $(1 - \tilde{S})W$ inside the 16-direction tensor before `jnp.mean(axis=-1)`.
5. **Pathology 5 — Inner `@jax.jit` Decorators on Helper Functions**:
   - Prevents whole-step XLA fusion across `_filter`, `_fast_conv`, `_forward_full`, and `_propagate_full_mlp_list`.
6. **Pathology 6 — Python Per-Batch Training Loop & Un-JITted Evaluation**:
   - Causes 800 host-device dispatch round-trips per epoch and a 12.4x slowdown during accuracy evaluation.

---

## 4. Summary of Design Decisions Approved for Milestones 1–5

| Decision ID | Architectural Choice | Justification |
| :--- | :--- | :--- |
| **ADR-01** | Replace `hk.Module` (`UniverseLP`) with pure-functional `ChipTopology2D` + `VNNParams` `NamedTuple` PyTree | Eliminates `hk.set_state` `1.97 MB` HBM writes, enables final-step DCE, and passes natively into `jax.jit`, `jax.lax.scan`, and `optax` |
| **ADR-02** | Pack `[W, B, S_raw]` into `(3, H, W)` and store filters as `OIHW` `(M, 1, Kh, Kw)` | Coalesces parameter reads, eliminates runtime transposes, and supports pre-step fusion of `W_eff` and `S_eff` |
| **ADR-03** | Execute standardization, Codd gating, and direction reduction strictly on the `(40, 50)` valid ROI; slice to `(8, 1)` output port at step $t=d$ | Cuts 4D tensor footprint by 48% (`31.5 MB -> 16.4 MB`, fitting in 24 MB L2), cuts hidden-step `tanh` ops by 48%, and cuts final-step Codd ops by 480x |
| **ADR-04** | Implement both Batched `NCHW` Direct Conv (`nchw_direct`) and Hoisted $64\times 64$ Overlap-Save 2D RFFT (`rfft_hoisted`) | `NCHW` ensures contiguous spatial reductions; $64\times 64$ RFFT hoists $\hat{G}$ outside batch $B=128$ and depth $d=4$, cutting conv FLOPs by 8.35x with zero circular aliasing |
| **ADR-05** | Implement Custom JAX Pallas / Triton Kernels (`pallas_fused` and `pallas_sram`) | Fuses spatial standardization + `tanh` + $M=16$ reduction in registers (`16 KB` saved stats vs `16.4 MB` tensor) and fits the `62x62` universe + params (`66.1 KB`) inside `100 KB/SM` SRAM |
