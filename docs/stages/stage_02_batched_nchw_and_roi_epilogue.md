# Stage 02 Design Decision Record: Batched `NCHW` Direct Convolution, Pure-ROI Decomposition & Output-Only Final Step

## 1. Context & Problem Statement
In Stage 00 ([`stage_00_baseline_audit.md`](file:///home/shakes/Work/neurocell-release/docs/stages/stage_00_baseline_audit.md)), we identified that the legacy `contrib/` VNN Full operator ([`forward._propagate_full_mlp_list`](file:///home/shakes/Work/neurocell-release/contrib/forward.py#L162) and [`forward._forward_full`](file:///home/shakes/Work/neurocell-release/contrib/forward.py#L737)) took **`23.411 ms/batch`** (`19.48 s` per 800-batch epoch) in training and **`24.147 ms/batch`** in un-JITted evaluation due to four device-side bottlenecks:
1. Wrapping single-sample `(1, 62, 62, 1)` `NHWC` convolutions inside `jax.vmap(B=128)`, causing strided (`stride=16`) spatial standardization reductions.
2. Calling `jnp.pad` *before* the Codd update and direction reduction, expanding the 4D intermediate tensor from `(128, 40, 50, 16)` (**16.38 MB**, fitting in the RTX 4060's **24 MB L2 cache**) to `(128, 62, 62, 16)` (**31.49 MB**, thrashing L2 cache!) and evaluating `tanh` + FMA across **48.0% dead zero-padded border cells**.
3. Evaluating $(1 - \tilde{S})W$ and $\tilde{S}$ unfactored inside the $M=16$ direction axis.
4. Computing the full $62 \times 62 = 3,844$ cellular grid at the final step $t = d = 4$ and writing `(128, 62, 62)` to Haiku state, even though the loss function only reads **8 output cells** `(128, 8, 1)`.

---

## 2. Design Decisions & Hardware/Mathematical Justifications

### Decision 2.1: Native Batched `NCHW` Direct Convolution ([`neurocell/kernels_greens.py`](file:///home/shakes/Work/neurocell-release/neurocell/kernels_greens.py))
- **Decision**: Eliminate `jax.vmap` and drive `jax.lax.conv_general_dilated` directly with dimension numbers `("NCHW", "OIHW", "NCHW")` (`cells_nchw` of shape `(B, 1, H_in, W_in)` and `filters_oihw` of shape `(M, 1, Kh, Kw)`).
- **Justification**:
  - In `NCHW`, each `(b, m)` spatial plane `(H_v, W_v) = (40, 50)` is stored as $2,000$ contiguous `float32` elements in memory, enabling coalesced 128-byte cache-line reads and warp-shuffle reductions during `jax.nn.standardize(z, axis=(-2, -1))`.

### Decision 2.2: Exact Pure-ROI `(B, 40, 50)` Decomposition with Hoisted `B=1` Border Convolution ([`neurocell/propagator.py`](file:///home/shakes/Work/neurocell-release/neurocell/propagator.py))
- **Discovery During Stage 02 Profiling**:
  - In our initial Stage 02 prototype (`Stage 02A`), we kept the hidden cellular field `cells` at full `(128, 62, 62)` shape by embedding `roi_next` `(128, 40, 50)` into a broadcasted border template `c_border.at[:, 11:51, 6:56].set(roi_next)`.
  - Profiling on the RTX 4060 revealed that `Stage 02A` reduced training step time from `23.411 ms` to `16.857 ms`, because in the backward pass XLA still had to differentiate `(128, 1, 62, 62)` convolutions and reduce gradients across `B=128` into `c_border` (`(128, 62, 62)`)!
- **Mathematical Solution (`Stage 02B` — Pure-ROI Decomposition)**:
  - For any step $t \ge 1$, the full-grid cellular field $C^{(t)}_b \in \mathbb{R}^{62 \times 62}$ is the sum of its interior ROI $\mathring{C}^{(t)}_b \in \mathbb{R}^{40 \times 50}$ (zero-padded by $r_h=11, r_w=6$) and the batch-invariant 2D border field $C_{\text{border, only}} \in \mathbb{R}^{62 \times 62}$ (which equals $\tanh(B) \odot \sigma(S)$ outside $[11:51, 6:56]$ and $0$ inside $[11:51, 6:56]$).
  - By linearity of the Green's function convolution:
    $$\text{conv}_{\text{VALID}}\!\left(C^{(t)}_b, \; G\right) \equiv \text{conv}_{\text{SAME}}\!\left(\mathring{C}^{(t)}_b, \; G\right) + \underbrace{\text{conv}_{\text{VALID}}\!\left(C_{\text{border, only}}, \; G\right)}_{\Delta_{\text{border}} \in \mathbb{R}^{1 \times M \times 40 \times 50}}$$
  - And at step $t = 0$, because all three ALU input ports (`x1` at `(23, 23)`, `x2` at `(32, 23)`, `ctrl` at `(18, 28)`) lie strictly inside $[11:51, 6:56]$, the outer border of $C^{(0)}_b$ is identically zero:
    $$\text{conv}_{\text{VALID}}\!\left(C^{(0)}_b, \; G\right) \equiv \text{conv}_{\text{SAME}}\!\left(\mathring{C}^{(0)}_b, \; G\right)$$
- **Hardware Impact**:
  1. **48% Smaller Convolution Input Domain**: Every batched convolution ($B=128$) operates on **`(128, 1, 40, 50)` ($2,000$ cells)** instead of `(128, 1, 62, 62)` ($3,844$ cells)!
  2. **128x Smaller Border Convolution**: $\Delta_{\text{border}}$ has batch size **`B = 1`** and is computed **once per batch** outside the depth loop (`1` convolution instead of $128 \times 3 = 384$ border convolutions!).
  3. **Zero `.at[...].set` Slice Copies in Hidden Steps**: `roi_cells` remains `(128, 40, 50)` (`1.024 MB`) throughout the network, dropping training step latency from `16.857 ms` down to **`11.007 ms/batch` (2.13x faster than baseline)**!

### Decision 2.3: Factored ROI Epilogue & Output-Only Final Step ([`neurocell/epilogue.py`](file:///home/shakes/Work/neurocell-release/neurocell/epilogue.py))
- **Decision**:
  - In [`fused_roi_step_epilogue`](file:///home/shakes/Work/neurocell-release/neurocell/epilogue.py#L24-L53), compute $\Sigma^{\text{lin}} = \sum_{m=0}^{M-1} \hat{Z}_m$ and $\Sigma^{\text{act}} = \sum_{m=0}^{M-1} \tanh(\hat{Z}_m + B_{\text{roi}})$ on `(B, 40, 50)` and combine via $W_{\text{eff}} \Sigma^{\text{lin}} + S_{\text{eff}} \Sigma^{\text{act}}$.
  - In [`fused_output_port_epilogue`](file:///home/shakes/Work/neurocell-release/neurocell/epilogue.py#L75-L115), at step $t = d$, slice `z_final` to `roi_out_offset = (16, 35)` of shape `(B, M, 8, 1)` immediately after spatial standardization, reducing final-step Codd work from $3,844$ cells to **$8$ cells (480x reduction)**.

### Decision 2.4: Preserving Input Slice Rank in `inject_inputs_2d`
- **Discovery During Parity Verification**:
  - In [`contrib/test_jax_vnn_alu_arithmetic.py`](file:///home/shakes/Work/neurocell-release/contrib/test_jax_vnn_alu_arithmetic.py#L90-L124), `ctrl_size` is declared as `(1, 8)` for parameter initialization (`mlp.initialize`), but the runtime input batch passes `C[:, :, jnp.newaxis]` of shape `(B, 8, 1)` into `jax.lax.dynamic_update_slice`.
  - In [`inject_inputs_2d`](file:///home/shakes/Work/neurocell-release/neurocell/propagator.py#L25-L49) and [`inject_inputs_roi_2d`](file:///home/shakes/Work/neurocell-release/neurocell/propagator.py#L52-L87), when a 3D tensor `(B, ih, iw)` is provided, we preserve its explicit `(ih, iw)` slice dimensions so runtime injection matches `jax.lax.dynamic_update_slice` identically.

---

## 3. Verification & Empirical Benchmark Results (RTX 4060 Laptop GPU)

All **12 unit and parity tests** across [`tests/unit_topology.py`](file:///home/shakes/Work/neurocell-release/tests/unit_topology.py), [`tests/unit_params.py`](file:///home/shakes/Work/neurocell-release/tests/unit_params.py), [`tests/unit_epilogue.py`](file:///home/shakes/Work/neurocell-release/tests/unit_epilogue.py), and [`tests/unit_vnn_alu_parity.py`](file:///home/shakes/Work/neurocell-release/tests/unit_vnn_alu_parity.py) pass:

| Parity Verification Test (`B=32`, Non-Zero Biases) | Reference (`contrib/`) | Measured Error (`nchw_direct`) |
| :--- | :--- | :---: |
| Single-step `_forward_full` vs. `fused_roi_epilogue` | `jax.vmap(_forward_full)` | **`3.58e-7` (Max Abs Diff)** |
| Final-step `_forward_full` vs. `fused_output_port_epilogue` (8 cells) | `jax.vmap(_forward_full)` | **`2.38e-7` (Max Abs Diff)** |
| End-to-End Forward Outputs (`return_universe=False`) | `UniverseLP.__call__` | **`4.17e-7` (Max Abs Diff)** |
| End-to-End Full Universe (`return_universe=True`, `62x62`) | `UniverseLP` `hk.get_state("universe")` | **`4.77e-7` (Max Abs Diff)** |
| Backward Gradients ($\nabla_W, \nabla_B, \nabla_S, \nabla_G$) | `jax.value_and_grad` on `UniverseLP` | **`< 4.11e-5` (Relative L2)** |
| 5-Step `optax.adamw` Training Trajectory | `UniverseLP` + `optax.adamw` | **`1.22e-5` (Loss Diff)** |

### Stage 02 Performance Progression on RTX 4060 Laptop GPU ($B = 128$, 800 Train Batches, 200 Test Batches)

| Execution Mode | Stage 00 Legacy (`contrib/`) | Stage 02A (Full-Grid Conv + ROI Epilogue) | Stage 02B (Pure-ROI `40x50` + `B=1` Hoisted Border) | Speedup vs. Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Pure Device Train Step** | 23.411 ms/batch<br/>(`5,468 samples/s`) | 16.857 ms/batch<br/>(`7,593 samples/s`) | **11.007 ms/batch**<br/>(**`11,629 samples/s`**) | **2.13x** |
| **800-Batch Training Epoch** | 19.48 s / epoch | 14.03 s / epoch | **8.81 s / epoch** | **2.21x** |
| **200-Batch Test Evaluation** | 4.829 s (`24.147 ms/batch`,<br/>`5,301 samples/s`) | 0.437 s (`2.185 ms/batch`,<br/>`58,581 samples/s`) | **0.437 s** (`2.185 ms/batch`,<br/>**`58,581 samples/s`**) | **11.05x** |
