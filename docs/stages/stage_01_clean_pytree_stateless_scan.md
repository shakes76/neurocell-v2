# Stage 01 Design Decision Record: Clean-Slate PyTree Architecture, Static Topology & On-Device `lax.scan` Trainer

## 1. Context & Problem Statement
In Stage 00 ([`stage_00_baseline_audit.md`](file:///home/shakes/Work/neurocell-release/docs/stages/stage_00_baseline_audit.md)), we identified three foundational architectural bottlenecks in the legacy Neurocell framework ([`contrib/universe.py`](file:///home/shakes/Work/neurocell-release/contrib/universe.py), [`contrib/data.py`](file:///home/shakes/Work/neurocell-release/contrib/data.py), and [`contrib/test_jax_vnn_alu_arithmetic.py`](file:///home/shakes/Work/neurocell-release/contrib/test_jax_vnn_alu_arithmetic.py)):

1. **Conflated Stateful `hk.Module` (`UniverseLP`)**:
   - Required calling `mlp.initialize(...)` inside `_forward(batch)` on every forward pass, resetting `hk.set_state("universe", jnp.zeros((62, 62)))` and then writing the entire `vmap`ped `(128, 62, 62)` batch universe (`1.97 MB`) back into Haiku state at the end of `__call__`.
   - Returning `state` from `loss` and `update` forced XLA to keep all $3,844$ cells at the final depth step $t=d$ alive, preventing Dead-Code Elimination (DCE) of non-output cells.
2. **Fragmented Parameter Tensors & Runtime `dynamic_update_slice` Chains**:
   - Stored `w`, `b`, `s` as 3 separate dictionary entries, executed 5 sequential `jax.lax.dynamic_update_slice` calls on every forward pass to pin input/output states and output weights, and recomputed `jax.nn.sigmoid(states)` inside the depth loop and unfactored across all $M=16$ directions.
3. **Host-Bound Data Pipeline & Python Per-Batch Training / Un-JITted Evaluation Loops**:
   - `contrib/data.py` used a nested Python loop over $128,000$ samples to compute ALU truth tables (`579.6 ms`).
   - Driver scripts iterated over 800 batches in Python per epoch with per-batch host-to-device slice transfers and called `network.apply` eagerly without `@jax.jit` during evaluation.

---

## 2. Design Decisions & Hardware/Mathematical Justifications

### Decision 1.1: Static Compile-Time Geometry via `ChipTopology2D` and `PortSpec` ([`neurocell/topology.py`](file:///home/shakes/Work/neurocell-release/neurocell/topology.py))
- **Decision**: Represent the cellular array geometry, input/output port locations (`PortSpec`), active `VALID` convolution Region-of-Interest (ROI) bounds (`roi_shape = (40, 50)`, `pad_h = 11`, `pad_w = 6`, `roi_out_offset = (16, 35)`), and static boolean pinning masks (`io_state_mask`, `out_weight_mask`) in an immutable `@dataclass(frozen=True)` (`ChipTopology2D`).
- **Justification**:
  - Pinning coordinates are static properties of the cellular topology and never change across training steps.
  - Replacing 5 sequential `jax.lax.dynamic_update_slice` calls with two precomputed boolean masks (`jnp.where(topology.io_state_mask, 1.0, s_raw)` and `jnp.where(topology.out_weight_mask, 1.0, w_raw)`) allows XLA to fuse I/O pinning, `sigmoid(S)`, and $\frac{1}{M}$ scaling into a single elementwise kernel over `(H, W)` outside the temporal loop.

### Decision 1.2: Unified `VNNParams` and `PrecomputedStepParams` PyTrees ([`neurocell/params.py`](file:///home/shakes/Work/neurocell-release/neurocell/params.py))
- **Decision**:
  - Define `VNNParams(NamedTuple)` with two contiguous device tensors:
    1. `cell_params` of shape **`(3, H, W)`** packing `[weights, biases, states_raw]` contiguously in HBM.
    2. `greens_filters` of shape **`(M, 1, Kh, Kw)`** in native `OIHW` layout.
  - Define `PrecomputedStepParams(NamedTuple)` computed once per forward pass via `precompute_step_params`:
    $$W_{\text{eff}} = \frac{1}{M} W_{\text{pinned}}[r_h:H-r_h, r_w:W-r_w] \odot \left(1 - \sigma(S_{\text{pinned}}[r_h:H-r_h, r_w:W-r_w])\right)$$
    $$S_{\text{eff}} = \frac{1}{M} \sigma(S_{\text{pinned}}[r_h:H-r_h, r_w:W-r_w]), \qquad B_{\text{roi}} = B[r_h:H-r_h, r_w:W-r_w]$$
- **Justification**:
  - `typing.NamedTuple` is a native JAX PyTree with zero third-party wrapper overhead, passing directly into `jax.jit`, `jax.value_and_grad`, `jax.lax.scan`, and `optax`.
  - Precomputing `W_eff` and `S_eff` on the $(40, 50)$ ROI once per batch reduces per-step gating operations from $4 \times 128 \times 62 \times 62 \times 16 = 31.49\text{M}$ ops to $2,000$ ops (**15,745x reduction**).
  - By matching Haiku's `PRNGSequence` key splitting order in `init_vnn_params(..., legacy_compat=True)` and initializing filters in `(1, Kh, Kw, M)` before transposing to `(M, 1, Kh, Kw)`, we preserve **bit-for-bit (`0.0` difference) initialization parity** with `contrib/universe.py`.

### Decision 1.3: Vectorized Dataset Generation & Contiguous Device Staging ([`neurocell/datasets.py`](file:///home/shakes/Work/neurocell-release/neurocell/datasets.py))
- **Decision**: Replace the nested Python `for` loop in `generate_arithmetic_logic` with vectorized `np.select` over `uint8` arrays, and stage the entire split dataset as contiguous GPU tensors (`ALUDeviceDataset` with `x_train` of shape `(800, 128, 3, 8, 1)`).
- **Justification**:
  - Preserves exact RNG call order and `uint8` modular arithmetic so generated datasets are bit-for-bit identical to `contrib/data.py` while accelerating CPU generation by **25.1x** (`579.6 ms -> 23.1 ms`).
  - Contiguous device staging (`9.83 MB` for `x_train`, `3.28 MB` for `y_train`) easily fits in the RTX 4060's 8 GB VRAM and eliminates 3,200 host-to-device slice transfers per epoch.

### Decision 1.4: Whole-Epoch On-Device `jax.lax.scan` & Matmul Bit-Packing ([`neurocell/trainer.py`](file:///home/shakes/Work/neurocell-release/neurocell/trainer.py))
- **Decision**: Implement `VNNTrainer` with `@jax.jit`-compiled `train_epoch` and `evaluate_packed` using `jax.lax.scan`, and replace `jnp.packbits` with `packbits_u8_on_device` (`jnp.dot(bits_u8, jnp.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.uint8))`).
- **Justification**:
  - Collapses 800 Python-to-XLA launch dispatches per training epoch into a single compiled XLA `While` loop on GPU.
  - Compiles the entire 200-batch test evaluation and bit-packing comparison into a single on-device kernel launch.

---

## 3. Alternatives Considered & Rejected
1. **Retaining Haiku `hk.Module` Without `hk.set_state`**:
   - *Rejected*: Even stateless `hk.transform` requires dictionary-based parameter trees (`params['universe_lp']['w']`) and re-tracing module constructors inside `_forward`. A native `NamedTuple` PyTree (`VNNParams`) is simpler, flatter, and directly compatible with custom JAX Pallas/Triton `custom_vjp` primitives.
2. **Storing `cell_params` as `(H, W, 3)` (Interleaved) Instead of `(3, H, W)` (Planar)**:
   - *Rejected*: During `precompute_step_params`, `W`, `B`, and `S` undergo distinct operations (`out_weight_mask` on `W`, `io_state_mask` + `sigmoid` on `S`). Planar `(3, H, W)` allows zero-copy contiguous slicing (`params.cell_params[0]`) with 100% coalesced 128-byte cache-line reads, whereas `(H, W, 3)` would produce strided views (`stride = 3`) when sliced by parameter type.

---

## 4. Verification & Empirical Benchmark Results (RTX 4060 Laptop GPU)

All 7 unit tests in [`tests/unit_topology.py`](file:///home/shakes/Work/neurocell-release/tests/unit_topology.py) and [`tests/unit_params.py`](file:///home/shakes/Work/neurocell-release/tests/unit_params.py) passed:

| Verification Check | Target Reference | Measured Error / Result |
| :--- | :--- | :---: |
| `init_vnn_params` (`W, B, S, G`) vs. `contrib/universe.py` | `UniverseLP.initialize` (`PRNGKey(42)`) | **`0.0` (Bit-Exact)** |
| `io_state_mask` & `out_weight_mask` vs. 5x `dynamic_update_slice` | `UniverseLP.__call__` | **`0.0` (Bit-Exact)** |
| `precompute_step_params` Factored Codd Update vs. Unfactored | `_forward_full` Codd formula | **`2.38e-7` (FP32 machine eps)** |
| Vectorized `generate_arithmetic_logic` vs. `contrib/data.py` | `seed=2026`, $2,560$ samples | **`0.0` (Bit-Exact)** |
| `packbits_u8_on_device` vs. `np.packbits` / `jnp.packbits` | $2,048$ 8-bit labels | **`0.0` (Bit-Exact)** |

### Stage 01 Performance Progression (Pipeline & `VNNTrainer` Scan Harness)

| Benchmark Workload ($N_{\text{samples}} = 128,000, B = 128$) | Stage 00 Legacy (`contrib/`) | Stage 01 (`neurocell/`) | Measured Speedup |
| :--- | :---: | :---: | :---: |
| **ALU Dataset Generation (128k samples, CPU)** | 579.6 ms | **23.1 ms** | **25.1x** |
| **MLP ALU Training Epoch (800 batches)** | 14.29 s (`17.863 ms/batch`) | **0.0672 s** (`0.0840 ms/batch`) | **212.6x** (vs. Python loop)<br/>**2.87x** (vs. single-step JIT) |
| **MLP ALU Test Evaluation (200 batches, 25.6k samples)** | 2.628 s (`9,741 samples/s`) | **0.00317 s** (`8,064,534 samples/s`) | **828.9x** |
