# VNN MLP XOR Ablation Study Details

This document outlines the architecture, data pipeline, hyperparameters, and exact ablations run for the XOR arithmetic task using the Von Neumann Network (VNN).

## 1. VNN Model Architecture

The experiment uses the `UniverseLP` (Von Neumann Network) model configured to mimic a Multi-Layer Perceptron (MLP) for arithmetic tasks.

* **Base Grid (Universe) Size**: $N \times N$, where baseline $N = 50$.
* **Depth**: The VNN is configured with $layers = 5$, resulting in an internal logical depth of $11$ ($2 \times layers + 1$) representing alternating decision and signal layers, plus an output layer.
* **Activations**: Internal states utilize `jax.nn.tanh` for non-linear activations.
* **Layout**: Inputs and outputs are spatially separated and offset from the center of the grid to allow signals to propagate dynamically through the defined depth.

## 2. Data Pipeline

The dataset evaluates the model's ability to learn the highly non-linear boolean exclusive OR (XOR) operation.

* **Task**: Bitwise XOR.
* **Bit Depth**: $n = 8$ bits per operand.
* **Samples**: $32,000$ total generated samples.
* **Inputs**: The inputs are unpacked into 8-bit arrays and continuous noise ($\delta = 0.2$) is added to synthesize floating-point observations.
* **Outputs**: Raw binary XOR results, packed for final accuracy evaluation.
* **Data Split**: Standard 80/20 train-test split (yielding 25,600 training samples and 6,400 test samples).

## 3. Training Hyperparameters

All ablations share the following core training configurations:

* **Epochs**: $100$
* **Batch Size**: $128$ (yielding $250$ batches per epoch)
* **Optimizer**: `optax.adamw`
* **Learning Rate Schedule**: Warmup cosine decay
    * Initial LR: $0.001$
    * Peak LR: $0.03$
    * Warmup Steps: $2.5\%$ of total steps
    * End LR: $0.0005$
* **Loss Function**: Mean Squared Error (MSE) on predicted logits.

## 4. Ablations Setup

The ablation study executes 13 total runs, testing variations one-at-a-time against the following **Baseline Configuration**:
* **Directions (Green's functions)**: $8$
* **Grid Size ($N$)**: $50$
* **Kernel Size**: $(17, 3)$

### Ablation 1: Number of Green's Functions (Directions)
Tests the effect of the number of filter directions the VNN uses to route signals.
* **Values Tested**: $2, 4, 8 \text{ (baseline)}, 12, 16$

### Ablation 2: Grid Size ($N$)
Tests the effect of the VNN universe's spatial capacity. The baseline $N=50$ is derived from the input dimension and standard kernel padding.
* **Values Tested ($N$)**: $40, 50 \text{ (baseline)}, 60, 70, 80$

### Ablation 3: Kernel Size
Tests the effect of the spatial receptive field of the VNN updates.
* **Values Tested**: $(5, 3), (9, 3), (13, 3), (17, 3) \text{ (baseline)}, (21, 3)$

---
*Note: Memory tracking (`Params Count` and `Model Size (MB)`) and Performance metrics (`Total Time`, `Time per Epoch`) are automatically appended to the output results during execution.*
