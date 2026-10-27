# Formal Verification of VNN Computational Universality

This document details the formal mathematical proof of the Turing completeness of Von Neumann Networks (VNNs). The proof has been written and verified using the **LEAN 4** theorem prover.

## Motivation

To rigorously support the claims in the [Von Neumann Networks paper](https://arxiv.org/abs/2605.05780), we show a mathematically verified bisimulation between a VNN and a known Turing-complete system. While Game of Life is also Turing complete, this proof simulates a 1D tape-based Turing Machine (TM) directly, mirroring the 1D tape concept introduced in the paper.

## Proof Strategy

The mathematical formalization relies on the following components:

1. **The Turing Machine (TM):**
   A deterministic TM is defined with states, a tape alphabet, and a standard transition function mapping state and symbol to a new state, symbol, and head movement direction (Left, Right, Stay).

2. **The VNN Structure (The Tape):**
   The 1D Chua field is defined as an infinite array of cells (`Int -> Cell`). The cell structure contains variables to accommodate tape symbols and the presence of the TM head, effectively encoding the TM configuration directly onto the VNN spatial grid.

3. **The Green's Functions (The Program):**
   We define a specific Green's function $G$ that acts as a local shift kernel, allowing cells to communicate with their immediate neighbors. Coupled with specifically constructed weights $W$, biases $B$, and an activation function $\sigma$ (the Codd state transitions), the VNN applies the TM's local transition rule across the entire grid simultaneously.

4. **The Theorem (`vnn_simulates_tm`):**
   The core theorem proves that applying one forward pass of this constructed VNN to an encoded TM configuration yields the exact encoding of the subsequent TM configuration. The proof involves rigorous case analysis of the TM head's location and evaluates all possible branches of the TM transition function.

## Code Explanation

The LEAN 4 project is structured into three main files, progressively building up to the final proof:

### 1. `TM.lean`
This file defines the mathematical structure of the deterministic Turing Machine. 
*   It declares the possible head movement directions (`Dir.left`, `Dir.right`, `Dir.stay`).
*   It defines the `TM` structure containing a blank symbol, initial state, and the transition function (`delta`).
*   It defines `TMConfig` to hold the current state of the machine (the `tape`, `head` location, and current `state`).
*   Finally, `stepTM` formally defines how a `TMConfig` is updated into the next configuration based on the `delta` function.

### 2. `VNN.lean`
This file abstracts the core mechanics of a Von Neumann Network.
*   The `VNN` structure requires definitions for the Green's function `G`, weights `W`, biases `B`, a Codd state `S`, and mathematical operations (`add`, `sub`, `mul`). It also requires a non-linear activation function `sigma`.
*   The `conv` function defines a local 1D spatial convolution over a cellular field (using a neighborhood of `{-1, 0, 1}`).
*   The `stepVNN` function implements the exact forward pass algorithm of a VNN: taking the convolution, applying the Codd state (`S`) for mixing passive and active signals, applying the non-linear `sigma`, and superimposing the results.

### 3. `Equiv.lean`
This is where the bisimulation is constructed and the actual theorem is proven.
*   **The Encoding:** A custom type `Cell Γ Q` is defined to store the local state of a tape cell (its symbol, and whether the TM head is currently positioned there). The `encodeConfig` function maps the global `TMConfig` directly onto an infinite 1D grid of these `Cell`s.
*   **The Construction:** The `constructVNN` function defines a highly specialized set of VNN parameters ($G, W, B, \sigma$). For instance, the Green's function `G` acts as a spatial shift operator, pulling the neighboring left and right cells into the current cell's receptive field. The activation function `vnnSigma` acts as the TM transition rule, checking if the head is present or arriving, and updating the local cell's symbol and head status accordingly.
*   **The Proof (`vnn_simulates_tm`):** This is the main theorem. It asserts mathematically that `stepVNN(encodeConfig(c)) = encodeConfig(stepTM(c))`. The proof script uses structural equality (`congr`) and exhaustively branches into all possible spatial cases (`by_cases i = c.head`, `by_cases i = c.head + 1`, etc.). It utilizes the LEAN `omega` tactic to automatically resolve integer inequalities regarding the head position, proving that the VNN simulates the TM identically on every cell across the infinite grid.

## Verification Instructions

The full LEAN 4 project is located in the `VNNProof` directory. To independently verify the proof:

1. Install [LEAN 4 and elan](https://leanprover.github.io/lean4/doc/setup.html).
2. Navigate to the `VNNProof` directory.
3. Run the following command:
   ```bash
   lake build
   ```
4. A successful, error-free build indicates that the LEAN kernel has formally verified the bisimulation proof. The core theorem can be found in `VNNProof/VNNProof/Equiv.lean`.
