# Turing Machine Von Neumann Network (TM VNN) Model

This document describes the purely 1D Turing Machine model implemented in the `UniverseTM` class, its associated training script, and the proposed visualization script.

## 1. TM VNN Model (`UniverseTM`)
The `UniverseTM` class (found in `universe.py`) implements a 1D tape model of Von Neumann Networks (VNNs) designed to mimic a Turing Machine. 
- **Architecture**: The universe resides entirely in a 1D grid (a tape) where computations happen exclusively through 1D spatial convolutions rather than multiple feed-forward layers. 
- **Forward Operation**: It utilizes `_propagate_tape` from `forward.py`, which repeatedly applies a 1D horizontal operation (`_forward_row`) to the same row of cells `depth` times. This mimics a TM executing instructions step-by-step in place.
- **Components**: Like other VNNs, it has trainable states (passive/active), weights, biases, and filters (the rules for the neighborhood).
- **Initialization**: `initialize_tm` initializes all cells as a horizontal line with specific inputs and output sizes and offsets.

## 2. Training Script (`test_jax_tm_arithmetic.py`)
This script trains the `UniverseTM` model to perform arithmetic operations (specifically, XOR).
- **Data**: Generates XOR arithmetic problems (e.g., adding two numbers bit by bit) and maps them to a 1D input tensor representing the initial TM tape.
- **Model Setup**: Sets up a `UniverseTM` model inside a Haiku transform.
- **Training**: Uses optax with AdamW and a cosine decay schedule over several epochs. 
- **Outputs**: After training, it extracts the learned weights, biases, states, and filters and visualizes them using matplotlib, saving them as images (`tm_arith_states_init.png`, `tm_arith_weights.png`, etc.). It also saves the model data to `tm_arith_cells.npz`.

## 3. Visualization Script (`run_tm_animation.py`) - *Proposed*
The proposed animation script will visually demonstrate the step-by-step computation of the 1D TM, showing the evolution of the Green's functions (cell states) over time.
- **Concept**: Since the universe is 1D, stacking the states of the tape at each time step vertically creates a 2D image, much like standard visualizations of Wolfram's 1D cellular automata.
- **Mechanism**: The script will load the trained `tm_arith_cells.npz` model. It will use a new unrolled function `_propagate_tape_unrolled` (added to `forward.py`) to capture the state of the 1D universe at every single propagation step.
- **Output**: By animating this stacked 2D tensor row by row using `matplotlib.animation`, it will generate a GIF (`tm_computation_evolution.gif`) illustrating the Turing Machine processing the input tape over time.
