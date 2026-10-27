# VNN Architectural Choices & Design Decisions

This document details the architectural choices, network parameters, and hyperparameters for the VNN model training scripts.

## Hyper-parameters Comparison Table

| Parameter / Decision | `test_jax_vnn_mlp_classify_mnist.py` | `test_jax_vnn_3D_classify_cifar_hyperplane_grain.py` | `test_jax_vnn_alu_arithmetic.py` |
| :--- | :--- | :--- | :--- |
| **Model Type** | `UniverseLP` (VNN MLP) | `Universe3D` (VNN 3D Hyperplane) | `UniverseLP` (VNN Chip Layout) |
| **Dataset** | MNIST | CIFAR-10 (via Grain Dataloader) | Synthetic ALU Arithmetic (128k samples) |
| **Epochs** | 8 | 35 | 200 |
| **Batch Size** | 64 | 100 | 128 |
| **Layers** | 4 | 20 | N/A |
| **Network Depth** | 9 (`layers * 2 + 1`) | 40 (`layers * 2`) | 4 |
| **Kernel Size** | `(71, 3)` | `(9, 9, 41)` (channel first, depth last) | `(23, 13)` |
| **Directions (Filters)** | 1 (implied single filter/Green's function) | 32 | 16 |
| **Universe Grid Size ($N$)**| $1024 \times 17$ ($N_x \times N_y$) | $44 \times 44 \times 89$ ($N_x \times N_y \times N_z$) | $62 \times 62$ (Square Chip Layout) |
| **Optimizer** | Adam | AdamW | AdamW |
| **Learning Rate** | `0.01` (Constant) | Warmup Cosine Decay (`peak=0.03`, `init=0.001`, `end=0.0005`) | Warmup Cosine Decay (`peak=0.03`, `init=0.001`, `end=0.0005`) |
| **Activation (Hidden)** | `jax.nn.tanh` | `jax.nn.leaky_relu` | `jax.nn.tanh` |
| **Activation (Final)** | `jax.nn.sigmoid` | `jax.nn.leaky_relu` | None |
| **Loss Function** | Softmax Cross Entropy (`optax.softmax_cross_entropy`) | Softmax Cross Entropy (`optax.softmax_cross_entropy`) | Mean Squared Error (`optax.squared_error`) |
| **Normalization** | Standardized/Layer Norm (`norm=True`) | Standardized/Layer Norm (`norm=True`) | Standardized/Layer Norm (`norm=True`) |
| **Input Shape Mapping** | 1D Vector (`28*28, 1`) | 2D/3D Tensor mapped into field | Sequential 1D Inputs ($X_1, X_2, C$) mapped via offsets |

## Design Rationale

### MNIST MLP VNN
- **Architecture**: Emulates a Multi-Layer Perceptron using the `UniverseLP` setup. The depth is strictly calculated based on the requested layers, interleaving active decision states with passive signal passing states.
- **State Thresholding**: At the end of training, the continuous Codd states ($s \in [0,1]$) are heavily binarized/thresholded to harden the architecture (e.g. states $\ge 0.6$ are active, $(0, 0.6)$ are passive/wiring).

### CIFAR-10 3D Hyperplane VNN
- **Architecture**: A much more complex 3D arrangement (`Universe3D`) employing hyperplane-based convolutions to remain tractable and efficient in 3D.
- **Directional Filters**: Unlike the MNIST network, the CIFAR-10 variant uses multiple directions (`directions = 32`) which translates to multiple learnable Green's functions/filters that give the network flexibility to route information spatially across the 3D grid. 
- **Memory Management**: Explicit JAX environment variables are set (`XLA_PYTHON_CLIENT_MEM_FRACTION`) to control memory spikes induced by 3D FFTs and deep volume allocations.
- **Learning Schedule**: Uses a Warmup Cosine Decay schedule (compared to standard Adam for MNIST) because finding optimal connectivity in 3D with 32 directions is a significantly harder optimization landscape.

### ALU Arithmetic VNN
- **Architecture**: Emulates a 2D computational "chip" where distinct inputs ($X_1, X_2, C$) and outputs are mapped to sequentially spaced coordinates like a tape. It utilizes a square-like `UniverseLP` cell grid.
- **Learning**: Optimizes a continuous routing over 200 epochs using 16 directional filters to map inputs to perform Arithmetic Logic Unit operations. The objective is to minimize Mean Squared Error (MSE) directly against the bits.
