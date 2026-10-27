# Von Neumann Networks (VNN) Algorithms

This document outlines the core algorithms behind Von Neumann Networks (VNNs) as implemented in the Neurocell framework. 

## 1. VNN Forward Pass Algorithm

The forward pass models the propagation of signals (or impulses) across a cellular array using learnable Green's functions (acting as Point Spread Functions or PSFs). Each cell maintains a Codd state $s \in [0, 1]$ that determines its role: either passing the signal through (as a connection) or activating (making a non-linear decision).

**Algorithm 1:** VNN Forward Pass
***
**Require:** Cellular field $C^{(0)}$, Weights $W$, Biases $B$, States $S$, Green's functions $\{G_j\}_{j=1}^M$, Depth $d$  
**Require:** Activation function $\sigma$  
1. **for** $n = 0$ **to** $d-1$ **do**
   1. Initialize $C^{(n+1)} \leftarrow 0$
   2. **for** $j = 1$ **to** $M$ **do**
      1. Compute propagated field: $C_j^{(n)} = G_j * C^{(n)}$
      2. Apply standardization/layer normalization (optional) on $C_j^{(n)}$
      3. Compute passive signal: $P_j = (I - S) \cdot (C_j^{(n)} \cdot W)$
      4. Compute active decision: $A_j = S \cdot \sigma(C_j^{(n)} + B)$
      5. Blend states: $F_j = P_j + A_j$
      6. Accumulate: $C^{(n+1)} \leftarrow C^{(n+1)} + F_j$
   3. **end for**
   4. Average across filters: $C^{(n+1)} \leftarrow \frac{1}{M} C^{(n+1)}$
2. **end for**
3. **return** $C^{(d)}$
***

In practice, the JAX implementation iterates this forward step up to a specified `depth`, ensuring that information can propagate from input locations to output locations across the cellular universe.

---

## 2. VNN Model Training Procedure Algorithm

VNNs are trained using standard Stochastic Gradient Descent (SGD) and backpropagation, treating the cellular machine as a differentiable computation graph.

**Algorithm 2:** VNN Training Procedure
***
**Require:** Training dataset $\mathcal{D}$, Learning rate schedule, Optimizer (e.g., AdamW)  
1. **Initialization:**
   1. Initialize cellular universe field $C \leftarrow 0$
   2. Initialize Weights $W \sim \mathcal{N}(\mu, \sigma^2)$ and Biases $B \leftarrow 0$
   3. Initialize States $S$ randomly, then bound via sigmoid $S \leftarrow \sigma_{sig}(S_{raw})$
   4. Fix $S \leftarrow 1.0$ at input and output coordinates
   5. Initialize Green's functions $G_j$
2. **while** not converged **do**
   1. Sample a minibatch of inputs $X$ and targets $Y$ from $\mathcal{D}$
   2. Place $X$ at designated input coordinates in $C^{(0)}$
   3. Compute forward pass $\hat{Y} = \text{VNN\_Forward}(C^{(0)}, W, B, S, \{G_j\})$ sampled at output coordinates
   4. Compute loss $\mathcal{L}(\hat{Y}, Y)$
   5. Compute gradients $\nabla_{W, B, S_{raw}, G_j} \mathcal{L}$ via reverse-mode auto-differentiation
   6. Update parameters $W, B, S_{raw}, G_j$ using Optimizer
3. **end while**
4. **return** $W, B, S, \{G_j\}$
***

During training, the backward pass simulates the inverse linear differential operator $\mathcal{D}^{-1}$, distributing gradients back through the continuous cellular field. Over time, the Codd states $S$ converge to binary-like roles, effectively allowing the network to self-engineer its own high-dimensional architecture and connectivity directly on the cellular array.
