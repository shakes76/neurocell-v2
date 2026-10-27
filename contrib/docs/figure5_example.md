# VNN Forward Pass Algebraic Example

This document provides a tabular representation of the algebraic calculation of the forward pass in a Von Neumann Network (VNN), as illustrated in Figure 5 of the VNN paper appendix.

## (a) The Initial Arrays

The following tables represent the initial state of the VNN, with the inputs, kernel, and weights broken up into separate arrays as originally shown for illustrative purposes.

### Input Image ($x$)
|   |   |   |   |
|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | 0 |
| 0 | $x_0$ | 0 | 0 |
| 0 | 0 | 0 | 0 |
| 0 | $x_1$ | 0 | 0 |
| 0 | 0 | 0 | 0 |

### Kernel ($k$)
|   |   |   |
|:---:|:---:|:---:|
| 0 | 0 | 1 |
| 0 | 0 | 1 |
| 0 | 0 | 1 |

### Weights 1 ($w$)
|   |   |   |   |
|:---:|:---:|:---:|:---:|
| 0 | 0 | $W_0$ | 0 |
| 0 | 0 | $W_1$ | 0 |
| 0 | 0 | $W_2$ | 0 |
| 0 | 0 | $W_3$ | 0 |
| 0 | 0 | $W_4$ | 0 |

### Weights 2 ($u$)
|   |   |   |   |
|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | $U_0$ |
| 0 | 0 | 0 | $U_1$ |
| 0 | 0 | 0 | $U_2$ |
| 0 | 0 | 0 | $U_3$ |
| 0 | 0 | 0 | $U_4$ |

## (b) Intermediate and Final Results

The following tables show the intermediate convolutions and element-wise multiplications, maintained as separate arrays to illustrate the algebraic progression through each layer.

### Intermediate Conv 1 ($x * k$)
|   |   |   |   |
|:---:|:---:|:---:|:---:|
| 0 | 0 | $x_0$ | 0 |
| 0 | 0 | $x_0$ | 0 |
| 0 | 0 | $x_0 + x_1$ | 0 |
| 0 | 0 | $x_1$ | 0 |
| 0 | 0 | $x_1$ | 0 |

### Output 1 ($(x * k) \circ w$)
|   |   |   |   |
|:---:|:---:|:---:|:---:|
| 0 | 0 | $W_0 x_0$ | 0 |
| 0 | 0 | $W_1 x_0$ | 0 |
| 0 | 0 | $W_2 (x_0 + x_1)$ | 0 |
| 0 | 0 | $W_3 x_1$ | 0 |
| 0 | 0 | $W_4 x_1$ | 0 |

### Intermediate Conv 2 $(((x * k) \circ w) * k)$
|   |   |   |   |
|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | $W_0 x_0 + W_1 x_0$ |
| 0 | 0 | 0 | $W_0 x_0 + W_1 x_0 + W_2(x_0 + x_1)$ |
| 0 | 0 | 0 | $W_1 x_0 + W_2(x_0 + x_1) + W_3 x_1$ |
| 0 | 0 | 0 | $W_2(x_0 + x_1) + W_3 x_1 + W_4 x_1$ |
| 0 | 0 | 0 | $W_3 x_1 + W_4 x_1$ |

### Final Output $(((x * k) \circ w) * k) \circ u$
|   |   |   |   |
|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | $U_0 (W_0 x_0 + W_1 x_0)$ |
| 0 | 0 | 0 | $U_1 (W_0 x_0 + W_1 x_0 + W_2(x_0 + x_1))$ |
| 0 | 0 | 0 | $U_2 (W_1 x_0 + W_2(x_0 + x_1) + W_3 x_1)$ |
| 0 | 0 | 0 | $U_3 (W_2(x_0 + x_1) + W_3 x_1 + W_4 x_1)$ |
| 0 | 0 | 0 | $U_4 (W_3 x_1 + W_4 x_1)$ |

## Explanation

The example illustrates how a simple two-neuron Von Neumann Network (VNN) computes a forward pass algebraically, embedded entirely within a cellular array. While the VNN itself evaluates as a single unified field, we have separated the weights and input into individual arrays for illustrative purposes exactly as done in Figure 5 of the paper.

1. **Propagation via Convolution:** The input signals $x_0$ and $x_1$ are propagated to the right and smeared vertically across adjacent cells by convolving the input image $x$ with a $3 \times 3$ kernel $k$. The kernel acts as a shift-and-spread operator, modeling the diffusion-like propagation of the Green's function across the Chua field.
2. **First Layer Weights:** The propagated signals are then element-wise multiplied by the first set of weights $w$, which are spatially distributed in the third column of the array. This forms the intermediate output (`Output 1`).
3. **Second Propagation:** The signals are convolved with the kernel $k$ a second time, propagating the weighted signals further to the right (into the fourth column) and mixing them vertically once again. Note how the spatial overlapping naturally produces the required algebraic terms (e.g., $W_1 x_0 + W_2(x_0 + x_1) + W_3 x_1$).
4. **Second Layer Weights:** Finally, the signals are multiplied element-wise by the second set of weights $u$, situated in the fourth column, resulting in the final output.

This sequence of operations algebraically mimics a two-layer neural network embedded entirely within the 2D spatial grid, utilizing convolution to simulate connectivity pathways and element-wise multiplication to apply weights. Biases can be incorporated similarly through addition, and non-linear activations can be applied per layer as needed.
