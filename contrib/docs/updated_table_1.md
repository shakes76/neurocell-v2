# Updated Table 1: Performance Comparison

Here is the updated Table 1 containing the new Random Forest and Linear Regression baseline models for the XOR (Arithmetic) task. Two new columns, **Inference Time (s)** and **Throughput (samples/s)**, have been added to track the newly requested metrics.

| Dataset | Task | Model | Accuracy (%) | # Parameters | Inference Time (s) | Training Time (s) | Throughput (samples/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MNIST** | Classify | MLP | 96.5 | 244.5K | 2.8921 | 151.70 | 3,457.66 |
| | | VNN Hyperplanes | 96.7 | 41.2K | 4.4076 | 562.57 | 2,268.79 |
| | | Logistic Regression | 92.6 | 7.9K | 0.0469 | 156.68 | 213,319.23 |
| | | Random Forest | 96.8 | 707.6K | 0.1251 | 3.91 | 79,965.23 |
| **CIFAR10** | Classify | MLP | 62.4 | 11.1M | 11.2762 | 1614.03 | 886.82 |
| | | VNN Hyperplanes | 71.9 | 841.7K | 15.4264 | 8789.62 | 648.24 |
| | | VNN Full | 77.7 | 1.2M | 25.7458 | 18750.47 | 388.41 |
| **ALU** | Arithmetic | MLP | 66.8 | 2.0K | 2.9845 | 295.85 | 67.01 |
| | | VNN 1D Hyperplanes | 69.9 | 7.9K | 2.3770 | 661.08 | 84.14 |
| | | VNN Full | 99.8 | 16.3K | 6.1487 | 13154.22 | 32.53 |
| **XOR** | Arithmetic | MLP | 100.0 | 36.2K | 0.7814 | 21.19 | 8,190.13 |
| | | VNN Hyperplanes | 100.0 | 3.3K | 0.5438 | 113.28 | 91.94 |
| | | Pure Linear Regression | 1.1 | 136 | 0.0003 | 0.01 | 20,228,745.74 |
| | | Polynomial (Degree 2) LR | 5.8 | 1.2K | 0.0088 | 0.08 | 730,376.99 |
| | | Polynomial (Degree 3) LR | 66.3 | 7.8K | 0.0136 | 0.80 | 471,352.86 |
| | | Pure Random Forest | 36.3 | 6.1M | 0.2179 | 12.44 | 29,367.31 |
| | | Classifier Chain RF | 73.3 | 16.1M | 2.7383 | 51.72 | 2,337.21 |
| | | MultiOutput RF | 79.6 | 18.2M | 1.3617 | 35.14 | 4,700.08 |

### Notes on Baselines
*   **Pure Linear Regression**: Severely underfits the XOR logic (1.1% accuracy) because it lacks the capacity to represent non-linear relationships.
*   **Polynomial LR**: Once higher order polynomial features (AND gates) are added, the linear models begin approaching the MLP baseline (66.3% accuracy at Degree 3), but inference time increases as the parameter count jumps.
*   **Random Forests**: The independent MultiOutput RF achieves the best baseline performance (79.6% accuracy). All RF models require a massive parameter footprint (represented as tree nodes) compared to the neural networks.
