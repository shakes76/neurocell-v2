# VNN 3D Imagenette Classifier

This document describes the `test_jax_vnn_3D_classify_imagenette_hyperplane.py` script, which implements a 3D Von Neumann Network (VNN) for classifying the Imagenette dataset using hyperplane convolutional operators.

## Overview
The script adapts the 3D VNN architecture to accommodate larger, more complex inputs from the Imagenette dataset (10 classes). It utilizes the Google `grain` framework and `PIL` for efficient data loading and ImageNet-standard preprocessing, completely eliminating PyTorch and TIMM dependencies to avoid ROCm/HIP conflicts in WSL2/HPC environments.

## Model Architecture
The architecture is based on the `Universe3D` grid with hyperplane configurations.
- **Input Resolution**: 56x56
- **Batch Size**: 64
- **Channels**: 3 (RGB), padded to effectively spread spatial information across depth dimensions.
- **Grid Dimensions (Nx, Ny, Nz)**: (68, 68, 85) based on the specified kernel sizes and network depth.
- **Optimizer**: AdamW with Warmup Cosine Decay schedule.
- **Loss Function**: Softmax Cross-Entropy.

## Data Loading
The script fetches the dataset from a local directory structure: `E:/data/imagenet/imagenette2-160`.
A custom `FolderDataSource` uses `PIL.Image` to stream the images.
- **Training**: Uses `RandomResizedCrop` and `RandomHorizontalFlip` (mimicking standard ImageNet augmentations).
- **Validation**: Uses `Resize` (direct resize to 56x56).
Both pipelines apply `ImageNetNormalize` (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) and `SpreadChannelsTransform` directly via the Grain pipeline, so no further JAX JIT preprocessing wrapper is required for channel shuffling.

## Requirements
- `jax`, `jaxlib`, `haiku`, `optax`
- `grain`, `numpy`, `Pillow (PIL)`
- `matplotlib`, `tqdm`

## Usage
Run the script using python:
```bash
python test_jax_vnn_3D_classify_imagenette_hyperplane.py
```
Outputs including trained weights, biases, and visual plots of network states will be saved in the root directory.
