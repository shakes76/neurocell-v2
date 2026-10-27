'''
Grain based data pipelines and helpers
'''
import numpy as np
from typing import Dict, Any, Tuple

import tensorflow_datasets as tfds
import grain.python as grain
import jax

padding = 3

# ==============================================================================
# Grain Transformations (Augmentations & Preprocessing)
# ==============================================================================
class RandomHorizontalFlip(grain.RandomMapTransform):
    """Randomly flips the image horizontally with a 50% probability."""

    def random_map(self, element: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        # rng is a per-record deterministic random number generator provided by Grain
        if rng.random() > 0.5:
            # element['image'] is expected to be [H, W, C]
            element['image'] = np.fliplr(element['image'])
        return element

class RandomCrop(grain.RandomMapTransform):
    """Pads the image and extracts a random crop of the original size."""

    def __init__(self, padding: int = 4):
        self.padding = padding

    def random_map(self, element: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        image = element['image']
        h, w, c = image.shape

        # Pad spatially (H, W), leave channels (C) alone
        padded_image = np.pad(
            image,
            ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode='reflect'
        )

        # Select random origin for crop
        y = rng.integers(0, 2 * self.padding + 1)
        x = rng.integers(0, 2 * self.padding + 1)

        # Crop back to original dimensions
        element['image'] = padded_image[y : y + h, x : x + w]
        return element

class Normalize(grain.MapTransform):
    """Standardizes the image to [0, 1] then applies CIFAR-10 mean/std."""

    def map(self, element: Dict[str, Any]) -> Dict[str, Any]:
        # Convert uint8 to float32
        image = element['image'].astype(np.float32) / 255.0

        # Standard CIFAR-10 normalization metrics
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

        image = (image - mean) / std

        return {
            'image': image,
            'label': element['label']
        }

# placing of the image channels transform
class SpreadChannelsTransform(grain.MapTransform):
    """
    A Grain-compatible transform that spreads 3 color channels
    into a 3D volume with equal padding along the channel dimension.
    """
    def __init__(self, padding_size: int = 2):
        self.padding_size = padding_size

    def map(self, element: Dict[str, Any]) -> Dict[str, Any]:
        image = element['image']

        # TFDS/Grain images are typically (H, W, C)
        if len(image.shape) != 3 or image.shape[-1] != 3:
            raise ValueError(f"Input image must have exactly 3 channels (H, W, C). Got {image.shape[-1]}.")

        if self.padding_size == 0:
            return element

        h, w, c = image.shape

        # 1. Create a block of zeros to insert between channels.
        zeros = np.zeros((h, w, self.padding_size), dtype=image.dtype)

        # 2. Split the input image into its 3 individual channels.
        # Slicing keeps the spatial dimensions intact: (H, W, 1)
        c0 = image[:, :, 0:1]
        c1 = image[:, :, 1:2]
        c2 = image[:, :, 2:3]

        # 3. Concatenate the channels with the zero blocks interleaved in between
        # Concatenating along the channel axis (-1)
        element['image'] = np.concatenate((c0, zeros, c1, zeros, c2), axis=-1)

        return element

# ==============================================================================
# 2. DataLoader Creation Factory
# ==============================================================================
def create_cifar10_dataloader(
    split: str = "train",
    batch_size: int = 128,
    shuffle: bool = True,
    seed: int = 42,
    num_epochs: int = 1, # Use None for infinite repeating
    worker_count: int = 4,
    drop_remainder: bool = True,
    data_dir: str = None,
    download: bool = True
) -> Tuple[grain.DataLoader, int]:
    """
    Creates a Grain DataLoader for the CIFAR-10 dataset.

    Returns:
        A tuple of (DataLoader, total_images_in_split)
    """
    # 1. Download and prepare data source
    builder = tfds.builder("cifar10", data_dir=data_dir, file_format="array_record")

    if download:
        builder.download_and_prepare()

    # Get total images dynamically for the requested split
    total_images = builder.info.splits[split].num_examples

    # Use tfds.data_source to read TFDS records efficiently without tf.data
    data_source = tfds.data_source("cifar10", split=split, data_dir=data_dir)

    # 2. Build the processing pipeline
    operations = []

    # Add data augmentations specifically for the training split
    if split == "train":
        operations.extend([
            RandomCrop(padding=4),
            RandomHorizontalFlip()
        ])

    # Normalize and Batch (applies to all splits)
    operations.extend([
        Normalize(),
        SpreadChannelsTransform(padding_size=padding),
        grain.Batch(batch_size=batch_size, drop_remainder=drop_remainder)
    ])

    # 3. Handle device sharding (Multi-GPU/TPU setup for JAX)
    # This ensures each JAX process (e.g., node) gets a distinct slice of the dataset.
    shard_options = grain.ShardOptions(
        shard_index=jax.process_index(),
        shard_count=jax.process_count(),
        drop_remainder=drop_remainder
    )

    # 4. Create the Sampler
    # IndexSampler decides the iteration order (shuffling/repeating) upfront.
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shard_options=shard_options,
        shuffle=shuffle,
        num_epochs=num_epochs,
        seed=seed
    )

    # 5. Instantiate the Grain DataLoader
    dataloader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
        worker_count=worker_count
    )

    return dataloader, total_images

# ==============================================================================
# ImageNet-Specific Grain Transformations
# ==============================================================================
from PIL import Image

class ImageNetNormalize(grain.MapTransform):
    """Standardizes the image to [0, 1] then applies ImageNet mean/std."""

    def map(self, element: Dict[str, Any]) -> Dict[str, Any]:
        # Convert uint8 to float32
        image = element['image'].astype(np.float32) / 255.0

        # Standard ImageNet normalization metrics
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image = (image - mean) / std

        return {
            'image': image,
            'label': element['label']
        }

class RandomResizedCrop(grain.RandomMapTransform):
    """Mimics TIMM's RandomResizedCrop."""
    def __init__(self, size: int, scale: Tuple[float, float] = (0.08, 1.0), ratio: Tuple[float, float] = (3. / 4., 4. / 3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def random_map(self, element: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        image = element['image'] # expected to be np.array from PIL
        h, w, c = image.shape
        area = h * w

        for _ in range(10):
            target_area = area * rng.uniform(self.scale[0], self.scale[1])
            log_ratio = rng.uniform(np.log(self.ratio[0]), np.log(self.ratio[1]))
            aspect_ratio = np.exp(log_ratio)

            w_t = int(round(np.sqrt(target_area * aspect_ratio)))
            h_t = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < w_t <= w and 0 < h_t <= h:
                i = rng.integers(0, h - h_t + 1)
                j = rng.integers(0, w - w_t + 1)
                
                # Crop and resize
                img_pil = Image.fromarray(image)
                img_cropped = img_pil.crop((j, i, j + w_t, i + h_t))
                img_resized = img_cropped.resize((self.size, self.size), Image.BILINEAR)
                element['image'] = np.array(img_resized)
                return element
                
        # Fallback to central crop and resize
        in_ratio = float(w) / float(h)
        if in_ratio < min(self.ratio):
            w_t = w
            h_t = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h_t = h
            w_t = int(round(h * max(self.ratio)))
        else:
            w_t = w
            h_t = h
        i = (h - h_t) // 2
        j = (w - w_t) // 2
        img_pil = Image.fromarray(image)
        img_cropped = img_pil.crop((j, i, j + w_t, i + h_t))
        img_resized = img_cropped.resize((self.size, self.size), Image.BILINEAR)
        element['image'] = np.array(img_resized)
        return element

class Resize(grain.MapTransform):
    """Directly resizes the image to the target size without cropping."""
    def __init__(self, size: int):
        self.size = size

    def map(self, element: Dict[str, Any]) -> Dict[str, Any]:
        image = element['image']
        img_pil = Image.fromarray(image)
        img_resized = img_pil.resize((self.size, self.size), Image.BILINEAR)
        element['image'] = np.array(img_resized)
        return element

class ResizeAndCenterCrop(grain.MapTransform):
    """Mimics TIMM's eval resize and center crop."""
    def __init__(self, size: int, crop_pct: float = 0.875):
        self.size = size
        self.crop_pct = crop_pct

    def map(self, element: Dict[str, Any]) -> Dict[str, Any]:
        image = element['image']
        h, w, c = image.shape
        
        # Resize shortest edge
        scale_size = int(np.floor(self.size / self.crop_pct))
        if h < w:
            new_h = scale_size
            new_w = int(round(w * scale_size / h))
        else:
            new_w = scale_size
            new_h = int(round(h * scale_size / w))
            
        img_pil = Image.fromarray(image)
        img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
        
        # Center crop
        left = (new_w - self.size) // 2
        top = (new_h - self.size) // 2
        right = left + self.size
        bottom = top + self.size
        
        img_cropped = img_resized.crop((left, top, right, bottom))
        element['image'] = np.array(img_cropped)
        
        return element
