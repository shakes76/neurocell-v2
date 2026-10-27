'''
MLP for classifying CIFAR10 dataset

Uses conventions similar to the VNN 3D MLP script test_jax_vnn_3D_classify_cifar_hyperplane_grain.py
'''
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.925"

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm import tqdm, trange #progbar
import time
import pickle

#dataset
import grains #local
import utils #local

utils.jax_status()

#parameters
epochs = 35
num_classes = 10
batch_size = 100
length = 32
channels = 3
grains.padding = 2
padding = grains.padding

depth_input = 2*padding+channels

print("Epochs:", epochs, "padding:", padding, "depth_input:", depth_input)

#network IO paths
path = '/home/shakes/'
# path = 'F:/'

#===========
# Data
# Create the dataloader and get total images
train_loader, total_images = grains.create_cifar10_dataloader(
    split="train",
    batch_size=batch_size,
    shuffle=True,
    num_epochs=epochs+1, #only upto this
    worker_count=0, # Use 0 for debugging in the main thread
    data_dir=path+'data'
)

test_loader, total_test_images = grains.create_cifar10_dataloader(
    split="test",
    batch_size=batch_size,
    shuffle=False,
    num_epochs=epochs+1, #only upto this
    worker_count=0, # Use 0 for debugging in the main thread
    data_dir=path+'data'
)

total_batches = total_images // batch_size
total_test_batches = total_test_images // batch_size

# Grain dataloaders yield standard python dictionaries with numpy arrays.
# For JAX, we iterate through the loader and move the arrays to the device.
train_ds = iter(train_loader)
test_ds = iter(test_loader)

#===========
##create model
def _forward(batch, is_training=True) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    # Flatten from [B, 32, 32, C] to [B, 32*32*C]
    x = hk.Flatten()(batch)

    # Initial projection block to get to our working dimension
    x = hk.Linear(1024)(x)
    x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
    x = jax.nn.gelu(x)

    # Stack 4 Residual Blocks (Dense -> Norm -> GeLU -> Dropout -> Dense -> Norm -> Add)
    # Decreasing dimensions to act as an encoder
    dims = [1024, 512, 256, 128]
    for dim in dims:
        orig_x = x
        
        # Project residual connection if dimensions change
        if orig_x.shape[-1] != dim:
            orig_x = hk.Linear(dim, with_bias=False)(orig_x)
            orig_x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(orig_x)

        x = hk.Linear(dim)(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), 0.2, x)

        x = hk.Linear(dim)(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)

        # The crucial residual connection
        x = x + orig_x

        x = jax.nn.gelu(x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), 0.2, x)

    # Output layer for 10 classes
    logits = hk.Linear(num_classes)(x)
    return logits

# Make the network and optimizer. Haiku standard
network = hk.transform(_forward)

total_steps = (total_batches*epochs)
schedule = optax.warmup_cosine_decay_schedule(init_value=0.001,peak_value=0.03,warmup_steps=0.02*total_steps,decay_steps=total_steps,end_value=0.0005)
opt = optax.adamw(learning_rate=schedule)

# Initialize network and optimiser; note we draw an input to get shapes.
jax_batch = jax.tree.map(jnp.asarray, next(train_ds))
x = jax_batch['image']

rng = jax.random.PRNGKey(42)
rng, init_rng = jax.random.split(rng)
params = network.init(init_rng, x, is_training=True)
opt_state = opt.init(params)

param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Total number of parameters: {param_count}")

#loss function
@jax.jit
def loss(params: hk.Params, rng, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    Cross entropy
    '''
    labels = jax.nn.one_hot(labels, num_classes)
    logits = network.apply(params, rng, batch, is_training=True)
    logits = logits.reshape(logits.shape[0],num_classes)

    return optax.softmax_cross_entropy(logits=logits, labels=labels).mean()

@jax.jit
def update(
        params: hk.Params,
        opt_state: optax.OptState,
        rng,
        batch,
        labels
    ) -> Tuple[hk.Params, optax.OptState, jnp.ndarray, jnp.ndarray]:
    """Learning rule (stochastic gradient descent)."""
    rng, subkey = jax.random.split(rng)
    loss_value, grads = jax.value_and_grad(loss)(params, subkey, batch, labels)
    updates, opt_state = opt.update(grads, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, rng

# Train/eval loop.
start = time.time() #time generation
for epoch in range(epochs):
    total_loss = 0
    # Do SGD on a batch of training examples.
    for step in trange(total_batches):
        # Convert numpy arrays to JAX DeviceArrays
        jax_batch = jax.tree.map(jnp.asarray, next(train_ds))

        x = jax_batch['image']
        y = jax_batch['label']

        params, opt_state, loss_value, rng = update(params, opt_state, rng, x, y)
        total_loss += loss_value/batch_size
        
    total_loss /= total_batches

    if epoch % 1 == 0:
        # Periodically evaluate on train & test sets.
        correct = 0
        total = 0
        dummy_rng = jax.random.PRNGKey(0)
        for test_step in trange(total_test_batches):
            # Convert numpy arrays to JAX DeviceArrays
            jax_batch = jax.tree.map(jnp.asarray, next(test_ds))

            x = jax_batch['image']
            y = jax_batch['label']

            logits = network.apply(params, dummy_rng, x, is_training=False)
            predicted = logits.reshape(logits.shape[0],num_classes)
            predicted = jnp.argmax(predicted, axis=-1)
            correct += jnp.sum(predicted == y)
            total += y.shape[0]
        accuracy = 100 * correct / total
        print(f"[Epoch {epoch}, Loss {total_loss}, Test Accuracy {accuracy}]")
    else:
        print(f"[Epoch {epoch}, Loss {total_loss}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

#test performance
correct = 0
total = 0
dummy_rng = jax.random.PRNGKey(0)
eval_start = time.time()
for test_step in trange(total_test_batches):
    # Convert numpy arrays to JAX DeviceArrays
    jax_batch = jax.tree.map(jnp.asarray, next(test_ds))

    x = jax_batch['image']
    y = jax_batch['label']

    logits = network.apply(params, dummy_rng, x, is_training=False)
    predicted = logits.reshape(logits.shape[0],num_classes)
    predicted = jnp.argmax(predicted, axis=-1)
    correct += jnp.sum(predicted == y)
    total += y.shape[0]

# Block on JAX computation to get accurate timing
jax.block_until_ready(logits)
eval_end = time.time()
eval_elapsed = eval_end - eval_start

accuracy = 100 * correct / total
print(f"[Final Test Accuracy {accuracy}]")

# Performance metrics
param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
memory_bytes = len(pickle.dumps(params))
throughput = total / eval_elapsed

print(f"\n--- Performance Metrics ---")
print(f"Total Parameters: {param_count}")
print(f"Memory Footprint: {memory_bytes / 1024 / 1024:.4f} MB")
print(f"Inference Time: {eval_elapsed:.4f} secs")
print(f"Throughput: {throughput:.2f} samples/sec")

print("END")