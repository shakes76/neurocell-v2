'''
ResNet18 baseline model for classifying Imagenette dataset
'''
import os
import glob
from typing import Dict, Any, Tuple
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.925"

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm, trange #progbar
import time
import pickle
import numpy as np
from PIL import Image

import grain.python as grain
import grains # local module

import utils #local

utils.jax_status()

#parameters
epochs = 35
num_classes = 10
batch_size = 64
length = 56
channels = 3

#network IO paths
path = 'E:/data/imagenet/'
output_path = ''

#===========
# Data Loader Source
class FolderDataSource:
    def __init__(self, root_dir):
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for ext in ('*.JPEG', '*.jpg', '*.png'):
                for img_path in glob.glob(os.path.join(cls_dir, ext)):
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_path, label = self.samples[index]
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            return {'image': np.array(img), 'label': label}

# Data Pipelines
print("> Loading Data via Grain")
train_source = FolderDataSource(os.path.join(path, 'imagenette2-160/train'))
train_sampler = grain.IndexSampler(num_records=len(train_source), shuffle=True, seed=42, num_epochs=None)
train_operations = [
    grains.RandomResizedCrop(size=length),
    grains.RandomHorizontalFlip(),
    grains.ImageNetNormalize(),
    grain.Batch(batch_size=batch_size, drop_remainder=True)
]
train_loader = grain.DataLoader(data_source=train_source, sampler=train_sampler, operations=train_operations, worker_count=0)
print("Training images found:", len(train_source))

val_source = FolderDataSource(os.path.join(path, 'imagenette2-160/val'))
val_sampler = grain.IndexSampler(num_records=len(val_source), shuffle=False, seed=42, num_epochs=None)
val_operations = [
    grains.RandomResizedCrop(size=length),
    grains.ImageNetNormalize(),
    grain.Batch(batch_size=batch_size, drop_remainder=True)
]
test_loader = grain.DataLoader(data_source=val_source, sampler=val_sampler, operations=val_operations, worker_count=0)
print("Test images found:", len(val_source))

total_batches = len(train_source) // batch_size
total_test_batches = len(val_source) // batch_size

#===========
##create model
def _forward(batch, is_training) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    resnet = hk.nets.ResNet18(num_classes, resnet_v2=True)
    return resnet(batch, is_training=is_training)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))

total_steps = (total_batches*epochs)
schedule = optax.warmup_cosine_decay_schedule(init_value=0.001,peak_value=0.03,warmup_steps=0.02*total_steps,decay_steps=total_steps,end_value=0.0005)
opt = optax.adamw(learning_rate=schedule)

# Initialize network and optimiser; note we draw an input to get shapes.
train_ds = iter(train_loader)
test_ds = iter(test_loader)

init_batch = jax.tree.map(jnp.asarray, next(train_ds))
x = init_batch['image']
y = init_batch['label']
params, state = network.init(jax.random.PRNGKey(42), x, is_training=True)
opt_state = opt.init(params)

param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
print(f"Total number of parameters: {param_count}")

#loss function
@jax.jit
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    Cross entropy
    '''
    labels = jax.nn.one_hot(labels, num_classes)
    logits, state = network.apply(params, state, batch, is_training=True)
    
    return optax.softmax_cross_entropy(logits=logits, labels=labels).mean(), state

@jax.jit
def update(
        params: hk.Params,
        opt_state: optax.OptState,
        state,
        batch,
        labels
    ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    (loss_value, state), grads = jax.value_and_grad(loss, has_aux=True)(params, state, batch, labels)
    updates, opt_state = opt.update(grads, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, state

@jax.jit
def eval_step(params, state, batch):
    logits, _ = network.apply(params, state, batch, is_training=False)
    return logits

# Train/eval loop.
start = time.time() #time generation
for epoch in range(epochs):
    total_loss = 0
    # Do SGD on a batch of training examples.
    for step in trange(total_batches):
        batch = jax.tree.map(jnp.asarray, next(train_ds))
        x = batch['image']
        y = batch['label']

        params, opt_state, loss_value, state = update(params, opt_state, state, x, y)
        total_loss += loss_value/batch_size
        
    total_loss /= total_batches

    if epoch % 1 == 0:
        # Periodically evaluate on train & test sets.
        correct = 0
        total = 0
        for test_step in trange(total_test_batches):
            batch = jax.tree.map(jnp.asarray, next(test_ds))
            x = batch['image']
            y = batch['label']

            logits = eval_step(params, state, x) 
            predicted = jnp.argmax(logits, axis=-1)
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
eval_start = time.time()
for test_step in trange(total_test_batches):
    batch = jax.tree.map(jnp.asarray, next(test_ds))
    x = batch['image']
    y = batch['label']

    logits = eval_step(params, state, x)
    predicted = jnp.argmax(logits, axis=-1)
    correct += jnp.sum(predicted == y)
    total += y.shape[0]
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
