'''
Von Neumann network that creates a 3D MLP for classifying Imagenette dataset

Uses hyperplanes (n-1 D) convs per layer as the forward operator, more efficient.
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

from universe import Universe3D #local
import utils #local
import visualize #local

utils.jax_status()

#parameters
epochs = 35
num_classes = 10
layers = 20
batch_size = 64
length = 56
channels = 3
padding = 2
grains.padding = padding
kernel_size = (9,9,41) #channel first, depth last
depth_full = 2*layers #-(channels-1) #(k-1)/2 hyperplanes per layer, counting from last channel
depth_input = 2*padding+channels

max_depth = depth_full  #count from first hplane after input
depth = max_depth//((kernel_size[-1]-1)//2)
blocks = 8
expand = True
blocks_depth = depth//blocks #depth per block
Nx = length + 1*(kernel_size[0]-1) + 4
Ny = length + 1*(kernel_size[1]-1) + 4
Nz = (depth_input-1) + max_depth + 1*(kernel_size[2]-1) + 4 #channel last
print("N:", (Nx,Ny,Nz), "depth full:", depth_full, "Epochs:", epochs, "Kernel:", kernel_size)
print("padding:", padding, "depth_input:", depth_input, "max_depth:", max_depth, "blocks", blocks, "blocks_depth", blocks_depth)
print("Expand Mode:", expand)

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
    grains.SpreadChannelsTransform(padding_size=padding),
    grain.Batch(batch_size=batch_size, drop_remainder=True)
]
train_loader = grain.DataLoader(data_source=train_source, sampler=train_sampler, operations=train_operations, worker_count=0)
print("Training images found:", len(train_source))

val_source = FolderDataSource(os.path.join(path, 'imagenette2-160/val'))
val_sampler = grain.IndexSampler(num_records=len(val_source), shuffle=False, seed=42, num_epochs=None)
val_operations = [
    grains.RandomResizedCrop(size=length),
    grains.ImageNetNormalize(),
    grains.SpreadChannelsTransform(padding_size=padding),
    grain.Batch(batch_size=batch_size, drop_remainder=True)
]
test_loader = grain.DataLoader(data_source=val_source, sampler=val_sampler, operations=val_operations, worker_count=0)
print("Test images found:", len(val_source))

total_batches = len(train_source) // batch_size
total_test_batches = len(val_source) // batch_size

#locations
#MLP with inputs and 10 outputs, 3 layers inbetween
#setup network position
centerx = Nx//2
centery = Ny//2
centerz = Nz//2
#input
input_top = centerx-length//2
input_left = centery-length//2
input_inner = centerz-((depth_input-1)+max_depth)//2
#output, doesn't have to be at end of layers, propagator will learn directions to output
output_top = centerx-num_classes//2
output_left = centery
output_inner = input_inner+depth_input+max_depth #include out layer
print("input_top", input_top, "input_left", input_left, "input_inner", input_inner)
print("output_top", output_top, "output_left", output_left, "output_inner", output_inner)
#setup parameters and lengths
input_size = (length,length,depth_input)
input_offset = (input_top,input_left,input_inner)
output_size = (num_classes,1,1)
output_offset = (output_top,output_left,output_inner)
activation = jax.nn.leaky_relu
final_activation = jax.nn.leaky_relu
directions = 32 #how many filters, i.e. possible directions, to learn
print("depth:", depth, "directions:", directions, "total batches:", total_batches)

#===========
##create model
def _forward(batch, is_training: bool = True) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = Universe3D((Nx,Ny,Nz), blocks=blocks, sharpen=False, expand=expand)
    mlp.initialize_hyperplane(input_size, output_size, input_offset, output_offset, depth, kernel_size, directions=directions)
    return mlp(batch, activation, final_activation, norm=True, is_training=is_training)

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
params, state = network.init(jax.random.PRNGKey(42), x)
opt_state = opt.init(params)

# plot first batch
center_point = (length//2, length//2, (depth_input-1)//2)
visualize.plot_orthogonal_views(x[0], center=center_point, cmap="gray")

param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
print(f"Total number of parameters: {param_count}")

visualize.plot_orthogonal_views(params['universe3_d']['s'], center=output_offset)

#loss function
@jax.jit
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    Cross entropy
    '''
    labels = jax.nn.one_hot(labels, num_classes)
    logits, state = network.apply(params, state, batch, is_training=True)
    logits = logits.reshape(logits.shape[0],num_classes)

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

            logits, state_test = network.apply(params, state, x, is_training=False) #if reshaped, use squeeze
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

jnp.savez("3D_hplane_classify_imagenette_cells.npz",
            weights=params['universe3_d']['w'],
            biases=params['universe3_d']['b'],
            filters=params['universe3_d']['f'],
            states=params['universe3_d']['s'],
            N=(Nx,Ny,Nz), length=length, num_classes=num_classes, input_size=input_size, output_size=output_size,
            input_offset=input_offset, output_offset=output_offset,
            kernel_size=kernel_size,
            allow_pickle=False)

#test performance
correct = 0
total = 0
eval_start = time.time()
for test_step in trange(total_test_batches):
    batch = jax.tree.map(jnp.asarray, next(test_ds))
    x = batch['image']
    y = batch['label']

    logits, state_test = network.apply(params, state, x, is_training=False)
    predicted = logits.reshape(logits.shape[0],num_classes)
    predicted = jnp.argmax(predicted, axis=-1)
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

plt.imshow(params['universe3_d']['w'][centerx,...])
plt.title("Final Weights")
plt.savefig("vnn_3D_hplane_classify_imagenette_weights.png", dpi=150)
plt.close()

plt.imshow(params['universe3_d']['b'][centerx,...])
plt.title("Final Biases")
plt.savefig("vnn_3D_hplane_classify_imagenette_biases.png", dpi=150)
plt.close()

center_point = (kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2)
visualize.plot_orthogonal_views(params['universe3_d']['f'][0,0,...,0], center=center_point)
visualize.plot_orthogonal_views(params['universe3_d']['f'][-1,0,...,0], center=center_point)

plt.imshow(params['universe3_d']['s'][centerx,...])
plt.title("Final States")
plt.savefig("vnn_3D_hplane_full_classify_imagenette_states.png", dpi=150)
plt.close()

states = params['universe3_d']['s']
states = jnp.where(states>=0.6, 2, states) #activate if passive, more accurate
states = jnp.where(jnp.logical_and(states>0,states<0.6), 1, states) #weight if signal

plt.imshow(states[centerx,...])
plt.title("Final States (Thresholded)")
plt.savefig("vnn_3D_hplane_full_classify_imagenette_states_thres.png", dpi=150)
plt.close()

plt.imshow(state['universe3_d']['universe'][centerx,...])
plt.title("Final Universe")
plt.savefig("vnn_3D_hplane_full_classify_imagenette_universe.png", dpi=150)
plt.close()

print("Layer Norm Parameters\n", params['universe3_d']['scale'], "\n", params['universe3_d']['shift'])

print("END")
