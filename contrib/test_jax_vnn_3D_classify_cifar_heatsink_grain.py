'''
Von Neumann network that creates a 3D MLP for classifying CIFAR10 dataset

Uses full 3D convs as the forward operator, expensive.
'''
import os
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# 1. Prevent JAX from taking all GPU memory at start
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# 2. (Optional) Limit JAX to a specific percentage of GPU memory
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

# 3. (Optional) Bypasses JAX's internal memory pool and forces it to use the standard CUDA malloc and free. Slower!
# This is often much better at handling the massive, temporary workspace spikes caused by 3D FFTs without fragmenting.
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm import tqdm, trange #progbar
import time
import pickle

#dataset
import grains
from grains import create_cifar10_dataloader #local

from universe import Universe3D #local
import utils #local
import visualize #local

utils.jax_status()

#parameters
epochs = 35
num_classes = 10
layers = 16
batch_size = 100
length = 32
channels = 3
grains.padding = 3
padding = grains.padding
kernel_size = (11,11,41) #channel first, depth last
depth_full = 2*layers #-(channels-1) #(k-1)/2 hyperplanes per layer, counting from last channel
depth_input = 2*padding+channels

max_depth = depth_full + depth_input - 1  #first plane is untouched, doesn't count
# depth = (depth_full+length)//( (kernel_size[0]-1)//2 ) # propagation speed is (kernel_size-1)//2 per step, go upto first channel
depth = 4
blocks = 4
expand = True
blocks_depth = depth//blocks #depth per block
Nx = length + 1*(kernel_size[0]-1)
Ny = length + 1*(kernel_size[1]-1)
Nz = max_depth + 1*(kernel_size[2]-1) + 4 #channel last
print("N:", (Nx,Ny,Nz), "depth full:", depth_full, "Epochs:", epochs, "Kernel:", kernel_size)
print("padding:", padding, "depth_input:", depth_input, "max_depth:", max_depth, "blocks", blocks, "blocks_depth", blocks_depth)
print("Expand Mode:", expand)

#network IO paths
path = '/home/shakes/'
#~ path = 'F:/'

#===========
# Data
# Create the dataloader and get total images
train_loader, total_images = create_cifar10_dataloader(
    split="train",
    batch_size=batch_size,
    shuffle=True,
    num_epochs=epochs+1, #only upto this
    worker_count=0, # Use 0 for debugging in the main thread
    data_dir=path+'data'
)

test_loader, total_test_images = create_cifar10_dataloader(
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

#locations
#MLP with 784 inputs and 10 outputs, 3 layers inbetween
#setup network position
centerx = Nx//2
centery = Ny//2
centerz = Nz//2
#input
input_top = centerx-length//2
input_left = centery-length//2
input_inner = centerz-max_depth//2
#output, doesn't have to be at end of layers, propagator will learn directions to output
output_top = centerx-num_classes//2
output_left = centery
# output_inner = input_inner+max_depth+1 #include out layer
output_inner = input_inner+(Nz-depth_input)//2+1 #middle of circular distance
print("input_top", input_top, "input_left", input_left, "input_inner", input_inner)
print("output_top", output_top, "output_left", output_left, "output_inner", output_inner)
#setup parameters and lengths
input_size = (length,length,depth_input)
input_offset = (input_top,input_left,input_inner)
output_size = (num_classes,1,1)
output_offset = (output_top,output_left,output_inner)
# activation = utils.sqnl
activation = jax.nn.leaky_relu #leaky_relu, swish, mish
final_activation = jax.nn.leaky_relu #softmax done in loss
directions = 32 #how many filters, i.e. possible directions, to learn
print("depth:", depth, "directions:", directions, "total batches:", total_batches)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = Universe3D((Nx,Ny,Nz), blocks=blocks, sharpen=False, expand=expand)
    mlp.initialize_3D(input_size, output_size, input_offset, output_offset, depth, kernel_size, directions=directions)
    return mlp(batch, activation, final_activation, norm=True)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))

# Make the network and optimizer. Haiku standard
total_steps = (total_batches*epochs)
schedule = optax.warmup_cosine_decay_schedule(init_value=0.001,peak_value=0.03,warmup_steps=0.02*total_steps,decay_steps=total_steps,end_value=0.0005)
# opt = optax.adam(learning_rate=schedule)
opt = optax.adamw(learning_rate=schedule)

# AGC should typically be applied BEFORE the optimizer (SGD/Adam) updates are calculated
# or AFTER, depending on interpretation. In NFNets, it clips the raw gradients.
# clip_factor = 0.01 #AGC parameter
# opt = optax.chain(
#     utils.adaptive_gradient_clipping(clipping_threshold=clip_factor),
#     optax.adamw(learning_rate=schedule),
# )

# Initialize network and optimiser; note we draw an input to get shapes.
jax_batch = jax.tree.map(jnp.asarray, next(train_ds))
x = jax_batch['image']
# x = jnp.zeros((batch_size,length,length,channels))
# print("INIT SHAPE", x.shape)
params, state = network.init(jax.random.PRNGKey(42), x)
opt_state = opt.init(params)
# print(params)

# plot first batch
center_point = (length//2, length//2, (depth_input-1)//2)
visualize.plot_orthogonal_views(x[0], center=center_point, cmap="gray")

param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Total number of parameters: {param_count}")

# plt.imshow(params['universe3_d']['s'][:,center,:])
# plt.title("Initial States")
# plt.savefig("vnn_3D_classify_cifar_states_init.png", dpi=150)
# plt.show()
visualize.plot_orthogonal_views(params['universe3_d']['s'], center=output_offset)

#loss function
@jax.jit
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    Cross entropy
    '''
    # labels = jax.nn.one_hot(labels, num_classes)
    logits, state = network.apply(params, state, batch)
    logits = logits.reshape(logits.shape[0],num_classes)
    # print("LOSS SHAPE SQUEEZE", labels.shape, logits.shape)

    # return -jnp.mean(labels * jax.nn.log_softmax(logits)) + l2*l2_scaling, state
    # return -jnp.mean(labels * jax.nn.log_softmax(logits)), state

    # return optax.softmax_cross_entropy(logits=logits, labels=labels).mean(), state
    return optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean(), state

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
    updates, opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, state

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

        params, opt_state, loss_value, state = update(params, opt_state, state, x, y)
        total_loss += loss_value/batch_size
        # break
    total_loss /= total_batches

    # visualize.plot_orthogonal_views(state['universe3_d']['universe'], center=input_offset)
    # visualize.plot_orthogonal_views(state['universe3_d']['universe'], center=output_offset)
    # print("universe shape", state['universe3_d']['universe'].shape)
    # break

    if epoch % 1 == 0:
        # Periodically evaluate on train & test sets.
        correct = 0
        total = 0
        for test_step in trange(total_test_batches):
            # Convert numpy arrays to JAX DeviceArrays
            jax_batch = jax.tree.map(jnp.asarray, next(test_ds))

            x = jax_batch['image']
            y = jax_batch['label']

            logits, state_test = network.apply(params, state, x) #if reshaped, use squeeze
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

visualize.plot_orthogonal_views(state['universe3_d']['universe'], center=output_offset)

jnp.savez("3D_classify_cifar_cells.npz",
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
    # Convert numpy arrays to JAX DeviceArrays
    jax_batch = jax.tree.map(jnp.asarray, next(test_ds))

    x = jax_batch['image']
    y = jax_batch['label']

    logits, state_test = network.apply(params, state, x)
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

plt.imshow(params['universe3_d']['w'][:,centery,:])
plt.title("Final Weights")
plt.savefig("vnn_3D_classify_cifar_weights.png", dpi=150)
plt.show()

plt.imshow(params['universe3_d']['b'][:,centery,:])
plt.title("Final Biases")
plt.savefig("vnn_3D_classify_cifar_biases.png", dpi=150)
plt.show()

center_point = (kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2)
# plt.imshow(params['universe3_d']['f'][0,:,center,:,0])
# plt.title("Final Filter")
# plt.savefig("vnn_3D_classify_cifar_filter.png", dpi=150)
# plt.show()
visualize.plot_orthogonal_views(params['universe3_d']['f'][0,0,...,0], center=center_point)
visualize.plot_orthogonal_views(params['universe3_d']['f'][1,0,...,0], center=center_point)

plt.imshow(params['universe3_d']['s'][:,centery,:])
plt.title("Final States")
plt.savefig("vnn_3D_classify_cifar_states.png", dpi=150)
plt.show()

center_point = (input_top, centery, centerz)
# plt.imshow(params['universe3_d']['s'][:,centery,:])
# plt.title("Final States")
# plt.savefig("vnn_3D_classify_cifar_states.png", dpi=150)
# plt.show()
visualize.plot_orthogonal_views(params['universe3_d']['s'], center=center_point)

states = params['universe3_d']['s']
states = jnp.where(states>=0.6, 2, states) #activate if passive, more accurate
states = jnp.where(jnp.logical_and(states>0,states<0.6), 1, states) #weight if signal

plt.imshow(states[:,centery,:])
plt.title("Final States (Thresholded)")
plt.savefig("vnn_3D_classify_cifar_states_thres.png", dpi=150)
plt.show()

plt.imshow(state['universe3_d']['universe'][:,centery,:])
plt.title("Final Universe")
plt.savefig("vnn_3D_classify_cifar_universe.png", dpi=150)
plt.show()

print("END")
