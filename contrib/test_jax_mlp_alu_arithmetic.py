'''
Von Neumann network that mimics an MLP for creating an ALU.

We use cells like a MLP
'''
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple
import time
import pickle

from sklearn.model_selection import train_test_split

from universe import UniverseLP #local
import data
import utils #local

utils.jax_status()

#parameters
epochs = 100
n = 8
bit_depth = 2**(n-1)
ctrl_bits = 4
ctrl_bit_depth = 2**ctrl_bits
samples = 128000
batch_size = 128
kernel_size = (17,3)
num_batches = samples//batch_size
total_steps = num_batches*epochs
delta = 0.2
padding = 2*max(kernel_size[0], kernel_size[1]) #on one side
epsilon = 1e-8
start_state = 1 #set None for random state
N = 2*n + padding
layers = 5
ctrl_gap = 1
print("N:", N, "Epochs:", epochs, "bit depth:", bit_depth)

#===========
#data
X1f, X2f, Cf, Y_raw = data.generate_arithmetic_logic(samples, n, ctrl_bits, batch_size, delta)

#stack
X_preproc = np.concatenate([X1f, X2f, Cf], axis=-1)
print("X_preproc:", X_preproc.shape)

X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)
print("X shape", X.shape, "Y shape:", Y.shape, "Y test shape:", Y_test.shape)
# print("X", X[:1])
# print("Y", Y[:1])

#split back to separate inputs/ctrl after train/test splits
X_splits = np.dsplit(X, 3)
print("X Split:", X_splits[0].shape)
X1 = X_splits[0]
X2 = X_splits[1]
C = X_splits[2]

X_test_splits = np.dsplit(X_test, 3)
# print("X Test Split:", X_test_splits[0].shape)
X1_test = X_test_splits[0]
X2_test = X_test_splits[1]
C_test = X_test_splits[2]

Y_packed = np.packbits(Y,axis=2)
Y_test_packed = np.packbits(Y_test,axis=2)
print("Y_packed shape", Y_packed.shape)
# print("Y_packed", Y_packed[:1])

#===========
##create model
class MLP(hk.Module):
    '''
    Multi-layer perceptron
    Applies flattening to input and returns logits
    '''
    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size

    def __call__(self, x):
        '''
        MLP Network
        '''
        #define layers
        net=hk.Flatten()(x)
        net=hk.Linear(24)(net)
        net=hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(20)(net)
        net=hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(16)(net)
        net=hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(12)(net)
        net=hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(10)(net)
        net=hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(self.output_size)(net)

        return net

def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = MLP(n)
    return mlp(batch)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))
# opt = optax.sgd(learning_rate=0.005)
lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0.001, peak_value=0.03, warmup_steps=int(0.025*total_steps), decay_steps=total_steps)
opt = optax.adamw(lr_schedule)
# opt = optax.adam(learning_rate=0.01)

# Initialize network and optimiser; note we draw an input to get shapes.
print("X0 shape:", X[0].shape)
x = X[0]
x = x[..., jnp.newaxis] #fake 2D
params, state = network.init(jax.random.PRNGKey(42), x)
opt_state = opt.init(params)
# print(params)
# print(state)

param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Total number of parameters: {param_count}")

#loss function
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    MSE
    '''
    #MSE loss
    logits, state = network.apply(params, state, batch)

    return optax.squared_error(predictions=logits, targets=labels).mean(), state

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
for step in range(1, epochs+1):
    # Do SGD on a batch of training examples.
    loss_value = 0
    for x, y in zip(X, Y):
        # print(x.shape)
        x = x[..., jnp.newaxis] #fake 2D
        # print("X1 shape:", x[0].shape)
        params, opt_state, value, state = update(params, opt_state, state, x, y)
        loss_value += value/batch_size

    if step % 10 == 0:
        Y_pred = []
        done = 0
        for x, y in zip(X, Y):
            x = x[..., jnp.newaxis] #fake 2D
            logits, state_test = network.apply(params, state, x)
            predictions = jnp.round(logits).astype(np.uint8)
            predictions = jnp.packbits(predictions,axis=1)
            Y_pred.append(predictions)
            if done < 16:
                y = y[0].astype(np.uint8)
                print(predictions[0], jnp.packbits(y,axis=0))
                done += 1
        # Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
        Y_pred = jnp.array(Y_pred)
        # print(Y_pred.shape, Y_packed.shape)
        accuracy = jnp.mean(Y_pred == Y_packed)
        print(f"[Step {step}, Loss {loss_value}, Accuracy {accuracy}]")
    else:
        print(f"[Step {step}, Loss {loss_value}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

#test performance
Y_pred = []
eval_start = time.time()
for x, y in zip(X_test, Y_test):
    x = x[..., jnp.newaxis] #fake 2D
    logits, state_test = network.apply(params, state, x)
    predictions = jnp.round(logits).astype(np.uint8)
    predictions = jnp.packbits(predictions,axis=1)
    Y_pred.append(predictions)
eval_end = time.time()
eval_elapsed = eval_end - eval_start

# Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
# print("After", Y_pred.shape, Y_pred)
Y_pred = jnp.array(Y_pred)
accuracy = jnp.mean(Y_pred == Y_test_packed)
print(f"[Final Accuracy {accuracy}]")

# Performance metrics
param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
memory_bytes = len(pickle.dumps(params))
throughput = len(X_test) / eval_elapsed

print(f"\n--- Performance Metrics ---")
print(f"Total Parameters: {param_count}")
print(f"Memory Footprint: {memory_bytes / 1024 / 1024:.4f} MB")
print(f"Inference Time: {eval_elapsed:.4f} secs")
print(f"Throughput: {throughput:.2f} samples/sec")

print("END")
