'''
Simple MLP trained as a classify network for a simple dataset - MNIST
'''
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import Tuple
import time
import pickle
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms

#parameters
epochs = 8
batch_size = 64
path = '/home/shakes/'

#===========
#transform preproc
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,)), transforms.Lambda(lambda x: x.permute(1, 2, 0))])

##data
trainset = torchvision.datasets.MNIST(root=path+'data/mnist', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root=path+'data/mnist', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

num_classes = 10
length = 28*28
total_batches = len(train_loader)
print("Input Length:", length, "Classes:", num_classes)

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
        net=hk.Linear(256)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(128)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(64)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(32)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(self.output_size)(net)

        return net

def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = MLP(num_classes)
    return mlp(batch)

# Make the network and optimiser.
network = hk.without_apply_rng(hk.transform(_forward))
# opt = optax.sgd(learning_rate=0.005)
opt = optax.adam(learning_rate=0.0001)

#loss function
def loss(params: hk.Params, batch, labels) -> jnp.ndarray:
    '''
    Cross entropy 
    '''
    labels = jax.nn.one_hot(labels, num_classes)
    logits = network.apply(params, batch)
    return -jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))

@jax.jit
def update(
        params: hk.Params,
        opt_state: optax.OptState,
        batch,
        labels
    ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value

# Initialize network and optimiser; note we draw an input to get shapes.
dataset_iter = iter(train_loader)
x, y = next(dataset_iter)
x = jnp.reshape(x.numpy(), (x.shape[0], length))
params = network.init(jax.random.PRNGKey(42), x)
opt_state = opt.init(params)

# Train/eval loop.
start = time.time() #time generation
for step in range(epochs+1):
    total_loss = 0
    # Do SGD on a batch of training examples.
    for x, y in tqdm(train_loader):
        x = jnp.asarray(x.numpy())
        y = jnp.asarray(y.numpy())
        x = jnp.reshape(x, (x.shape[0], length))

        params, opt_state, loss_value = update(params, opt_state, x, y)
        total_loss += loss_value
        
    total_loss /= total_batches

    if step % 1 == 0:
        # Periodically evaluate on train & test sets.
        correct = 0
        total = 0
        for x, y in tqdm(test_loader):
            x = jnp.asarray(x.numpy())
            y = jnp.asarray(y.numpy())
            x = jnp.reshape(x, (x.shape[0], length))

            logits = network.apply(params, x)
            predicted = jnp.argmax(jnp.array(logits), axis=-1)
            correct += jnp.sum(predicted == y)
            total += y.shape[0]
        accuracy = 100 * correct / total
        print(f"[Step {step}, Loss {total_loss}, Test Accuracy {accuracy}]")

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

#test performance
correct = 0
total = 0
eval_start = time.time()
for x, y in tqdm(test_loader):
    x = jnp.asarray(x.numpy())
    y = jnp.asarray(y.numpy())
    x = jnp.reshape(x, (x.shape[0], length))

    logits = network.apply(params, x)
    predicted = jnp.argmax(jnp.array(logits), axis=-1)
    correct += jnp.sum(predicted == y)
    total += y.shape[0]
eval_end = time.time()
eval_elapsed = eval_end - eval_start

accuracy = 100 * correct / total
print(f"[Final Test Accuracy {accuracy}]")

# Performance metrics
param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
memory_bytes = len(pickle.dumps(params))
throughput = total / eval_elapsed

print(f"--- Performance Metrics ---")
print(f"Total Parameters: {param_count}")
print(f"Memory Footprint: {memory_bytes / 1024 / 1024:.4f} MB")
print(f"Inference Time: {eval_elapsed:.4f} secs")
print(f"Throughput: {throughput:.2f} samples/sec")

print("END")
