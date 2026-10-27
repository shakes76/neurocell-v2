'''
JAX baseline MLP for XOR task matching VNN script data pipeline
'''
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import time
from sklearn.model_selection import train_test_split

import data
import utils

utils.jax_status()

# Parameters matching VNN arithmetic pipeline
epochs = 100
n = 8
length = 2 * n
samples = 32000
batch_size = 128
num_batches = samples // batch_size
delta = 0.2
hidden_size = 128

# Data Pipeline
print(f"Generating data: samples={samples}, n={n}, delta={delta}")
X1f, X2f, Y_raw = data.generate_arithmetic(samples, "xor", n, batch_size, delta)
X_preproc = np.concatenate([X1f, X2f], axis=-1)

X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)

# Packbits for exact match accuracy evaluation
Y_test_packed = np.packbits(Y_test, axis=2)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("X_test shape:", X_test.shape)

# Create model
def _forward(batch) -> jnp.ndarray:
    # 4 layers (3 hidden layers + 1 output layer) to match PyTorch script depth
    mlp = hk.nets.MLP([hidden_size, hidden_size, hidden_size, n], activation=jax.nn.tanh)
    return jax.nn.sigmoid(mlp(batch))

network = hk.without_apply_rng(hk.transform(_forward))

# Optimizer
total_steps = epochs * num_batches
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.001, peak_value=0.03, warmup_steps=0.025 * total_steps, 
    decay_steps=total_steps, end_value=0.0005
)
opt = optax.adamw(learning_rate=schedule)

# Initialization
x_init = X[0]
params = network.init(jax.random.PRNGKey(42), x_init)
opt_state = opt.init(params)

# Loss and Update functions
def loss(params: hk.Params, batch, labels) -> jnp.ndarray:
    logits = network.apply(params, batch)
    return optax.squared_error(predictions=logits, targets=labels).mean()

@jax.jit
def update(params: hk.Params, opt_state: optax.OptState, batch, labels):
    loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
    updates, opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value

# Training Loop
print("> Training")
start = time.time()
for step in range(1, epochs + 1):
    loss_value = 0
    for x, y in zip(X, Y):
        params, opt_state, value = update(params, opt_state, x, y)
        loss_value += float(value) / batch_size
        
    if step % 10 == 0 or step == epochs:
        print(f"[Epoch {step}/{epochs}] Loss: {loss_value:.4f}")
        
elapsed = time.time() - start
print(f"Training took {elapsed:.2f} seconds.")

# Evaluation
print("> Testing")
eval_start = time.time()
Y_pred = []
for x, y in zip(X_test, Y_test):
    logits = network.apply(params, x)
    predictions = jnp.round(logits).astype(np.uint8)
    predictions = jnp.packbits(predictions, axis=1)
    Y_pred.append(predictions)
    
eval_end = time.time()
eval_elapsed = eval_end - eval_start

Y_pred = jnp.array(Y_pred)
accuracy = jnp.mean(Y_pred == Y_test_packed)
print(f"Final Accuracy: {accuracy:.4f}")

# Performance metrics
param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
param_bytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(params))
model_size_mb = param_bytes / (1024 ** 2)

total_samples = X_test.shape[0] * X_test.shape[1]
throughput = total_samples / eval_elapsed

print(f"\n--- Performance Metrics ---")
print(f"Total Parameters: {param_count}")
print(f"Model Size (MB): {model_size_mb:.4f} MB")
print(f"Training Time: {elapsed:.2f} seconds")
print(f"Inference Time: {eval_elapsed:.4f} secs")
print(f"Throughput: {throughput:.2f} samples/sec")
print("END")
