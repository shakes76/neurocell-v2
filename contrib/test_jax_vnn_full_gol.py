'''
Von Neumann network full model that mimics GoL
'''
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple
import time, math

from universe import UniverseLP #local
import patterns
import evolve
import utils #local

utils.jax_status()

#parameters
epochs = 100
N = 64
samples = 25600
batch_size = 128
l_steps = 8
kernel_size = (3,3)
directions = 64
depth = 2
num_batches = samples//batch_size
total_steps = num_batches*epochs

print(f"N: {N}, Epochs: {epochs}, Directions: {directions}, Depth: {depth}, Samples: {samples}, Batch Size: {batch_size}, L-steps: {l_steps}")

#===========
#data generation
def generate_gol_data(num_samples, steps, grid_size):
    X = []
    Y = []
    
    # Define patterns and their required dimensions (rows, cols)
    available_patterns = [
        (patterns.insertBlinker, 3, 2),
        (patterns.insertGlider, 3, 3),
        (patterns.insertToad, 4, 4),
        (patterns.insertPulsar, 13, 13),
        (patterns.insertGliderGun, 10, 37)
    ]
    
    for _ in range(num_samples // steps):
        universe = np.zeros((grid_size, grid_size))
        
        # Randomly select a pattern
        idx = np.random.randint(0, len(available_patterns))
        pattern_func, r_req, c_req = available_patterns[idx]
        
        # Ensure it fits within the grid size and won't crop during evolution
        # Buffer by steps on all sides to allow movement without hitting the border
        rx = np.random.randint(steps, max(steps + 1, grid_size - r_req - steps))
        ry = np.random.randint(steps, max(steps + 1, grid_size - c_req - steps))
        
        universe = pattern_func(universe, index=(rx, ry))
        
        for _ in range(steps):
            next_universe = evolve.evolve_fast(universe)
            X.append(np.array(universe))
            Y.append(np.array(next_universe))
            universe = next_universe
    return np.array(X), np.array(Y)

print("Generating data...")
X, Y = generate_gol_data(samples, l_steps, N)
print("X shape:", X.shape, "Y shape:", Y.shape)

input_size = (N, N)
output_size = (N, N)
input_offset = (0, 0)
output_offset = (0, 0)
activation = jax.nn.tanh
final_activation = None

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = UniverseLP(N, sharpen=False, gated_filters=False)
    mlp.initialize([input_size], output_size, [input_offset], output_offset, depth, kernel_size, directions=directions)
    return mlp(batch, activation, final_activation, norm=True)

# Make the network and optimizer.
network = hk.without_apply_rng(hk.transform_with_state(_forward))
lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0.0001, peak_value=0.003, warmup_steps=int(0.025*total_steps), decay_steps=total_steps, end_value=0.00005)
opt = optax.adamw(lr_schedule)

# Initialize network and optimiser
x = [X[0:1]] 

params, state = network.init(jax.random.PRNGKey(42), x)
opt_state = opt.init(params)

param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
print(f"Total number of parameters: {param_count}")

#loss function
def loss(params: hk.Params, state, batch, labels) -> jnp.ndarray:
    '''
    MSE
    '''
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

# Train loop.
start = time.time()
print("Starting training...")
for step in range(1, epochs+1):
    # Shuffle for epoch
    perm = np.random.permutation(len(X))
    X_shuffled = X[perm]
    Y_shuffled = Y[perm]
    
    total_loss = 0
    for i in range(0, len(X), batch_size):
        x_batch = X_shuffled[i:i+batch_size]
        y_batch = Y_shuffled[i:i+batch_size]
        
        x_inp = [x_batch]
        params, opt_state, value, state = update(params, opt_state, state, x_inp, y_batch)
        total_loss += value
        
    total_loss /= num_batches

    if step % 10 == 0:
        Y_pred = []
        for i in range(0, min(1000, len(X)), batch_size):
            x_batch = X[i:i+batch_size]
            x_inp = [x_batch]
            logits, _ = network.apply(params, state, x_inp)
            predictions = jnp.round(logits).astype(np.uint8)
            Y_pred.append(predictions)
            
        Y_pred = jnp.concatenate(Y_pred, axis=0)
        accuracy = jnp.mean(Y_pred == Y[:len(Y_pred)])
        print(f"[Step {step}, Loss {total_loss}, Accuracy {accuracy}]")
    else:
        print(f"[Step {step}, Loss {total_loss}]")

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

# Plots
# Green's Functions (Filters)
print("Saving filter plot...")
cols = 8
rows = math.ceil(directions / cols)
fig = plt.figure(figsize=(2*cols, 2*rows))
for c in range(directions):
    plt.subplot(rows, cols, c+1)
    plt.imshow(params['universe_lp']['f'][0,:,:,c])
    plt.tight_layout()
    plt.axis('off')
    plt.title("Filter "+str(c))
plt.savefig("vnn_full_gol_filter.png", dpi=150)
# plt.show()

# Final weights
print("Saving weights plot...")
plt.figure()
plt.imshow(params['universe_lp']['w'])
plt.title("Final Weights")
plt.savefig("vnn_full_gol_weights.png", dpi=150)

# Final biases
print("Saving biases plot...")
plt.figure()
plt.imshow(params['universe_lp']['b'])
plt.title("Final Biases")
plt.savefig("vnn_full_gol_biases.png", dpi=150)

# Final states
print("Saving states plot...")
states_arr = params['universe_lp']['s']
states_arr = jnp.where(states_arr>=0.6, 2, states_arr) 
states_arr = jnp.where(jnp.logical_and(states_arr>0,states_arr<0.6), 1, states_arr) 
plt.figure()
plt.imshow(states_arr)
plt.title("Final States (Thresholded)")
plt.savefig("vnn_full_gol_states_final_thres.png", dpi=150)
# plt.show()

# Animation
print("Generating animation...")
fig = plt.figure()
plt.gray()

# fresh glider
test_universe = np.zeros((N, N))
test_universe = patterns.insertGlider(test_universe, index=(2, 2))

img = plt.imshow(test_universe, animated=True)

def animate(i):
    global test_universe
    x_inp = [np.asarray(test_universe, dtype=np.float32)[np.newaxis, :, :]]
    logits, _ = network.apply(params, state, x_inp)
    test_universe = np.asarray(jnp.round(logits[0])).astype(np.uint8)
    img.set_array(test_universe)
    return img,

ani = animation.FuncAnimation(fig, animate, frames=30, interval=200, blit=False)
try:
    ani.save("vnn_full_gol_animation.gif", writer='pillow', fps=5)
    print("Saved animation to vnn_full_gol_animation.gif")
except Exception as e:
    print("Could not save gif:", e)
# plt.show()

# Save model
print("Saving model to vnn_full_gol_model.npz...")
jnp.savez("vnn_full_gol_model.npz",
            weights=params['universe_lp']['w'],
            biases=params['universe_lp']['b'],
            filters=params['universe_lp']['f'],
            states=params['universe_lp']['s'],
            N=N, kernel_size=kernel_size, directions=directions, depth=depth,
            allow_pickle=False)

# Save filters to Markdown
print("Saving filters to Markdown...")
filters_arr = params['universe_lp']['f']
utils.save_filters_to_markdown(filters_arr, filename="vnn_full_gol_filters.md")

print("END")
