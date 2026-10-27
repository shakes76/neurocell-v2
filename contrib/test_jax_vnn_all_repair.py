'''
Von Neumann network that mimics an MLP with a Turing machine structure.

We use cells like a tape
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
import time, math

from sklearn.model_selection import train_test_split

from universe import UniverseLP, VectorCells #local
from data import generate_arithmetic_logic
import utils #local

utils.jax_status()

#repair parameters
dmg_percentage_pixels = 0.03
dmg_pixel_value = -jnp.inf  # Constant value for the random pixels (-infinity)
fine_tune = True

#model hyper-parameters
N = 1 #unused
n = None
bit_depth = None
samples = 64000
batch_size = 128
gap = None
ctrl_gap = None
delta = 0.2
depth = None
# filename = "mlp_classify_cells.npz"
filename = "alu_arithmetic_cells.npz"
# filename = "alu_tm_arithmetic_cells.npz"
# length = None
# num_classes = None
input_size = None
input_offset = None
input_offset_x1 = None
input_offset_x2 = None
ctrl_size = None
ctrl_offset = None
ctrl_bits = None
output_size = None
output_offset = None
# layer_size = None
# layer_offset = None
kernel_size = None
activation = jax.nn.tanh
final_activation = None
directions = None

#fine tuning parameters
epochs = 50
num_batches = samples//batch_size
total_steps = num_batches*epochs

"""
TM ALU file format

weights=params['universe_lp']['w'],
biases=params['universe_lp']['b'],
filters=params['universe_lp']['f'],
states=params['universe_lp']['s'],
N=N, ctrl_bits=ctrl_bits, bit_depth=bit_depth, n=n, input_size=input_size, output_size=output_size,
input_offset_x1=input_offset_x1, input_offset_x2=input_offset_x2,
crtl_size=ctrl_size, ctrl_offset=ctrl_offset, output_offset=output_offset,
kernel_size=kernel_size,
"""
# load model parameters
with np.load(filename, allow_pickle=False) as data:
    weights = data['weights']
    biases = data['biases']
    states = data['states']
    filters = data['filters']
    N=int(data['N'])
    # length=data['length']
    depth=int(data['depth'])
    gap=int(data['gap'])
    ctrl_gap=int(data['ctrl_gap'])
    directions=int(data['directions'])
    # num_classes=data['num_classes']
    input_size=tuple(data['input_size'])
    output_size=tuple(data['output_size'])
    # input_offset=data['input_offset']
    input_offset_x1=tuple(data['input_offset_x1'])
    input_offset_x2=tuple(data['input_offset_x2'])
    output_offset=tuple(data['output_offset'])
    ctrl_size=tuple(data['ctrl_size'])
    ctrl_offset=tuple(data['ctrl_offset'])
    ctrl_bits=int(data['ctrl_bits'])
    bit_depth=int(data['bit_depth'])
    n=int(data['n'])
    # layer_size=data['layer_size']
    # layer_offset=data['layer_offset']
    kernel_size=tuple(data['kernel_size'])

#damage cells
dmg_num_pixels = int(N * N * dmg_percentage_pixels)   # Number of random pixels to generate (5% of total)
key = jax.random.PRNGKey(42)

# Generate the dmg image
dmg_image = utils.create_random_pixel_image(key, N, dmg_num_pixels, dmg_pixel_value)

#dmg model
# Apply the random pixel mask to the model parameters.
# Since the mask is 0 everywhere except the random pixels (-inf),
# adding it will turn the corresponding pixels in the weights to -inf.
weights[dmg_image == dmg_pixel_value] = 0
biases[dmg_image == dmg_pixel_value] = 0
dmg_states = states + dmg_image

#input and output states
s_init = VectorCells([input_size, input_size, ctrl_size], output_size, [input_offset_x1, input_offset_x2, ctrl_offset], output_offset, is_states=True, clear=False, random_init=None)
io_states = s_init(shape=(N,N), dtype=jnp.float32)
# io_states = hk.get_parameter("io", shape=(N,N), dtype=jnp.float32, init=s_init)

#restore io states
# dmg_states[io_states > 0] = io_states
# dmg_states = dmg_states.at[io_states > 0].set(io_states)
updated_dmg_states = jnp.where(
        io_states == 1, # Condition
        states,         # Restore values to take if condition is True
        dmg_states      # Values to take if condition is False
    )

print("N:", N, "Number of Bits:", n, "kernel_size", kernel_size)
print("Dmg Percentage Pixels:", dmg_percentage_pixels, "Dmg Num Pixels:", dmg_num_pixels)
print("input_size:", input_size, "output_size:", output_size, "ctrl_size:", ctrl_size)
print("depth:", depth, "directions:", directions)

plt.imshow(weights)
plt.title("Initial Damaged Weights")
plt.savefig("vnn_alu_arithmetic_repair_weights_init.png", dpi=150)
plt.show()

plt.imshow(biases)
plt.title("Initial Damaged Biases")
plt.savefig("vnn_alu_arithmetic_repair_biases_init.png", dpi=150)
plt.show()

plt.imshow(dmg_states)
plt.title("Initial Damaged States")
plt.savefig("vnn_alu_arithmetic_repair_states_init.png", dpi=150)
plt.show()

plt.imshow(io_states)
plt.title("Input-Output States")
plt.savefig("vnn_alu_arithmetic_repair_states_io.png", dpi=150)
plt.show()

plt.imshow(updated_dmg_states)
plt.title("Initial Damaged States IO Restored")
plt.savefig("vnn_alu_arithmetic_repair_states_restored.png", dpi=150)
plt.show()

#===========
#data
#data
X1f, X2f, Cf, Y_raw = generate_arithmetic_logic(samples, n, ctrl_bits, batch_size, delta)

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

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = UniverseLP(N, sharpen=False)
    mlp.initialize([input_size, input_size, ctrl_size], output_size, [input_offset_x1, input_offset_x2, ctrl_offset], output_offset, depth, kernel_size, directions=directions)
    return mlp(batch, activation, final_activation, norm=True)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))

# Initialize network and optimiser; note we draw an input to get shapes.
x = [X1[0], X2[0], C[0]] #one batch
for index, input in enumerate(x):
    x[index] = x[index][:, :, jnp.newaxis] #fake 2D
print("X0 shape:", x[0].shape)
params, state = network.init(jax.random.PRNGKey(42), x)

params['universe_lp']['w'] = weights
params['universe_lp']['b'] = biases
params['universe_lp']['s'] = updated_dmg_states
params['universe_lp']['f'] = filters

#optimizer stuff
# lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0.0002, peak_value=0.0001, warmup_steps=int(0.025*total_steps), decay_steps=total_steps, end_value=0.0001)
# opt = optax.adamw(lr_schedule)
opt = optax.adam(learning_rate=0.0001)
opt_state = opt.init(params)

#loss function
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    MSE
    '''
    #MSE loss
    logits, state = network.apply(params, state, batch)
    logits = jnp.squeeze(logits, axis=-1) #remove extra dim, must match output vector type

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

if fine_tune:
    # Train/eval loop.
    start = time.time() #time generation
    for step in range(1, epochs+1):
        # Do SGD on a batch of training examples.
        loss_value = 0
        for x1, x2, c, y in zip(X1, X2, C, Y):
            # print(x.shape)
            x = [x1, x2, c] #batch of inputs
            for index, input in enumerate(x):
                x[index] = x[index][:, :, jnp.newaxis] #fake 2D
            # print("X1 shape:", x[0].shape)
            params, opt_state, value, state = update(params, opt_state, state, x, y)
            loss_value += value/batch_size

            # x1 = jnp.round(x[0,:n,0]).astype(np.uint8)
            # x2 = jnp.round(x[0,n:,0]).astype(np.uint8)
            # y = y[0].astype(np.uint8)
            # print(jnp.packbits(x1,axis=0),jnp.packbits(x2,axis=0), jnp.packbits(y,axis=0))
            # break

        if step % 10 == 0:
            Y_pred = []
            for x1, x2, c, y in zip(X1, X2, C, Y):
                x = [x1, x2, c] #batch of inputs
                for index, input in enumerate(x):
                    x[index] = x[index][:, :, jnp.newaxis] #fake 2D
                logits, state_test = network.apply(params, state, x)
                predictions = jnp.squeeze(logits, axis=-1) #remove extra dim, must match output vector type
                predictions = jnp.round(predictions).astype(np.uint8)
                predictions = jnp.packbits(predictions,axis=1)
                Y_pred.append(predictions)
            # Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
            Y_pred = jnp.array(Y_pred)
            # print(Y_pred.shape, Y_packed.shape)
            accuracy = jnp.mean(Y_pred == Y_packed)
            print(f"[Step {step}, Loss {loss_value}, Accuracy {accuracy}]")
        else:
            print(f"[Step {step}, Loss {loss_value}]")
    end = time.time()
    elapsed = end - start
    print("Fine tuning took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

    jnp.savez("alu_arithmetic_repair_cells.npz",
                weights=params['universe_lp']['w'],
                biases=params['universe_lp']['b'],
                filters=params['universe_lp']['f'],
                states=params['universe_lp']['s'],
                N=N, ctrl_bits=ctrl_bits, bit_depth=bit_depth, n=n, input_size=input_size, output_size=output_size,
                depth=depth, input_offset_x1=input_offset_x1, input_offset_x2=input_offset_x2,
                ctrl_size=ctrl_size, ctrl_offset=ctrl_offset, output_offset=output_offset,
                kernel_size=kernel_size, directions=directions, gap=gap, ctrl_gap=ctrl_gap,
                allow_pickle=False)

    plt.imshow(params['universe_lp']['w'])
    plt.title("Final Weights")
    plt.savefig("vnn_alu_arithmetic_repair_weights.png", dpi=150)
    plt.show()

    plt.imshow(params['universe_lp']['b'])
    plt.title("Final Biases")
    plt.savefig("vnn_alu_arithmetic_repair_biases.png", dpi=150)
    plt.show()

    plt.imshow(params['universe_lp']['s'])
    plt.title("Final States")
    plt.savefig("vnn_alu_arithmetic_repair_states.png", dpi=150)
    plt.show()

    plt.imshow(states-params['universe_lp']['s'])
    plt.title("Difference States")
    plt.savefig("vnn_alu_arithmetic_repair_states_diff.png", dpi=150)
    plt.show()

#test performance
Y_pred = []
for x1, x2, c, y in zip(X1_test, X2_test, C_test, Y_test):
    x = [x1, x2, c] #batch of inputs
    for index, input in enumerate(x):
        x[index] = x[index][:, :, jnp.newaxis] #fake 2D
    logits, state_test = network.apply(params, state, x)
    predictions = jnp.squeeze(logits, axis=-1) #remove extra dim, must match output vector type
    predictions = jnp.round(predictions).astype(np.uint8)
    predictions = jnp.packbits(predictions,axis=1)
    Y_pred.append(predictions)
# Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
# print("After", Y_pred.shape, Y_pred)
Y_pred = jnp.array(Y_pred)
accuracy = jnp.mean(Y_pred == Y_test_packed)
print(f"[Final Accuracy {accuracy}]")

print("END")
