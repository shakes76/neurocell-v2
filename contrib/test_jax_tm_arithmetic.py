'''
Von Neumann network that mimics a TM with a tape model for arithmetic
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

from sklearn.model_selection import train_test_split

from universe import UniverseTM #local
import data
import utils #local

utils.jax_status()

#parameters
epochs = 30
n = 8
# bit_depth = 2**(n-1)
length = 2*n
blanks = 2*n
samples = 32000
batch_size = 128
num_batches = samples//batch_size
delta = 0.2
padding = 4*(length+n+blanks)
N = length + n + blanks + 16*padding
plot_ratio = 24
plot_height = 1
print("N:", N, "Epochs:", epochs)

#===========
#data
X1f, X2f, Y_raw = data.generate_arithmetic(samples, "xor", n, batch_size, delta)

#stack
X_preproc = np.concatenate([X1f, X2f], axis=-1)
# print(X_preproc.shape)

X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)
print("X shape", X.shape, "Y shape:", Y.shape, "Y test shape:", Y_test.shape)
# print("X", X[:1])
# print("Y", Y[:1])

Y_packed = np.packbits(Y,axis=2)
Y_test_packed = np.packbits(Y_test,axis=2)
print("Y_packed shape", Y_packed.shape)
# print("Y_packed", Y_packed[:1])

#locations
#MLP with inputs and outputs, depth layers inbetween
#setup
center = (1, N//2)
input_size = (1,length)
input_offset = (0,center[1]-(2*n+1)-blanks//2) #add blanks
output_size = (1,n)
output_offset = (0,center[1]+(n+1)+blanks//2) #+1 output layer, add blanks
kernel_size = (1,8*n+1) #(1,3*n+2)
activation = jax.nn.tanh
final_activation = None
directions = 16 #how many instructions, i.e. possible directions, to learn
depth = 4
print("depth:", depth, "in offset", input_offset, "out offset", output_offset)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    tm = UniverseTM(N, sharpen=False)
    tm.initialize(input_size, output_size, input_offset, output_offset, depth, kernel_size, directions=directions)
    return tm(batch, activation, final_activation, norm=True)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))

#optimizer
total_steps = epochs*num_batches
schedule = optax.warmup_cosine_decay_schedule(init_value=0.001,peak_value=0.03,warmup_steps=0.025*total_steps,decay_steps=total_steps)
# opt = optax.sgd(learning_rate=0.005)
# opt = optax.adam(learning_rate=schedule)
opt = optax.adamw(learning_rate=schedule)

# Initialize network and optimiser; note we draw an input to get shapes.
x = X[0]
x = x[:, jnp.newaxis, :] #fake 2D
print("X0 shape:", x.shape)
params, state = network.init(jax.random.PRNGKey(42), x)
opt_state = opt.init(params)
# print(params)
# print(state)

fig = plt.figure(figsize=(plot_ratio,plot_height))
plt.imshow(params['universe_tm']['s'], aspect='auto')
plt.title("Initial States")
plt.tight_layout()
plt.savefig("tm_arith_states_init.png", dpi=75)
plt.show()

#loss function
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    Cross entropy
    '''
    #MSE loss
    logits, state = network.apply(params, state, batch)
    logits = jnp.squeeze(logits, axis=-2) #remove extra dim, row in this case

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
        x = x[:, jnp.newaxis, :] #fake 2D

        params, opt_state, value, state = update(params, opt_state, state, x, y)
        loss_value += value/batch_size
        # print("x:",x)
        # print("cells:",state['universe_tm']['universe'])

        # x1 = jnp.round(x[0,:n,0]).astype(np.uint8)
        # x2 = jnp.round(x[0,n:,0]).astype(np.uint8)
        # y = y[0].astype(np.uint8)
        # print(jnp.packbits(x1,axis=0),jnp.packbits(x2,axis=0), jnp.packbits(y,axis=0))
        # break

    if step % 10 == 0:
        Y_pred = []
        done = 0
        for x, y in zip(X, Y):
            x = x[:, jnp.newaxis, :] #fake 2D
            logits, state_test = network.apply(params, state, x)
            predictions = jnp.squeeze(logits, axis=-2) #remove extra dim, row in this case
            predictions = jnp.round(predictions).astype(np.uint8)
            predictions = jnp.packbits(predictions,axis=1)
            Y_pred.append(predictions)
            if done < 3:
                x1 = jnp.round(x[0,:n]).astype(np.uint8)
                x2 = jnp.round(x[0,n:]).astype(np.uint8)
                y = y[0].astype(np.uint8)
                print(jnp.packbits(x1,axis=0),jnp.packbits(x2,axis=0), predictions[0], jnp.packbits(y,axis=0))
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

jnp.savez("tm_arith_cells.npz",
            weights=params['universe_tm']['w'],
            biases=params['universe_tm']['b'],
            filters=params['universe_tm']['f'],
            states=params['universe_tm']['s'],
            N=N, length=length, n=n, input_size=input_size, output_size=output_size,
            input_offset=input_offset, output_offset=output_offset,
            kernel_size=kernel_size,
            allow_pickle=False)

fig = plt.figure(figsize=(plot_ratio,plot_height))
plt.imshow(params['universe_tm']['w'], aspect='auto')
plt.title("Final Weights")
plt.tight_layout()
plt.savefig("tm_arith_weights.png", dpi=150)
plt.show()

fig = plt.figure(figsize=(plot_ratio,plot_height))
plt.imshow(params['universe_tm']['b'], aspect='auto')
plt.title("Final Biases")
plt.tight_layout()
plt.savefig("tm_arith_biases.png", dpi=150)
plt.show()

fig = plt.figure(figsize=(plot_ratio//4,directions*2))
for c in range(directions):
    plt.subplot(directions, 1, c+1)
    plt.imshow(params['universe_tm']['f'][0,:,:,c], aspect='auto')
    plt.axis('off') # Hides both x and y axes
    # plt.title("Filter "+str(c))
    plt.title("") #disable
    plt.tight_layout()
plt.savefig("tm_arith_filter.png", dpi=75)
plt.show()

fig = plt.figure(figsize=(plot_ratio,plot_height))
plt.imshow(params['universe_tm']['s'], aspect='auto')
plt.title("Final States")
plt.tight_layout()
plt.savefig("tm_arith_states_final.png", dpi=150)
plt.show()

states = params['universe_tm']['s']
states = jnp.where(states>=0.6, 2, states) #activate if passive, more accurate
states = jnp.where(jnp.logical_and(states>0,states<0.6), 1, states) #weight if signal

fig = plt.figure(figsize=(plot_ratio,plot_height))
plt.imshow(states, aspect='auto')
plt.title("Final States (Thresholded)")
plt.tight_layout()
plt.savefig("tm_arith_states_final_thres.png", dpi=150)
plt.show()

fig = plt.figure(figsize=(plot_ratio,plot_height))
plt.imshow(state['universe_tm']['universe'][0], aspect='auto')
plt.title("Final Universe")
plt.tight_layout()
plt.savefig("tm_arith_universe.png", dpi=150)
plt.show()

#function plot
Y_pred = []
for x, y in zip(X_test, Y_test):
    x = x[:, jnp.newaxis, :] #fake 2D
    logits, state_test = network.apply(params, state, x)
    predictions = jnp.squeeze(logits, axis=-2) #remove extra dim, row in this case
    predictions = jnp.round(predictions).astype(np.uint8)
    predictions = jnp.packbits(predictions,axis=1)
    Y_pred.append(predictions)
# Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
# print("After", Y_pred.shape, Y_pred)
Y_pred = jnp.array(Y_pred)
accuracy = jnp.mean(Y_pred == Y_test_packed)
print(f"[Final Accuracy {accuracy}]")

print("END")
