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
import time

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from universe import UniverseLP #local
import utils #local

utils.jax_status()

#parameters
epochs = 20
batch_size = 2
kernel_size = (7,7)
padding = 4*max(kernel_size[0], kernel_size[1]) #on one side
epsilon = 1e-8
start_state = 1 #set None for random state

#===========
##data
data_dict = datasets.load_wine(return_X_y=False)
X_raw = data_dict['data']
Y_raw = data_dict['target']
length = len(data_dict['feature_names'])
samples = len(Y_raw)
num_classes = len(data_dict['target_names'])
N = (length+num_classes+1)+2*padding #padded both sides
print("N:", N, "Epochs:", epochs)
print("Data Shape:", X_raw.shape, "Samples:", samples, "Input Length:", length, "Classes:", num_classes)

#scaling
scaler = StandardScaler()
X_preproc = scaler.fit_transform(X_raw)

#split
data_split = 18
X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=data_split, random_state=42)
num_batches = (samples-data_split)//batch_size
print("Y shape:", Y.shape, "Y test shape:", Y_test.shape, "Num Batches", num_batches)

#reshape to number of batches
X = np.reshape(X, (num_batches,batch_size,length))
Y = np.reshape(Y, (num_batches,batch_size))
X_test = np.reshape(X_test, (data_split//batch_size,batch_size,length))
Y_test = np.reshape(Y_test, (data_split//batch_size,batch_size))
print("X shape:", X.shape, "X test shape:", X_test.shape)
print("Y shape:", Y.shape, "Y test shape:", Y_test.shape)

#locations
#MLP with 13 inputs and 3 outputs, 2 layers inbetween
#setup network position, but in a line like a tape
#inputs and outputs sequentially in 1D with gap
center = N//2
gap = 1
input_mid = center #centre vertically
input_left = padding #one pad away from edge
output_mid = center
output_left = input_left+length+gap #include out layer
#setup parameters and lengths
input_size = (1,length)
input_offset = (input_mid,input_left)
output_size = (1,num_classes)
output_offset = (output_mid,output_left)
activation = jax.nn.tanh
final_activation = jax.nn.sigmoid
directions = 4 #how many filters, i.e. possible directions, to learn
depth = output_left-input_left
print("depth:", depth)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = UniverseLP(N, sharpen=False)
    mlp.initialize_tm(input_size, output_size, input_offset, output_offset, depth, kernel_size, directions=directions)
    return mlp(batch, activation, final_activation, norm=True)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))
# opt = optax.sgd(learning_rate=0.005)
opt = optax.adam(learning_rate=0.01)

# Initialize network and optimiser; note we draw an input to get shapes.
x = X[0]
print("X0 shape:", x.shape)
x = x[jnp.newaxis,:] #fake 2D
params, state = network.init(jax.random.PRNGKey(42), x)
opt_state = opt.init(params)
# print(params)
# print(state)

plt.imshow(params['universe_lp']['s'])
plt.title("Initial States")
plt.savefig("vnn_tm_classify_states_init.png", dpi=75)
plt.show()

#loss function
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    Cross entropy 
    '''
    labels = jax.nn.one_hot(labels, num_classes)

    #CE loss
    logits, state = network.apply(params, state, batch)
    print(logits.shape, labels.shape)
    logits = jnp.squeeze(logits, axis=1) #remove extra dim at row
    ce = -1 * jnp.mean(labels * jnp.log(logits + epsilon) + (1 - labels) * jnp.log(1. - logits + epsilon))

    return ce, state

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
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, state

# Train/eval loop.
start = time.time() #time generation
for step in range(1, epochs+1):
    # Do SGD on a batch of training examples.
    loss_value = 0
    for x, y in zip(X, Y):
        x = x[:, jnp.newaxis, :] #fake 2D, but horizontal
        params, opt_state, value, state = update(params, opt_state, state, x, y)
        loss_value += value/batch_size
        # break

    if step % 10 == 0:
        Y_pred = []
        for x, y in zip(X, Y):
            x = x[:, jnp.newaxis, :] #fake 2D, but horizontal
            logits, state_test = network.apply(params, state, x)
            predictions = jnp.squeeze(logits, axis=1) #remove extra dim at front
            Y_pred.append(predictions)
        Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
        # Y_pred = Y_pred[:, jnp.newaxis] #fake 2D
        accuracy = jnp.mean(Y_pred == Y)
        print(f"[Step {step}, Loss {loss_value}, Accuracy {accuracy}]")
    else:
        print(f"[Step {step}, Loss {loss_value}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

jnp.savez("tm_classify_cells.npz", 
            weights=params['universe_lp']['w'], 
            biases=params['universe_lp']['b'], 
            filters=params['universe_lp']['f'], 
            states=params['universe_lp']['s'], 
            N=N, length=length, num_classes=num_classes, input_size=input_size, output_size=output_size,
            input_offset=input_offset, output_offset=output_offset,
            kernel_size=kernel_size,
            allow_pickle=False)

plt.imshow(params['universe_lp']['w'])
plt.title("Final Weights")
plt.savefig("vnn_tm_classify_weights.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['b'])
plt.title("Final Biases")
plt.savefig("vnn_tm_classify_biases.png", dpi=150)
plt.show()

fig = plt.figure(figsize=(4,4))
for c in range(directions):
    plt.subplot(2, 2, c+1)
    plt.imshow(params['universe_lp']['f'][0,:,:,c])
    plt.tight_layout()
    plt.title("Filter "+str(c))
plt.savefig("vnn_tm_classify_filter.png", dpi=75)
plt.show()

plt.imshow(params['universe_lp']['s'])
plt.title("Final States")
plt.savefig("vnn_tm_classify_states_final.png", dpi=75)
plt.show()

states = params['universe_lp']['s']
states = jnp.where(states>=0.6, 2, states) #activate if passive, more accurate
states = jnp.where(jnp.logical_and(states>0,states<0.6), 1, states) #weight if signal

plt.imshow(states)
plt.title("Final States (Thresholded)")
plt.savefig("vnn_tm_classify_states_final_thres.png", dpi=75)
plt.show()

#test performance
correct = 0
total = 0
for x, y in zip(X_test, Y_test):
    x = x[:, jnp.newaxis, :] #fake 2D, but horizontal
    logits, state_test = network.apply(params, state, x)
    predicted = jnp.squeeze(logits, axis=1) #remove extra dim at front
    predicted = jnp.argmax(jnp.array(predicted), axis=-1)
    correct += jnp.sum(predicted == y)
    total += batch_size
accuracy = 100 * correct / total
print(f"[Final Test Accuracy {accuracy}]")

print("END")