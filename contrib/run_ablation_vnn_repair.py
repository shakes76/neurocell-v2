import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import time
import csv
from typing import Tuple
from sklearn.model_selection import train_test_split
from tqdm import trange

from universe import UniverseLP, VectorCells
from data import generate_arithmetic_logic
import utils

utils.jax_status()

# Base Configuration from test_jax_vnn_all_repair.py
filename = "alu_arithmetic_cells.npz"
samples = 64000
batch_size = 128
delta = 0.2
finetune_epochs = 50
num_batches = samples // batch_size
total_steps = num_batches * finetune_epochs

# Load config once to setup things
with np.load(filename, allow_pickle=False) as data:
    N = int(data['N'])
    depth = int(data['depth'])
    directions = int(data['directions'])
    input_size = tuple(data['input_size'])
    output_size = tuple(data['output_size'])
    input_offset_x1 = tuple(data['input_offset_x1'])
    input_offset_x2 = tuple(data['input_offset_x2'])
    output_offset = tuple(data['output_offset'])
    ctrl_size = tuple(data['ctrl_size'])
    ctrl_offset = tuple(data['ctrl_offset'])
    ctrl_bits = int(data['ctrl_bits'])
    bit_depth = int(data['bit_depth'])
    n = int(data['n'])
    kernel_size = tuple(data['kernel_size'])

# Generate Data
def get_data():
    X1f, X2f, Cf, Y_raw = generate_arithmetic_logic(samples, n, ctrl_bits, batch_size, delta)
    X_preproc = np.concatenate([X1f, X2f, Cf], axis=-1)
    X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)
    
    X_splits = np.dsplit(X, 3)
    X1, X2, C = X_splits[0], X_splits[1], X_splits[2]
    
    X_test_splits = np.dsplit(X_test, 3)
    X1_test, X2_test, C_test = X_test_splits[0], X_test_splits[1], X_test_splits[2]
    
    Y_packed = np.packbits(Y, axis=2)
    Y_test_packed = np.packbits(Y_test, axis=2)
    
    return X1, X2, C, Y, Y_packed, X1_test, X2_test, C_test, Y_test, Y_test_packed

# Forward Model
activation = jax.nn.tanh
final_activation = None

def _forward(batch) -> jnp.ndarray:
    mlp = UniverseLP(N, sharpen=False)
    mlp.initialize([input_size, input_size, ctrl_size], output_size, [input_offset_x1, input_offset_x2, ctrl_offset], output_offset, depth, kernel_size, directions=directions)
    return mlp(batch, activation, final_activation, norm=True)

network = hk.without_apply_rng(hk.transform_with_state(_forward))

def evaluate_model(params, state, X1_test, X2_test, C_test, Y_test_packed):
    Y_pred = []
    for x1, x2, c in zip(X1_test, X2_test, C_test):
        x = [x1, x2, c]
        for index in range(len(x)):
            x[index] = x[index][:, :, jnp.newaxis]
        logits, state_test = network.apply(params, state, x)
        predictions = jnp.squeeze(logits, axis=-1)
        predictions = jnp.round(predictions).astype(np.uint8)
        predictions = jnp.packbits(predictions, axis=1)
        Y_pred.append(predictions)
    
    Y_pred = jnp.array(Y_pred)
    accuracy = jnp.mean(Y_pred == Y_test_packed)
    return float(accuracy)

def get_damaged_params(dmg_percentage, rng_key):
    with np.load(filename, allow_pickle=False) as data:
        weights = data['weights'].copy()
        biases = data['biases'].copy()
        states = data['states'].copy()
        filters = data['filters'].copy()
        
    dmg_num_pixels = int(N * N * dmg_percentage)
    dmg_pixel_value = -jnp.inf
    
    if dmg_num_pixels > 0:
        dmg_image = utils.create_random_pixel_image(rng_key, N, dmg_num_pixels, dmg_pixel_value)
        
        weights[dmg_image == dmg_pixel_value] = 0
        biases[dmg_image == dmg_pixel_value] = 0
        dmg_states = states + dmg_image
        
        s_init = VectorCells([input_size, input_size, ctrl_size], output_size, [input_offset_x1, input_offset_x2, ctrl_offset], output_offset, is_states=True, clear=False, random_init=None)
        io_states = s_init(shape=(N,N), dtype=jnp.float32)
        
        updated_dmg_states = jnp.where(io_states == 1, states, dmg_states)
    else:
        updated_dmg_states = states
    
    return weights, biases, updated_dmg_states, filters

def train_model(params, state, X1, X2, C, Y, epochs):
    opt = optax.adam(learning_rate=0.0001)
    opt_state = opt.init(params)
    
    def loss_fn(params: hk.Params, state, batch, labels) -> jnp.ndarray:
        logits, state = network.apply(params, state, batch)
        logits = jnp.squeeze(logits, axis=-1)
        return optax.squared_error(predictions=logits, targets=labels).mean(), state

    @jax.jit
    def update(params: hk.Params, opt_state: optax.OptState, state, batch, labels):
        (loss_value, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, batch, labels)
        updates, opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss_value, state
    
    pbar = trange(1, epochs + 1, desc="Fine-tuning")
    for step in pbar:
        loss_value = 0
        for x1, x2, c, y in zip(X1, X2, C, Y):
            x = [x1, x2, c]
            for index in range(len(x)):
                x[index] = x[index][:, :, jnp.newaxis]
            params, opt_state, value, state = update(params, opt_state, state, x, y)
            loss_value += float(value) / batch_size
        pbar.set_postfix(loss=f"{loss_value:.4f}")
        
    return params, state

def main():
    print("==================================================")
    print("        VNN REPAIR ABLATION STUDY SETUP           ")
    print("==================================================")
    
    X1, X2, C, Y, Y_packed, X1_test, X2_test, C_test, Y_test, Y_test_packed = get_data()
    
    # Initialize network state to get structure
    x_init = [X1[0][:, :, jnp.newaxis], X2[0][:, :, jnp.newaxis], C[0][:, :, jnp.newaxis]]
    rng = jax.random.PRNGKey(42)
    params_struct, state = network.init(rng, x_init)
    
    # Get Baseline Accuracy
    print("\n=== Evaluating Baseline (No Damage) ===")
    w, b, s, f = get_damaged_params(0.0, rng)
    params_base = dict(params_struct)
    params_base['universe_lp'] = dict(params_base['universe_lp'])
    params_base['universe_lp']['w'] = w
    params_base['universe_lp']['b'] = b
    params_base['universe_lp']['s'] = s
    params_base['universe_lp']['f'] = f
    
    base_accuracy = evaluate_model(params_base, state, X1_test, X2_test, C_test, Y_test_packed)
    print(f"Base Model Accuracy: {base_accuracy:.4f}")
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params_base))
    param_bytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(params_base))
    state_bytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(state))
    model_size_mb = (param_bytes + state_bytes) / (1024 ** 2)

    results = []
    results.append({
        'Damage Level': '0.00 (Baseline)',
        'Accuracy Before Repair': base_accuracy,
        'Accuracy After Repair': base_accuracy,
        'Repair Time (s)': 0.0,
        'Params Count': param_count,
        'Model Size (MB)': model_size_mb
    })
    
    damage_levels = [0.01, 0.02, 0.03, 0.04, 0.05]
    
    print("\n=== Starting Ablation over Damage Levels ===")
    
    for dmg in damage_levels:
        print(f"\n--- Damage Level: {dmg} ---")
        rng, subkey = jax.random.split(rng)
        
        # 1. Apply damage
        w, b, s, f = get_damaged_params(dmg, subkey)
        params_dmg = dict(params_struct)
        params_dmg['universe_lp'] = dict(params_dmg['universe_lp'])
        params_dmg['universe_lp']['w'] = w
        params_dmg['universe_lp']['b'] = b
        params_dmg['universe_lp']['s'] = s
        params_dmg['universe_lp']['f'] = f
        
        # 2. Evaluate before repair
        acc_before = evaluate_model(params_dmg, state, X1_test, X2_test, C_test, Y_test_packed)
        print(f"Accuracy after {dmg*100:.1f}% damage (before repair): {acc_before:.4f}")
        
        # 3. Fine-tune to repair
        start_repair = time.time()
        repaired_params, repaired_state = train_model(
            params_dmg, state, X1, X2, C, Y, finetune_epochs
        )
        repair_time = time.time() - start_repair
        
        # 4. Evaluate after repair
        acc_after = evaluate_model(repaired_params, repaired_state, X1_test, X2_test, C_test, Y_test_packed)
        print(f"Accuracy after {finetune_epochs} epochs repair: {acc_after:.4f}")
        
        results.append({
            'Damage Level': f"{dmg:.2f}",
            'Accuracy Before Repair': acc_before,
            'Accuracy After Repair': acc_after,
            'Repair Time (s)': repair_time,
            'Params Count': param_count,
            'Model Size (MB)': model_size_mb
        })
        
    # Save results to CSV
    csv_filename = "ablation_results_repair.csv"
    with open(csv_filename, "w", newline='') as f:
        fieldnames = ['Damage Level', 'Accuracy Before Repair', 'Accuracy After Repair', 'Repair Time (s)', 'Params Count', 'Model Size (MB)']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_filename}")
    
    # Save results to Markdown
    md_filename = "ablation_results_repair.md"
    with open(md_filename, "w") as f:
        f.write("# VNN Repair Ablation Study Results\n\n")
        f.write(f"This document contains the accuracies of the VNN MLP on the arithmetic task before and after {finetune_epochs}-epoch repair at varying damage levels.\n\n")
        
        f.write("| Damage Level | Accuracy Before Repair | Accuracy After Repair | Repair Time (s) | Params Count | Model Size (MB) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['Damage Level']} | {r['Accuracy Before Repair']:.4f} | {r['Accuracy After Repair']:.4f} | {r['Repair Time (s)']:.2f} | {r['Params Count']} | {r['Model Size (MB)']:.4f} |\n")
            
    print(f"Markdown results saved to {md_filename}")

if __name__ == "__main__":
    main()
