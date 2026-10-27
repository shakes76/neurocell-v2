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

from universe import UniverseLP
import data
import utils

utils.jax_status()

# Fixed parameters based on test_jax_vnn_mlp_arithmetic.py
epochs = 100
n = 8
length = 2 * n
samples = 32000
batch_size = 128
num_batches = samples // batch_size
delta = 0.2
layers = 5

def run_training(directions: int, N: int, kernel_size: Tuple[int, int]) -> dict:
    print(f"\n--- Starting run: directions={directions}, N={N}, kernel_size={kernel_size} ---")
    
    # 1. Data generation (using "xor" for the ablation)
    X1f, X2f, Y_raw = data.generate_arithmetic(samples, "xor", n, batch_size, delta)
    X_preproc = np.concatenate([X1f, X2f], axis=-1)
    
    X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)
    Y_test_packed = np.packbits(Y_test, axis=2)
    
    # 2. Setup VNN locations
    center = (N // 2, N // 2)
    depth = 2 * layers + 1
    input_size = (length, 1)
    input_offset = (center[0] - length // 2, center[1] - (depth - 1) // 2)
    output_size = (n, 1)
    output_offset = (center[0] - output_size[0] // 2, input_offset[1] + depth + 1)
    activation = jax.nn.tanh
    final_activation = None
    
    # 3. Create model
    def _forward(batch) -> jnp.ndarray:
        mlp = UniverseLP(N, sharpen=False)
        mlp.initialize_mlp(input_size, output_size, input_offset, output_offset, depth, kernel_size, directions=directions)
        return mlp(batch, activation, final_activation, norm=True)

    network = hk.without_apply_rng(hk.transform_with_state(_forward))
    
    # 4. Optimizer
    total_steps = epochs * num_batches
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.001, peak_value=0.03, warmup_steps=0.025 * total_steps, 
        decay_steps=total_steps, end_value=0.0005
    )
    opt = optax.adamw(learning_rate=schedule)
    
    # 5. Initialization
    x_init = X[0][:, :, jnp.newaxis]
    params, state = network.init(jax.random.PRNGKey(42), x_init)
    opt_state = opt.init(params)
    
    # Calculate Memory Use
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    param_bytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(params))
    state_bytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(state))
    model_size_mb = (param_bytes + state_bytes) / (1024 ** 2)
    
    # 6. Loss and Update functions
    def loss(params: hk.Params, state, batch, labels) -> jnp.ndarray:
        logits, state = network.apply(params, state, batch)
        logits = jnp.squeeze(logits, axis=-1)
        return optax.squared_error(predictions=logits, targets=labels).mean(), state

    @jax.jit
    def update(params: hk.Params, opt_state: optax.OptState, state, batch, labels):
        (loss_value, state), grads = jax.value_and_grad(loss, has_aux=True)(params, state, batch, labels)
        updates, opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss_value, state
    
    # 7. Training Loop
    start = time.time()
    pbar = trange(1, epochs + 1, desc="Training")
    for step in pbar:
        loss_value = 0
        for x, y in zip(X, Y):
            x = x[:, :, jnp.newaxis]
            params, opt_state, value, state = update(params, opt_state, state, x, y)
            loss_value += float(value) / batch_size
            
        pbar.set_postfix(loss=f"{loss_value:.4f}")
            
    elapsed = time.time() - start
    print(f"Training took {elapsed:.2f} seconds.")
    
    # 8. Evaluation
    Y_pred = []
    for x, y in zip(X_test, Y_test):
        x = x[:, :, jnp.newaxis]
        logits, state_test = network.apply(params, state, x)
        predictions = jnp.squeeze(logits, axis=-1)
        predictions = jnp.round(predictions).astype(np.uint8)
        predictions = jnp.packbits(predictions, axis=1)
        Y_pred.append(predictions)
        
    Y_pred = jnp.array(Y_pred)
    accuracy = jnp.mean(Y_pred == Y_test_packed)
    print(f"Final Accuracy: {accuracy:.4f}")
    
    return {
        'Accuracy': float(accuracy),
        'Total Time (s)': elapsed,
        'Time per Epoch (s)': elapsed / epochs,
        'Params Count': param_count,
        'Model Size (MB)': model_size_mb
    }

def main():
    results = []
    
    # Baselines
    base_kernel = (17, 3)
    base_directions = 8
    base_padding = 2 * max(base_kernel[0], base_kernel[1])
    base_N = 2 * n + base_padding
    
    # Ablation values
    directions_to_test = [2, 4, 12, 16]
    grid_size_offsets = [-10, 10, 20, 30]
    kernel_sizes_to_test = [(5, 3), (9, 3), (13, 3), (21, 3)]
    
    total_ablations = 3
    total_runs = 1 + len(directions_to_test) + len(grid_size_offsets) + len(kernel_sizes_to_test)
    
    print("==================================================")
    print("        VNN MLP XOR ABLATION STUDY SETUP          ")
    print("==================================================")
    print(f"Total Runs Scheduled: {total_runs}")
    print(f"Total Ablations (Parameters Varied): {total_ablations}")
    print("Strategy: One-at-a-time variations from baseline")
    print("==================================================\n")
    
    # Run Baseline
    print("=== Running Baseline ===")
    base_metrics = run_training(base_directions, base_N, base_kernel)
    results.append({'Ablation': 'Baseline', 'Value': 'Baseline', **base_metrics})
    
    # 1. Ablation: Number of Green's functions (directions)
    print("\n=== Ablating Directions ===")
    for d in directions_to_test:
        metrics = run_training(d, base_N, base_kernel)
        results.append({'Ablation': 'Directions', 'Value': d, **metrics})
        
    # 2. Ablation: Grid Size (N)
    print("\n=== Ablating Grid Size (N) ===")
    for offset in grid_size_offsets:
        n_val = base_N + offset
        metrics = run_training(base_directions, n_val, base_kernel)
        results.append({'Ablation': 'Grid Size', 'Value': n_val, **metrics})
        
    # 3. Ablation: Kernel Size
    print("\n=== Ablating Kernel Size ===")
    for k in kernel_sizes_to_test:
        # Ensure N is large enough for the new kernel size padding
        required_padding = 2 * max(k[0], k[1])
        required_N = 2 * n + required_padding
        test_N = max(base_N, required_N)
        metrics = run_training(base_directions, test_N, k)
        results.append({'Ablation': 'Kernel Size', 'Value': str(k), **metrics})
        
    # Save results to CSV
    csv_filename = "ablation_results_xor.csv"
    with open(csv_filename, "w", newline='') as f:
        fieldnames = ['Ablation', 'Value', 'Accuracy', 'Total Time (s)', 'Time per Epoch (s)', 'Params Count', 'Model Size (MB)']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_filename}")
    
    # Save results to Markdown
    md_filename = "ablation_results_xor.md"
    with open(md_filename, "w") as f:
        f.write("# VNN MLP XOR Ablation Study Results\n\n")
        f.write("This document contains the final accuracies of the VNN MLP on the XOR arithmetic task while ablating various hyperparameters.\n\n")
        
        # Get unique ablation types
        ablations = []
        for r in results:
            if r['Ablation'] not in ablations and r['Ablation'] != 'Baseline':
                ablations.append(r['Ablation'])
                
        # Generate a table for each ablation type
        for ab in ablations:
            f.write(f"## Ablation: {ab}\n\n")
            f.write("| Value | Accuracy | Total Time (s) | Time per Epoch (s) | Params Count | Model Size (MB) |\n")
            f.write("|---|---|---|---|---|---|\n")
            # Always write the baseline for comparison
            f.write(f"| Baseline | {base_metrics['Accuracy']:.4f} | {base_metrics['Total Time (s)']:.2f} | {base_metrics['Time per Epoch (s)']:.4f} | {base_metrics['Params Count']} | {base_metrics['Model Size (MB)']:.4f} |\n")
            
            for r in results:
                if r['Ablation'] == ab:
                    f.write(f"| {r['Value']} | {r['Accuracy']:.4f} | {r['Total Time (s)']:.2f} | {r['Time per Epoch (s)']:.4f} | {r['Params Count']} | {r['Model Size (MB)']:.4f} |\n")
            f.write("\n")
            
    print(f"Markdown results saved to {md_filename}")

if __name__ == "__main__":
    main()
