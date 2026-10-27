import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import data
import forward

def main():
    print("Loading model parameters...")
    loaded = jnp.load("tm_arith_cells.npz")
    weights = loaded['weights']
    biases = loaded['biases']
    states = loaded['states']
    filters = loaded['filters']
    N = loaded['N'].item()
    n = loaded['n'].item()
    input_offset = tuple(loaded['input_offset'])
    output_offset = tuple(loaded['output_offset'])
    output_size = tuple(loaded['output_size'])
    input_size = tuple(loaded['input_size'])
    
    depth = 4 # Default depth used in training script

    print("Generating input data for animation...")
    # Generate 1 sample of XOR arithmetic
    X1f, X2f, Y_raw = data.generate_arithmetic(1, "xor", n, 1, 0.2)
    x = np.concatenate([X1f, X2f], axis=-1)[0] 
    
    # Ensure it matches the expected input shape
    x_input = jnp.array(x).reshape(input_size)
    
    # Initialize cells
    cells = jnp.zeros((1, N), dtype=jnp.float32)
    
    # Setup convolution parameters
    img_shape = (1, 1, N, 1)
    dn = jax.lax.conv_dimension_numbers(img_shape, filters.shape, ('NHWC', 'IHWO', 'NHWC'))
    window_strides = (1, 1)

    print("Evaluating test set accuracy...")
    test_samples = 1000
    X1f_test, X2f_test, Y_raw_test = data.generate_arithmetic(test_samples, "xor", n, 1, 0.2)
    X_test = np.concatenate([X1f_test, X2f_test], axis=-1)
    Y_pred = []
    
    # Propagate for each sample
    for i in range(test_samples):
        x_t = X_test[i]
        x_in = jnp.array(x_t).reshape(input_size)
        cells_eval = jnp.zeros((1, N), dtype=jnp.float32)
        _, out = forward._propagate_tape(
            x_in, input_offset, output_offset, output_size, depth, cells_eval, 
            weights, biases, states, filters, jax.nn.tanh, None, True, dn, window_strides
        )
        pred = jnp.squeeze(out) # shape (n,)
        pred = jnp.round(pred).astype(np.uint8)
        pred_packed = jnp.packbits(pred)
        Y_pred.append(pred_packed)
    
    Y_pred = jnp.array(Y_pred).flatten()
    Y_test_packed = np.packbits(Y_raw_test.astype(np.uint8), axis=-1).flatten()
    accuracy = jnp.mean(Y_pred == Y_test_packed)
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    
    print("Running unrolled forward pass for animation...")
    history_stacked, output = forward._propagate_tape_unrolled(
        x_input, input_offset, output_offset, output_size, depth, cells, 
        weights, biases, states, filters, jax.nn.tanh, None, True, dn, window_strides
    )
    
    # Single sample prediction accuracy
    anim_pred = jnp.round(jnp.squeeze(output)).astype(np.uint8)
    anim_pred_packed = jnp.packbits(anim_pred)[0]
    anim_true_packed = jnp.packbits(Y_raw[0].astype(np.uint8))[0]
    sample_correct = (anim_pred_packed == anim_true_packed)
    
    history_2d = np.squeeze(np.array(history_stacked)) # Shape (time_steps, N)
    time_steps = history_2d.shape[0]
    
    print(f"Creating animation with {time_steps} frames...")
    # Make the plot ratio match test script for consistency
    plot_ratio = 24
    plot_height = max(2, time_steps // 2)
    fig, ax = plt.subplots(figsize=(plot_ratio, plot_height))
    
    # We will display an image that is empty at first, and gets filled row by row
    im = ax.imshow(np.zeros_like(history_2d), aspect='auto', cmap='viridis', 
                   vmin=history_2d.min(), vmax=history_2d.max())
    
    title_str = f"1D Turing Machine Evolution\nTest Set Accuracy: {accuracy*100:.1f}% | Animated Sample Correct: {sample_correct}"
    ax.set_title(title_str)
    ax.set_xlabel("Tape Cells")
    ax.set_ylabel("Time Steps")
    
    def update(frame):
        # Frame goes from 0 to time_steps-1
        current_data = np.zeros_like(history_2d)
        current_data[:frame+1, :] = history_2d[:frame+1, :]
        im.set_array(current_data)
        return [im]
        
    anim = animation.FuncAnimation(fig, update, frames=time_steps, blit=True, interval=500)
    
    out_file = "tm_computation_evolution.gif"
    anim.save(out_file, writer='pillow')
    print(f"Animation saved to {out_file}")

if __name__ == '__main__':
    main()
