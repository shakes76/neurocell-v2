'''
GPU-optimal Von Neumann Network (VNN) 8-bit Arithmetic Logic Unit (ALU) driver.

Trains and evaluates a 2D cellular array (`62 x 62` cells, `K = 23 x 13` Green's function
receptive field, `M = 16` directions, `depth = 4`) on 8-bit arithmetic and logic operations
using Neurocell v2's `ChipTopology2D`, `VNNParams`, `VNNTrainer` (`jax.lax.scan`), and
custom Pallas/Triton & Plancherel RFFT GPU backends.

Copyright (c) 2026 Shekhar S. Chandra. All rights reserved.
Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

@author Shakes
'''
import argparse
import math
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax

from neurocell import (
    SUPPORTED_BACKENDS,
    SUPPORTED_CODD_MODES,
    ChipTopology2D,
    VonNeumannNetwork,
    VNNTrainer,
    compute_annealed_temperature,
    prepare_alu_device_dataset,
)


def save_vnn_plots(params, directions: int, prefix: str = "vnn_alu_arithmetic"):
    """Saves weight, bias, Green's filter, and Codd state plots matching `contrib/`."""
    plt.figure()
    plt.imshow(params.weights)
    plt.title("Final Weights")
    plt.savefig(f"{prefix}_weights.png", dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(params.biases)
    plt.title("Final Biases")
    plt.savefig(f"{prefix}_biases.png", dpi=150)
    plt.close()

    fig_width = int(math.sqrt(directions))
    plt.figure(figsize=(2 * fig_width, 2 * fig_width))
    filters_ihwo = params.filters_ihwo
    for c1 in range(fig_width):
        for c2 in range(fig_width):
            c = c1 * fig_width + c2
            plt.subplot(fig_width, fig_width, c + 1)
            plt.imshow(filters_ihwo[0, :, :, c])
            plt.tight_layout()
            plt.axis("off")
            plt.title(f"Filter {c}")
    plt.savefig(f"{prefix}_filter.png", dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(params.states)
    plt.title("Final States")
    plt.savefig(f"{prefix}_states_final.png", dpi=150)
    plt.close()

    states_thres = VonNeumannNetwork.threshold_states(params.states, active_threshold=0.6)
    plt.figure()
    plt.imshow(states_thres)
    plt.title("Final States (Thresholded)")
    plt.savefig(f"{prefix}_states_final_thres.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train and benchmark the Neurocell v2 GPU-optimal VNN 8-bit ALU model."
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--samples", type=int, default=128000, help="Total dataset samples.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--backend",
        type=str,
        default="pallas_fused",
        choices=SUPPORTED_BACKENDS,
        help="Green's function & epilogue GPU backend.",
    )
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate accuracy every N epochs.")
    parser.add_argument(
        "--output-npz",
        type=str,
        default="alu_arithmetic_cells.npz",
        help="Output checkpoint filename.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip saving PNG plots.")
    parser.add_argument(
        "--adaptive-margin",
        type=int,
        default=None,
        help="Optional safety margin (in cells) for pre-epoch adaptive spatial extent masking.",
    )
    parser.add_argument(
        "--codd-mode",
        type=str,
        default="continuous",
        choices=SUPPORTED_CODD_MODES,
        help="Codd state gating mode ('continuous', 'annealed', 'ste_binary', 'ste_ternary').",
    )
    parser.add_argument(
        "--init-temp",
        type=float,
        default=1.0,
        help="Initial Codd state temperature when --codd-mode != 'continuous'.",
    )
    parser.add_argument(
        "--final-temp",
        type=float,
        default=0.2,
        help="Final Codd state temperature when --codd-mode != 'continuous'.",
    )
    args = parser.parse_args()

    print("JAX version:", jax.__version__)
    print("Devices:", jax.devices())

    n = 8
    bit_depth = 2 ** (n - 1)
    ctrl_bits = 4
    kernel_size = (23, 13)
    directions = 16
    depth = 4
    gap = 5
    ctrl_gap = 5
    delta = 0.2

    topology = ChipTopology2D.alu_chip(
        n=n,
        kernel_size=kernel_size,
        directions=directions,
        depth=depth,
        gap=gap,
        ctrl_gap=ctrl_gap,
    )
    print(
        f"N: {topology.N} | ROI: {topology.roi_shape} | Epochs: {args.epochs} | "
        f"Bit depth: {bit_depth} | Backend: {args.backend}"
    )

    # 1. Vectorized dataset generation and single-shot device placement
    t_data0 = time.perf_counter()
    dataset = prepare_alu_device_dataset(
        samples=args.samples,
        n=n,
        ctrl_bits=ctrl_bits,
        batch_size=args.batch_size,
        noise_level=delta,
    )
    t_data1 = time.perf_counter()
    print(
        f"Dataset prepared on GPU in {(t_data1 - t_data0) * 1000:.1f} ms: "
        f"x_train={dataset.x_train.shape}, x_test={dataset.x_test.shape}"
    )

    # 2. Initialize model, parameters, and warmup-cosine AdamW schedule
    num_batches = args.samples // args.batch_size
    total_steps = num_batches * args.epochs
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.001,
        peak_value=0.03,
        warmup_steps=int(0.025 * total_steps),
        decay_steps=total_steps,
        end_value=0.0005,
    )
    optimizer = optax.adamw(lr_schedule)

    model = VonNeumannNetwork(
        topology=topology,
        backend=args.backend,
        codd_mode=args.codd_mode,
        state_temperature=args.init_temp,
    )
    params = model.init(jax.random.PRNGKey(42), legacy_compat=True)
    trainer = VNNTrainer(
        apply_fn=model.apply,
        optimizer=optimizer,
        batch_size=args.batch_size,
    )
    opt_state = trainer.optimizer.init(params)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Total number of parameters: {param_count}")

    if not args.no_plots:
        plt.figure()
        plt.imshow(params.states)
        plt.title("Initial States")
        plt.savefig("vnn_alu_arithmetic_states_init.png", dpi=150)
        plt.close()

    # 3. Pre-compile XLA executables before timing training loop
    t_comp0 = time.perf_counter()
    mask_spec = None
    if args.adaptive_margin is not None:
        mask_spec = model.compute_spatial_mask(params, margin=args.adaptive_margin, bucket_multiple=8)
        step_fn, epoch_fn, eval_fn, _ = trainer._get_adaptive_fns(mask_spec)
        _ = epoch_fn.lower(params, opt_state, dataset.x_train, dataset.y_train).compile()
        _ = eval_fn.lower(params, dataset.x_train, dataset.y_train_packed).compile()
        _ = eval_fn.lower(params, dataset.x_test, dataset.y_test_packed).compile()
        print(
            f"Adaptive mask initialized: active_shape={mask_spec.active_shape} "
            f"({mask_spec.active_fraction * 100.0:.1f}% of ROI {topology.roi_shape}), "
            f"roi_bbox={mask_spec.roi_bbox}, margin={args.adaptive_margin}"
        )
    else:
        _ = trainer.train_epoch.lower(
            params, opt_state, dataset.x_train, dataset.y_train
        ).compile()
        _ = trainer.evaluate_packed.lower(
            params, dataset.x_train, dataset.y_train_packed
        ).compile()
        _ = trainer.evaluate_packed.lower(
            params, dataset.x_test, dataset.y_test_packed
        ).compile()
    t_comp1 = time.perf_counter()
    print(f"JIT compilation completed in {t_comp1 - t_comp0:.2f} secs")

    # 4. On-device `lax.scan` training loop
    start = time.perf_counter()
    for step in range(1, args.epochs + 1):
        if args.adaptive_margin is not None:
            mask_spec = model.compute_spatial_mask(
                params, margin=args.adaptive_margin, bucket_multiple=8, prev_mask=mask_spec
            )
        if args.codd_mode != "continuous":
            tau = compute_annealed_temperature(
                step - 1, args.epochs, init_temp=args.init_temp, final_temp=args.final_temp
            )
            params, opt_state, epoch_loss = trainer.train_epoch_annealed(
                params,
                opt_state,
                dataset.x_train,
                dataset.y_train,
                state_temperature=tau,
                codd_mode=args.codd_mode,
                mask_spec=mask_spec,
            )
        elif args.adaptive_margin is not None:
            params, opt_state, epoch_loss = trainer.train_epoch_adaptive(
                params, opt_state, dataset.x_train, dataset.y_train, mask_spec
            )
        else:
            params, opt_state, epoch_loss = trainer.train_epoch(
                params, opt_state, dataset.x_train, dataset.y_train
            )
        # Match legacy loss scaling (sum of batch MSEs / batch_size)
        scaled_loss = float(epoch_loss) * (dataset.x_train.shape[0] / args.batch_size)
        if step % args.eval_every == 0 or step == args.epochs:
            if args.adaptive_margin is not None:
                train_acc, _ = trainer.evaluate_packed_adaptive(
                    params, dataset.x_train, dataset.y_train_packed, mask_spec
                )
            else:
                train_acc, _ = trainer.evaluate_packed(
                    params, dataset.x_train, dataset.y_train_packed
                )
            print(f"[Step {step:3d}, Loss {scaled_loss:.6f}, Accuracy {float(train_acc):.6f}]")
        else:
            print(f"[Step {step:3d}, Loss {scaled_loss:.6f}]")

    jax.block_until_ready(params.cell_params)
    elapsed = time.perf_counter() - start
    print(
        f"Training took {elapsed:.2f} secs ({elapsed / 60:.2f} mins) in total "
        f"({elapsed / args.epochs:.3f} secs/epoch)"
    )

    # 5. Save checkpoint and plots
    model.save_checkpoint(
        args.output_npz,
        params,
        ctrl_bits=ctrl_bits,
        bit_depth=bit_depth,
        n=n,
        input_size=topology.in_ports[0].size,
        output_size=topology.out_port.size,
        input_offset_x1=topology.in_ports[0].offset,
        input_offset_x2=topology.in_ports[1].offset,
        ctrl_size=topology.in_ports[2].size,
        ctrl_offset=topology.in_ports[2].offset,
        output_offset=topology.out_port.offset,
        gap=gap,
        ctrl_gap=ctrl_gap,
    )
    if not args.no_plots:
        save_vnn_plots(params, directions=directions)

    # 6. Final test evaluation and throughput benchmark
    eval_start = time.perf_counter()
    if args.adaptive_margin is not None:
        test_acc, y_pred_packed = trainer.evaluate_packed_adaptive(
            params, dataset.x_test, dataset.y_test_packed, mask_spec
        )
    else:
        test_acc, y_pred_packed = trainer.evaluate_packed(
            params, dataset.x_test, dataset.y_test_packed
        )
    jax.block_until_ready(test_acc)
    eval_elapsed = time.perf_counter() - eval_start

    num_test_samples = dataset.x_test.shape[0] * dataset.x_test.shape[1]
    memory_bytes = len(pickle.dumps(params))
    throughput = num_test_samples / eval_elapsed

    print(f"[Final Accuracy {float(test_acc):.6f}]")
    print("\n--- Performance Metrics ---")
    print(f"Backend: {args.backend}")
    print(f"Total Parameters: {param_count}")
    print(f"Memory Footprint: {memory_bytes / 1024 / 1024:.4f} MB")
    print(f"Training Time ({args.epochs} epochs): {elapsed:.4f} secs ({elapsed / args.epochs:.4f} secs/epoch)")
    print(f"Inference Time ({num_test_samples} test samples): {eval_elapsed:.4f} secs")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print("END")


if __name__ == "__main__":
    main()
