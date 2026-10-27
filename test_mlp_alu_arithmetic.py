'''
GPU-optimal Multi-Layer Perceptron (MLP) 8-bit Arithmetic Logic Unit (ALU) baseline driver.

Uses the exact 6-layer LayerNorm+ReLU MLP architecture (`24 -> 20 -> 16 -> 12 -> 10 -> 8`,
`1,978` parameters) from `contrib/test_jax_mlp_alu_arithmetic.py`, driven by Neurocell v2's
vectorized `prepare_alu_device_dataset` and `VNNTrainer` (`jax.lax.scan` whole-epoch execution)
to provide a hardware-saturated deep learning reference on the RTX 4060 GPU.

Copyright (c) 2026 Shekhar S. Chandra. All rights reserved.
Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

@author Shakes
'''
import argparse
import pickle
import time

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from neurocell import VNNTrainer, prepare_alu_device_dataset


class MLP(hk.Module):
    """6-layer LayerNorm + ReLU MLP matching `contrib/test_jax_mlp_alu_arithmetic.py`."""

    def __init__(self, output_size: int, name=None):
        super().__init__(name=name)
        self.output_size = output_size

    def __call__(self, x: jax.Array) -> jax.Array:
        net = hk.Flatten()(x)
        for hidden_dim in (24, 20, 16, 12, 10):
            net = hk.Linear(hidden_dim)(net)
            net = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(net)
            net = jax.nn.relu(net)
        return hk.Linear(self.output_size)(net)


def main():
    parser = argparse.ArgumentParser(
        description="Train and benchmark the GPU-optimal MLP 8-bit ALU baseline."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--samples", type=int, default=128000, help="Total dataset samples.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate accuracy every N epochs.")
    args = parser.parse_args()

    print("JAX version:", jax.__version__)
    print("Devices:", jax.devices())

    n = 8
    bit_depth = 2 ** (n - 1)
    ctrl_bits = 4
    delta = 0.2
    print(f"Model: 6-Layer MLP | Epochs: {args.epochs} | Bit depth: {bit_depth}")

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

    def _forward(batch: jax.Array) -> jax.Array:
        # In `contrib/test_jax_mlp_alu_arithmetic.py`, inputs X1, X2, C are concatenated
        # along the feature axis (B, 24, 1); `dataset.x_train` has shape (B, 3, 8, 1),
        # whose flattened layout per sample is identical (24 floats).
        mlp = MLP(n)
        return mlp(batch)

    network = hk.without_apply_rng(hk.transform(_forward))

    num_batches = args.samples // args.batch_size
    total_steps = num_batches * args.epochs
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.001,
        peak_value=0.03,
        warmup_steps=int(0.025 * total_steps),
        decay_steps=total_steps,
    )
    optimizer = optax.adamw(lr_schedule)

    params = network.init(jax.random.PRNGKey(42), dataset.x_train[0])
    trainer = VNNTrainer(
        apply_fn=network.apply,
        optimizer=optimizer,
        batch_size=args.batch_size,
    )
    opt_state = trainer.optimizer.init(params)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Total number of parameters: {param_count}")

    # 2. Pre-compile XLA executables before timing training loop
    t_comp0 = time.perf_counter()
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

    # 3. On-device `lax.scan` training loop
    start = time.perf_counter()
    for step in range(1, args.epochs + 1):
        params, opt_state, epoch_loss = trainer.train_epoch(
            params, opt_state, dataset.x_train, dataset.y_train
        )
        scaled_loss = float(epoch_loss) * (dataset.x_train.shape[0] / args.batch_size)
        if step % args.eval_every == 0 or step == args.epochs:
            train_acc, _ = trainer.evaluate_packed(
                params, dataset.x_train, dataset.y_train_packed
            )
            print(f"[Step {step:3d}, Loss {scaled_loss:.6f}, Accuracy {float(train_acc):.6f}]")
        else:
            print(f"[Step {step:3d}, Loss {scaled_loss:.6f}]")

    jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])
    elapsed = time.perf_counter() - start
    print(
        f"Training took {elapsed:.2f} secs ({elapsed / 60:.2f} mins) in total "
        f"({elapsed / args.epochs:.4f} secs/epoch)"
    )

    # 4. Final test evaluation and throughput benchmark
    eval_start = time.perf_counter()
    test_acc, _ = trainer.evaluate_packed(
        params, dataset.x_test, dataset.y_test_packed
    )
    jax.block_until_ready(test_acc)
    eval_elapsed = time.perf_counter() - eval_start

    num_test_samples = dataset.x_test.shape[0] * dataset.x_test.shape[1]
    memory_bytes = len(pickle.dumps(params))
    throughput = num_test_samples / eval_elapsed

    print(f"[Final Accuracy {float(test_acc):.6f}]")
    print("\n--- Performance Metrics ---")
    print("Model: 6-Layer MLP")
    print(f"Total Parameters: {param_count}")
    print(f"Memory Footprint: {memory_bytes / 1024 / 1024:.4f} MB")
    print(f"Training Time ({args.epochs} epochs): {elapsed:.4f} secs ({elapsed / args.epochs:.4f} secs/epoch)")
    print(f"Inference Time ({num_test_samples} test samples): {eval_elapsed:.4f} secs")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print("END")


if __name__ == "__main__":
    main()
