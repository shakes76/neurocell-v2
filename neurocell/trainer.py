'''
On-device `jax.lax.scan` training and evaluation engine for Neurocell VNNs.

Supports arbitrary loss functions (`loss_fn`), scalar carry evaluation metrics
(`eval_metric_fn`), 8-bit packed evaluation (`evaluate_packed`), single-epoch scans
(`train_epoch`), and multi-epoch on-device training scans (`train_epochs`).

Copyright (c) 2026 Shekhar S. Chandra. All rights reserved.
Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

@author Shakes
'''
from typing import Callable
import math
import jax
import jax.numpy as jnp
import optax


BIT_WEIGHTS_U8 = jnp.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.uint8)


def compute_annealed_temperature(
    epoch: int,
    total_epochs: int,
    init_temp: float = 1.0,
    final_temp: float = 0.1,
    schedule: str = "exponential",
) -> float:
    """
    Computes the Codd state sigmoid temperature `tau` at `epoch` (in `0 .. total_epochs - 1`)
    using an `"exponential"`, `"cosine"`, or `"linear"` annealing schedule.
    """
    if total_epochs <= 1:
        return float(final_temp)
    progress = min(1.0, max(0.0, float(epoch) / float(total_epochs - 1)))
    if schedule == "exponential":
        return float(init_temp * ((final_temp / init_temp) ** progress))
    if schedule == "cosine":
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(final_temp + (init_temp - final_temp) * cosine_decay)
    if schedule == "linear":
        return float(init_temp + (final_temp - init_temp) * progress)
    raise ValueError(
        f"Unsupported temperature schedule '{schedule}'; choose from ('exponential', 'cosine', 'linear')."
    )


def packbits_u8_on_device(bits_u8: jax.Array) -> jax.Array:
    """
    Packs an 8-bit binary tensor of shape `(..., 8)` into uint8 bytes `(..., 1)`
    entirely on GPU via an integer dot product, avoiding `jnp.packbits` overhead.
    """
    return jnp.dot(bits_u8, BIT_WEIGHTS_U8)[..., jnp.newaxis]


def default_mse_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Default mean-squared error loss squeezing trailing singleton dimension when needed."""
    if logits.ndim == 3 and logits.shape[-1] == 1 and targets.ndim == 2:
        logits = jnp.squeeze(logits, axis=-1)
    return optax.squared_error(predictions=logits, targets=targets).mean()


def default_exact_match_metric(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Default binary/rounded exact-match accuracy metric."""
    if logits.ndim == 3 and logits.shape[-1] == 1 and targets.ndim == 2:
        logits = jnp.squeeze(logits, axis=-1)
    preds = jnp.round(logits)
    return jnp.mean(preds == jnp.round(targets))


class VNNTrainer:
    """
    High-throughput, task-agnostic on-device trainer and evaluator using `jax.lax.scan`.

    Compiles single-step updates, single-epoch scans, and multi-epoch scans into
    fused XLA programs.
    """

    def __init__(
        self,
        apply_fn: Callable[[optax.Params, jax.Array], jax.Array],
        optimizer: optax.GradientTransformation,
        batch_size: int = 128,
        loss_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        eval_metric_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    ):
        self.apply_fn = apply_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.custom_loss_fn = loss_fn or default_mse_loss
        self.eval_metric_fn = eval_metric_fn or default_exact_match_metric

        def _loss_fn(params: optax.Params, batch_x: jax.Array, batch_y: jax.Array) -> jax.Array:
            logits = self.apply_fn(params, batch_x)
            return self.custom_loss_fn(logits, batch_y)

        self.loss_fn = _loss_fn
        self._val_and_grad = jax.value_and_grad(_loss_fn)

        @jax.jit
        def _train_step(params, opt_state, batch_x, batch_y):
            loss_val, grads = self._val_and_grad(params, batch_x, batch_y)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss_val

        @jax.jit
        def _train_epoch(params, opt_state, x_epoch, y_epoch):
            def scan_step(carry, batch_xy):
                p, o_s = carry
                bx, by = batch_xy
                p, o_s, l_val = _train_step(p, o_s, bx, by)
                return (p, o_s), l_val

            (new_params, new_opt_state), losses = jax.lax.scan(
                scan_step, (params, opt_state), (x_epoch, y_epoch)
            )
            epoch_loss = jnp.sum(losses) / float(self.batch_size)
            return new_params, new_opt_state, epoch_loss

        @jax.jit
        def _evaluate_packed(params, x_batches, y_packed_batches):
            def eval_step(acc_correct, batch_xy):
                batch_x, batch_y_packed = batch_xy
                logits = self.apply_fn(params, batch_x)
                if logits.ndim == 3 and logits.shape[-1] == 1:
                    logits = jnp.squeeze(logits, axis=-1)
                preds_u8 = jnp.round(logits).astype(jnp.uint8)
                preds_packed = packbits_u8_on_device(preds_u8)
                batch_acc = jnp.mean(preds_packed == batch_y_packed)
                return acc_correct + batch_acc, preds_packed

            total_acc, all_preds_packed = jax.lax.scan(
                eval_step, jnp.array(0.0, dtype=jnp.float32), (x_batches, y_packed_batches)
            )
            accuracy = total_acc / float(x_batches.shape[0])
            return accuracy, all_preds_packed

        @jax.jit
        def _evaluate_metric(params, x_batches, y_batches):
            def eval_step(acc_metric, batch_xy):
                batch_x, batch_y = batch_xy
                logits = self.apply_fn(params, batch_x)
                m_val = self.eval_metric_fn(logits, batch_y)
                return acc_metric + m_val, None

            total_metric, _ = jax.lax.scan(
                eval_step, jnp.array(0.0, dtype=jnp.float32), (x_batches, y_batches)
            )
            return total_metric / float(x_batches.shape[0])

        self.train_step = _train_step
        self.train_epoch = _train_epoch
        self.evaluate_packed = _evaluate_packed
        self.evaluate_metric = _evaluate_metric
        self._adaptive_step_cache = {}
        self._adaptive_epoch_cache = {}
        self._adaptive_eval_packed_cache = {}
        self._adaptive_eval_metric_cache = {}
        self._annealed_step_cache = {}
        self._annealed_epoch_cache = {}

    def _get_adaptive_fns(self, mask_spec):
        """
        Returns JIT-compiled `(train_step, train_epoch, evaluate_packed, evaluate_metric)`
        functions specialized to `mask_spec.roi_bbox`, cached by `(roi_bbox, is_full_roi)`
        so XLA compiles once per quantized bounding-box bucket.
        """
        if mask_spec is None or mask_spec.is_full_roi:
            return (
                self.train_step,
                self.train_epoch,
                self.evaluate_packed,
                self.evaluate_metric,
            )
        key = (mask_spec.roi_bbox, mask_spec.is_full_roi)
        if key in self._adaptive_epoch_cache:
            return (
                self._adaptive_step_cache[key],
                self._adaptive_epoch_cache[key],
                self._adaptive_eval_packed_cache[key],
                self._adaptive_eval_metric_cache[key],
            )

        def _masked_loss(params, batch_x, batch_y):
            logits = self.apply_fn(params, batch_x, spatial_mask=mask_spec)
            return self.custom_loss_fn(logits, batch_y)

        val_and_grad_m = jax.value_and_grad(_masked_loss)

        @jax.jit
        def _step_m(params, opt_state, batch_x, batch_y):
            loss_val, grads = val_and_grad_m(params, batch_x, batch_y)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss_val

        @jax.jit
        def _epoch_m(params, opt_state, x_epoch, y_epoch):
            def scan_step(carry, batch_xy):
                p, o_s = carry
                bx, by = batch_xy
                p, o_s, l_val = _step_m(p, o_s, bx, by)
                return (p, o_s), l_val

            (new_params, new_opt_state), losses = jax.lax.scan(
                scan_step, (params, opt_state), (x_epoch, y_epoch)
            )
            epoch_loss = jnp.sum(losses) / float(self.batch_size)
            return new_params, new_opt_state, epoch_loss

        @jax.jit
        def _eval_packed_m(params, x_batches, y_packed_batches):
            def eval_step(acc_correct, batch_xy):
                batch_x, batch_y_packed = batch_xy
                logits = self.apply_fn(params, batch_x, spatial_mask=mask_spec)
                if logits.ndim == 3 and logits.shape[-1] == 1:
                    logits = jnp.squeeze(logits, axis=-1)
                preds_u8 = jnp.round(logits).astype(jnp.uint8)
                preds_packed = packbits_u8_on_device(preds_u8)
                batch_acc = jnp.mean(preds_packed == batch_y_packed)
                return acc_correct + batch_acc, preds_packed

            total_acc, all_preds_packed = jax.lax.scan(
                eval_step, jnp.array(0.0, dtype=jnp.float32), (x_batches, y_packed_batches)
            )
            accuracy = total_acc / float(x_batches.shape[0])
            return accuracy, all_preds_packed

        @jax.jit
        def _eval_metric_m(params, x_batches, y_batches):
            def eval_step(acc_metric, batch_xy):
                batch_x, batch_y = batch_xy
                logits = self.apply_fn(params, batch_x, spatial_mask=mask_spec)
                m_val = self.eval_metric_fn(logits, batch_y)
                return acc_metric + m_val, None

            total_metric, _ = jax.lax.scan(
                eval_step, jnp.array(0.0, dtype=jnp.float32), (x_batches, y_batches)
            )
            return total_metric / float(x_batches.shape[0])

        self._adaptive_step_cache[key] = _step_m
        self._adaptive_epoch_cache[key] = _epoch_m
        self._adaptive_eval_packed_cache[key] = _eval_packed_m
        self._adaptive_eval_metric_cache[key] = _eval_metric_m
        return _step_m, _epoch_m, _eval_packed_m, _eval_metric_m

    def train_step_adaptive(self, params, opt_state, batch_x, batch_y, mask_spec):
        """Executes a single training step using the pre-epoch `AdaptiveMaskSpec`."""
        step_fn, _, _, _ = self._get_adaptive_fns(mask_spec)
        return step_fn(params, opt_state, batch_x, batch_y)

    def train_epoch_adaptive(self, params, opt_state, x_epoch, y_epoch, mask_spec):
        """Executes a full epoch `jax.lax.scan` using the pre-epoch `AdaptiveMaskSpec`."""
        _, epoch_fn, _, _ = self._get_adaptive_fns(mask_spec)
        return epoch_fn(params, opt_state, x_epoch, y_epoch)

    def evaluate_packed_adaptive(self, params, x_batches, y_packed_batches, mask_spec):
        """Executes bit-packed 8-bit accuracy evaluation using `AdaptiveMaskSpec`."""
        _, _, eval_packed_fn, _ = self._get_adaptive_fns(mask_spec)
        return eval_packed_fn(params, x_batches, y_packed_batches)

    def evaluate_metric_adaptive(self, params, x_batches, y_batches, mask_spec):
        """Executes scalar-carry metric evaluation using `AdaptiveMaskSpec`."""
        _, _, _, eval_metric_fn = self._get_adaptive_fns(mask_spec)
        return eval_metric_fn(params, x_batches, y_batches)

    def _get_annealed_fns(self, codd_mode: str = "annealed", mask_spec=None):
        """
        Returns JIT-compiled `(train_step_ann, train_epoch_ann)` functions that accept a
        dynamic traced scalar `state_temperature` (requiring zero XLA recompilation when
        temperature changes across steps or epochs), cached by `(codd_mode, roi_bbox)`.
        """
        bbox_key = (
            None
            if (mask_spec is None or mask_spec.is_full_roi)
            else (mask_spec.roi_bbox, mask_spec.is_full_roi)
        )
        key = (codd_mode, bbox_key)
        if key in self._annealed_epoch_cache:
            return self._annealed_step_cache[key], self._annealed_epoch_cache[key]

        def _ann_loss(params, batch_x, batch_y, temp_scalar):
            logits = self.apply_fn(
                params,
                batch_x,
                spatial_mask=mask_spec,
                codd_mode=codd_mode,
                state_temperature=temp_scalar,
            )
            return self.custom_loss_fn(logits, batch_y)

        val_and_grad_ann = jax.value_and_grad(_ann_loss)

        @jax.jit
        def _step_ann(params, opt_state, batch_x, batch_y, temp_scalar):
            loss_val, grads = val_and_grad_ann(params, batch_x, batch_y, temp_scalar)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss_val

        @jax.jit
        def _epoch_ann(params, opt_state, x_epoch, y_epoch, temp_scalar):
            def scan_step(carry, batch_xy):
                p, o_s = carry
                bx, by = batch_xy
                p, o_s, l_val = _step_ann(p, o_s, bx, by, temp_scalar)
                return (p, o_s), l_val

            (new_params, new_opt_state), losses = jax.lax.scan(
                scan_step, (params, opt_state), (x_epoch, y_epoch)
            )
            epoch_loss = jnp.sum(losses) / float(self.batch_size)
            return new_params, new_opt_state, epoch_loss

        self._annealed_step_cache[key] = _step_ann
        self._annealed_epoch_cache[key] = _epoch_ann
        return _step_ann, _epoch_ann

    def train_step_annealed(
        self,
        params,
        opt_state,
        batch_x,
        batch_y,
        state_temperature: float | jax.Array = 1.0,
        codd_mode: str = "annealed",
        mask_spec=None,
    ):
        """
        Executes a single training step with dynamic `state_temperature` and `codd_mode`
        (`"annealed"`, `"ste_binary"`, or `"ste_ternary"`) without XLA recompilation.
        """
        step_fn, _ = self._get_annealed_fns(codd_mode=codd_mode, mask_spec=mask_spec)
        temp_arr = jnp.asarray(state_temperature, dtype=jnp.float32)
        return step_fn(params, opt_state, batch_x, batch_y, temp_arr)

    def train_epoch_annealed(
        self,
        params,
        opt_state,
        x_epoch,
        y_epoch,
        state_temperature: float | jax.Array = 1.0,
        codd_mode: str = "annealed",
        mask_spec=None,
    ):
        """
        Executes a full epoch `jax.lax.scan` with dynamic `state_temperature` and `codd_mode`
        (`"annealed"`, `"ste_binary"`, or `"ste_ternary"`) without XLA recompilation.
        """
        _, epoch_fn = self._get_annealed_fns(codd_mode=codd_mode, mask_spec=mask_spec)
        temp_arr = jnp.asarray(state_temperature, dtype=jnp.float32)
        return epoch_fn(params, opt_state, x_epoch, y_epoch, temp_arr)

    def train_epochs(
        self,
        params: optax.Params,
        opt_state: optax.OptState,
        x_train: jax.Array,
        y_train: jax.Array,
        rng: jax.Array,
        num_epochs: int,
    ):
        """
        Executes `num_epochs` training epochs entirely on GPU inside a nested `jax.lax.scan`,
        performing on-device shuffling and batching at each epoch boundary without returning
        to Python.
        """
        num_samples = x_train.shape[0]
        steps_per_epoch = num_samples // self.batch_size
        usable = steps_per_epoch * self.batch_size
        x_item_shape = x_train.shape[1:]
        y_item_shape = y_train.shape[1:]

        @jax.jit
        def _run_epochs(p0, o0, key0):
            def epoch_body(carry, _):
                p, o_s, key = carry
                key, subkey = jax.random.split(key)
                perm = jax.random.permutation(subkey, num_samples)[:usable]
                x_ep = jnp.reshape(x_train[perm], (steps_per_epoch, self.batch_size) + x_item_shape)
                y_ep = jnp.reshape(y_train[perm], (steps_per_epoch, self.batch_size) + y_item_shape)
                p, o_s, ep_loss = self.train_epoch(p, o_s, x_ep, y_ep)
                return (p, o_s, key), ep_loss

            (pf, of, _), epoch_losses = jax.lax.scan(
                epoch_body, (p0, o0, key0), None, length=int(num_epochs)
            )
            return pf, of, epoch_losses

        return _run_epochs(params, opt_state, rng)
