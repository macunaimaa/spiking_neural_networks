from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from training import generate_sine_wave
from utils.brax_flow_matching import (
    BraxFlowConfig,
    FlowBatch,
    ModelType,
    flow_matching_loss,
    init_ann_flow_params,
    init_snn_flow_params,
    sample_actions_euler,
    train_step,
)


class SineTrainingHistory(NamedTuple):
    steps: jnp.ndarray
    loss: jnp.ndarray
    grad_norm: jnp.ndarray
    spike_rate: jnp.ndarray
    membrane_mean: jnp.ndarray


class SineFlowResult(NamedTuple):
    eval_loss: jnp.ndarray
    generated_sequence_mse: jnp.ndarray
    spike_rate: jnp.ndarray
    generated_sequences: jnp.ndarray
    targets: jnp.ndarray
    nfe: jnp.ndarray
    history: SineTrainingHistory | None = None


def make_sine_flow_batch(
    key: jax.Array,
    config: BraxFlowConfig,
    sampling_size: int,
    data_dt: float,
) -> FlowBatch:
    inputs, targets = generate_sine_wave(
        key,
        config.batch_size,
        config.horizon,
        dt=data_dt,
        sampling_size=sampling_size,
    )
    return FlowBatch(
        observations=inputs,
        actions=targets,
        rewards=jnp.zeros((config.batch_size, config.horizon)),
    )


def _init_params(
    key: jax.Array,
    config: BraxFlowConfig,
    model_type: ModelType,
    observation_size: int,
    action_size: int,
):
    if model_type == "ann":
        return init_ann_flow_params(key, config, observation_size, action_size)
    if model_type == "snn":
        return init_snn_flow_params(key, config, observation_size, action_size)
    raise ValueError(f"Unknown model_type: {model_type}")


@jax.jit(static_argnames=("config", "model_type", "sampling_size", "data_dt", "chunk_steps"))
def _train_sine_chunk(
    params,
    key_data: jax.Array,
    key_train: jax.Array,
    config: BraxFlowConfig,
    model_type: ModelType,
    sampling_size: int,
    data_dt: float,
    chunk_steps: int,
):
    def chunk_step(carry, _):
        current_params, current_key, current_train_key = carry
        current_key, current_train_key, sub_data, sub_train = jax.random.split(
            current_key, 4
        )
        batch = make_sine_flow_batch(sub_data, config, sampling_size, data_dt)
        current_params, metrics = train_step(
            current_params, config, batch, sub_train, model_type
        )
        metric_values = jnp.array(
            [
                metrics["loss"],
                metrics["grad_norm"],
                metrics["spike_rate"],
                metrics["membrane_mean"],
            ]
        )
        return (current_params, current_key, current_train_key), metric_values

    (params, key_data, key_train), metrics = jax.lax.scan(
        chunk_step, (params, key_data, key_train), xs=None, length=chunk_steps
    )
    return params, key_data, key_train, metrics[-1]


def train_sine_flow_model(
    key: jax.Array,
    config: BraxFlowConfig,
    model_type: ModelType,
    train_steps: int,
    sampling_size: int,
    data_dt: float,
    progress_callback: Callable[[ModelType, int, int], None] | None = None,
    progress_interval: int = 1,
    compiled: bool = True,
    chunk_steps: int = 50,
    record_history: bool = False,
) -> SineFlowResult:
    key_params, key_data, key_train, key_sample = jax.random.split(key, 4)
    params = _init_params(
        key_params,
        config,
        model_type,
        observation_size=sampling_size,
        action_size=1,
    )

    progress_interval = max(1, progress_interval)
    chunk_steps = max(1, chunk_steps)
    history_steps = []
    history_loss = []
    history_grad_norm = []
    history_spike_rate = []
    history_membrane_mean = []

    def record_metric(step_number, metric_values):
        if not record_history:
            return
        history_steps.append(step_number)
        history_loss.append(float(metric_values[0]))
        history_grad_norm.append(float(metric_values[1]))
        history_spike_rate.append(float(metric_values[2]))
        history_membrane_mean.append(float(metric_values[3]))

    if compiled:
        completed_steps = 0
        next_progress_step = progress_interval
        while completed_steps < train_steps:
            current_chunk_steps = min(chunk_steps, train_steps - completed_steps)
            params, key_data, key_train, metric_values = _train_sine_chunk(
                params,
                key_data,
                key_train,
                config,
                model_type,
                sampling_size,
                data_dt,
                current_chunk_steps,
            )
            completed_steps += current_chunk_steps
            record_metric(completed_steps, metric_values)
            if progress_callback is not None and (
                completed_steps >= next_progress_step
                or completed_steps == train_steps
            ):
                progress_callback(model_type, completed_steps, train_steps)
                while next_progress_step <= completed_steps:
                    next_progress_step += progress_interval
    else:
        for step in range(train_steps):
            key_data, key_train, sub_data, sub_train = jax.random.split(key_data, 4)
            batch = make_sine_flow_batch(sub_data, config, sampling_size, data_dt)
            params, metrics = train_step(params, config, batch, sub_train, model_type)
            step_number = step + 1
            if record_history and (
                step_number % progress_interval == 0 or step_number == train_steps
            ):
                record_metric(
                    step_number,
                    jnp.array(
                        [
                            metrics["loss"],
                            metrics["grad_norm"],
                            metrics["spike_rate"],
                            metrics["membrane_mean"],
                        ]
                    ),
                )
            if progress_callback is not None and (
                step_number % progress_interval == 0 or step_number == train_steps
            ):
                progress_callback(model_type, step_number, train_steps)

    eval_batch = make_sine_flow_batch(key_train, config, sampling_size, data_dt)
    eval_loss, aux = flow_matching_loss(
        params, config, eval_batch, key_sample, model_type
    )
    generated_sequences = sample_actions_euler(
        params, config, eval_batch.observations, key_sample, model_type
    )
    generated_sequence_mse = jnp.mean((generated_sequences - eval_batch.actions) ** 2)
    history = None
    if record_history:
        history = SineTrainingHistory(
            steps=jnp.array(history_steps),
            loss=jnp.array(history_loss),
            grad_norm=jnp.array(history_grad_norm),
            spike_rate=jnp.array(history_spike_rate),
            membrane_mean=jnp.array(history_membrane_mean),
        )

    return SineFlowResult(
        eval_loss=eval_loss,
        generated_sequence_mse=generated_sequence_mse,
        spike_rate=aux["spike_rate"],
        generated_sequences=generated_sequences,
        targets=eval_batch.actions,
        nfe=jnp.array(config.sampler_steps),
        history=history,
    )
