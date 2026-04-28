from typing import NamedTuple

import numpy as np

from utils.brax_flow_matching import BraxFlowConfig, ModelType
from utils.sine_flow_matching import SineFlowResult


class MlxFlowBatch(NamedTuple):
    observations: object
    actions: object


def _require_mlx():
    try:
        import mlx.core as mx
    except Exception as exc:
        raise RuntimeError(
            "MLX backend is unavailable. Install it with `uv sync --extra mlx` "
            "on an Apple Silicon Mac."
        ) from exc
    return mx


def _make_sine_np(seed: int, batch_size: int, horizon: int, sampling_size: int, dt: float):
    rng = np.random.default_rng(seed)
    freq = rng.uniform(0.5, 2.0, size=(batch_size,))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(batch_size,))
    t = np.arange(horizon + sampling_size, dtype=np.float32) * dt
    signals = np.sin(2.0 * np.pi * freq[:, None] * t[None, :] + phase[:, None])
    window_idx = np.arange(horizon)[:, None] + np.arange(sampling_size)[None, :]
    inputs = np.take(signals, window_idx, axis=1).astype(np.float32)
    targets = signals[:, sampling_size : sampling_size + horizon, None].astype(
        np.float32
    )
    return inputs, targets


def _make_sine_batch(mx, seed, config, sampling_size, data_dt):
    inputs, targets = _make_sine_np(
        seed, config.batch_size, config.horizon, sampling_size, data_dt
    )
    return MlxFlowBatch(observations=mx.array(inputs), actions=mx.array(targets))


def _dense_params(mx, rng, in_dim, out_dim, scale=0.1):
    return {
        "W": mx.array((rng.normal(size=(in_dim, out_dim)) * scale).astype(np.float32)),
        "b": mx.array((rng.normal(size=(out_dim,)) * scale).astype(np.float32)),
    }


def _init_params(mx, seed, config, model_type, observation_size, action_size):
    rng = np.random.default_rng(seed)
    input_size = observation_size + action_size + 1
    if model_type == "ann":
        return {
            "dense1": _dense_params(mx, rng, input_size, config.hidden_size),
            "dense2": _dense_params(mx, rng, config.hidden_size, config.hidden_size),
            "out": _dense_params(mx, rng, config.hidden_size, action_size),
        }
    if model_type == "snn":
        return {
            "input": _dense_params(mx, rng, input_size, config.hidden_size, scale=0.4),
            "recurrent": _dense_params(
                mx, rng, config.hidden_size, config.hidden_size, scale=0.02
            ),
            "out": _dense_params(mx, rng, config.hidden_size * 2, action_size),
        }
    raise ValueError(f"Unknown model_type: {model_type}")


def _tree_map(fn, tree_a, tree_b=None):
    if isinstance(tree_a, dict):
        return {
            key: _tree_map(fn, value, None if tree_b is None else tree_b[key])
            for key, value in tree_a.items()
        }
    if tree_b is None:
        return fn(tree_a)
    return fn(tree_a, tree_b)


def _apply_dense(mx, params, x):
    return x @ params["W"] + params["b"]


def _model_inputs(mx, observations, x_tau, tau):
    tau_features = mx.broadcast_to(
        tau[:, None, None], (x_tau.shape[0], x_tau.shape[1], 1)
    )
    return mx.concatenate([x_tau, observations, tau_features], axis=-1)


def _flow_inputs(mx, batch, seed):
    rng = np.random.default_rng(seed)
    noise = mx.array(rng.normal(size=batch.actions.shape).astype(np.float32))
    tau_np = rng.uniform(0.0, 1.0, size=(batch.actions.shape[0],)).astype(np.float32)
    tau = mx.array(tau_np)
    x_tau = (1.0 - tau[:, None, None]) * noise + tau[:, None, None] * batch.actions
    return _model_inputs(mx, batch.observations, x_tau, tau), batch.actions - noise


def _apply_ann(mx, params, model_inputs):
    hidden = mx.tanh(_apply_dense(mx, params["dense1"], model_inputs))
    hidden = mx.tanh(_apply_dense(mx, params["dense2"], hidden))
    return _apply_dense(mx, params["out"], hidden), mx.array(0.0)


def _spike_ste(mx, voltage, threshold):
    hard = (voltage >= threshold).astype(mx.float32)
    smooth = mx.sigmoid(5.0 * (voltage - threshold))
    return hard + smooth - mx.stop_gradient(smooth)


def _apply_snn(mx, params, config, model_inputs):
    batch_size = model_inputs.shape[0]
    voltage = mx.zeros((batch_size, config.hidden_size))
    spikes = mx.zeros((batch_size, config.hidden_size))
    velocities = []
    spike_rates = []

    for step in range(model_inputs.shape[1]):
        input_t = model_inputs[:, step, :]
        current = _apply_dense(mx, params["input"], input_t)
        recurrent = _apply_dense(mx, params["recurrent"], spikes)
        voltage_raw = config.snn_beta * voltage + current + recurrent
        spikes = _spike_ste(mx, voltage_raw, config.snn_threshold)
        voltage = voltage_raw - spikes * config.snn_threshold
        readout = mx.concatenate([spikes, voltage], axis=-1)
        velocities.append(_apply_dense(mx, params["out"], readout))
        spike_rates.append(mx.mean(spikes))

    return mx.stack(velocities, axis=1), mx.mean(mx.stack(spike_rates))


def _loss(mx, params, config, batch, seed, model_type):
    model_inputs, target_velocity = _flow_inputs(mx, batch, seed)
    if model_type == "ann":
        predicted_velocity, spike_rate = _apply_ann(mx, params, model_inputs)
    else:
        predicted_velocity, spike_rate = _apply_snn(mx, params, config, model_inputs)
    loss = mx.mean(mx.square(predicted_velocity - target_velocity))
    return loss, spike_rate


def _train_step(mx, params, config, batch, seed, model_type):
    def loss_only(current_params):
        loss, _ = _loss(mx, current_params, config, batch, seed, model_type)
        return loss

    loss, grads = mx.value_and_grad(loss_only)(params)
    params = _tree_map(
        lambda param, grad: param - config.learning_rate * grad, params, grads
    )
    mx.eval(params, loss)
    return params, loss


def _sample(mx, params, config, observations, seed, model_type):
    rng = np.random.default_rng(seed)
    actions = mx.array(
        rng.normal(size=(observations.shape[0], observations.shape[1], 1)).astype(
            np.float32
        )
    )
    batch_size = observations.shape[0]
    for step in range(config.sampler_steps):
        tau = mx.full((batch_size,), step / float(config.sampler_steps))
        model_inputs = _model_inputs(mx, observations, actions, tau)
        if model_type == "ann":
            velocity, _ = _apply_ann(mx, params, model_inputs)
        else:
            velocity, _ = _apply_snn(mx, params, config, model_inputs)
        actions = actions + velocity / float(config.sampler_steps)
    return mx.tanh(actions)


def train_mlx_sine_flow_model(
    seed: int,
    config: BraxFlowConfig,
    model_type: ModelType,
    train_steps: int,
    sampling_size: int,
    data_dt: float,
    progress_callback=None,
    progress_interval: int = 1,
    compiled: bool = True,
    chunk_steps: int = 50,
) -> SineFlowResult:
    mx = _require_mlx()
    params = _init_params(mx, seed, config, model_type, sampling_size, 1)
    progress_interval = max(1, progress_interval)
    next_progress_step = progress_interval

    for step in range(train_steps):
        batch = _make_sine_batch(mx, seed + step + 1, config, sampling_size, data_dt)
        params, _ = _train_step(mx, params, config, batch, seed + 10_000 + step, model_type)
        step_number = step + 1
        if progress_callback is not None and (
            step_number >= next_progress_step or step_number == train_steps
        ):
            progress_callback(model_type, step_number, train_steps)
            while next_progress_step <= step_number:
                next_progress_step += progress_interval

    eval_batch = _make_sine_batch(mx, seed + 20_000, config, sampling_size, data_dt)
    eval_loss, spike_rate = _loss(mx, params, config, eval_batch, seed + 30_000, model_type)
    generated = _sample(mx, params, config, eval_batch.observations, seed + 40_000, model_type)
    sequence_mse = mx.mean(mx.square(generated - eval_batch.actions))
    mx.eval(eval_loss, spike_rate, generated, sequence_mse)

    return SineFlowResult(
        eval_loss=np.array(eval_loss),
        generated_sequence_mse=np.array(sequence_mse),
        spike_rate=np.array(spike_rate),
        generated_sequences=np.array(generated),
        targets=np.array(eval_batch.actions),
        nfe=np.array(config.sampler_steps),
    )
