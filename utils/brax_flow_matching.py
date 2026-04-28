from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import optax

ModelType = Literal["ann", "snn"]


@dataclass(frozen=True)
class BraxFlowConfig:
    env_name: str = "fast"
    backend: str = "positional"
    horizon: int = 16
    batch_size: int = 32
    hidden_size: int = 64
    learning_rate: float = 1e-3
    sampler_steps: int = 8
    snn_beta: float = 0.8
    snn_threshold: float = 0.5
    surrogate_sigma: float = 5.0


class FlowBatch(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray


@jax.custom_vjp
def spike_function(voltage: jnp.ndarray, threshold: float, sigma: float) -> jnp.ndarray:
    return (voltage >= threshold).astype(jnp.float32)


def _spike_fwd(voltage: jnp.ndarray, threshold: float, sigma: float):
    spikes = (voltage >= threshold).astype(jnp.float32)
    return spikes, (voltage, threshold, sigma)


def _spike_bwd(res, grad_output: jnp.ndarray):
    voltage, threshold, sigma = res
    surrogate_grad = sigma / (1.0 + jnp.abs(sigma * (voltage - threshold))) ** 2
    return grad_output * surrogate_grad, None, None


spike_function.defvjp(_spike_fwd, _spike_bwd)


def collect_random_brax_batch(key: jax.Array, config: BraxFlowConfig) -> FlowBatch:
    from brax import envs

    env = envs.create(env_name=config.env_name, backend=config.backend)
    reset_keys = jax.random.split(key, config.batch_size + 1)
    action_key = reset_keys[0]
    reset_keys = reset_keys[1:]

    states = jax.vmap(env.reset)(reset_keys)
    action_keys = jax.random.split(action_key, config.horizon)

    def rollout_step(state, step_key):
        actions = jax.random.uniform(
            step_key,
            (config.batch_size, env.action_size),
            minval=-1.0,
            maxval=1.0,
        )
        next_state = jax.vmap(env.step)(state, actions)
        return next_state, (state.obs, actions, next_state.reward)

    _, (observations, actions, rewards) = jax.lax.scan(
        rollout_step, states, action_keys
    )
    return FlowBatch(
        observations=jnp.swapaxes(observations, 0, 1),
        actions=jnp.swapaxes(actions, 0, 1),
        rewards=jnp.swapaxes(rewards, 0, 1),
    )


def _dense_params(key: jax.Array, in_dim: int, out_dim: int, scale: float = 0.1):
    key_w, key_b = jax.random.split(key)
    return {
        "W": jax.random.normal(key_w, (in_dim, out_dim)) * scale,
        "b": jnp.zeros((out_dim,)) + jax.random.normal(key_b, (out_dim,)) * scale,
    }


def init_ann_flow_params(
    key: jax.Array,
    config: BraxFlowConfig,
    observation_size: int,
    action_size: int,
):
    input_size = observation_size + action_size + 1
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "dense1": _dense_params(k1, input_size, config.hidden_size),
        "dense2": _dense_params(k2, config.hidden_size, config.hidden_size),
        "out": _dense_params(k3, config.hidden_size, action_size),
    }


def init_snn_flow_params(
    key: jax.Array,
    config: BraxFlowConfig,
    observation_size: int,
    action_size: int,
):
    input_size = observation_size + action_size + 1
    k_in, k_rec, k_out = jax.random.split(key, 3)
    return {
        "input": _dense_params(k_in, input_size, config.hidden_size, scale=0.4),
        "recurrent": {
            "W": jax.random.normal(k_rec, (config.hidden_size, config.hidden_size))
            * 0.02,
            "b": jnp.zeros((config.hidden_size,)),
        },
        "out": _dense_params(k_out, config.hidden_size * 2, action_size),
    }


def _make_model_inputs(
    observations: jnp.ndarray, x_tau: jnp.ndarray, tau: jnp.ndarray
) -> jnp.ndarray:
    tau_broadcast = tau[:, None, None]
    tau_features = jnp.broadcast_to(tau_broadcast, (*x_tau.shape[:2], 1))
    return jnp.concatenate([x_tau, observations, tau_features], axis=-1)


def _flow_inputs(batch: FlowBatch, noise: jnp.ndarray, tau: jnp.ndarray):
    tau_broadcast = tau[:, None, None]
    x_tau = (1.0 - tau_broadcast) * noise + tau_broadcast * batch.actions
    model_inputs = _make_model_inputs(batch.observations, x_tau, tau)
    target_velocity = batch.actions - noise
    return x_tau, model_inputs, target_velocity


def _apply_dense(params, x: jnp.ndarray):
    return x @ params["W"] + params["b"]


def _apply_ann(params, model_inputs: jnp.ndarray):
    hidden = jnp.tanh(_apply_dense(params["dense1"], model_inputs))
    hidden = jnp.tanh(_apply_dense(params["dense2"], hidden))
    velocity = _apply_dense(params["out"], hidden)
    return velocity, {"spike_rate": jnp.array(0.0), "membrane_mean": jnp.array(0.0)}


def _apply_snn(params, config: BraxFlowConfig, model_inputs: jnp.ndarray):
    batch_size = model_inputs.shape[0]
    hidden_size = params["input"]["b"].shape[0]
    init_voltage = jnp.zeros((batch_size, hidden_size))
    init_spikes = jnp.zeros((batch_size, hidden_size))

    def step(carry, input_t):
        voltage_prev, spikes_prev = carry
        current = _apply_dense(params["input"], input_t)
        recurrent = spikes_prev @ params["recurrent"]["W"] + params["recurrent"]["b"]
        voltage_raw = config.snn_beta * voltage_prev + current + recurrent
        spikes = spike_function(
            voltage_raw, config.snn_threshold, config.surrogate_sigma
        )
        voltage = voltage_raw - spikes * config.snn_threshold
        readout = jnp.concatenate([spikes, voltage], axis=-1)
        velocity = _apply_dense(params["out"], readout)
        return (voltage, spikes), (velocity, spikes, voltage)

    _, (velocity, spikes, voltage) = jax.lax.scan(
        step, (init_voltage, init_spikes), jnp.swapaxes(model_inputs, 0, 1)
    )
    velocity = jnp.swapaxes(velocity, 0, 1)
    return velocity, {
        "spike_rate": jnp.mean(spikes),
        "membrane_mean": jnp.mean(voltage),
    }


def apply_flow(
    params,
    config: BraxFlowConfig,
    batch: FlowBatch,
    noise: jnp.ndarray,
    tau: jnp.ndarray,
    model_type: ModelType,
):
    _, model_inputs, _ = _flow_inputs(batch, noise, tau)
    if model_type == "ann":
        return _apply_ann(params, model_inputs)
    if model_type == "snn":
        return _apply_snn(params, config, model_inputs)
    raise ValueError(f"Unknown model_type: {model_type}")


def flow_matching_loss(
    params,
    config: BraxFlowConfig,
    batch: FlowBatch,
    key: jax.Array,
    model_type: ModelType,
):
    key_noise, key_tau = jax.random.split(key)
    noise = jax.random.normal(key_noise, batch.actions.shape)
    tau = jax.random.uniform(key_tau, (batch.actions.shape[0],))
    _, _, target_velocity = _flow_inputs(batch, noise, tau)
    predicted_velocity, aux = apply_flow(params, config, batch, noise, tau, model_type)
    loss = jnp.mean((predicted_velocity - target_velocity) ** 2)
    return loss, aux


def _tree_l2_norm(tree) -> jnp.ndarray:
    leaves = jax.tree.leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def train_step(
    params,
    config: BraxFlowConfig,
    batch: FlowBatch,
    key: jax.Array,
    model_type: ModelType,
):
    def loss_fn(current_params):
        return flow_matching_loss(current_params, config, batch, key, model_type)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates = jax.tree.map(lambda grad: -config.learning_rate * grad, grads)
    new_params = optax.apply_updates(params, updates)
    metrics = {
        "loss": loss,
        "grad_norm": _tree_l2_norm(grads),
        "spike_rate": aux["spike_rate"],
        "membrane_mean": aux["membrane_mean"],
        "nfe": jnp.array(1),
    }
    return new_params, metrics


def sample_actions_euler(
    params,
    config: BraxFlowConfig,
    observations: jnp.ndarray,
    key: jax.Array,
    model_type: ModelType,
) -> jnp.ndarray:
    action_size = params["out"]["b"].shape[0]
    batch_size, horizon = observations.shape[:2]
    actions = jax.random.normal(key, (batch_size, horizon, action_size))

    def sampler_step(current_actions, step_index):
        tau_value = step_index.astype(jnp.float32) / float(config.sampler_steps)
        tau = jnp.full((batch_size,), tau_value)
        batch = FlowBatch(
            observations=observations,
            actions=current_actions,
            rewards=jnp.zeros((batch_size, horizon)),
        )
        model_inputs = _make_model_inputs(batch.observations, current_actions, tau)
        if model_type == "ann":
            velocity, _ = _apply_ann(params, model_inputs)
        elif model_type == "snn":
            velocity, _ = _apply_snn(params, config, model_inputs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        return current_actions + velocity / float(config.sampler_steps), None

    sampled_actions, _ = jax.lax.scan(
        sampler_step, actions, jnp.arange(config.sampler_steps)
    )
    return jnp.tanh(sampled_actions)


def evaluate_action_sequences(
    key: jax.Array, config: BraxFlowConfig, actions: jnp.ndarray
):
    from brax import envs

    env = envs.create(env_name=config.env_name, backend=config.backend)
    batch_size = actions.shape[0]
    reset_keys = jax.random.split(key, batch_size)
    states = jax.vmap(env.reset)(reset_keys)

    def rollout_step(state, action_t):
        next_state = jax.vmap(env.step)(state, action_t)
        return next_state, next_state.reward

    _, rewards = jax.lax.scan(rollout_step, states, jnp.swapaxes(actions, 0, 1))
    rollout_rewards = jnp.sum(jnp.swapaxes(rewards, 0, 1), axis=-1)
    return {
        "rollout_rewards": rollout_rewards,
        "mean_rollout_reward": jnp.mean(rollout_rewards),
        "reward_std": jnp.std(rollout_rewards),
    }
