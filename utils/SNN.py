from typing import Any, Sequence, Tuple

import jax
import jax.numpy as jnp
from brax.training import distribution
from brax.training import networks as brax_networks
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen as nn

SURROGATE_SIGMA = 5.0


@jax.custom_vjp
def spike_function(x, threshold):
    return (x >= threshold).astype(jnp.float32)


def _spike_fwd(x, threshold):
    y = (x >= threshold).astype(jnp.float32)
    return y, (x, threshold)


def _spike_bwd(res, g):
    x, threshold = res
    sg = SURROGATE_SIGMA / (1.0 + jnp.abs(SURROGATE_SIGMA * (x - threshold))) ** 2
    return (g * sg, None)


spike_function.defvjp(_spike_fwd, _spike_bwd)


def rate_encode(obs: jnp.ndarray) -> jnp.ndarray:
    # simple squash; if PPO already normalizes observations, you can just "return obs"
    obs = jnp.tanh(obs)
    return (obs + 1.0) * 0.5


class SpikingDense(nn.Module):
    features: int
    threshold: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.features)(x)
        return spike_function(x, self.threshold)


class SpikingTrunk(nn.Module):
    hidden_sizes: Sequence[int]

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = rate_encode(obs)
        for h in self.hidden_sizes:
            x = SpikingDense(h)(x)
        return x


class SNNPolicy(nn.Module):
    action_size: int
    hidden_sizes: Sequence[int]

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = SpikingTrunk(self.hidden_sizes)(obs)
        mean = nn.Dense(self.action_size)(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_size,))
        return mean, log_std


class SNNValue(nn.Module):
    hidden_sizes: Sequence[int]

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = SpikingTrunk(self.hidden_sizes)(obs)
        v = nn.Dense(1)(x)
        return v.squeeze(-1)


def make_snn_ppo_networks(
    observation_size: int,
    action_size: int,
    hidden_sizes: Sequence[int] = (64, 64),
    **unused_kwargs: Any,
) -> ppo_networks.PPONetworks:
    # Most Brax continuous-control uses tanh-squashed Normal
    try:
        param_dist = distribution.NormalTanhDistribution(event_size=action_size)
    except Exception:
        param_dist = distribution.NormalDistribution(event_size=action_size)

    policy_module = SNNPolicy(action_size=action_size, hidden_sizes=hidden_sizes)
    value_module = SNNValue(hidden_sizes=hidden_sizes)

    def _obs_dummy(observation_size):
        # Brax may pass observation_size as int or as a shape tuple like (27,)
        if isinstance(observation_size, int):
            return jnp.zeros((observation_size,), dtype=jnp.float32)
        # already a shape (tuple/list)
        return jnp.zeros(tuple(observation_size), dtype=jnp.float32)

    def policy_init(key):
        dummy = _obs_dummy(observation_size)
        return policy_module.init(key, dummy)

    def _pick_params(args):
        # pick the first mapping-like object (flax params are FrozenDict/dict-like)
        for a in args:
            if isinstance(a, (dict,)) or hasattr(a, "keys"):
                return a
        return args[0]

    def policy_apply(*args):
        # args = (normalizer_state, flax_params, obs)
        params = args[1]
        obs = args[2]

        mean, log_std = policy_module.apply(params, obs)
        log_std_b = jnp.broadcast_to(log_std, mean.shape)
        return jnp.concatenate([mean, log_std_b], axis=-1)

    def value_init(key):
        dummy = _obs_dummy(observation_size)
        return value_module.init(key, dummy)

    def value_apply(*args):
        params = args[1]
        obs = args[2]
        return value_module.apply(params, obs)

    policy_ffn = brax_networks.FeedForwardNetwork(init=policy_init, apply=policy_apply)
    value_ffn = brax_networks.FeedForwardNetwork(init=value_init, apply=value_apply)

    return ppo_networks.PPONetworks(
        policy_network=policy_ffn,
        value_network=value_ffn,
        parametric_action_distribution=param_dist,
    )
