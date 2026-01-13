from dataclasses import dataclass
from re import T

import jax
import jax.numpy as jnp
import pandas as pd
import torch
from jax import grad, jit, lax, vmap
from numpy import arange


@dataclass
class SpikingNN:
    T: float = 200.0
    dt: float = 0.125
    Pref: float = 0.0
    Pmin: float = -1.0
    Pth: float = 25.0
    D: float = 0.25
    Pspike: float = 4.0
    t_ref: float = 5.0

    def time(self):
        return jnp.arange(0, self.T + self.dt, self.dt)

    def init_state(self):
        return (0.0, 0.0)

    def update(self, state, S):
        time = self.time()

        def step(carry, inputs):
            P_prev, t_rest = carry
            S_i, t = inputs

            def in_ref(_):
                return self.Pref

            def not_in_ref(_):
                P = jnp.where(P_prev > self.Pmin, P_prev + S_i - self.D, 0)
                return jnp.where(P >= self.Pth, P + self.Pspike, P)

            P = lax.cond(t <= t_rest, in_ref, not_in_ref, operand=None)

            spiked = P >= self.Pth

            P = jnp.where(spiked, P + self.Pspike, P)
            t_rest = jnp.where(spiked, t + self.t_ref, t_rest)

            return (P, t_rest), P

        carry0 = state

        new_state, P_trace = lax.scan(step, carry0, (S, time))

        return new_state, P_trace


if __name__ == "__main__":
    network = SpikingNN()
    state = network.init_state()

    N = 1601

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    input_vector = jax.random.normal(subkey, shape=(N,)) * 5

    update = jit(network.update)

    for time in range(100):
        P = update(state, input_vector)
        print("trace", P)
