import functools

import jax
from brax import envs
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train

from utils.SNN import make_snn_ppo_networks


def main():
    # 1) Create environment (pick any dm_control task Brax supports via envs.create)
    # Examples often include: "dm_control:cartpole-swingup", "dm_control:cheetah-run", etc.
    env_name = "ant"
    env = envs.create(env_name=env_name, backend="positional")

    # 2) PPO config (keep minimal)
    num_timesteps = 5_000_000
    num_envs = 1024
    episode_length = 1000

    unroll_length = 32
    num_minibatches = 32
    num_updates_per_batch = 4

    learning_rate = 3e-4
    entropy_cost = 1e-3
    discounting = 0.99
    seed = 0

    # 3) Custom network factory (your SNN)
    # Brax PPO will call this with observation_size and action_size.
    network_factory = functools.partial(
        make_snn_ppo_networks,
        hidden_sizes=(64, 64),  # simplest
    )

    # 4) Progress callback
    def progress(num_steps, metrics):
        # metrics commonly includes: eval/episode_reward, eval/episode_length, etc.
        if num_steps % 1_000_000 == 0:
            print(num_steps, {k: float(v) for k, v in metrics.items()})

    # 5) Train
    train_fn = functools.partial(
        ppo_train.train,
        num_timesteps=num_timesteps,
        num_envs=num_envs,
        episode_length=episode_length,
        unroll_length=unroll_length,
        num_minibatches=num_minibatches,
        num_updates_per_batch=num_updates_per_batch,
        learning_rate=learning_rate,
        entropy_cost=entropy_cost,
        discounting=discounting,
        network_factory=network_factory,
        progress_fn=progress,
        seed=seed,
    )

    make_inference_fn, params, _ = train_fn(environment=env)

    # 6) Quick rollout (sanity check)
    inference_fn = make_inference_fn(params)
    key = jax.random.PRNGKey(seed)
    state = env.reset(key)

    total_reward = 0.0
    for _ in range(1000):
        key, sub = jax.random.split(key)
        action, _ = inference_fn(state.obs, sub)
        state = env.step(state, action)
        total_reward += float(state.reward)

    print("Rollout reward (1000 steps):", total_reward)


if __name__ == "__main__":
    main()
