import argparse

import jax
import jax.numpy as jnp
from brax import envs

from utils.brax_flow_matching import (
    BraxFlowConfig,
    collect_random_brax_batch,
    evaluate_action_sequences,
    flow_matching_loss,
    init_ann_flow_params,
    init_snn_flow_params,
    sample_actions_euler,
    train_step,
)


def _init_params(key, config, model_type, observation_size, action_size):
    if model_type == "ann":
        return init_ann_flow_params(key, config, observation_size, action_size)
    if model_type == "snn":
        return init_snn_flow_params(key, config, observation_size, action_size)
    raise ValueError(f"Unknown model_type: {model_type}")


def _train_model(key, config, model_type, observation_size, action_size, num_steps):
    key_params, key_data, key_train = jax.random.split(key, 3)
    params = _init_params(key_params, config, model_type, observation_size, action_size)

    metrics = {}
    for step in range(num_steps):
        key_data, key_train, sub_data, sub_train = jax.random.split(key_data, 4)
        batch = collect_random_brax_batch(sub_data, config)
        params, metrics = train_step(params, config, batch, sub_train, model_type)
        if step % max(1, num_steps // 5) == 0 or step == num_steps - 1:
            print(
                f"{model_type:>3} step={step:04d} "
                f"loss={float(metrics['loss']):.6f} "
                f"grad={float(metrics['grad_norm']):.6f} "
                f"spikes={float(metrics['spike_rate']):.4f}"
            )

    eval_batch = collect_random_brax_batch(key_train, config)
    loss, aux = flow_matching_loss(params, config, eval_batch, key_data, model_type)
    sampled = sample_actions_euler(params, config, eval_batch.observations, key, model_type)
    action_mse = jnp.mean((sampled - eval_batch.actions) ** 2)
    rollout = evaluate_action_sequences(key_train, config, sampled)

    return {
        "params": params,
        "eval_loss": loss,
        "eval_spike_rate": aux["spike_rate"],
        "generated_action_mse": action_mse,
        "mean_rollout_reward": rollout["mean_rollout_reward"],
        "reward_std": rollout["reward_std"],
        "nfe": jnp.array(config.sampler_steps),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Brax-only ANN/SNN flow-matching smoke experiment."
    )
    parser.add_argument("--env", default="fast")
    parser.add_argument("--backend", default="positional")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sampler-steps", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = BraxFlowConfig(
        env_name=args.env,
        backend=args.backend,
        horizon=args.horizon,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        sampler_steps=args.sampler_steps,
    )
    env = envs.create(env_name=config.env_name, backend=config.backend)
    key = jax.random.PRNGKey(args.seed)
    key_ann, key_snn = jax.random.split(key)

    print(
        f"env={config.env_name} obs={env.observation_size} actions={env.action_size} "
        f"horizon={config.horizon} batch={config.batch_size}"
    )

    results = {}
    for model_type, model_key in (("ann", key_ann), ("snn", key_snn)):
        results[model_type] = _train_model(
            model_key,
            config,
            model_type,
            env.observation_size,
            env.action_size,
            args.train_steps,
        )

    print("\nFinal comparison")
    for model_type, result in results.items():
        print(
            f"{model_type:>3}: "
            f"eval_loss={float(result['eval_loss']):.6f} "
            f"action_mse={float(result['generated_action_mse']):.6f} "
            f"reward={float(result['mean_rollout_reward']):.6f} "
            f"reward_std={float(result['reward_std']):.6f} "
            f"spikes={float(result['eval_spike_rate']):.4f} "
            f"nfe={int(result['nfe'])}"
        )


if __name__ == "__main__":
    main()
