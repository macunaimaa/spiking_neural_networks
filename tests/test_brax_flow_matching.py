import unittest

import jax
import jax.numpy as jnp

from utils.brax_flow_matching import (
    BraxFlowConfig,
    FlowBatch,
    apply_flow,
    collect_random_brax_batch,
    flow_matching_loss,
    init_ann_flow_params,
    init_snn_flow_params,
    evaluate_action_sequences,
    sample_actions_euler,
    train_step,
)


class BraxFlowMatchingTest(unittest.TestCase):
    def test_collect_random_brax_batch_uses_brax_environment(self):
        config = BraxFlowConfig(env_name="fast", horizon=4, batch_size=3, hidden_size=8)
        batch = collect_random_brax_batch(jax.random.PRNGKey(0), config)

        self.assertIsInstance(batch, FlowBatch)
        self.assertEqual(batch.observations.shape, (3, 4, 2))
        self.assertEqual(batch.actions.shape, (3, 4, 1))
        self.assertEqual(batch.rewards.shape, (3, 4))
        self.assertTrue(jnp.all(batch.actions <= 1.0))
        self.assertTrue(jnp.all(batch.actions >= -1.0))

    def test_ann_and_snn_predict_velocity_for_brax_action_sequences(self):
        config = BraxFlowConfig(env_name="fast", horizon=4, batch_size=3, hidden_size=8)
        batch = collect_random_brax_batch(jax.random.PRNGKey(1), config)
        key_noise, key_tau, key_ann, key_snn = jax.random.split(jax.random.PRNGKey(2), 4)
        noise = jax.random.normal(key_noise, batch.actions.shape)
        tau = jax.random.uniform(key_tau, (config.batch_size,))

        ann_params = init_ann_flow_params(key_ann, config, observation_size=2, action_size=1)
        ann_velocity, ann_aux = apply_flow(ann_params, config, batch, noise, tau, model_type="ann")

        snn_params = init_snn_flow_params(key_snn, config, observation_size=2, action_size=1)
        snn_velocity, snn_aux = apply_flow(snn_params, config, batch, noise, tau, model_type="snn")

        self.assertEqual(ann_velocity.shape, batch.actions.shape)
        self.assertEqual(snn_velocity.shape, batch.actions.shape)
        self.assertEqual(ann_aux["spike_rate"].shape, ())
        self.assertEqual(snn_aux["spike_rate"].shape, ())
        self.assertEqual(float(ann_aux["spike_rate"]), 0.0)
        self.assertGreaterEqual(float(snn_aux["spike_rate"]), 0.0)
        self.assertLessEqual(float(snn_aux["spike_rate"]), 1.0)

    def test_train_step_sampler_and_metrics_are_finite(self):
        config = BraxFlowConfig(
            env_name="fast",
            horizon=4,
            batch_size=3,
            hidden_size=8,
            learning_rate=1e-3,
            sampler_steps=3,
        )
        batch = collect_random_brax_batch(jax.random.PRNGKey(3), config)
        params = init_snn_flow_params(jax.random.PRNGKey(4), config, 2, 1)

        initial_loss, initial_aux = flow_matching_loss(
            params, config, batch, jax.random.PRNGKey(5), model_type="snn"
        )
        params, metrics = train_step(
            params, config, batch, jax.random.PRNGKey(6), model_type="snn"
        )
        sampled_actions = sample_actions_euler(
            params, config, batch.observations, jax.random.PRNGKey(7), model_type="snn"
        )

        self.assertTrue(jnp.isfinite(initial_loss))
        self.assertTrue(jnp.isfinite(metrics["loss"]))
        self.assertTrue(jnp.isfinite(metrics["grad_norm"]))
        self.assertTrue(jnp.isfinite(initial_aux["spike_rate"]))
        self.assertEqual(sampled_actions.shape, batch.actions.shape)
        self.assertTrue(jnp.all(sampled_actions <= 1.0))
        self.assertTrue(jnp.all(sampled_actions >= -1.0))
        self.assertEqual(int(metrics["nfe"]), 1)

    def test_evaluate_action_sequences_rolls_actions_in_brax(self):
        config = BraxFlowConfig(env_name="fast", horizon=4, batch_size=3, hidden_size=8)
        batch = collect_random_brax_batch(jax.random.PRNGKey(8), config)

        metrics = evaluate_action_sequences(
            jax.random.PRNGKey(9), config, batch.actions
        )

        self.assertEqual(metrics["rollout_rewards"].shape, (3,))
        self.assertTrue(jnp.isfinite(metrics["mean_rollout_reward"]))
        self.assertTrue(jnp.isfinite(metrics["reward_std"]))


if __name__ == "__main__":
    unittest.main()
