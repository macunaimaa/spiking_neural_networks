import unittest

import jax
import jax.numpy as jnp

from utils.brax_flow_matching import BraxFlowConfig
from utils.sine_flow_matching import (
    SineFlowResult,
    SineTrainingHistory,
    make_sine_flow_batch,
    train_sine_flow_model,
)


class SineFlowMatchingTest(unittest.TestCase):
    def test_make_sine_flow_batch_reuses_existing_dataset_shape(self):
        config = BraxFlowConfig(horizon=12, batch_size=5, hidden_size=8)
        batch = make_sine_flow_batch(
            jax.random.PRNGKey(0), config, sampling_size=5, data_dt=0.1
        )

        self.assertEqual(batch.observations.shape, (5, 12, 5))
        self.assertEqual(batch.actions.shape, (5, 12, 1))
        self.assertEqual(batch.rewards.shape, (5, 12))
        self.assertLessEqual(float(jnp.max(batch.actions)), 1.0)
        self.assertGreaterEqual(float(jnp.min(batch.actions)), -1.0)

    def test_train_sine_flow_model_runs_ann_and_snn(self):
        config = BraxFlowConfig(
            horizon=12,
            batch_size=5,
            hidden_size=8,
            learning_rate=1e-3,
            sampler_steps=3,
        )

        ann = train_sine_flow_model(
            jax.random.PRNGKey(1),
            config,
            model_type="ann",
            train_steps=2,
            sampling_size=5,
            data_dt=0.1,
        )
        snn = train_sine_flow_model(
            jax.random.PRNGKey(2),
            config,
            model_type="snn",
            train_steps=2,
            sampling_size=5,
            data_dt=0.1,
        )

        self.assertIsInstance(ann, SineFlowResult)
        self.assertIsInstance(snn, SineFlowResult)
        self.assertTrue(jnp.isfinite(ann.eval_loss))
        self.assertTrue(jnp.isfinite(snn.eval_loss))
        self.assertTrue(jnp.isfinite(ann.generated_sequence_mse))
        self.assertTrue(jnp.isfinite(snn.generated_sequence_mse))
        self.assertEqual(ann.generated_sequences.shape, (5, 12, 1))
        self.assertEqual(snn.generated_sequences.shape, (5, 12, 1))
        self.assertEqual(float(ann.spike_rate), 0.0)
        self.assertGreaterEqual(float(snn.spike_rate), 0.0)
        self.assertLessEqual(float(snn.spike_rate), 1.0)

    def test_train_sine_flow_model_progress_callback_is_step_only(self):
        config = BraxFlowConfig(
            horizon=8,
            batch_size=3,
            hidden_size=6,
            learning_rate=1e-3,
            sampler_steps=2,
        )
        calls = []

        def progress_callback(model_type, step, total_steps):
            calls.append((model_type, step, total_steps))

        train_sine_flow_model(
            jax.random.PRNGKey(3),
            config,
            model_type="ann",
            train_steps=4,
            sampling_size=5,
            data_dt=0.1,
            progress_callback=progress_callback,
            progress_interval=2,
            compiled=False,
        )

        self.assertEqual(calls, [("ann", 2, 4), ("ann", 4, 4)])

    def test_train_sine_flow_model_can_use_compiled_chunks(self):
        config = BraxFlowConfig(
            horizon=8,
            batch_size=3,
            hidden_size=6,
            learning_rate=1e-3,
            sampler_steps=2,
        )
        calls = []

        result = train_sine_flow_model(
            jax.random.PRNGKey(4),
            config,
            model_type="snn",
            train_steps=4,
            sampling_size=5,
            data_dt=0.1,
            progress_callback=lambda model, step, total: calls.append(
                (model, step, total)
            ),
            progress_interval=2,
            compiled=True,
            chunk_steps=2,
        )

        self.assertIsInstance(result, SineFlowResult)
        self.assertTrue(jnp.isfinite(result.eval_loss))
        self.assertEqual(calls, [("snn", 2, 4), ("snn", 4, 4)])

    def test_compiled_chunk_one_matches_uncompiled_key_flow(self):
        config = BraxFlowConfig(
            horizon=8,
            batch_size=3,
            hidden_size=6,
            learning_rate=1e-3,
            sampler_steps=2,
        )
        key = jax.random.PRNGKey(5)

        compiled = train_sine_flow_model(
            key,
            config,
            model_type="snn",
            train_steps=3,
            sampling_size=5,
            data_dt=0.1,
            compiled=True,
            chunk_steps=1,
        )
        uncompiled = train_sine_flow_model(
            key,
            config,
            model_type="snn",
            train_steps=3,
            sampling_size=5,
            data_dt=0.1,
            compiled=False,
        )

        self.assertAlmostEqual(
            float(compiled.eval_loss), float(uncompiled.eval_loss), places=6
        )
        self.assertAlmostEqual(
            float(compiled.generated_sequence_mse),
            float(uncompiled.generated_sequence_mse),
            places=6,
        )

    def test_train_sine_flow_model_records_chunk_metrics(self):
        config = BraxFlowConfig(
            horizon=8,
            batch_size=3,
            hidden_size=6,
            learning_rate=1e-3,
            sampler_steps=2,
        )

        result = train_sine_flow_model(
            jax.random.PRNGKey(5),
            config,
            model_type="snn",
            train_steps=4,
            sampling_size=5,
            data_dt=0.1,
            compiled=True,
            chunk_steps=2,
            record_history=True,
        )

        self.assertIsInstance(result.history, SineTrainingHistory)
        self.assertEqual(result.history.steps.shape, (2,))
        self.assertEqual(result.history.loss.shape, (2,))
        self.assertEqual(result.history.grad_norm.shape, (2,))
        self.assertEqual(result.history.spike_rate.shape, (2,))
        self.assertEqual(result.history.membrane_mean.shape, (2,))
        self.assertTrue(jnp.all(jnp.isfinite(result.history.loss)))
        self.assertEqual(result.history.steps.tolist(), [2, 4])


if __name__ == "__main__":
    unittest.main()
