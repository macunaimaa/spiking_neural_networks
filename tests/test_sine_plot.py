import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp

from sine_flow_experiment import save_prediction_plot
from utils.sine_flow_matching import SineFlowResult


class SinePlotTest(unittest.TestCase):
    def test_save_prediction_plot_writes_png(self):
        target = jnp.linspace(-1.0, 1.0, 8).reshape(1, 8, 1)
        ann = SineFlowResult(
            eval_loss=jnp.array(0.1),
            generated_sequence_mse=jnp.array(0.2),
            spike_rate=jnp.array(0.0),
            generated_sequences=target + 0.1,
            targets=target,
            nfe=jnp.array(4),
        )
        snn = SineFlowResult(
            eval_loss=jnp.array(0.2),
            generated_sequence_mse=jnp.array(0.1),
            spike_rate=jnp.array(0.3),
            generated_sequences=target - 0.1,
            targets=target,
            nfe=jnp.array(4),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "prediction.png"
            save_prediction_plot({"ann": ann, "snn": snn}, output_path)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
