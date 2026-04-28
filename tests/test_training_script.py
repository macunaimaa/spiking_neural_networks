import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sine_training.sh"


class TrainingScriptTest(unittest.TestCase):
    def test_script_help_and_dry_run(self):
        help_result = subprocess.run(
            [str(SCRIPT), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("quick", help_result.stdout)
        self.assertIn("serious", help_result.stdout)

        dry_run = subprocess.run(
            [str(SCRIPT), "--preset", "quick", "--dry-run"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(dry_run.returncode, 0)
        self.assertIn("sine_flow_experiment.py", dry_run.stdout)
        self.assertIn("--train-steps 50", dry_run.stdout)
        self.assertIn("--snn-threshold", dry_run.stdout)
        self.assertIn("--snn-beta", dry_run.stdout)
        self.assertIn("--surrogate-sigma", dry_run.stdout)
        self.assertIn("UV_CACHE_DIR", dry_run.stdout)

        serious_dry_run = subprocess.run(
            [str(SCRIPT), "--preset", "serious", "--dry-run"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(serious_dry_run.returncode, 0)
        self.assertIn("--hidden-size 64", serious_dry_run.stdout)
        self.assertIn("--snn-threshold 0.6", serious_dry_run.stdout)
        self.assertIn("--snn-beta 0.7", serious_dry_run.stdout)
        self.assertIn("--surrogate-sigma 8.0", serious_dry_run.stdout)

    def test_sine_experiment_help_lists_backends(self):
        result = subprocess.run(
            ["uv", "run", "python", "sine_flow_experiment.py", "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--backend", result.stdout)
        self.assertIn("--chunk-steps", result.stdout)
        self.assertIn("--no-compile", result.stdout)
        self.assertIn("--snn-threshold", result.stdout)
        self.assertIn("--snn-beta", result.stdout)
        self.assertIn("--surrogate-sigma", result.stdout)


if __name__ == "__main__":
    unittest.main()
