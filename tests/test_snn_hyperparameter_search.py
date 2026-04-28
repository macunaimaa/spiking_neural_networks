import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SnnHyperparameterSearchTest(unittest.TestCase):
    def test_help_and_dry_run(self):
        help_result = subprocess.run(
            ["uv", "run", "python", "snn_hyperparameter_search.py", "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--learning-rates", help_result.stdout)
        self.assertIn("--thresholds", help_result.stdout)
        self.assertIn("--betas", help_result.stdout)

        dry_run = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "snn_hyperparameter_search.py",
                "--learning-rates",
                "0.001,0.0005",
                "--hidden-sizes",
                "8",
                "--thresholds",
                "0.5",
                "--betas",
                "0.8",
                "--sigmas",
                "5.0",
                "--dry-run",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(dry_run.returncode, 0)
        self.assertIn("planned_trials=2", dry_run.stdout)


if __name__ == "__main__":
    unittest.main()
