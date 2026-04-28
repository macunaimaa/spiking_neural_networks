#!/usr/bin/env bash
set -euo pipefail

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache}"

uv run python - <<'PY'
import importlib.util

import jax

print("JAX")
print(f"  backend: {jax.default_backend()}")
print(f"  devices: {jax.devices()}")

print("MLX")
print(f"  installed: {importlib.util.find_spec('mlx') is not None}")
print("  runtime probe: run `uv run python sine_flow_experiment.py --backend mlx --train-steps 1 --no-plot --no-progress`")
PY
