#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PRESET="default"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Run ANN vs SNN flow matching on the existing sine-wave dataset.

Usage:
  scripts/run_sine_training.sh [--preset quick|default|serious] [--dry-run] [extra sine_flow_experiment.py args]

Presets:
  quick    50 train steps, small shapes. Use for smoke tests.
  default  1000 train steps with the current best SNN params.
  serious  5000 train steps with the current best SNN params.

Examples:
  scripts/run_sine_training.sh --preset quick
  scripts/run_sine_training.sh --preset default -- --seed 3
  scripts/run_sine_training.sh --preset serious -- --hidden-size 64 --horizon 128
  scripts/run_sine_training.sh --preset default -- --plot-path artifacts/seed_3.png
  scripts/run_sine_training.sh --preset default -- --backend mlx

Notes:
  Current best SNN params from artifacts/snn_hparam_serious:
  learning_rate=0.001, hidden_size=64, threshold=0.6, beta=0.7, sigma=8.0.
  This script uses train steps, not classic epochs. Each train step draws a fresh
  generated sine batch and applies one optimizer update for both ANN and SNN.
  The JAX backend trains in compiled chunks by default; use -- --no-compile to disable.
  The progress bar is counter-based and does not read JAX metrics during training.
  A prediction plot is saved to artifacts/sine_prediction.png unless --no-plot is passed.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --help | -h)
    usage
    exit 0
    ;;
  --preset)
    PRESET="${2:-}"
    shift 2
    ;;
  --dry-run)
    DRY_RUN=1
    shift
    ;;
  --)
    shift
    EXTRA_ARGS+=("$@")
    break
    ;;
  *)
    EXTRA_ARGS+=("$1")
    shift
    ;;
  esac
done

case "$PRESET" in
quick)
  HORIZON=32
  BATCH_SIZE=16
  HIDDEN_SIZE=16
  SAMPLER_STEPS=4
  TRAIN_STEPS=50
  LEARNING_RATE=0.001
  SNN_THRESHOLD=0.5
  SNN_BETA=0.8
  SURROGATE_SIGMA=5.0
  ;;
default)
  HORIZON=64
  BATCH_SIZE=32
  HIDDEN_SIZE=64
  SAMPLER_STEPS=8
  TRAIN_STEPS=1000
  LEARNING_RATE=0.001
  SNN_THRESHOLD=0.6
  SNN_BETA=0.7
  SURROGATE_SIGMA=8.0
  ;;
serious)
  HORIZON=128
  BATCH_SIZE=64
  HIDDEN_SIZE=64
  SAMPLER_STEPS=16
  TRAIN_STEPS=5000
  LEARNING_RATE=0.001
  SNN_THRESHOLD=0.6
  SNN_BETA=0.7
  SURROGATE_SIGMA=8.0
  ;;
*)
  echo "Unknown preset: $PRESET" >&2
  usage >&2
  exit 2
  ;;
esac

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache}"

CMD=(
  uv run python sine_flow_experiment.py
  --horizon "$HORIZON"
  --batch-size "$BATCH_SIZE"
  --hidden-size "$HIDDEN_SIZE"
  --learning-rate "$LEARNING_RATE"
  --sampler-steps "$SAMPLER_STEPS"
  --snn-threshold "$SNN_THRESHOLD"
  --snn-beta "$SNN_BETA"
  --surrogate-sigma "$SURROGATE_SIGMA"
  --train-steps "$TRAIN_STEPS"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Preset: $PRESET"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo "MPLCONFIGDIR=$MPLCONFIGDIR"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

cd "$ROOT_DIR"
exec "${CMD[@]}"
