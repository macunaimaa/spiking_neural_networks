# spiking_neural_networks

This repo exist for 2 main reasons, study jax lib and its fundamentals and to start my studies on bioinspired computing systems currently throught the lenses of Deep Learning.

Other bioinspired things that are of my interest are Compliant mechanisms which Ive already designed some on the past years and I have other interests on the subject but these are the ones I can work with.

Currently this repo only possess a rather simplistic implementatation on predicting a sine wave using SNNs, in a later stage of this project I aim to port these tests onto robotic systems, and possibly analyze the capabilities of generalization of this architecture on rapidly changing enviroments such as crowded spaces or maybe even playing some games.

## Setup

This project uses `uv` and Python 3.12.

```bash
uv sync
```

On machines where the default uv cache is not writable, use:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

The default dependency set installs CPU-compatible JAX. For Linux machines with CUDA 12 support, install the optional CUDA extra:

```bash
uv sync --extra cuda
```

Quick local checks:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m py_compile main.py training.py utils/SNN.py test_brax_nn.py
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv run python -c "import jax, optax, training; key=jax.random.PRNGKey(0); x,y=training.generate_sine_wave(key, 8, 20, training.Data_DT, training.SAMPLING_SIZE); params=training.init_network_params(key, training.LAYER_SIZES); config=training.SpikingNN(T=20*training.DT, dt=training.DT); training.optimizer=optax.adam(training.LEARNING_RATE); opt_state=training.optimizer.init(params); params,opt_state,loss=training.train_step(params,opt_state,config,x,y); print(float(loss))"
```

`training.py` runs the sine-wave SNN experiment. `test_brax_nn.py` runs a much heavier Brax PPO experiment and should be treated as a long training script rather than a unit test.

## Project map

- `main.py`: first scalar LIF neuron experiment using `jax.lax.scan`.
- `training.py`: trainable surrogate-gradient SNN for sine-wave next-step prediction.
- `utils/SNN.py`: Flax/Brax-compatible spiking policy/value network factory.
- `test_brax_nn.py`: Brax PPO experiment using the spiking policy/value network.
- `utils/brax_flow_matching.py`: Brax-only flow-matching research utilities for ANN/SNN vector-field baselines, Euler action sampling, and rollout metrics.
- `utils/sine_flow_matching.py`: adapter that reuses the existing sine-wave generator with the same ANN/SNN flow-matching models.
- `brax_flow_experiment.py`: small CLI experiment comparing ANN and SNN flow matching on Brax trajectory chunks.
- `sine_flow_experiment.py`: small CLI experiment comparing ANN and SNN flow matching on the existing sine-wave dataset.

## Brax flow matching

Run a small ANN-vs-SNN flow-matching experiment on Brax:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python brax_flow_experiment.py --env fast --horizon 8 --batch-size 16 --hidden-size 32 --sampler-steps 8 --train-steps 20
```

This experiment collects random Brax action trajectories, trains a model to predict the flow-matching velocity field from noise to those action trajectories, samples generated action sequences with Euler integration, and evaluates them with:

- `eval_loss`: flow-matching velocity MSE.
- `action_mse`: generated action sequence MSE against held-out random Brax actions.
- `reward`: open-loop Brax rollout reward from generated actions.
- `spikes`: mean SNN spike rate.
- `nfe`: number of vector-field evaluations used by the sampler.

## Sine flow matching

Run a Mac-friendly ANN-vs-SNN flow-matching experiment on the existing sine-wave dataset:

```bash
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv run python sine_flow_experiment.py --horizon 64 --batch-size 32 --hidden-size 32 --sampler-steps 8 --train-steps 50
```

This uses `training.generate_sine_wave` for the data. The sine windows are the conditioning sequence, and the next sine values are the target generated sequence.

You can also use the training launcher:

```bash
scripts/run_sine_training.sh --preset quick
scripts/run_sine_training.sh --preset default
scripts/run_sine_training.sh --preset serious
```

The launcher uses train steps instead of classic epochs. Each train step draws a fresh sine batch and applies one optimizer update for both ANN and SNN. Good starting points:

- `quick`: 50 train steps for smoke testing.
- `default`: 1000 train steps using the current best SNN search parameters.
- `serious`: 5000 train steps using the current best SNN search parameters.

Current best SNN search parameters:

- `learning_rate=0.001`
- `hidden_size=64`
- `snn_threshold=0.6`
- `snn_beta=0.7`
- `surrogate_sigma=8.0`

The default JAX backend trains in compiled chunks. Tune or disable this with:

```bash
scripts/run_sine_training.sh --preset serious -- --chunk-steps 100
scripts/run_sine_training.sh --preset serious -- --no-compile
```

The progress bar is intentionally cheap: it only updates from the Python step counter and does not read JAX loss values during training. Disable it with `-- --no-progress` if you want fully quiet output:

```bash
scripts/run_sine_training.sh --preset default -- --no-progress
```

Training also saves a prediction plot by default:

```text
artifacts/sine_prediction.png
```

Override or disable it with:

```bash
scripts/run_sine_training.sh --preset default -- --plot-path artifacts/seed_0.png
scripts/run_sine_training.sh --preset default -- --no-plot
```

Check the active compute backend:

```bash
scripts/check_backends.sh
```

Run the MLX backend on Apple Silicon:

```bash
uv sync --extra mlx
scripts/run_sine_training.sh --preset serious -- --backend mlx
```

Try the experimental JAX Metal plugin:

```bash
uv sync --extra metal
ENABLE_PJRT_COMPATIBILITY=1 scripts/check_backends.sh
ENABLE_PJRT_COMPATIBILITY=1 scripts/run_sine_training.sh --preset serious -- --backend jax
```

JAX Metal support is experimental and may not support every JAX operation used by this repo.

## SNN hyperparameter search

Run a compact SNN-only search and save CSV plus plots:

```bash
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv run python snn_hyperparameter_search.py \
  --learning-rates 0.001,0.0005 \
  --hidden-sizes 32,64 \
  --thresholds 0.4,0.5,0.6 \
  --betas 0.7,0.8,0.9 \
  --sigmas 5.0 \
  --horizon 64 \
  --batch-size 32 \
  --sampler-steps 8 \
  --train-steps 300 \
  --chunk-steps 25 \
  --output-dir artifacts/snn_hparam
```

Outputs:

- `results.csv`: one row per trial with final loss, sequence MSE, spike rate, gradient norm, and elapsed time.
- `training_curves.png`: loss, gradient norm, and spike-rate curves for the best trials.
- `search_summary.png`: final flow loss vs generated sequence MSE, colored by spike rate.
- `best_prediction.png`: target vs generated sequence for the best SNN trial.

### Articles used as inspiration:

- https://sampathkumaran.medium.com/brain-inspired-robot-navigation-part-4-f88e32eb8856
- https://arxiv.org/abs/1804.08150

### important docs:
- https://docs.jax.dev/en/latest/jax-101.html
- https://arxiv.org/pdf/2503.02013 (could be useful idk)

### Could be useful:
- weight initialization: https://arxiv.org/pdf/2410.00580
