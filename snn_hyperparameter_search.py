import argparse
import csv
import itertools
import time
from pathlib import Path

import jax
import numpy as np

from utils.brax_flow_matching import BraxFlowConfig
from utils.sine_flow_matching import SineFlowResult, train_sine_flow_model


def _csv_floats(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _csv_ints(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _trial_label(row):
    return (
        f"t{row['trial']} lr={row['learning_rate']} h={row['hidden_size']} "
        f"thr={row['threshold']} beta={row['beta']} sig={row['sigma']} seed={row['seed']}"
    )


def _write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_training_curves(trials, output_path, top_k):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = sorted(
        trials, key=lambda item: item["row"]["generated_sequence_mse"]
    )[:top_k]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    metrics = [
        ("loss", "Flow loss"),
        ("grad_norm", "Gradient norm"),
        ("spike_rate", "Spike rate"),
    ]

    for trial in selected:
        history = trial["result"].history
        if history is None or len(history.steps) == 0:
            continue
        steps = np.asarray(history.steps)
        for axis, (field, title) in zip(axes, metrics):
            axis.plot(steps, np.asarray(getattr(history, field)), label=_trial_label(trial["row"]))
            axis.set_ylabel(title)
            axis.grid(True, alpha=0.25)

    axes[-1].set_xlabel("train step")
    axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_summary(rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sequence_mse = np.array([row["generated_sequence_mse"] for row in rows])
    eval_loss = np.array([row["eval_loss"] for row in rows])
    spike_rate = np.array([row["spike_rate"] for row in rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(eval_loss, sequence_mse, c=spike_rate, cmap="viridis", s=70)
    ax.set_xlabel("eval flow loss")
    ax.set_ylabel("generated sequence MSE")
    ax.set_title("SNN hyperparameter search")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("spike rate")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_best_prediction(result: SineFlowResult, output_path, sample_index=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = np.asarray(result.targets[sample_index]).squeeze()
    generated = np.asarray(result.generated_sequences[sample_index]).squeeze()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(target, label="target", linewidth=2.0, color="black")
    ax.plot(generated, label="best SNN generated", linewidth=1.7)
    ax.set_title("Best SNN generated sine sequence")
    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _planned_trials(args):
    grid = list(
        itertools.product(
            _csv_floats(args.learning_rates),
            _csv_ints(args.hidden_sizes),
            _csv_floats(args.thresholds),
            _csv_floats(args.betas),
            _csv_floats(args.sigmas),
            _csv_ints(args.seeds),
        )
    )
    return grid


def main():
    parser = argparse.ArgumentParser(
        description="Run an SNN-only hyperparameter search on the sine flow task."
    )
    parser.add_argument("--learning-rates", default="0.001,0.0005")
    parser.add_argument("--hidden-sizes", default="32,64")
    parser.add_argument("--thresholds", default="0.4,0.5,0.6")
    parser.add_argument("--betas", default="0.7,0.8,0.9")
    parser.add_argument("--sigmas", default="5.0")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sampler-steps", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--chunk-steps", type=int, default=25)
    parser.add_argument("--sampling-size", type=int, default=5)
    parser.add_argument("--data-dt", type=float, default=0.1)
    parser.add_argument("--output-dir", default="artifacts/snn_hparam")
    parser.add_argument("--top-k-plots", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    grid = _planned_trials(args)
    print(f"planned_trials={len(grid)}")
    if args.dry_run:
        return

    output_dir = Path(args.output_dir)
    rows = []
    trials = []
    total_start = time.perf_counter()

    for trial_index, (lr, hidden_size, threshold, beta, sigma, seed) in enumerate(grid, start=1):
        config = BraxFlowConfig(
            horizon=args.horizon,
            batch_size=args.batch_size,
            hidden_size=hidden_size,
            learning_rate=lr,
            sampler_steps=args.sampler_steps,
            snn_beta=beta,
            snn_threshold=threshold,
            surrogate_sigma=sigma,
        )
        if not args.no_progress:
            print(
                f"[{trial_index}/{len(grid)}] lr={lr} hidden={hidden_size} "
                f"threshold={threshold} beta={beta} sigma={sigma} seed={seed}"
            )

        start = time.perf_counter()
        result = train_sine_flow_model(
            jax.random.PRNGKey(seed),
            config,
            model_type="snn",
            train_steps=args.train_steps,
            sampling_size=args.sampling_size,
            data_dt=args.data_dt,
            progress_interval=args.chunk_steps,
            compiled=True,
            chunk_steps=args.chunk_steps,
            record_history=True,
        )
        elapsed_s = time.perf_counter() - start

        history = result.history
        last_grad_norm = float(history.grad_norm[-1]) if history is not None else float("nan")
        last_loss = float(history.loss[-1]) if history is not None else float("nan")
        row = {
            "trial": trial_index,
            "seed": seed,
            "learning_rate": lr,
            "hidden_size": hidden_size,
            "threshold": threshold,
            "beta": beta,
            "sigma": sigma,
            "eval_loss": float(result.eval_loss),
            "generated_sequence_mse": float(result.generated_sequence_mse),
            "spike_rate": float(result.spike_rate),
            "last_train_loss": last_loss,
            "last_grad_norm": last_grad_norm,
            "elapsed_s": elapsed_s,
        }
        rows.append(row)
        trials.append({"row": row, "result": result})

    rows = sorted(rows, key=lambda row: row["generated_sequence_mse"])
    csv_path = output_dir / "results.csv"
    _write_csv(rows, csv_path)
    _plot_summary(rows, output_dir / "search_summary.png")
    _plot_training_curves(trials, output_dir / "training_curves.png", args.top_k_plots)

    best_trial = min(trials, key=lambda item: item["row"]["generated_sequence_mse"])
    _plot_best_prediction(best_trial["result"], output_dir / "best_prediction.png")

    elapsed_total = time.perf_counter() - total_start
    print(f"saved_results={csv_path}")
    print(f"saved_summary_plot={output_dir / 'search_summary.png'}")
    print(f"saved_training_plot={output_dir / 'training_curves.png'}")
    print(f"saved_best_prediction_plot={output_dir / 'best_prediction.png'}")
    print(f"elapsed_s={elapsed_total:.2f}")
    print("best_trial")
    for key, value in rows[0].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
