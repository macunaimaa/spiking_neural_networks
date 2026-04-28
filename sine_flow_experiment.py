import argparse
import sys
from pathlib import Path

import jax
import numpy as np

from utils.brax_flow_matching import BraxFlowConfig
from utils.sine_flow_matching import train_sine_flow_model


class ProgressBar:
    def __init__(self, enabled=True, width=28):
        self.enabled = enabled
        self.width = width

    def update(self, model_type, step, total_steps):
        if not self.enabled:
            return
        filled = int(self.width * step / total_steps)
        bar = "#" * filled + "." * (self.width - filled)
        percent = int(100 * step / total_steps)
        end = "\n" if step == total_steps else ""
        print(
            f"\r{model_type:>3} [{bar}] {step}/{total_steps} {percent:3d}%",
            end=end,
            file=sys.stderr,
            flush=False,
        )


def save_prediction_plot(results, output_path, sample_index=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ann = results["ann"]
    snn = results["snn"]
    target = np.asarray(ann.targets[sample_index]).squeeze()
    ann_pred = np.asarray(ann.generated_sequences[sample_index]).squeeze()
    snn_pred = np.asarray(snn.generated_sequences[sample_index]).squeeze()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(target, label="target", linewidth=2.0, color="black")
    ax.plot(ann_pred, label="ann generated", linewidth=1.6)
    ax.plot(snn_pred, label="snn generated", linewidth=1.6)
    ax.set_title("Sine Flow Matching: Target vs Generated Sequence")
    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare ANN and SNN flow matching on the existing sine dataset."
    )
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sampler-steps", type=int, default=8)
    parser.add_argument("--snn-threshold", type=float, default=0.5)
    parser.add_argument("--snn-beta", type=float, default=0.8)
    parser.add_argument("--surrogate-sigma", type=float, default=5.0)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--sampling-size", type=int, default=5)
    parser.add_argument("--data-dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-updates", type=int, default=50)
    parser.add_argument("--backend", choices=("jax", "mlx"), default="jax")
    parser.add_argument("--chunk-steps", type=int, default=50)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--plot-path", default="artifacts/sine_prediction.png")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-sample", type=int, default=0)
    args = parser.parse_args()

    config = BraxFlowConfig(
        horizon=args.horizon,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        sampler_steps=args.sampler_steps,
        snn_threshold=args.snn_threshold,
        snn_beta=args.snn_beta,
        surrogate_sigma=args.surrogate_sigma,
    )
    key_ann, key_snn = jax.random.split(jax.random.PRNGKey(args.seed))

    print(
        f"sine_flow backend={args.backend} horizon={config.horizon} batch={config.batch_size} "
        f"sampling_size={args.sampling_size} train_steps={args.train_steps}"
    )
    progress_interval = max(1, args.train_steps // max(1, args.progress_updates))
    progress = ProgressBar(enabled=not args.no_progress)

    if args.backend == "jax":
        results = {
            "ann": train_sine_flow_model(
                key_ann,
                config,
                model_type="ann",
                train_steps=args.train_steps,
                sampling_size=args.sampling_size,
                data_dt=args.data_dt,
                progress_callback=progress.update,
                progress_interval=progress_interval,
                compiled=not args.no_compile,
                chunk_steps=args.chunk_steps,
            ),
            "snn": train_sine_flow_model(
                key_snn,
                config,
                model_type="snn",
                train_steps=args.train_steps,
                sampling_size=args.sampling_size,
                data_dt=args.data_dt,
                progress_callback=progress.update,
                progress_interval=progress_interval,
                compiled=not args.no_compile,
                chunk_steps=args.chunk_steps,
            ),
        }
    else:
        from utils.mlx_sine_flow_matching import train_mlx_sine_flow_model

        try:
            results = {
                "ann": train_mlx_sine_flow_model(
                    args.seed,
                    config,
                    model_type="ann",
                    train_steps=args.train_steps,
                    sampling_size=args.sampling_size,
                    data_dt=args.data_dt,
                    progress_callback=progress.update,
                    progress_interval=progress_interval,
                    compiled=not args.no_compile,
                    chunk_steps=args.chunk_steps,
                ),
                "snn": train_mlx_sine_flow_model(
                    args.seed + 1,
                    config,
                    model_type="snn",
                    train_steps=args.train_steps,
                    sampling_size=args.sampling_size,
                    data_dt=args.data_dt,
                    progress_callback=progress.update,
                    progress_interval=progress_interval,
                    compiled=not args.no_compile,
                    chunk_steps=args.chunk_steps,
                ),
            }
        except RuntimeError as exc:
            raise SystemExit(
                "MLX backend failed. Run `uv sync --extra mlx` on an Apple Silicon "
                "Mac with Metal access, then retry with `--backend mlx`.\n"
                f"Original error: {exc}"
            ) from exc

    print("\nFinal sine comparison")
    for model_type, result in results.items():
        print(
            f"{model_type:>3}: "
            f"eval_loss={float(result.eval_loss):.6f} "
            f"sequence_mse={float(result.generated_sequence_mse):.6f} "
            f"spikes={float(result.spike_rate):.4f} "
            f"nfe={int(result.nfe)}"
        )

    if not args.no_plot:
        plot_path = save_prediction_plot(results, args.plot_path, args.plot_sample)
        print(f"\nSaved prediction plot: {plot_path}")


if __name__ == "__main__":
    main()
