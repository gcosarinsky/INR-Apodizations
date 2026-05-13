"""Sweep Gaussian-mask lambda with decomposition-based MaskedMAE.

The evaluated metric is defined as:

MaskedMAE(lambda) = MAE + lambda * mean(abs(E * Mg))

where ``E = y_true - y_pred`` and ``Mg`` is the gaussian mask in ``[0, 1]``.

The script computes the two lambda-independent terms in one pass over the
evaluation subset and then evaluates many lambdas without recomputing
predictions or absolute error tensors.

Usage (from project root):

python sandbox/inr_das_experiment/sweep_mask_lambda.py \
    --dataset-folder data/delayed_samples_dataset/20260406_155448 \
    --lambdas 0,0.5,1,2,3,5,10, 15, 20
    
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import inr_apodizations.experiment_helpers as helpers

from inr_apodizations.apodizations import compute_dynamic_apodizations_tf


def parse_lambda_list(value: str) -> list[float]:
    """Parse comma-separated lambda list or special keyword ``log``."""
    token = value.strip().lower()
    if token == "log":
        return list(np.logspace(-3, 1, num=7).tolist())
    parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    lambdas = [float(chunk) for chunk in parts]
    if any(lam < 0.0 for lam in lambdas):
        raise ValueError("All lambda values must be >= 0")
    return lambdas


def _compute_method_predictions(
    delayed_chunk: tf.Tensor,
    hanning_weights_b: tf.Tensor | None,
    boxcar_weights_b: tf.Tensor | None,
) -> dict[str, tf.Tensor]:
    """Compute baseline predictions for one delayed-sample batch."""
    predictions = {
        "zero": tf.zeros_like(tf.abs(tf.reduce_sum(delayed_chunk, axis=1))),
        "uniform": tf.abs(tf.reduce_sum(delayed_chunk, axis=1)),
    }
    if hanning_weights_b is not None:
        predictions["hanning"] = tf.abs(
            tf.reduce_sum(delayed_chunk * tf.cast(hanning_weights_b, delayed_chunk.dtype), axis=1)
        )
    if boxcar_weights_b is not None:
        predictions["boxcar"] = tf.abs(
            tf.reduce_sum(delayed_chunk * tf.cast(boxcar_weights_b, delayed_chunk.dtype), axis=1)
        )
    return predictions


def decompose_masked_mae_terms(
    delayed: np.ndarray,
    targets: np.ndarray,
    gaussian_masks: np.ndarray,
    val_idx: np.ndarray,
    cm,
    baseline_f_number: float,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    """Compute MAE and masked-error terms for each baseline method in one pass."""
    apods = compute_dynamic_apodizations_tf(
        cm=cm,
        f_number=baseline_f_number,
        methods=("hanning", "boxcar"),
        scaled=False,
    )
    hanning_weights = apods.get("hanning")
    boxcar_weights = apods.get("boxcar")
    hanning_weights_b = tf.expand_dims(hanning_weights, axis=0) if hanning_weights is not None else None
    boxcar_weights_b = tf.expand_dims(boxcar_weights, axis=0) if boxcar_weights is not None else None

    accumulators: dict[str, dict[str, float]] = {
        "zero": {"abs_sum": 0.0, "masked_abs_sum": 0.0, "count": 0.0},
        "uniform": {"abs_sum": 0.0, "masked_abs_sum": 0.0, "count": 0.0},
    }
    if hanning_weights_b is not None:
        accumulators["hanning"] = {"abs_sum": 0.0, "masked_abs_sum": 0.0, "count": 0.0}
    if boxcar_weights_b is not None:
        accumulators["boxcar"] = {"abs_sum": 0.0, "masked_abs_sum": 0.0, "count": 0.0}

    n_val = int(len(val_idx))
    for start in range(0, n_val, batch_size):
        end = min(start + batch_size, n_val)
        idx_chunk = val_idx[start:end]

        delayed_chunk = tf.convert_to_tensor(delayed[idx_chunk].astype(np.complex64, copy=False))
        y_true_chunk = tf.convert_to_tensor(targets[idx_chunk].astype(np.float32, copy=False))
        mask_chunk = tf.convert_to_tensor(gaussian_masks[idx_chunk].astype(np.float32, copy=False))

        predictions = _compute_method_predictions(
            delayed_chunk=delayed_chunk,
            hanning_weights_b=hanning_weights_b,
            boxcar_weights_b=boxcar_weights_b,
        )
        for method_name, y_pred_chunk in predictions.items():
            abs_error = tf.abs(tf.cast(y_true_chunk, tf.float32) - tf.cast(y_pred_chunk, tf.float32))
            abs_sum = float(tf.reduce_sum(abs_error).numpy())
            masked_abs_sum = float(tf.reduce_sum(abs_error * mask_chunk).numpy())
            pixel_count = float(tf.size(abs_error).numpy())

            accumulators[method_name]["abs_sum"] += abs_sum
            accumulators[method_name]["masked_abs_sum"] += masked_abs_sum
            accumulators[method_name]["count"] += pixel_count

    decomposition = {}
    for method_name, values in accumulators.items():
        count = values["count"]
        if count <= 0.0:
            raise ValueError("Evaluation subset is empty; cannot compute decomposition terms")
        mae = values["abs_sum"] / count
        masked_term = values["masked_abs_sum"] / count
        decomposition[method_name] = {
            "mae": float(mae),
            "masked_term": float(masked_term),
            "count": float(count),
        }
    return decomposition


def evaluate_lambdas(
    decomposition: dict[str, dict[str, float]],
    lambdas: Sequence[float],
) -> dict[float, dict[str, float]]:
    """Evaluate MaskedMAE(lambda) from decomposition terms."""
    results: dict[float, dict[str, float]] = {}
    for lam in lambdas:
        results[float(lam)] = {
            method_name: values["mae"] + float(lam) * values["masked_term"]
            for method_name, values in decomposition.items()
        }
    return results


def save_results_csv(results: dict[float, dict[str, float]], output_path: str) -> None:
    """Save lambda sweep values as CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    method_keys = sorted(next(iter(results.values())).keys())
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["lambda"] + method_keys)
        for lam in sorted(results.keys()):
            writer.writerow([lam] + [results[lam][method_name] for method_name in method_keys])


def save_decomposition_csv(
    decomposition: dict[str, dict[str, float]],
    output_path: str,
) -> None:
    """Save MAE and masked-error decomposition terms per method."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["method", "mae", "masked_term", "count"])
        for method_name in sorted(decomposition.keys()):
            values = decomposition[method_name]
            writer.writerow(
                [method_name, values["mae"], values["masked_term"], values["count"]]
            )


def plot_results(results: dict[float, dict[str, float]], output_path: str) -> None:
    """Plot MaskedMAE(lambda) curves for all methods."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lambdas = sorted(results.keys())
    methods = sorted(next(iter(results.values())).keys())

    plt.figure(figsize=(7, 4.5))
    for method_name in methods:
        values = [results[lam][method_name] for lam in lambdas]
        plt.plot(lambdas, values, marker="o", label=method_name)

    plt.xlabel("lambda")
    plt.ylabel("MaskedMAE")
    plt.title("MaskedMAE(lambda) = MAE + lambda * mean(|E * Mg|)")
    plt.legend()
    plt.grid(True)
    positive = [lam for lam in lambdas if lam > 0]
    if positive and max(lambdas) / min(positive) > 50:
        plt.xscale("log")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    """Run lambda sweep using decomposition-based MaskedMAE evaluation."""
    parser = argparse.ArgumentParser(
        description="Sweep Gaussian-mask lambda for MaskedMAE decomposition"
    )
    parser.add_argument(
        "--dataset-folder",
        required=True,
        help="Dataset folder path containing delayed samples and gaussian masks",
    )
    parser.add_argument(
        "--lambdas",
        default="0,0.5,1,2,3,5,10,15,20",
        help="Comma-separated lambda values or 'log'",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--use-validation",
        action="store_true",
        help="Use validation split; otherwise evaluate on all examples",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of examples used for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory root (default: script outputs folder)",
    )
    args = parser.parse_args()

    lambdas = parse_lambda_list(args.lambdas)
    delayed, noise, targets, gaussian_masks, info = helpers.load_delayed_samples_dataset(
        args.dataset_folder
    )
    helpers.validate_dataset_shapes(delayed, targets, gaussian_masks)
    kp, cm = helpers.build_coordinate_manager(args.dataset_folder)

    if args.use_validation:
        _, val_idx = helpers.split_train_validation_indices(
            n_examples=delayed.shape[0],
            train_fraction=0.8,
            seed=0,
        )
    else:
        val_idx = np.arange(delayed.shape[0], dtype=np.int64)

    if args.max_examples is not None:
        val_idx = val_idx[: int(args.max_examples)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir or os.path.join(SCRIPT_DIR, "outputs/lambda_sweep_outputs")
    output_dir = os.path.join(output_root, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print("Running decomposition pass...")
    decomposition = decompose_masked_mae_terms(
        delayed=delayed,
        targets=targets,
        gaussian_masks=gaussian_masks,
        val_idx=val_idx,
        cm=cm,
        baseline_f_number=kp.f_number,
        batch_size=int(args.batch_size),
    )

    print("Evaluating lambdas:", lambdas)
    results = evaluate_lambdas(decomposition=decomposition, lambdas=lambdas)

    csv_path = os.path.join(output_dir, "mae_vs_lambda.csv")
    decomp_csv_path = os.path.join(output_dir, "mae_decomposition_terms.csv")
    plot_path = os.path.join(output_dir, "mae_vs_lambda.png")
    save_results_csv(results, csv_path)
    save_decomposition_csv(decomposition, decomp_csv_path)
    plot_results(results, plot_path)

    print(f"Results saved: {csv_path}")
    print(f"Decomposition saved: {decomp_csv_path}")
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
