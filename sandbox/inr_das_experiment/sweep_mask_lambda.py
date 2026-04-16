"""Sweep Gaussian-mask lambda and evaluate MaskedMAE for reference apodizations.

This script evaluates how the `MaskedMAE` metric varies when per-pixel
loss weights are computed as `1 + lambda * gaussian_mask`. It computes
MaskedMAE for the following prediction methods:

- zero image (all zeros)
- uniform (simple sum over elements)
- Hanning apodization
- Boxcar apodization

The script iterates over a list of `lambda` values, computes per-pixel
weights using the saved `gaussian_masks_dataset.npy`, and accumulates the
masked MAE across the validation subset (configurable). Results are saved
as CSV and a PNG plot in a timestamped output folder.

Usage (from project root):

python sandbox/inr_das_experiment/sweep_mask_lambda.py \
    --dataset-folder data/delayed_samples_dataset/20260406_155448 \
    --lambdas 0,0.5,1,2,3,5,10, 15, 20
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Ensure local helpers module is importable regardless of CWD when running
# from project root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import helpers  # type: ignore

from inr_apodizations.apodizations import compute_dynamic_apodizations_tf
from inr_apodizations.modeling.metrics import MaskedMAE


def parse_lambda_list(value: str) -> list[float]:
    """Parse comma-separated lambda list or special keywords.

    Accepts comma-separated numeric values or the keyword 'log' which
    generates a default logarithmic sweep.
    """
    v = value.strip().lower()
    if v == "log":
        return list(np.logspace(-3, 1, num=7).tolist())
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [float(p) for p in parts]


def evaluate_on_indices(
    delayed: np.ndarray,
    targets: np.ndarray,
    gaussian_masks: np.ndarray,
    val_idx: np.ndarray,
    cm,
    baseline_f_number: float,
    lambdas: Sequence[float],
    batch_size: int = 8,
) -> dict:
    """Evaluate MaskedMAE for each lambda and each baseline method.

    Returns a mapping: lambda -> {method_name: mae_value}
    """
    results = {}

    # Precompute apodization maps (E, Z, X)
    apods = compute_dynamic_apodizations_tf(
        cm=cm, f_number=baseline_f_number, methods=("hanning", "boxcar"), scaled=False
    )
    hanning_weights = apods.get("hanning")  # (E, Z, X) or None
    boxcar_weights = apods.get("boxcar")

    n_val = int(len(val_idx))

    for lam in lambdas:
        # instantiate fresh metrics per lambda
        m_zero = MaskedMAE(name="zero")
        m_uniform = MaskedMAE(name="uniform")
        m_hanning = MaskedMAE(name="hanning") if hanning_weights is not None else None
        m_boxcar = MaskedMAE(name="boxcar") if boxcar_weights is not None else None

        # Build per-lambda global weights array once (N, Z, X)
        wcfg = {"enabled": True, "lambda": float(lam)}
        loss_weights = helpers.build_gaussian_loss_weights(gaussian_masks, wcfg)
        if loss_weights is None:
            # Fallback to ones
            loss_weights = np.ones_like(gaussian_masks, dtype=np.float32)

        # Iterate in batches
        for start in range(0, n_val, batch_size):
            end = min(start + batch_size, n_val)
            idx_chunk = val_idx[start:end]

            # Prepare delayed chunk (batch, E, Z, X)
            val_delayed_chunk = tf.convert_to_tensor(delayed[idx_chunk].astype(np.complex64, copy=False))
            y_true_chunk = tf.convert_to_tensor(targets[idx_chunk].astype(np.float32, copy=False))

            weights_chunk = tf.convert_to_tensor(loss_weights[idx_chunk].astype(np.float32, copy=False))

            # Zero prediction
            m_zero.update_state(y_true_chunk, tf.zeros_like(y_true_chunk), sample_weight=weights_chunk)

            # Uniform (sum over elements)
            uniform_pred = tf.abs(tf.reduce_sum(val_delayed_chunk, axis=1))
            m_uniform.update_state(y_true_chunk, uniform_pred, sample_weight=weights_chunk)

            # Hanning
            if hanning_weights is not None and m_hanning is not None:
                # convert to float tensor first, then cast to the complex dtype of delayed chunk
                h_w = tf.convert_to_tensor(hanning_weights, dtype=tf.float32)
                h_w_b = tf.expand_dims(tf.cast(h_w, val_delayed_chunk.dtype), axis=0)
                hanning_pred = tf.abs(tf.reduce_sum(val_delayed_chunk * h_w_b, axis=1))
                m_hanning.update_state(y_true_chunk, hanning_pred, sample_weight=weights_chunk)

            # Boxcar
            if boxcar_weights is not None and m_boxcar is not None:
                b_w = tf.convert_to_tensor(boxcar_weights, dtype=tf.float32)
                b_w_b = tf.expand_dims(tf.cast(b_w, val_delayed_chunk.dtype), axis=0)
                boxcar_pred = tf.abs(tf.reduce_sum(val_delayed_chunk * b_w_b, axis=1))
                m_boxcar.update_state(y_true_chunk, boxcar_pred, sample_weight=weights_chunk)

        # Collect results
        res = {
            "zero": float(m_zero.result().numpy()),
            "uniform": float(m_uniform.result().numpy()),
        }
        if m_hanning is not None:
            res["hanning"] = float(m_hanning.result().numpy())
        if m_boxcar is not None:
            res["boxcar"] = float(m_boxcar.result().numpy())

        results[float(lam)] = res

    return results


def save_results_csv(results: dict, output_path: str) -> None:
    """Save results mapping to CSV with columns: lambda,zero,uniform,hanning,boxcar"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Determine all method keys
    method_keys = set()
    for r in results.values():
        method_keys.update(r.keys())
    method_keys = sorted(method_keys)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda"] + method_keys)
        for lam in sorted(results.keys()):
            row = [lam] + [results[lam].get(k, "") for k in method_keys]
            writer.writerow(row)


def plot_results(results: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lams = sorted(results.keys())
    methods = sorted(next(iter(results.values())).keys())

    plt.figure(figsize=(7, 4.5))
    for m in methods:
        ys = [results[lam].get(m, np.nan) for lam in lams]
        plt.plot(lams, ys, marker="o", label=m)

    plt.xlabel("lambda")
    plt.ylabel("Masked MAE")
    plt.title("MaskedMAE vs Gaussian mask lambda")
    plt.legend()
    plt.grid(True)
    # Use log scale if lambdas span several orders
    try:
        if max(lams) / (min([x for x in lams if x > 0]) if any(x > 0 for x in lams) else 1) > 50:
            plt.xscale("log")
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Sweep gaussian-mask lambda and evaluate MaskedMAE")
    parser.add_argument("--dataset-folder", required=True, help="Dataset folder path containing delayed_samples and masks")
    parser.add_argument("--lambdas", default="0,0.5,1,2,3,5,10,15,20", help="Comma-separated lambda values or 'log' for default log sweep")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-validation", action="store_true", help="Use a validation split; otherwise evaluate on full dataset")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap on number of validation examples to use")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: sandbox_output/<timestamp>)")
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    lambdas = parse_lambda_list(args.lambdas)

    delayed, noise, targets, gaussian_masks, info = helpers.load_delayed_samples_dataset(dataset_folder)
    helpers.validate_dataset_shapes(delayed, targets, gaussian_masks)

    # Build coordinate manager and kernel params
    kp, cm = helpers.build_coordinate_manager(dataset_folder)

    # Determine validation indices
    if args.use_validation:
        train_idx, val_idx = helpers.split_train_validation_indices(n_examples=delayed.shape[0], train_fraction=0.8, seed=0)
    else:
        val_idx = np.arange(delayed.shape[0], dtype=np.int64)

    if args.max_examples is not None:
        val_idx = val_idx[: args.max_examples]

    # Baseline f-number
    baseline_f_number = kp.f_number

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.output_dir or os.path.join(SCRIPT_DIR, "outputs/lambda_sweep_outputs")
    out_dir = os.path.join(out_root, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    print("Starting lambda sweep:", lambdas)
    results = evaluate_on_indices(
        delayed=delayed,
        targets=targets,
        gaussian_masks=gaussian_masks,
        val_idx=val_idx,
        cm=cm,
        baseline_f_number=baseline_f_number,
        lambdas=lambdas,
        batch_size=args.batch_size,
    )

    csv_path = os.path.join(out_dir, "mae_vs_lambda.csv")
    png_path = os.path.join(out_dir, "mae_vs_lambda.png")
    save_results_csv(results, csv_path)
    plot_results(results, png_path)

    print(f"Results saved: {csv_path}")
    print(f"Plot saved: {png_path}")


if __name__ == "__main__":
    main()
