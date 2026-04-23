"""Manual tuning script: iterate candidate architectures and train each with fixed hyperparameters.

This script reuses the same candidate-generation logic as `tune_inr_das.py` (via
`generate_candidate_architectures`) but does a manual loop over candidates and
saves per-run results plus a `best_config.yml` in the output folder.
"""
from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
import os
import yaml
import csv

import numpy as np

# Configure XLA path like the other tuning scripts (non-fatal if CONDA_PREFIX unset)
conda_prefix = os.environ.get("CONDA_PREFIX", "").replace("\\", "/")
if conda_prefix:
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={conda_prefix}"
    os.environ["PATH"] = f"{conda_prefix}/bin;" + os.environ.get("PATH", "")

import tensorflow as tf

import inr_apodizations.sandbox_helpers as helpers
from inr_apodizations import config
from inr_apodizations.modeling.trainer import DasInrTrainer, build_mlp_inr
from inr_apodizations.tuning.utils import generate_candidate_architectures
from inr_apodizations.apodizations import compute_dynamic_apodizations_tf
from scatterer_metrics import compute_scatterer_metrics


CONFIG_PATH = Path("configs/tune_inr_das_manual.yml")
cfg = helpers.load_experiment_config(str(CONFIG_PATH))

seed = int(cfg["training"]["seed"])
tf.keras.utils.set_random_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Generate candidates using existing helper (same behavior as tune_inr_das.py)
candidate_architectures = generate_candidate_architectures(cfg["tuning"], fallback_seed=seed)
print(f"Generated {len(candidate_architectures)} candidate INR architectures.")

dataset_folder = Path(cfg["io"]["dataset_folder"])
if not dataset_folder.is_absolute():
    dataset_folder = config.PROJ_ROOT / dataset_folder

print(f"Loading dataset from: {dataset_folder}")
delayed, noise, targets, gaussian_masks, info = helpers.load_delayed_samples_dataset(str(dataset_folder))
kp, cm = helpers.build_coordinate_manager(str(dataset_folder), physical_feature_set=cfg["model"]["physical_feature_set"])
scatterers_all = helpers.load_saved_scatterers(str(dataset_folder))

train_idx, val_idx = helpers.split_train_validation_indices(
    n_examples=delayed.shape[0],
    train_fraction=float(cfg["training"]["train_fraction"]),
    seed=seed,
)

val_delayed = delayed[val_idx].astype(np.complex64)
val_scatterers = []
for idx in val_idx:
    s = np.asarray(scatterers_all[int(idx)], dtype=np.float32).copy()
    s[:, :2] *= 1000.0
    val_scatterers.append(s[:, :2])

# Optional mask weighting
mask_weighting_cfg = dict(cfg["training"].get("mask_weighting", {}))
train_loss_weights = helpers.build_gaussian_loss_weights(gaussian_masks, mask_weighting_cfg)
use_pixelwise_weights = train_loss_weights is not None


def _pixelwise_mae(y_true, y_pred):
    return tf.abs(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32))


# Pre-compute Hanning baseline SNR
print("Computing Hanning baseline SNR...")
baseline_f = float(cfg["training"]["baseline_f_number"])
apods_h = compute_dynamic_apodizations_tf(
    cm=cm, f_number=baseline_f, methods=("hanning",), scaled=bool(cfg["model"]["scaled_features"]) 
)
hanning_weights = apods_h["hanning"]
hanning_weights_b = tf.expand_dims(hanning_weights, axis=0)

val_delayed_tf = tf.convert_to_tensor(val_delayed)
hanning_complex = tf.reduce_sum(val_delayed_tf * tf.cast(hanning_weights_b, val_delayed_tf.dtype), axis=1)
hanning_abs = tf.abs(hanning_complex).numpy()

h_metrics = compute_scatterer_metrics(hanning_abs, val_scatterers, cm, radius_mm=cfg["scatterer_eval"]["radius_mm"])
h_peaks = h_metrics["aggregated"]["peak_amplitudes"]
h_bg_rms = h_metrics["aggregated"]["point_background_rms"]
hanning_snrs = h_peaks / (h_bg_rms + 1e-12)

print(f"Hanning baseline ready. Mean SNR: {np.mean(hanning_snrs):.4f}")


class SnrImprovementCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_delayed_tf, val_scatterers, hanning_snrs, cm):
        super().__init__()
        self.val_delayed_tf = val_delayed_tf
        self.val_scatterers = val_scatterers
        self.hanning_snrs = hanning_snrs
        self.cm = cm

    def on_epoch_end(self, epoch, logs=None):
        pred_complex, _ = self.model.reconstruct_image(self.val_delayed_tf, training=False)
        pred_abs = np.abs(pred_complex.numpy())
        metrics = compute_scatterer_metrics(pred_abs, self.val_scatterers, self.cm, radius_mm=cfg["scatterer_eval"]["radius_mm"]) 
        inr_peaks = metrics["aggregated"]["peak_amplitudes"]
        inr_bg_rms = metrics["aggregated"]["point_background_rms"]
        inr_snrs = inr_peaks / (inr_bg_rms + 1e-12)
        snr_ratios = inr_snrs / (self.hanning_snrs + 1e-12)
        better_count = int(np.sum(snr_ratios > 1.0))
        better_fraction = better_count / len(snr_ratios)
        logs["val_snr_better_count"] = better_count
        logs["val_snr_better_fraction"] = better_fraction
        print(f" - val_snr_better_count: {better_count}/{len(snr_ratios)} ({better_fraction:.2%})")


def evaluate_on_val(trainer):
    pred_complex, _ = trainer.reconstruct_image(val_delayed_tf, training=False)
    pred_abs = np.abs(pred_complex.numpy())
    metrics = compute_scatterer_metrics(pred_abs, val_scatterers, cm, radius_mm=cfg["scatterer_eval"]["radius_mm"]) 
    inr_peaks = metrics["aggregated"]["peak_amplitudes"]
    inr_bg_rms = metrics["aggregated"]["point_background_rms"]
    inr_snrs = inr_peaks / (inr_bg_rms + 1e-12)
    snr_ratios = inr_snrs / (hanning_snrs + 1e-12)
    better_count = int(np.sum(snr_ratios > 1.0))
    better_fraction = better_count / len(snr_ratios)
    return dict(better_count=int(better_count), better_fraction=float(better_fraction))



timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_name = f"inr_das_manual_{timestamp}"
tuning_dir = Path(cfg["io"]["sandbox_output_root"]) / project_name
tuning_dir.mkdir(parents=True, exist_ok=True)

# Search history structures (persisted after each candidate)
search_history: list[dict] = []
search_history_path = tuning_dir / "search_history.yml"
search_history_csv_path = tuning_dir / "search_history.csv"

# Prepare datasets
train_ds = helpers.build_tf_dataset_by_indices(
    delayed, targets, indices=train_idx,
    sample_weights=(train_loss_weights if train_loss_weights is not None else None),
    batch_size=cfg["training"]["batch_size"], shuffle=True, seed=seed,
)
val_ds = helpers.build_tf_dataset_by_indices(
    delayed, targets, indices=val_idx,
    sample_weights=(train_loss_weights if train_loss_weights is not None else None),
    batch_size=cfg["training"]["batch_size"], shuffle=False, seed=seed,
)

best_score = -1
best_record = None

for i, hidden_units in enumerate(candidate_architectures):
    print(f"\n=== Running candidate {i+1}/{len(candidate_architectures)}: {hidden_units} ===")
    inr_mlp = build_mlp_inr(
        input_dim=cm.n_physical_features,
        hidden_units_config=hidden_units,
        activation=cfg["model"]["activation"],
        output_activation=cfg["model"]["output_activation"],
    )

    trainer = DasInrTrainer(
        apodization_model=inr_mlp,
        features_grid=cm.get_features_grid(scaled=cfg["model"]["scaled_features"]),
        feature_chunk_size=cfg["model"]["feature_chunk_size"],
        weight_regularization_enabled=True,
        weight_regularization_lambda=float(cfg["tuning"]["reg_lambda"]),
        weight_regularization_tau=float(cfg["tuning"]["reg_tau"]),
    )

    lr = float(cfg["tuning"].get("lr", cfg["training"].get("learning_rate", 1e-3)))
    loss_fn = _pixelwise_mae if use_pixelwise_weights else tf.keras.losses.MeanAbsoluteError()
    trainer.compile(
        jit_compile=True,
        optimizer=tf.keras.optimizers.experimental.AdamW(
            weight_decay=float(cfg["training"]["weight_decay"]),
            learning_rate=lr,
            jit_compile=True,
        ),
        loss=loss_fn if not isinstance(loss_fn, tf.keras.losses.Loss) else loss_fn,
        weighted_metrics=[],
    )

    run_name = f"run_{i:02d}_" + "x".join([str(int(x)) for x in hidden_units])
    run_dir = tuning_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        SnrImprovementCallback(val_delayed_tf, val_scatterers, hanning_snrs, cm),
        tf.keras.callbacks.EarlyStopping(monitor="val_snr_better_count", mode="max", patience=cfg["training"]["early_stopping_patience"]),
    ]

    history = trainer.fit(
        train_ds,
        epochs=cfg["training"]["epochs"],
        validation_data=val_ds,
        callbacks=callbacks,
    )

    # Evaluate final performance on validation set
    eval_res = evaluate_on_val(trainer)
    print(f"Candidate result: {eval_res}")

    # Save run summary
    run_summary = {
        "hidden_units": [int(v) for v in hidden_units],
        "reg_lambda": float(cfg["tuning"]["reg_lambda"]),
        "reg_tau": float(cfg["tuning"]["reg_tau"]),
        "lr": lr,
        "eval": eval_res,
        "history": {k: v for k, v in history.history.items()},
    }
    with open(run_dir / "run_summary.yml", "w") as f:
        yaml.safe_dump(run_summary, f, sort_keys=False)

    # Update search history (persist immediately to avoid data loss)
    entry = {
        "run_index": int(i),
        "hidden_units": [int(v) for v in hidden_units],
        "reg_lambda": float(cfg["tuning"]["reg_lambda"]),
        "reg_tau": float(cfg["tuning"]["reg_tau"]),
        "lr": float(lr),
        "eval": {
            "better_count": int(eval_res.get("better_count", 0)),
            "better_fraction": float(eval_res.get("better_fraction", 0.0)),
        },
        "run_dir": str(run_dir.name),
    }
    search_history.append(entry)

    # Write YAML history (overwrites atomically each candidate)
    with open(search_history_path, "w") as f:
        yaml.safe_dump(search_history, f, sort_keys=False)

    # Also export a compact CSV for quick analysis
    with open(search_history_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_index",
            "hidden_units",
            "reg_lambda",
            "reg_tau",
            "lr",
            "better_count",
            "better_fraction",
            "run_dir",
        ])
        for e in search_history:
            writer.writerow([
                e["run_index"],
                "x".join([str(int(x)) for x in e["hidden_units"]]),
                e["reg_lambda"],
                e["reg_tau"],
                e["lr"],
                e["eval"]["better_count"],
                e["eval"]["better_fraction"],
                e["run_dir"],
            ])

    score = int(eval_res["better_count"])
    if score > best_score:
        best_score = score
        best_record = run_summary

# Save best config
if best_record is not None:
    best_config_path = tuning_dir / "best_config.yml"
    with open(best_config_path, "w") as f:
        yaml.safe_dump(best_record, f, sort_keys=False)
    print(f"Best candidate saved to: {best_config_path}")
else:
    print("No candidates produced a result.")

