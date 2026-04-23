"""Hyperband Hyperparameter Tuning for INR-based DAS using Keras Tuner.

This script is a copy of `tune_inr_das_hyperband.py` but uses validation
RelativeMAE normalized by ``y_pred`` as the objective to be minimized.
"""
from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import os
import numpy as np

# Configure XLA using the active Conda environment before importing TensorFlow.
conda_prefix = os.environ.get("CONDA_PREFIX", "").replace("\\", "/")
if not conda_prefix:
    raise RuntimeError(
        "CONDA_PREFIX is not set. Activate the Conda environment before running this script."
    )

os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={conda_prefix}"
os.environ["PATH"] = f"{conda_prefix}/bin;" + os.environ["PATH"]

print(f"XLA search path configured to: {conda_prefix}")
print(f"Effective XLA env: XLA_FLAGS={os.environ.get('XLA_FLAGS', '(unset)')}")

import tensorflow as tf
import keras_tuner as kt
from inr_apodizations.tuning.utils import (
    LiveTrialScorePlot,
    PlottingHyperband,
    generate_candidate_architectures,
)
import yaml
import sys
import json
import csv

print("TF GPUs:", tf.config.list_physical_devices("GPU"))

# Ensure Keras Tuner runs on a specific GPU device by using a OneDeviceStrategy.
# This places variable creation and model building on the chosen GPU.
strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
print(f"Using distribution strategy: {strategy}")

import inr_apodizations.sandbox_helpers as helpers
from inr_apodizations import config
from inr_apodizations.modeling.metrics import RelativeMAE
from inr_apodizations.modeling.metrics import PixelWeightedMAE
from inr_apodizations.modeling.losses import PixelWeightedMAELoss
from inr_apodizations.modeling.trainer import DasInrTrainer, build_mlp_inr


# --- Configuration & Data Loading ---
CONFIG_PATH = Path("configs/tune_config.yml")
cfg = helpers.load_experiment_config(str(CONFIG_PATH))
seed = int(cfg["training"]["seed"])
tf.keras.utils.set_random_seed(seed)
random.seed(seed)
np.random.seed(seed)

candidate_architectures = generate_candidate_architectures(cfg["tuning"], fallback_seed=seed)
print(f"Generated {len(candidate_architectures)} candidate INR architectures.")

# Print all candidate architectures and ask for user confirmation before tuning.
# Messages to the user are in Spanish per workspace conventions; code/comments remain in English.
print("\nCandidate architectures:")
for i, arch in enumerate(candidate_architectures):
    print(f"  [{i}] {arch}")

print(f"Total candidate architectures: {len(candidate_architectures)}")

try:
    resp = input("¿Desea continuar con el tuning? (y/N): ").strip().lower()
except EOFError:
    print("No hay entrada disponible. Cancelando el tuning.")
    sys.exit(0)

if resp not in ("y", "yes", "s", "si"):
    print("Tuning cancelado por el usuario.")
    sys.exit(0)

dataset_folder = Path(cfg["io"]["dataset_folder"])
if not dataset_folder.is_absolute():
    dataset_folder = config.PROJ_ROOT / dataset_folder

print(f"Loading dataset from: {dataset_folder}")
delayed, noise, targets, gaussian_masks, info = helpers.load_delayed_samples_dataset(str(dataset_folder))
_kp, cm = helpers.build_coordinate_manager(
    str(dataset_folder), physical_feature_set=cfg["model"]["physical_feature_set"]
)

train_idx, val_idx = helpers.split_train_validation_indices(
    n_examples=delayed.shape[0],
    train_fraction=float(cfg["training"]["train_fraction"]),
    seed=seed
)
# Optionally reduce the number of validation examples (config: training.val_subset_size)
val_subset_size = int(cfg["training"].get("val_subset_size", 0))
if val_subset_size > 0 and val_subset_size < len(val_idx):
    rnd = np.random.RandomState(seed)
    val_idx = list(rnd.choice(val_idx, size=val_subset_size, replace=False))
    val_idx.sort()
    print(f"Validation set reduced to {len(val_idx)} examples (subset_size={val_subset_size}).")

# --- Optional: Mask weighting (pixelwise sample weights)
mask_weighting_cfg = dict(cfg["training"].get("mask_weighting", {}))
use_pixelwise_weights = bool(mask_weighting_cfg.get("enabled", False))
pixel_weight_lambda_min = float(cfg["tuning"].get("pixel_weight_lambda_min", 0.1))
pixel_weight_lambda_max = float(cfg["tuning"].get("pixel_weight_lambda_max", 2.0))


def _build_trial_loss_weights(pixel_weight_lambda: float | None) -> np.ndarray | None:
    """Build per-pixel loss weights for the current trial.

    Args:
        pixel_weight_lambda: Trial-specific lambda value. When ``None``, the
            configured mask weighting lambda is used.

    Returns:
        Optional per-example sample-weight tensor.
    """
    if not mask_weighting_cfg.get("enabled", False):
        return None

    trial_mask_cfg = dict(mask_weighting_cfg)
    if pixel_weight_lambda is not None:
        trial_mask_cfg["pixel_weight_lambda"] = float(pixel_weight_lambda)
    return helpers.build_gaussian_loss_weights(gaussian_masks, trial_mask_cfg)

RELATIVE_MAE_NAME = "relative_mae_y_pred"
OBJECTIVE_NAME = f"val_{RELATIVE_MAE_NAME}"


def build_trial_datasets(trial):
    """Build train and validation datasets for the current tuning trial."""
    sample_weights = None
    if use_pixelwise_weights:
        pixel_weight_lambda = trial.hyperparameters.get("pixel_weight_lambda")
        sample_weights = _build_trial_loss_weights(pixel_weight_lambda)

    train_ds = helpers.build_tf_dataset_by_indices(
        delayed,
        targets,
        indices=train_idx,
        sample_weights=sample_weights,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        seed=seed,
    )
    val_ds = helpers.build_tf_dataset_by_indices(
        delayed,
        targets,
        indices=val_idx,
        sample_weights=sample_weights,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        seed=seed,
    )
    return train_ds, val_ds

def _pixelwise_mae(y_true, y_pred):
    """Pixelwise MAE that returns per-pixel absolute error (for sample weighting).

    This returns an unreduced tensor so Keras can multiply by `sample_weight`.
    """
    return tf.abs(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32))

# --- Tuner Model Builder ---
def build_model(hp):
    # 1. Explicit architecture candidate selected by index.
    arch_index = hp.Int(
        "architecture_index",
        min_value=0,
        max_value=len(candidate_architectures) - 1,
        step=1,
    )
    hidden_units = candidate_architectures[arch_index]
    
    inr_mlp = build_mlp_inr(
        input_dim=cm.n_physical_features,
        hidden_units_config=hidden_units,
        activation=cfg["model"]["activation"],
        output_activation=cfg["model"]["output_activation"]
    )
    
    # 2. Physical Trainer with Tunable Regularization
    trainer = DasInrTrainer(
        apodization_model=inr_mlp,
        features_grid=cm.get_features_grid(scaled=cfg["model"]["scaled_features"]),
        feature_chunk_size=cfg["model"]["feature_chunk_size"],
        weight_regularization_enabled=True,
        weight_regularization_lambda=hp.Float("reg_lambda", 
                                              cfg["tuning"]["reg_lambda_min"], 
                                              cfg["tuning"]["reg_lambda_max"], 
                                              sampling="log"),
        weight_regularization_tau=hp.Float("reg_tau", 
                                           cfg["tuning"]["reg_tau_min"], 
                                           cfg["tuning"]["reg_tau_max"]) 
    )
    
    # 3. Compilation
    lr = hp.Float("lr", cfg["tuning"]["lr_min"], cfg["tuning"]["lr_max"], sampling="log")
    loss_obj = PixelWeightedMAELoss(name="pixel_weighted_mae_loss")
    metrics = []
    weighted_metrics = [
        PixelWeightedMAE(name="pixel_weighted_mae"),
        RelativeMAE(name="relative_mae_y_pred", normalize_by="y_pred"),
        RelativeMAE(name="relative_mae_y_true", normalize_by="y_true"),
    ]
    if use_pixelwise_weights:
        hp.Float(
            "pixel_weight_lambda",
            min_value=pixel_weight_lambda_min,
            max_value=pixel_weight_lambda_max,
            sampling="linear",
        )

    trainer.compile(
        jit_compile=False, # Hyperband may not benefit from JIT due to short epochs; set to True if desired.
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr,
            decay=float(cfg["training"]["weight_decay"]),
        ),
        loss=loss_obj,
        metrics=metrics,
        weighted_metrics=weighted_metrics,
    )
    
    return trainer

# --- Run Tuning ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_name = f"inr_das_tuning_{timestamp}"
tuning_dir = Path(cfg["io"]["sandbox_output_root"]) / project_name
tuning_dir.mkdir(parents=True, exist_ok=True)
live_score_plot = LiveTrialScorePlot(
    tuning_dir / "hyperband_relative_mae_progress.png",
    OBJECTIVE_NAME,
    architecture_lookup=candidate_architectures,
)

# Hyperband params (from config)
max_epochs = int(cfg["tuning"].get("hyperband_max_epochs", cfg["training"]["epochs"]))
factor = int(cfg["tuning"].get("hyperband_factor", 3))
hyperband_iterations = int(cfg["tuning"].get("hyperband_iterations", 1))
print(f"Using Hyperband: max_epochs={max_epochs}, factor={factor}, iterations={hyperband_iterations}")

with strategy.scope():
    tuner = PlottingHyperband(
        build_model,
        objective=kt.Objective(
            OBJECTIVE_NAME,
            direction="min",
        ),
        max_epochs=max_epochs,
        factor=factor,
        hyperband_iterations=hyperband_iterations,
        directory=str(tuning_dir),
        project_name="kt_hyperband",
        live_plot=live_score_plot,
        trial_data_builder=build_trial_datasets,
    )

# Prepare datasets for fit
# Pass sample_weights when mask weighting is enabled
train_ds = None
val_ds = None

print("\nStarting Hyperband Optimization...")
tuner.search(
    train_ds,
    epochs=max_epochs,
    validation_data=val_ds,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor=OBJECTIVE_NAME,
            mode="min",
            patience=cfg["training"]["early_stopping_patience"],
        )
    ],
    verbose=True
)

# --- Results & Artifacts ---
print("\nTuning Finished!")
# Save per-architecture scores (trials grouped by candidate architecture)
try:
    helpers.save_tuner_architecture_scores(
        tuner,
        candidate_architectures,
        str(tuning_dir),
        objective_name=OBJECTIVE_NAME,
    )
    print(f"Per-architecture scores saved to: {tuning_dir}")
except Exception as e:
    print("Warning: failed to save per-architecture scores:", e)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_arch_index = int(best_hps.get("architecture_index"))
best_hidden_units = candidate_architectures[best_arch_index]

print("Best Hyperparameters:")
for key in best_hps.values:
    print(f"  {key}: {best_hps.get(key)}")
print(f"  hidden_units: {best_hidden_units}")

# Save best config as YAML.
best_config_path = tuning_dir / "best_config.yml"
best_config = {
    "hidden_units": [int(v) for v in best_hidden_units],
    "reg_lambda": float(best_hps.get("reg_lambda")),
    "reg_tau": float(best_hps.get("reg_tau")),
    "lr": float(best_hps.get("lr")),
    # Include architecture index and pixel-wise weight hyperparameter when available
    "architecture_index": int(best_hps.get("architecture_index")) if best_hps.get("architecture_index") is not None else None,
    "pixel_weight_lambda": (float(best_hps.get("pixel_weight_lambda")) if best_hps.get("pixel_weight_lambda") is not None else None),
}

with open(best_config_path, "w") as f:
    yaml.safe_dump(best_config, f, sort_keys=False)

# Save all candidate architectures for reference
candidates_path = tuning_dir / "candidate_architectures.yml"
with open(candidates_path, "w") as f:
    yaml.safe_dump({"candidates": candidate_architectures}, f, sort_keys=False)


print(f"Artifacts saved in: {tuning_dir}")

# --- Generate a flat CSV with all trials (one row per trial) ---
tuner_trials_path = tuning_dir / "tuner_trials.json"
csv_trials_path = tuning_dir / "tuner_trials.csv"
try:
    if tuner_trials_path.exists():
        with open(tuner_trials_path, "r", encoding="utf-8") as f:
            trials = json.load(f)

        # Discover all hyperparameter keys across trials
        hp_keys = set()
        for t in trials:
            hp = t.get("hyperparameters") or {}
            if isinstance(hp, dict):
                hp_keys.update(hp.keys())
        hp_keys = sorted(hp_keys)

        # CSV header: basic trial fields + discovered hyperparameters
        header = ["trial_id", "architecture_index", "score", "hidden_units"] + hp_keys

        with open(csv_trials_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for t in trials:
                row = []
                row.append(t.get("trial_id", ""))
                ai = t.get("architecture_index")
                row.append("" if ai is None else ai)
                row.append(t.get("score", ""))
                # hidden_units as JSON string (keeps list structure)
                row.append(json.dumps(t.get("hidden_units", None)))
                hp = t.get("hyperparameters") or {}
                for k in hp_keys:
                    v = hp.get(k, None)
                    try:
                        row.append(json.dumps(v))
                    except Exception:
                        row.append(str(v))
                writer.writerow(row)
        print(f"Saved trials CSV to: {csv_trials_path}")
    else:
        print(f"tuner_trials.json not found at: {tuner_trials_path}; skipping CSV generation")
except Exception as e:
    print("Warning: failed to generate tuner_trials.csv:", e)