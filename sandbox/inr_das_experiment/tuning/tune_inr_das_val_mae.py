"""Hyperband Hyperparameter Tuning for INR-based DAS using Keras Tuner.

This script is a copy of `tune_inr_das_hyperband.py` but uses `val_mae`
as the Keras Tuner objective (to be minimized).
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
from inr_apodizations.tuning.utils import generate_candidate_architectures
import yaml
import sys

print("TF GPUs:", tf.config.list_physical_devices("GPU"))

# Ensure Keras Tuner runs on a specific GPU device by using a OneDeviceStrategy.
# This places variable creation and model building on the chosen GPU.
strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
print(f"Using distribution strategy: {strategy}")

# Make `helpers.py` importable when running this script from the project root.
# This inserts the parent folder (`sandbox/inr_das_experiment`) into `sys.path`.
script_dir = Path(__file__).resolve().parent
sandbox_pkg_dir = script_dir.parent
if str(sandbox_pkg_dir) not in sys.path:
    sys.path.insert(0, str(sandbox_pkg_dir))

import helpers
from inr_apodizations import config
from inr_apodizations.modeling.trainer import DasInrTrainer, build_mlp_inr
from inr_apodizations.modeling.losses import ScaledLoss
from inr_apodizations.apodizations import compute_dynamic_apodizations_tf
from scatterer_metrics import compute_scatterer_metrics


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
kp, cm = helpers.build_coordinate_manager(str(dataset_folder), 
                                          physical_feature_set=cfg["model"]["physical_feature_set"])
scatterers_all = helpers.load_saved_scatterers(str(dataset_folder))

train_idx, val_idx = helpers.split_train_validation_indices(
    n_examples=delayed.shape[0],
    train_fraction=float(cfg["training"]["train_fraction"]),
    seed=seed
)
# Optionally reduce the number of validation examples (config: training.val_subset_size)
val_subset_size = int(cfg["training"].get("val_subset_size", 1))
if val_subset_size > 0 and val_subset_size < len(val_idx):
    rnd = np.random.RandomState(seed)
    val_idx = list(rnd.choice(val_idx, size=val_subset_size, replace=False))
    val_idx.sort()
    print(f"Validation set reduced to {len(val_idx)} examples (subset_size={val_subset_size}).")

# Prepare validation data and scatterers for the custom metric
val_delayed = delayed[val_idx].astype(np.complex64)
val_scatterers = []
for idx in val_idx:
    s = np.asarray(scatterers_all[int(idx)], dtype=np.float32).copy()
    s[:, :2] *= 1000.0  # m to mm
    val_scatterers.append(s[:, :2])

# --- Optional: Mask weighting (pixelwise sample weights)
mask_weighting_cfg = dict(cfg["training"].get("mask_weighting", {}))
train_loss_weights = helpers.build_gaussian_loss_weights(gaussian_masks, mask_weighting_cfg)
use_pixelwise_weights = train_loss_weights is not None

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
    loss_fn = _pixelwise_mae if use_pixelwise_weights else tf.keras.losses.MeanAbsoluteError()
    trainer.compile(
        jit_compile=False, # Hyperband may not benefit from JIT due to short epochs; set to True if desired.
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr,
            decay=float(cfg["training"]["weight_decay"]),
        ),
        loss=ScaledLoss(loss_fn),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        weighted_metrics=[],
    )
    
    return trainer

# --- Run Tuning ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_name = f"inr_das_tuning_{timestamp}"
tuning_dir = Path(cfg["io"]["sandbox_output_root"]) / project_name

# Hyperband params (from config)
max_epochs = int(cfg["tuning"].get("hyperband_max_epochs", cfg["training"]["epochs"]))
factor = int(cfg["tuning"].get("hyperband_factor", 3))
print(f"Using Hyperband: max_epochs={max_epochs}, factor={factor}")

with strategy.scope():
    tuner = kt.Hyperband(
        build_model,
        objective=kt.Objective("val_mae", direction="min"),
        max_epochs=max_epochs,
        factor=factor,
        directory=str(tuning_dir),
        project_name="kt_hyperband"
    )

# Prepare datasets for fit
# Pass sample_weights when mask weighting is enabled
train_ds = helpers.build_tf_dataset_by_indices(
    delayed, targets, indices=train_idx,
    sample_weights=(train_loss_weights if train_loss_weights is not None else None),
    batch_size=cfg["training"]["batch_size"], shuffle=True, seed=seed
)
# Use the full validation set for Keras `validation_data` (only ~20 examples).
# The SnrImprovementCallback also uses `val_delayed_tf` (precomputed full val set).
val_ds = helpers.build_tf_dataset_by_indices(
    delayed, targets, indices=val_idx, # Use full validation indices
    sample_weights=(train_loss_weights if train_loss_weights is not None else None),
    batch_size=cfg["training"]["batch_size"], shuffle=False, seed=seed
)

print("\nStarting Hyperband Optimization...")
tuner.search(
    train_ds,
    epochs=max_epochs,
    validation_data=val_ds,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", mode="min", patience=cfg["training"]["early_stopping_patience"]) 
    ],
    verbose=True
)

# --- Results & Artifacts ---
print("\nTuning Finished!")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_arch_index = int(best_hps.get("architecture_index"))
best_hidden_units = candidate_architectures[best_arch_index]

print("Best Hyperparameters:")
for key in best_hps.values:
    print(f"  {key}: {best_hps.get(key)}")
print(f"  hidden_units: {best_hidden_units}")

# Save best config as YAML
best_config_path = tuning_dir / "best_config.yml"
best_config = {
    "hidden_units": [int(v) for v in best_hidden_units],
    "reg_lambda": float(best_hps.get("reg_lambda")),
    "reg_tau": float(best_hps.get("reg_tau")),
    "lr": float(best_hps.get("lr")),
}
with open(best_config_path, "w") as f:
    yaml.safe_dump(best_config, f, sort_keys=False)

print(f"Artifacts saved in: {tuning_dir}")