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

import inr_apodizations.experiment_helpers as helpers
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


def _is_number(value) -> bool:
    """Return True for numeric scalar values excluding booleans."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _coerce_float(name: str, value) -> float:
    """Validate and convert numeric scalar value to float."""
    if _is_number(value):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} must be a numeric scalar. Received: {value!r}")
        try:
            return float(stripped)
        except ValueError as exc:
            raise ValueError(f"{name} must be a numeric scalar. Received: {value!r}") from exc
    raise ValueError(f"{name} must be a numeric scalar. Received: {value!r}")


def _validate_sampling(name: str, sampling: str | None, default: str = "linear") -> str:
    """Normalize and validate sampling mode for Keras Tuner Float parameters."""
    sampling_value = default if sampling is None else str(sampling).strip().lower()
    if sampling_value not in ("linear", "log"):
        raise ValueError(f"{name}.sampling must be 'linear' or 'log'. Received: {sampling!r}")
    return sampling_value


def _validate_numeric_bounds(
    name: str,
    value: float,
    *,
    min_allowed: float | None = None,
    strictly_positive: bool = False,
) -> None:
    """Validate scalar numeric constraints."""
    if min_allowed is not None and value < min_allowed:
        raise ValueError(f"{name} must be >= {min_allowed}. Received: {value}.")
    if strictly_positive and value <= 0.0:
        raise ValueError(f"{name} must be > 0. Received: {value}.")


def _resolve_numeric_hparam(
    *,
    name: str,
    cfg_value,
    legacy_fixed=None,
    legacy_min=None,
    legacy_max=None,
    default_sampling: str = "linear",
    min_allowed: float | None = None,
    strictly_positive: bool = False,
) -> dict:
    """Resolve fixed/search numeric hyperparameter with legacy compatibility.

    Accepted formats:
    - Scalar: fixed value
    - Dict with mode=fixed/value or mode=search/min/max
    - Legacy fallback via `legacy_fixed` or `legacy_min`/`legacy_max`
    """
    if isinstance(cfg_value, dict):
        mode = str(cfg_value.get("mode", "")).strip().lower()
        if not mode:
            if "value" in cfg_value:
                mode = "fixed"
            elif "min" in cfg_value and "max" in cfg_value:
                mode = "search"
            else:
                raise ValueError(
                    f"tuning.{name} dict requires mode='fixed'/'search' or compatible keys."
                )

        if mode == "fixed":
            fixed_value = _coerce_float(f"tuning.{name}.value", cfg_value.get("value"))
            _validate_numeric_bounds(
                f"tuning.{name}.value",
                fixed_value,
                min_allowed=min_allowed,
                strictly_positive=strictly_positive,
            )
            return {
                "mode": "fixed",
                "value": fixed_value,
                "sampling": _validate_sampling(f"tuning.{name}", cfg_value.get("sampling"), default_sampling),
                "source": "config",
            }

        if mode == "search":
            min_value = _coerce_float(f"tuning.{name}.min", cfg_value.get("min"))
            max_value = _coerce_float(f"tuning.{name}.max", cfg_value.get("max"))
            if min_value > max_value:
                raise ValueError(
                    f"tuning.{name}.min must be <= tuning.{name}.max. "
                    f"Received: {min_value} > {max_value}."
                )
            _validate_numeric_bounds(
                f"tuning.{name}.min",
                min_value,
                min_allowed=min_allowed,
                strictly_positive=strictly_positive,
            )
            _validate_numeric_bounds(
                f"tuning.{name}.max",
                max_value,
                min_allowed=min_allowed,
                strictly_positive=strictly_positive,
            )
            sampling = _validate_sampling(
                f"tuning.{name}", cfg_value.get("sampling"), default_sampling
            )
            if sampling == "log" and min_value <= 0.0:
                raise ValueError(f"tuning.{name}.min must be > 0 when sampling='log'.")
            return {
                "mode": "search",
                "min": min_value,
                "max": max_value,
                "sampling": sampling,
                "source": "config",
            }

        raise ValueError(f"tuning.{name}.mode must be 'fixed' or 'search'. Received: {mode!r}")

    if _is_number(cfg_value):
        fixed_value = float(cfg_value)
        _validate_numeric_bounds(
            f"tuning.{name}",
            fixed_value,
            min_allowed=min_allowed,
            strictly_positive=strictly_positive,
        )
        return {
            "mode": "fixed",
            "value": fixed_value,
            "sampling": default_sampling,
            "source": "legacy_scalar",
        }

    if legacy_min is not None and legacy_max is not None:
        min_value = _coerce_float(f"legacy.{name}_min", legacy_min)
        max_value = _coerce_float(f"legacy.{name}_max", legacy_max)
        if min_value > max_value:
            raise ValueError(
                f"legacy {name}_min must be <= {name}_max. Received: {min_value} > {max_value}."
            )
        _validate_numeric_bounds(
            f"legacy.{name}_min",
            min_value,
            min_allowed=min_allowed,
            strictly_positive=strictly_positive,
        )
        _validate_numeric_bounds(
            f"legacy.{name}_max",
            max_value,
            min_allowed=min_allowed,
            strictly_positive=strictly_positive,
        )
        if default_sampling == "log" and min_value <= 0.0:
            raise ValueError(f"legacy {name}_min must be > 0 when sampling='log'.")
        return {
            "mode": "search",
            "min": min_value,
            "max": max_value,
            "sampling": default_sampling,
            "source": "legacy_range",
        }

    if legacy_fixed is not None:
        fixed_value = _coerce_float(f"legacy.{name}", legacy_fixed)
        _validate_numeric_bounds(
            f"legacy.{name}",
            fixed_value,
            min_allowed=min_allowed,
            strictly_positive=strictly_positive,
        )
        return {
            "mode": "fixed",
            "value": fixed_value,
            "sampling": default_sampling,
            "source": "legacy_fixed",
        }

    raise ValueError(
        f"tuning.{name} is not configured. Provide fixed/search config or legacy fallback keys."
    )


def _resolve_trial_param_value_from_hp_dict(hp_values: dict, resolved_cfg: dict, hp_key: str) -> float:
    """Resolve trial value from hp dict for searchable params, else use fixed value."""
    if resolved_cfg["mode"] == "search":
        value = hp_values.get(hp_key)
        if value is None:
            raise ValueError(f"Missing hyperparameter '{hp_key}' in trial values.")
        return float(value)
    return float(resolved_cfg["value"])


def _extract_trial_total_epochs_fallback(
    tuning_output_dir: Path,
    trial_id: str,
    objective_name: str,
) -> int | None:
    """Fallback epoch extraction from Keras Tuner trial JSON observations."""
    trial_json = tuning_output_dir / "kt_hyperband" / f"trial_{trial_id}" / "trial.json"
    if not trial_json.exists():
        return None
    try:
        with open(trial_json, "r", encoding="utf-8") as file:
            payload = json.load(file)
        metrics = payload.get("metrics", {})
        objective_data = metrics.get(objective_name, {})
        objective_observations = objective_data.get("observations", [])
        if isinstance(objective_observations, list) and objective_observations:
            return int(len(objective_observations))

        max_observations = 0
        for metric_payload in metrics.values():
            observations = metric_payload.get("observations", [])
            if isinstance(observations, list):
                max_observations = max(max_observations, len(observations))
        return int(max_observations) if max_observations > 0 else None
    except Exception:
        return None

# --- Optional: Mask weighting (pixelwise sample weights)
mask_weighting_cfg = dict(cfg["training"].get("mask_weighting", {}))
use_pixelwise_weights = bool(mask_weighting_cfg.get("enabled", False))
autoinit_cfg = dict(cfg["tuning"].get("weight_regularization_autoinit", {}))
autoinit_enabled = bool(autoinit_cfg.get("enabled", False))
autoinit_ratio = float(autoinit_cfg.get("ratio", 0.5))
autoinit_epsilon = float(autoinit_cfg.get("epsilon", 1e-12))
autoinit_norm_fraction = float(autoinit_cfg.get("norm_fraction", 0.9))

learning_rate_cfg = _resolve_numeric_hparam(
    name="learning_rate",
    cfg_value=cfg["tuning"].get("learning_rate"),
    legacy_min=cfg["tuning"].get("lr_min"),
    legacy_max=cfg["tuning"].get("lr_max"),
    default_sampling="log",
    strictly_positive=True,
)
reg_tau_cfg = _resolve_numeric_hparam(
    name="reg_tau",
    cfg_value=cfg["tuning"].get("reg_tau"),
    legacy_min=cfg["tuning"].get("reg_tau_min"),
    legacy_max=cfg["tuning"].get("reg_tau_max"),
    default_sampling="linear",
    min_allowed=0.0,
)
reg_lambda_cfg = _resolve_numeric_hparam(
    name="reg_lambda",
    cfg_value=cfg["tuning"].get("reg_lambda"),
    default_sampling="log",
    min_allowed=0.0,
)
pixel_weight_lambda_cfg = _resolve_numeric_hparam(
    name="pixel_weight_lambda",
    cfg_value=cfg["tuning"].get("pixel_weight_lambda"),
    legacy_fixed=mask_weighting_cfg.get(
        "pixel_weight_lambda",
        mask_weighting_cfg.get("lambda", 0.0),
    ),
    default_sampling="linear",
    min_allowed=0.0,
)

if autoinit_ratio < 0.0:
    raise ValueError("tuning.weight_regularization_autoinit.ratio must be >= 0")
if autoinit_epsilon <= 0.0:
    raise ValueError("tuning.weight_regularization_autoinit.epsilon must be > 0")
if autoinit_norm_fraction <= 0.0:
    raise ValueError("tuning.weight_regularization_autoinit.norm_fraction must be > 0")

print(
    "Regularization configuration:",
    {
        "autoinit_enabled": autoinit_enabled,
        "reg_lambda_mode": reg_lambda_cfg["mode"],
        "reg_lambda": reg_lambda_cfg,
        "reg_tau": reg_tau_cfg,
        "learning_rate": learning_rate_cfg,
        "ratio": autoinit_ratio,
        "epsilon": autoinit_epsilon,
        "norm_fraction": autoinit_norm_fraction,
    },
)
print(
    "Pixel-weight configuration:",
    {
        "enabled": use_pixelwise_weights,
        "pixel_weight_lambda": pixel_weight_lambda_cfg,
    },
)


def _build_trial_loss_weights(pixel_weight_lambda_value: float) -> np.ndarray | None:
    """Build per-pixel loss weights for the current trial.

    Returns:
        Optional per-example sample-weight tensor.
    """
    if not mask_weighting_cfg.get("enabled", False):
        return None

    local_mask_weighting_cfg = dict(mask_weighting_cfg)
    local_mask_weighting_cfg["pixel_weight_lambda"] = float(pixel_weight_lambda_value)

    return helpers.build_gaussian_loss_weights(gaussian_masks, local_mask_weighting_cfg)

RELATIVE_MAE_NAME = "relative_mae_y_pred"
OBJECTIVE_NAME = f"val_{RELATIVE_MAE_NAME}"


def build_trial_datasets(trial):
    """Build train and validation datasets for the current tuning trial."""
    sample_weights = None
    if use_pixelwise_weights:
        hp_values = dict(getattr(trial.hyperparameters, "values", {}))
        trial_pixel_weight_lambda = _resolve_trial_param_value_from_hp_dict(
            hp_values,
            pixel_weight_lambda_cfg,
            hp_key="pixel_weight_lambda",
        )
        sample_weights = _build_trial_loss_weights(trial_pixel_weight_lambda)

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
    
    # Register searchable pixel weighting as tuner HP, even though the value is
    # consumed in trial dataset building.
    if pixel_weight_lambda_cfg["mode"] == "search":
        hp.Float(
            "pixel_weight_lambda",
            pixel_weight_lambda_cfg["min"],
            pixel_weight_lambda_cfg["max"],
            sampling=pixel_weight_lambda_cfg["sampling"],
        )

    # 2. Physical Trainer with Tunable Regularization
    if reg_lambda_cfg["mode"] == "search":
        reg_lambda = hp.Float(
            "reg_lambda",
            reg_lambda_cfg["min"],
            reg_lambda_cfg["max"],
            sampling=reg_lambda_cfg["sampling"],
        )
    else:
        reg_lambda = float(reg_lambda_cfg["value"])

    if reg_tau_cfg["mode"] == "search":
        reg_tau = hp.Float(
            "reg_tau",
            reg_tau_cfg["min"],
            reg_tau_cfg["max"],
            sampling=reg_tau_cfg["sampling"],
        )
    else:
        reg_tau = float(reg_tau_cfg["value"])

    # If auto-init is enabled, `reg_lambda` is treated as trial initial value and
    # then overwritten before the first optimizer step.

    trainer = DasInrTrainer(
        apodization_model=inr_mlp,
        features_grid=cm.get_features_grid(scaled=cfg["model"]["scaled_features"]),
        feature_chunk_size=cfg["model"]["feature_chunk_size"],
        weight_regularization_enabled=True,
        weight_regularization_lambda=reg_lambda,
        weight_regularization_tau=reg_tau,
    )
    
    # 3. Compilation
    if learning_rate_cfg["mode"] == "search":
        lr = hp.Float(
            "lr",
            learning_rate_cfg["min"],
            learning_rate_cfg["max"],
            sampling=learning_rate_cfg["sampling"],
        )
    else:
        lr = float(learning_rate_cfg["value"])

    loss_obj = PixelWeightedMAELoss(name="pixel_weighted_mae_loss")
    metrics = []
    weighted_metrics = [
        PixelWeightedMAE(name="pixel_weighted_mae"),
        RelativeMAE(name="relative_mae_y_pred", normalize_by="y_pred"),
        RelativeMAE(name="relative_mae_y_true", normalize_by="y_true"),
    ]

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
tuning_output_root = Path(
    cfg["io"].get(
        "scripts_output_root",
        cfg["io"].get("sandbox_output_root", "scripts/outputs/tuning"),
    )
)
if not tuning_output_root.is_absolute():
    tuning_output_root = config.PROJ_ROOT / tuning_output_root
tuning_dir = tuning_output_root / project_name
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
        weight_regularization_autoinit_enabled=autoinit_enabled,
        weight_regularization_autoinit_ratio=autoinit_ratio,
        weight_regularization_autoinit_epsilon=autoinit_epsilon,
        weight_regularization_autoinit_norm_fraction=autoinit_norm_fraction,
        autoinit_log_path=tuning_dir / "trial_autoinit_log.json",
        trial_epoch_log_path=tuning_dir / "trial_epoch_log.json",
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

trial_autoinit_log = dict(getattr(tuner, "trial_autoinit_log", {}))
if trial_autoinit_log:
    autoinit_log_path = tuning_dir / "trial_autoinit_log.json"
    with open(autoinit_log_path, "w", encoding="utf-8") as file:
        json.dump(trial_autoinit_log, file, indent=2)
    print(f"Saved trial auto-init diagnostics to: {autoinit_log_path}")

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
best_trial_id = str(best_trial.trial_id)
best_arch_index = int(best_hps.get("architecture_index"))
best_hidden_units = candidate_architectures[best_arch_index]
best_trial_autoinit = trial_autoinit_log.get(best_trial_id, {})
best_hp_values = dict(best_hps.values)
best_lr = _resolve_trial_param_value_from_hp_dict(best_hp_values, learning_rate_cfg, hp_key="lr")
best_reg_tau = _resolve_trial_param_value_from_hp_dict(best_hp_values, reg_tau_cfg, hp_key="reg_tau")
best_reg_lambda_candidate = _resolve_trial_param_value_from_hp_dict(
    best_hp_values,
    reg_lambda_cfg,
    hp_key="reg_lambda",
)
best_pixel_weight_lambda = _resolve_trial_param_value_from_hp_dict(
    best_hp_values,
    pixel_weight_lambda_cfg,
    hp_key="pixel_weight_lambda",
)

resolved_best_reg_lambda = (
    float(best_trial_autoinit.get("lambda_applied"))
    if best_trial_autoinit.get("lambda_applied") is not None
    else best_reg_lambda_candidate
)
best_reg_lambda_source = "autoinit" if best_trial_autoinit.get("lambda_applied") is not None else reg_lambda_cfg["mode"]

print("Best Hyperparameters:")
for key in best_hps.values:
    print(f"  {key}: {best_hps.get(key)}")
print(f"  hidden_units: {best_hidden_units}")
print(f"  pixel_weight_lambda: {best_pixel_weight_lambda}")
print(f"  lr: {best_lr}")
print(f"  reg_tau: {best_reg_tau}")
if best_trial_autoinit:
    print(
        "  reg_lambda_auto:",
        best_trial_autoinit.get("lambda_applied"),
    )
else:
    print(f"  reg_lambda: {resolved_best_reg_lambda}")

# Save best config as YAML.
best_config_path = tuning_dir / "best_config.yml"
best_config = {
    "hidden_units": [int(v) for v in best_hidden_units],
    "reg_lambda": resolved_best_reg_lambda,
    "reg_lambda_auto": (
        float(best_trial_autoinit.get("lambda_applied"))
        if best_trial_autoinit.get("lambda_applied") is not None
        else None
    ),
    "reg_lambda_source": best_reg_lambda_source,
    "reg_tau": best_reg_tau,
    "lr": best_lr,
    "trial_id": best_trial_id,
    "weight_regularization_autoinit": {
        "enabled": autoinit_enabled,
        "ratio": autoinit_ratio,
        "epsilon": autoinit_epsilon,
        "norm_fraction": autoinit_norm_fraction,
        "best_trial": best_trial_autoinit,
    },
    # Store fixed weights even though they are no longer tuner hyperparameters.
    "architecture_index": int(best_hps.get("architecture_index")) if best_hps.get("architecture_index") is not None else None,
    "pixel_weight_lambda": best_pixel_weight_lambda if use_pixelwise_weights else None,
    "learning_rate_mode": learning_rate_cfg["mode"],
    "reg_tau_mode": reg_tau_cfg["mode"],
    "reg_lambda_mode": reg_lambda_cfg["mode"],
    "pixel_weight_lambda_mode": pixel_weight_lambda_cfg["mode"],
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

        trial_autoinit_lookup = dict(getattr(tuner, "trial_autoinit_log", {}))

        # Discover all hyperparameter keys across trials
        hp_keys = set()
        for t in trials:
            hp = t.get("hyperparameters") or {}
            if isinstance(hp, dict):
                hp_keys.update(hp.keys())
        hp_keys.discard("architecture_index")
        hp_keys = sorted(hp_keys)

        trial_epoch_lookup = dict(getattr(tuner, "trial_epoch_log", {}))

        # CSV header: basic trial fields + discovered hyperparameters
        header = [
            "trial_id",
            "architecture_index",
            "score",
            "epochs_trained_total",
            "epochs_source",
            "hidden_units",
            "reg_lambda",
            "reg_lambda_auto",
            "reg_lambda_source",
            "autoinit_status",
            "autoinit_hinge_active",
            "autoinit_fallback_used",
            "pixel_weight_lambda",
        ] + hp_keys

        with open(csv_trials_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for t in trials:
                row = []
                trial_id = str(t.get("trial_id", ""))
                row.append(trial_id)
                ai = t.get("architecture_index")
                row.append("" if ai is None else ai)
                row.append(t.get("score", ""))

                trial_epochs_callback = trial_epoch_lookup.get(trial_id)
                trial_epochs_fallback = _extract_trial_total_epochs_fallback(
                    tuning_output_dir=tuning_dir,
                    trial_id=trial_id,
                    objective_name=OBJECTIVE_NAME,
                )
                if trial_epochs_callback is not None:
                    row.append(int(trial_epochs_callback))
                    row.append("callback")
                elif trial_epochs_fallback is not None:
                    row.append(int(trial_epochs_fallback))
                    row.append("trial_json")
                else:
                    row.append("")
                    row.append("missing")

                # hidden_units as JSON string (keeps list structure)
                row.append(json.dumps(t.get("hidden_units", None)))
                autoinit_info = trial_autoinit_lookup.get(trial_id, {})
                hp = t.get("hyperparameters") or {}

                trial_reg_lambda_candidate = _resolve_trial_param_value_from_hp_dict(
                    hp,
                    reg_lambda_cfg,
                    hp_key="reg_lambda",
                )
                trial_reg_lambda = autoinit_info.get("lambda_applied", trial_reg_lambda_candidate)
                row.append(trial_reg_lambda)
                row.append(autoinit_info.get("lambda_applied", ""))
                row.append(
                    "autoinit"
                    if autoinit_info.get("lambda_applied") is not None
                    else reg_lambda_cfg["mode"]
                )
                row.append(autoinit_info.get("status", ""))
                row.append(autoinit_info.get("hinge_active", ""))
                row.append(autoinit_info.get("fallback_used", ""))
                trial_pixel_weight_lambda = _resolve_trial_param_value_from_hp_dict(
                    hp,
                    pixel_weight_lambda_cfg,
                    hp_key="pixel_weight_lambda",
                )
                row.append(trial_pixel_weight_lambda if use_pixelwise_weights else "")
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