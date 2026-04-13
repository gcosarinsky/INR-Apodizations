"""Optuna hyperparameter tuning for INR-based DAS.

Reads configuration from `configs/tune_config.yml` (no CLI).

Notes:
- Does not use XLA/JIT flags or `jit_compile` to avoid libdevice/XLA issues.
- Uses `cfg["tuning"]["max_trials"]` as `n_trials` for Optuna.
"""
from __future__ import annotations

import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import tensorflow as tf
# Disable XLA/JIT for this script to avoid libdevice/XLA compilation warnings
tf.config.optimizer.set_jit(False)
import optuna

# Make `helpers.py` importable when running from project root
script_dir = Path(__file__).resolve().parent
sandbox_pkg_dir = script_dir.parent
if str(sandbox_pkg_dir) not in sys.path:
    sys.path.insert(0, str(sandbox_pkg_dir))

import helpers
from inr_apodizations import config
from inr_apodizations.modeling.trainer import DasInrTrainer, build_mlp_inr
from inr_apodizations.modeling.losses import ScaledLoss
from inr_apodizations.apodizations import compute_dynamic_apodizations_tf
from inr_apodizations.tuning.utils import generate_candidate_architectures
from sandbox.inr_das_experiment.scatterer_metrics import compute_scatterer_metrics


class OptunaPruningCallback(tf.keras.callbacks.Callback):
    """Callback that computes `val_snr_better_count`, reports to Optuna and prunes.

    The callback keeps the best observed `val_snr_better_count` during training and
    reports it to the `trial` each epoch.
    """

    def __init__(self, trial, val_delayed_tf, val_scatterers, hanning_snrs, cm, cfg):
        super().__init__()
        self.trial = trial
        self.val_delayed_tf = val_delayed_tf
        self.val_scatterers = val_scatterers
        self.hanning_snrs = hanning_snrs
        self.cm = cm
        self.cfg = cfg
        self.best = 0

    def on_epoch_end(self, epoch, logs=None):
        pred_complex, _ = self.model.reconstruct_image(self.val_delayed_tf, training=False)
        pred_abs = np.abs(pred_complex.numpy())
        metrics = compute_scatterer_metrics(pred_abs, self.val_scatterers, self.cm,
                                            radius_mm=self.cfg["scatterer_eval"]["radius_mm"])
        inr_peaks = metrics["aggregated"]["peak_amplitudes"]
        inr_bg_rms = metrics["aggregated"]["point_background_rms"]
        inr_snrs = inr_peaks / (inr_bg_rms + 1e-12)
        snr_ratios = inr_snrs / (self.hanning_snrs + 1e-12)
        better_count = int(np.sum(snr_ratios > 1.0))
        if better_count > self.best:
            self.best = better_count
        # report to Optuna and possibly prune
        self.trial.report(float(better_count), step=epoch)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()


def build_and_compile_trainer(cfg, cm, candidate_architectures, arch_index, reg_lambda, reg_tau, lr):
    hidden_units = candidate_architectures[int(arch_index)]
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
        weight_regularization_lambda=float(reg_lambda),
        weight_regularization_tau=float(reg_tau),
    )

    loss_fn = (lambda y_true, y_pred: tf.abs(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)))
    # Do NOT pass `jit_compile` to avoid XLA/libdevice dependency
    trainer.compile(
        optimizer=tf.keras.optimizers.experimental.AdamW(
            weight_decay=float(cfg["training"]["weight_decay"]),
            learning_rate=float(lr),
        ),
        loss=ScaledLoss(loss_fn),
    )
    return trainer


# --- Load config (no CLI) ---
cfg_path = Path("configs/tune_config.yml")
cfg = helpers.load_experiment_config(str(cfg_path))

seed = int(cfg["training"]["seed"])
tf.keras.utils.set_random_seed(seed)
random.seed(seed)
np.random.seed(seed)

candidate_architectures = generate_candidate_architectures(cfg["tuning"], fallback_seed=seed)
print(f"Generated {len(candidate_architectures)} candidate architectures")

# Data & coordinate manager
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
    seed=seed,
)

val_delayed = delayed[val_idx].astype(np.complex64)
val_scatterers = []
for idx in val_idx:
    s = np.asarray(scatterers_all[int(idx)], dtype=np.float32).copy()
    s[:, :2] *= 1000.0  # m -> mm
    val_scatterers.append(s[:, :2])

mask_weighting_cfg = dict(cfg["training"].get("mask_weighting", {}))
train_loss_weights = helpers.build_gaussian_loss_weights(gaussian_masks, mask_weighting_cfg)

# Hanning baseline
print("Computing Hanning baseline SNR...")
baseline_f = float(cfg["training"]["baseline_f_number"])
apods_h = compute_dynamic_apodizations_tf(cm=cm, f_number=baseline_f, methods=("hanning",),
                                          scaled=bool(cfg["model"]["scaled_features"]))
hanning_weights = apods_h["hanning"]
hanning_weights_b = tf.expand_dims(hanning_weights, axis=0)
val_delayed_tf = tf.convert_to_tensor(val_delayed)
hanning_complex = tf.reduce_sum(val_delayed_tf * tf.cast(hanning_weights_b, val_delayed_tf.dtype), axis=1)
hanning_abs = tf.abs(hanning_complex).numpy()
h_metrics = compute_scatterer_metrics(hanning_abs, val_scatterers, cm, radius_mm=cfg["scatterer_eval"]["radius_mm"])
h_peaks = h_metrics["aggregated"]["peak_amplitudes"]
h_bg_rms = h_metrics["aggregated"]["point_background_rms"]
hanning_snrs = h_peaks / (h_bg_rms + 1e-12)
print(f"Hanning baseline mean SNR: {np.mean(hanning_snrs):.4f}")

# TF datasets
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

# Smoke test: run a single-epoch training outside the Optuna study to verify
# `trainer.fit` works and to surface any TF/CUDA/XLA errors early.
print("Running 1-epoch smoke test (outside Optuna)...")
try:
    smoke_arch_index = 0
    smoke_reg_lambda = float(cfg["tuning"]["reg_lambda_min"])
    smoke_reg_tau = float(cfg["tuning"]["reg_tau_min"])
    smoke_lr = float(cfg["tuning"]["lr_min"])
    smoke_trainer = build_and_compile_trainer(cfg, cm, candidate_architectures,
                                              smoke_arch_index, smoke_reg_lambda,
                                              smoke_reg_tau, smoke_lr)
    smoke_trainer.fit(
        train_ds,
        epochs=1,
        validation_data=val_ds,
        callbacks=[],
        verbose=1,
    )
    print("Smoke test completed successfully.")
except Exception as e:
    print("Smoke test failed with exception:", repr(e))
    raise

# Output folder (use sandbox output root from config)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_name = f"inr_das_optuna_{timestamp}"
output_dir = Path(cfg["io"].get("sandbox_output_root", "sandbox/outputs/optuna_tuning"))
if not output_dir.is_absolute():
    output_dir = config.PROJ_ROOT / output_dir
output_dir = output_dir / project_name
output_dir.mkdir(parents=True, exist_ok=True)

# Optuna study
storage_path = output_dir / "optuna_study.db"
sampler = optuna.samplers.TPESampler(seed=seed)
pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                           study_name=f"inr_das_optuna_{timestamp}",
                           storage=f"sqlite:///{storage_path}", load_if_exists=True)


def objective(trial: optuna.trial.Trial):
    arch_index = trial.suggest_int("architecture_index", 0, len(candidate_architectures) - 1)
    reg_lambda = trial.suggest_float(
        "reg_lambda",
        float(cfg["tuning"]["reg_lambda_min"]),
        float(cfg["tuning"]["reg_lambda_max"]),
        log=True,
    )
    reg_tau = trial.suggest_float(
        "reg_tau",
        float(cfg["tuning"]["reg_tau_min"]),
        float(cfg["tuning"]["reg_tau_max"]),
    )
    lr = trial.suggest_float(
        "lr",
        float(cfg["tuning"]["lr_min"]),
        float(cfg["tuning"]["lr_max"]),
        log=True,
    )

    trainer = build_and_compile_trainer(cfg, cm, candidate_architectures, arch_index, reg_lambda, reg_tau, lr)

    optuna_cb = OptunaPruningCallback(trial, val_delayed_tf, val_scatterers, hanning_snrs, cm, cfg)

    try:
        trainer.fit(
            train_ds,
            epochs=cfg["training"]["epochs"],
            validation_data=val_ds,
            callbacks=[optuna_cb],
            verbose=0,
        )
    except optuna.exceptions.TrialPruned:
        raise

    return float(optuna_cb.best)


# Run optimization
n_trials = int(cfg["tuning"].get("max_trials", 20))
print(f"Starting Optuna optimization for {n_trials} trials...")
study.optimize(objective, n_trials=n_trials, n_jobs=1)

# Save best hyperparameters
best_trial = study.best_trial
best_hps = best_trial.params
best_arch = candidate_architectures[int(best_hps["architecture_index"])]
best_config = {
    "hidden_units": [int(v) for v in best_arch],
    "reg_lambda": float(best_hps.get("reg_lambda")),
    "reg_tau": float(best_hps.get("reg_tau")),
    "lr": float(best_hps.get("lr")),
}
best_config_path = output_dir / "best_config.yml"
with open(best_config_path, "w") as f:
    yaml.safe_dump(best_config, f, sort_keys=False)

# Export trials dataframe if possible
try:
    df = study.trials_dataframe()
    df_path = output_dir / "trials.csv"
    df.to_csv(df_path, index=False)
except Exception:
    pass

print(f"Optuna study finished. Best value: {study.best_value}")
print(f"Best params saved to: {best_config_path}")
