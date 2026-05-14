"""Training script for INR-based DAS apodization using DasInrApodMixer.

This script adapts the single-apodization pipeline from ``train_inr_das_by_indices.py``
to the multi-apodization mixer model (``DasInrApodMixer``). The INR outputs
``n_apodizations`` independent weight channels; each channel produces a separate
DAS magnitude image, and these images are combined pixel-wise by a trainable
linear layer followed by ReLU (the ``pixel_combiner`` inside the model).

Configuration lives in ``configs/train_mixer_config.yml``. The key addition
relative to ``train_config.yml`` is ``model.n_apodizations``.
"""
from __future__ import annotations

import json
import csv
import os
import pprint
import random
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

from inr_apodizations import config
import inr_apodizations.experiment_helpers as helpers

import numpy as np
import tensorflow as tf

from inr_apodizations.evaluation import (
    extract_scatterer_snr,
    find_latest_baseline_reference,
    load_validation_scatterers,
)
from inr_apodizations.modeling.losses import PixelWeightedMAELoss
from inr_apodizations.modeling.das_models import DasInrApodMixer, build_mlp_inr
from inr_apodizations.modeling.metrics import PixelWeightedMAE, RelativeMAE
from inr_apodizations.apodizations import compute_dynamic_apodizations_tf
from inr_apodizations.utils import relative_mae

import matplotlib.pyplot as plt


# ============================================================================
# 1) Configuration, reproducibility, and dataset loading
# ============================================================================

CONFIG_PATH = config.CONFIGS_DIR / "train_mixer_config.yml"
cfg = helpers.load_experiment_config(str(CONFIG_PATH))
seed = int(cfg["training"]["seed"])
tf.keras.utils.set_random_seed(seed)
tf.config.experimental.enable_op_determinism()
random.seed(seed)
np.random.seed(seed)

# Resolve dataset folder relative to project root when given as a relative path
dataset_folder = Path(cfg["io"]["dataset_folder"])
if not dataset_folder.is_absolute():
    dataset_folder = config.PROJ_ROOT / dataset_folder
dataset_folder = str(dataset_folder)
sigma_x_override, sigma_z_override, alpha_override = helpers.get_target_regeneration_override(cfg)
eval_noise_cfg = dict(cfg.get("eval_noise", {}))
eval_noise_enabled = bool(eval_noise_cfg.get("enabled", False))
print("Loading dataset from:", dataset_folder)
delayed, noise, targets, gaussian_masks, info = helpers.load_delayed_samples_dataset(
    dataset_folder,
    sigma_x=sigma_x_override,
    sigma_z=sigma_z_override,
    alpha_override=alpha_override,
    load_noise=eval_noise_enabled,
)

# Log noise provenance when present
if info.get("precomputed_noise_source", {}):
    pinfo = info.get("precomputed_noise_source", {})
    print("Dataset contains precomputed noise information:")
    print(f"  source={pinfo.get('source')}")
helpers.validate_dataset_shapes(delayed, targets, gaussian_masks)
physical_feature_set = str(
    cfg.get("model", {}).get("physical_feature_set", "distance_depth_edge")
)
kp, cm = helpers.build_coordinate_manager(
    dataset_folder,
    physical_feature_set=physical_feature_set,
)

# ============================================================================
# 2) Dataset slicing, weighting, and memory diagnostics
# ============================================================================

delayed_dataset_bytes = int(delayed.nbytes)
delayed_example_bytes = int(np.prod(delayed.shape[1:], dtype=np.int64) * delayed.dtype.itemsize)

max_examples = cfg["training"].get("max_examples")
if max_examples is not None:
    max_examples = int(max_examples)
    delayed = delayed[:max_examples]
    targets = targets[:max_examples]
    gaussian_masks = gaussian_masks[:max_examples]
    if noise is not None:
        noise = noise[:max_examples]

mask_weighting_cfg = dict(cfg["training"].get("mask_weighting", {}))
pixel_weight_lambda = float(
    mask_weighting_cfg.get(
        "pixel_weight_lambda",
        mask_weighting_cfg.get("lambda", 0.0),
    )
)
if pixel_weight_lambda < 0.0:
    raise ValueError(
        "training.mask_weighting.pixel_weight_lambda must be >= 0 "
        "(legacy key training.mask_weighting.lambda is also accepted)"
    )
train_loss_weights = 1.0 + pixel_weight_lambda * gaussian_masks.astype(np.float32, copy=False)
train_loss_weights = train_loss_weights.astype(np.float32, copy=False)
use_pixelwise_weights = True

train_idx, val_idx = helpers.split_train_validation_indices(
    n_examples=delayed.shape[0],
    train_fraction=float(cfg["training"]["train_fraction"]),
    seed=int(cfg["training"]["seed"]),
)

configured_batch_size = int(cfg["training"]["batch_size"])
effective_batch_examples = min(configured_batch_size, int(train_idx.shape[0]))
configured_batch_bytes = delayed_example_bytes * configured_batch_size
effective_batch_bytes = delayed_example_bytes * effective_batch_examples

gpu_mem_info = helpers.gpu_mem()
if isinstance(gpu_mem_info, list) and gpu_mem_info:
    gpu_vram_source = "nvidia-smi"
    gpu_total_vram_bytes = int(gpu_mem_info[0]["total"] * 1024 * 1024)
    gpu_free_vram_bytes = int(gpu_mem_info[0]["free"] * 1024 * 1024)
    gpu_used_vram_bytes = gpu_total_vram_bytes - gpu_free_vram_bytes
    half_free_vram_bytes = int(0.5 * gpu_free_vram_bytes)
    batch_exceeds_half_free_vram = configured_batch_bytes > half_free_vram_bytes
else:
    gpu_vram_source = "unavailable"
    gpu_total_vram_bytes = None
    gpu_free_vram_bytes = None
    gpu_used_vram_bytes = None
    half_free_vram_bytes = None
    batch_exceeds_half_free_vram = None

print("Dataset loaded. Shapes:")
pprint.pprint(
    {
        "delayed.shape": delayed.shape,
        "targets.shape": targets.shape,
        "n_train_examples": int(train_idx.shape[0]),
        "n_val_examples": int(val_idx.shape[0]),
        "n_elements": kp.n_elements,
        "nz": kp.nz,
        "nx": kp.nx,
    }
)
print("Memory diagnostics:")
print(f"  delayed_samples_dataset (full): {helpers.bytes_to_gb(delayed_dataset_bytes):.3f} GiB")
print(
    "  delayed_samples_dataset (one configured batch): "
    f"{helpers.bytes_to_gb(configured_batch_bytes):.3f} GiB "
    f"(batch_size={configured_batch_size})"
)
print(
    "  delayed_samples_dataset (one effective train batch): "
    f"{helpers.bytes_to_gb(effective_batch_bytes):.3f} GiB "
    f"(examples={effective_batch_examples})"
)
if gpu_free_vram_bytes is None:
    print("  GPU VRAM check: unavailable (could not query nvidia-smi free VRAM).")
else:
    print(
        "  GPU VRAM (GPU:0): "
        f"total={helpers.bytes_to_gb(gpu_total_vram_bytes):.3f} GiB; "
        f"free={helpers.bytes_to_gb(gpu_free_vram_bytes):.3f} GiB; "
        f"used={helpers.bytes_to_gb(gpu_used_vram_bytes):.3f} GiB; "
        f"source: {gpu_vram_source}"
    )
    print(
        "  Batch > 50% free VRAM: "
        f"{'YES' if batch_exceeds_half_free_vram else 'NO'} "
        f"(50% free threshold={helpers.bytes_to_gb(half_free_vram_bytes):.3f} GiB; "
        f"configured batch uses {helpers.bytes_to_gb(configured_batch_bytes):.3f} GiB)"
    )


# ============================================================================
# 3) Build tf.data pipelines and evaluation-noise policy
# ============================================================================

train_ds = helpers.build_tf_dataset_by_indices(
    delayed,
    targets,
    indices=train_idx,
    sample_weights=train_loss_weights,
    batch_size=int(cfg["training"]["batch_size"]),
    shuffle=True,
    seed=int(cfg["training"]["seed"]),
)
val_ds = helpers.build_tf_dataset_by_indices(
    delayed,
    targets,
    indices=val_idx,
    sample_weights=train_loss_weights,
    batch_size=int(cfg["training"]["batch_size"]),
    shuffle=False,
    seed=int(cfg["training"]["seed"]),
)

# Evaluation-time noise configuration (applies only to validation/eval)
eval_noise_cfg = dict(cfg.get("eval_noise", {}))
eval_noise_enabled = bool(eval_noise_cfg.get("enabled", False))
eval_noise_scale = float(eval_noise_cfg.get("scale", 1.0)) if eval_noise_enabled else 1.0

if eval_noise_enabled:
    # Compute max_abs over validation subset just for information (scale is explicit)
    try:
        val_max_abs = float(np.max(np.abs(delayed[val_idx])))
    except Exception:
        val_max_abs = float(np.max(np.abs(delayed)))
    print(f"Eval noise enabled: scale={eval_noise_scale}, val_max_abs={val_max_abs:.6g}")
# Note: Do NOT modify `val_ds` used for training validation. Evaluation-time
# noise will be applied only during post-training reconstruction/plotting.


# ============================================================================
# 4) Model and trainer configuration
# ============================================================================

features_grid = cm.get_features_grid(scaled=bool(cfg["model"]["scaled_features"]))
output_activation = cfg["model"].get("output_activation", "sigmoid")
hidden_units_raw = cfg["model"].get("hidden_units")
n_hidden_layers_raw = cfg["model"].get("n_hidden_layers")
n_apodizations = int(cfg["model"].get("n_apodizations", 1))
if n_apodizations <= 0:
    raise ValueError("model.n_apodizations must be > 0")
apodization_model = build_mlp_inr(
    input_dim=cm.n_physical_features,
    hidden_units_config=hidden_units_raw,
    n_hidden_layers=int(n_hidden_layers_raw) if isinstance(hidden_units_raw, (int, float)) else None,
    activation=cfg["model"]["activation"],
    output_activation=output_activation,
    n_apodizations=n_apodizations,
)
weight_reg_cfg = dict(cfg["training"].get("weight_regularization", {}))
weight_reg_enabled = bool(weight_reg_cfg.get("enabled", False))
weight_reg_type = str(weight_reg_cfg.get("type", "hinge_low_norm")).strip().lower()
if weight_reg_enabled and weight_reg_type not in ("hinge_low_norm", "hinge"):
    raise ValueError(
        "training.weight_regularization.type must be 'hinge_low_norm' or 'hinge'"
    )

weight_reg_lambda = float(weight_reg_cfg.get("lambda", 1e-3))
weight_reg_tau = float(weight_reg_cfg.get("tau", 0.30))
weight_reg_epsilon = float(weight_reg_cfg.get("epsilon", 1e-8))
weight_reg_normalize = bool(weight_reg_cfg.get("normalize_norm", True))
weight_reg_auto_cfg = dict(weight_reg_cfg.get("auto_init", {}))
weight_reg_auto_enabled = bool(weight_reg_auto_cfg.get("enabled", False))
weight_reg_auto_ratio = float(weight_reg_auto_cfg.get("ratio", 0.5))
weight_reg_auto_eps = float(weight_reg_auto_cfg.get("epsilon", 1e-12))
weight_reg_auto_norm_fraction = float(weight_reg_auto_cfg.get("norm_fraction", 0.5))

if weight_reg_lambda < 0.0:
    raise ValueError("training.weight_regularization.lambda must be >= 0")
if weight_reg_tau < 0.0:
    raise ValueError("training.weight_regularization.tau must be >= 0")
if weight_reg_epsilon <= 0.0:
    raise ValueError("training.weight_regularization.epsilon must be > 0")
if weight_reg_auto_ratio < 0.0:
    raise ValueError("training.weight_regularization.auto_init.ratio must be >= 0")
if weight_reg_auto_eps <= 0.0:
    raise ValueError("training.weight_regularization.auto_init.epsilon must be > 0")
if weight_reg_auto_norm_fraction <= 0.0:
    raise ValueError("training.weight_regularization.auto_init.norm_fraction must be > 0")

resolved_weight_reg_type = "hinge_low_norm" if weight_reg_type == "hinge" else weight_reg_type

trainer = DasInrApodMixer(
    apodization_model=apodization_model,
    features_grid=features_grid,
    feature_chunk_size=int(cfg["model"]["feature_chunk_size"]),
    n_apodizations=n_apodizations,
    weight_regularization_enabled=weight_reg_enabled,
    weight_regularization_lambda=weight_reg_lambda,
    weight_regularization_tau=weight_reg_tau,
    weight_regularization_epsilon=weight_reg_epsilon,
    weight_regularization_normalize=weight_reg_normalize,
)
print("DasInrApodMixer configuration:")
print(
    {
        "n_apodizations": n_apodizations,
        "weight_regularization": {
            "enabled": weight_reg_enabled,
            "type": resolved_weight_reg_type,
            "lambda": weight_reg_lambda,
            "tau": weight_reg_tau,
            "epsilon": weight_reg_epsilon,
            "normalize_norm": weight_reg_normalize,
            "auto_init": {
                "enabled": weight_reg_auto_enabled,
                "ratio": weight_reg_auto_ratio,
                "epsilon": weight_reg_auto_eps,
                "norm_fraction": weight_reg_auto_norm_fraction,
            },
        },
    }
)
weight_decay = float(cfg["training"].get("weight_decay", 0.0))
loss_obj = PixelWeightedMAELoss(name="pixel_weighted_mae_loss")
metrics_list = []
weighted_metrics_list = [
    PixelWeightedMAE(name="pixel_weighted_mae"),
    RelativeMAE(name="relative_mae_y_pred", normalize_by="y_pred"),
    RelativeMAE(name="relative_mae_y_true", normalize_by="y_true"),
]

trainer.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=float(cfg["training"]["learning_rate"]),
        decay=weight_decay,
    ),
    loss=loss_obj,
    metrics=metrics_list,
    weighted_metrics=weighted_metrics_list,
)


# ============================================================================
# 5) Pre-fit reference sample and optional regularization auto-init
# ============================================================================

# Keep a deterministic baseline prediction from random INR initialization.
sample_delayed = tf.convert_to_tensor(delayed[val_idx[:1]].astype(np.complex64, copy=False))
sample_target = np.expand_dims(targets[val_idx[0]].astype(np.float32, copy=False), axis=0)
predicted_before_image, weights_before_grid = trainer.reconstruct_image(sample_delayed, training=False)
# weights_before_grid has shape (E, Z, X, N) for DasInrApodMixer

mae_initial = None
norm_reference_auto = None
reg_loss_reference_auto = None
if weight_reg_enabled and weight_reg_auto_enabled:
    first_batch = next(iter(train_ds.take(1)))
    if not isinstance(first_batch, (tuple, list)) or len(first_batch) != 3:
        raise ValueError(
            "Auto-init of training.weight_regularization.lambda requires "
            "dataset batches as (delayed, target, sample_weight)."
        )

    x_init, y_init, sample_weight_init = first_batch
    y_pred_init, weights_grid_init = trainer.reconstruct_image(x_init, training=False)
    mae_initial = float(
        loss_obj(y_init, y_pred_init, sample_weight=sample_weight_init).numpy()
    )

    lambda_prev = float(trainer.weight_regularization_lambda)
    norm_reference_auto = float(weight_reg_auto_norm_fraction * weight_reg_tau)
    violation_reference = max(0.0, weight_reg_tau - norm_reference_auto)
    reg_loss_reference_auto = float(violation_reference * violation_reference)

    weight_reg_lambda = float(
        weight_reg_auto_ratio * mae_initial / max(reg_loss_reference_auto, weight_reg_auto_eps)
    )
    trainer.weight_regularization_lambda = float(weight_reg_lambda)

    print("Auto-initialized weight regularization lambda:")
    print(
        {
            "method": "tau_reference",
            "ratio": weight_reg_auto_ratio,
            "norm_fraction": weight_reg_auto_norm_fraction,
            "tau": weight_reg_tau,
            "mae_initial": mae_initial,
            "norm_reference": norm_reference_auto,
            "reg_loss_reference": reg_loss_reference_auto,
            "lambda_previous_config": lambda_prev,
            "lambda_applied": weight_reg_lambda,
        }
    )

print(f"Final weight regularization lambda used for training: {weight_reg_lambda:.6g}")


# ============================================================================
# 6) Baseline apodization weights
# ============================================================================

# Baseline f-number resolution (used in post-training eval and plotting)
_cfg_f_number = cfg["training"].get("baseline_f_number", None)
if _cfg_f_number is None:
    baseline_f_number = kp.f_number
else:
    try:
        baseline_f_number = float(_cfg_f_number)
    except Exception:
        baseline_f_number = kp.f_number

# Apodization weights for baselines
apods_h = compute_dynamic_apodizations_tf(
    cm=cm, f_number=baseline_f_number, methods=("hanning",), scaled=bool(cfg["model"].get("scaled_features", False))
)
apods_b = compute_dynamic_apodizations_tf(
    cm=cm, f_number=baseline_f_number, methods=("boxcar",), scaled=bool(cfg["model"].get("scaled_features", False))
)
hanning_weights = apods_h.get("hanning")
boxcar_weights = apods_b.get("boxcar")
hanning_weights_np = hanning_weights.numpy() if hanning_weights is not None else None
boxcar_weights_np = boxcar_weights.numpy() if boxcar_weights is not None else None
if hanning_weights_np is None or boxcar_weights_np is None:
    raise ValueError(
        "Reference apodizations are required but were not computed for "
        f"f_number={baseline_f_number}."
    )


# ============================================================================
# 7) Output folders, callbacks, and model training
# ============================================================================

# Resolve scripts output root and create a timestamped outputs folder.
scripts_output_root_cfg = Path(
    cfg["io"].get(
        "scripts_output_root",
        cfg["io"].get("sandbox_output_root", "scripts/outputs/train_mixer"),
    )
)
if not scripts_output_root_cfg.is_absolute():
    sandbox_root = config.PROJ_ROOT / scripts_output_root_cfg
else:
    sandbox_root = scripts_output_root_cfg

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sandbox_dir = str(Path(sandbox_root) / timestamp)
os.makedirs(sandbox_dir, exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=int(cfg["training"]["early_stopping_patience"]),
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        patience=int(cfg["training"]["reduce_lr_patience"]),
        factor=0.5,
    ),
]

print("Starting physical-forward training (DasInrApodMixer)...")
history = trainer.fit(
    train_ds,
    validation_data=val_ds,
    epochs=int(cfg["training"]["epochs"]),
    callbacks=callbacks,
    verbose=1,
)


# ============================================================================
# 8) Post-training reconstructions for plotting (clean and noisy eval paths)
# ============================================================================

# weights_after_grid has shape (E, Z, X, N) for DasInrApodMixer.
predicted_after_image, weights_after_grid = trainer.reconstruct_image(sample_delayed, training=False)
sample_delayed_np = sample_delayed.numpy()
uniform_image = tf.convert_to_tensor(
    helpers.compute_das_baseline_numpy(sample_delayed_np),
    dtype=tf.float32,
)
hanning_image = tf.convert_to_tensor(
    helpers.compute_das_baseline_numpy(
        delayed_samples=sample_delayed_np,
        apodization=hanning_weights_np,
    ),
    dtype=tf.float32,
)
boxcar_image = tf.convert_to_tensor(
    helpers.compute_das_baseline_numpy(
        delayed_samples=sample_delayed_np,
        apodization=boxcar_weights_np,
    ),
    dtype=tf.float32,
)

# If eval-time noise is requested, build noisy samples using precomputed noise.
# Note: for DasInrApodMixer, the pre-training "before" reconstruction on a noisy
# sample cannot be reproduced manually (it requires the pixel_combiner layer).
# The clean predicted_before_image is used as the "before" reference in that case.
if eval_noise_enabled:
    if noise is None:
        raise ValueError(
            "Eval noise requested but dataset does not contain precomputed noise. "
            "Provide delayed_samples_noise.npy or delayed_samples_combined.npy (and a signal file when needed)."
        )
    sample_signal_np = sample_delayed.numpy()
    sample_noise_np = noise[val_idx[:1]] if noise is not None else np.zeros_like(sample_signal_np)
    noisy_sample_np = sample_signal_np + eval_noise_scale * sample_noise_np
    noisy_sample = tf.convert_to_tensor(noisy_sample_np.astype(np.complex64, copy=False))

    # Compute post-training image using trained INR on noisy sample
    predicted_after_image_noisy, weights_after_grid_noisy = trainer.reconstruct_image(noisy_sample, training=False)
    uniform_image_noisy = tf.convert_to_tensor(
        helpers.compute_das_baseline_numpy(noisy_sample_np),
        dtype=tf.float32,
    )
    hanning_image_noisy = tf.convert_to_tensor(
        helpers.compute_das_baseline_numpy(
            delayed_samples=noisy_sample_np,
            apodization=hanning_weights_np,
        ),
        dtype=tf.float32,
    )
    boxcar_image_noisy = tf.convert_to_tensor(
        helpers.compute_das_baseline_numpy(
            delayed_samples=noisy_sample_np,
            apodization=boxcar_weights_np,
        ),
        dtype=tf.float32,
    )

    # Use clean pre-training prediction as "before" reference (mixer lacks manual reconstruction)
    uniform_for_plot = uniform_image_noisy
    inr_before_for_plot = predicted_before_image
    inr_after_for_plot = predicted_after_image_noisy
    hanning_for_plot = hanning_image_noisy
    boxcar_for_plot = boxcar_image_noisy
else:
    uniform_for_plot = uniform_image
    inr_before_for_plot = predicted_before_image
    inr_after_for_plot = predicted_after_image
    hanning_for_plot = hanning_image
    boxcar_for_plot = boxcar_image


# ============================================================================
# 9) Persist artifacts and generate main figures
# ============================================================================

effective_cfg = {
    "config_path": str(CONFIG_PATH),
    "dataset_folder": dataset_folder,
    "run_timestamp": timestamp,
    "beamforming": {
        "n_elements": kp.n_elements,
        "nz": kp.nz,
        "nx": kp.nx,
        "roi_effective": list(kp.roi_effective),
        "baseline_f_number_used": float(baseline_f_number),
    },
    "experiment": cfg,
    "resolved_mixer": {
        "n_apodizations": n_apodizations,
    },
    "resolved_weight_regularization": {
        "enabled": weight_reg_enabled,
        "type": resolved_weight_reg_type,
        "lambda": weight_reg_lambda,
        "tau": weight_reg_tau,
        "epsilon": weight_reg_epsilon,
        "normalize_norm": weight_reg_normalize,
        "auto_init": {
            "enabled": weight_reg_auto_enabled,
            "ratio": weight_reg_auto_ratio,
            "epsilon": weight_reg_auto_eps,
            "norm_fraction": weight_reg_auto_norm_fraction,
            "mae_initial": mae_initial,
            "norm_reference": norm_reference_auto,
            "reg_loss_reference": reg_loss_reference_auto,
        },
    },
}
helpers.save_artifacts(sandbox_dir, apodization_model, history.history, effective_cfg)

plot_cfg = cfg.get("plots", {})
normalize_each_image = bool(plot_cfg.get("normalize_each_image", False))
history_dir = Path(sandbox_dir) / "history"
history_dir.mkdir(parents=True, exist_ok=True)

helpers.plot_das_comparison_db(
    uniform_image=uniform_for_plot.numpy()[0],
    inr_before_image=inr_before_for_plot.numpy()[0],
    inr_after_image=inr_after_for_plot.numpy()[0],
    target_image=sample_target[0],
    output_path=str(Path(sandbox_dir) / "das_images_comparison_db.png"),
    extent=kp.get_imshow_extent(),
    cmap=str(plot_cfg.get("cmap", "gray")),
    vmin_db=float(plot_cfg.get("vmin_db", -60.0)),
    vmax_db=float(plot_cfg.get("vmax_db", 0.0)),
    normalize_each_image=normalize_each_image,
)
helpers.plot_das_comparison_db(
    uniform_image=hanning_for_plot.numpy()[0],
    inr_before_image=inr_before_for_plot.numpy()[0],
    inr_after_image=inr_after_for_plot.numpy()[0],
    target_image=sample_target[0],
    output_path=str(Path(sandbox_dir) / "das_images_comparison_db_hanning.png"),
    extent=kp.get_imshow_extent(),
    cmap=str(plot_cfg.get("cmap", "gray")),
    vmin_db=float(plot_cfg.get("vmin_db", -60.0)),
    vmax_db=float(plot_cfg.get("vmax_db", 0.0)),
    normalize_each_image=normalize_each_image,
    baseline_name="Hanning",
)
helpers.plot_das_comparison_db(
    uniform_image=boxcar_for_plot.numpy()[0],
    inr_before_image=inr_before_for_plot.numpy()[0],
    inr_after_image=inr_after_for_plot.numpy()[0],
    target_image=sample_target[0],
    output_path=str(Path(sandbox_dir) / "das_images_comparison_db_boxcar.png"),
    extent=kp.get_imshow_extent(),
    cmap=str(plot_cfg.get("cmap", "gray")),
    vmin_db=float(plot_cfg.get("vmin_db", -60.0)),
    vmax_db=float(plot_cfg.get("vmax_db", 0.0)),
    normalize_each_image=normalize_each_image,
    baseline_name="Boxcar",
)

# Energy comparison and before/after apodization plots: one figure per channel.
# weights_after_grid shape: (E, Z, X, N); weights_before_grid shape: (E, Z, X, N).
weights_after_grid_np = weights_after_grid.numpy()
weights_before_grid_np = weights_before_grid.numpy()
for ch_idx in range(n_apodizations):
    apod_after_ch = weights_after_grid_np[..., ch_idx]   # (E, Z, X)
    apod_before_ch = weights_before_grid_np[..., ch_idx]  # (E, Z, X)
    helpers.plot_apodization_energy_comparison(
        hanning_apod=hanning_weights_np,
        inr_apod_after=apod_after_ch,
        output_path=str(
            Path(sandbox_dir) / f"apodization_energy_comparison_hanning_vs_inr_ch{ch_idx}.png"
        ),
        extent=kp.get_imshow_extent(),
        cmap=str(plot_cfg.get("apod_cmap", "viridis")),
    )

# Save one apodization figure per selected x value, per channel.
x_values_cfg = plot_cfg.get("x_values_apod", None)
if x_values_cfg is None:
    raise ValueError("plots.x_values_apod must be provided in configs/train_mixer_config.yml")
if isinstance(x_values_cfg, (int, float)):
    x_values_apod = [float(x_values_cfg)]
else:
    x_values_apod = [float(x_val) for x_val in x_values_cfg]

z_profiles_cfg = plot_cfg.get("z_profiles_mm", None)
if z_profiles_cfg is None:
    z_profiles_mm = None
elif isinstance(z_profiles_cfg, (int, float)):
    z_profiles_mm = [float(z_profiles_cfg)]
else:
    z_profiles_mm = [float(z_val) for z_val in z_profiles_cfg]

for ch_idx in range(n_apodizations):
    apod_after_ch = weights_after_grid_np[..., ch_idx]   # (E, Z, X)
    apod_before_ch = weights_before_grid_np[..., ch_idx]  # (E, Z, X)
    for x_value in x_values_apod:
        x_token = f"{x_value:.2f}".replace("-", "m").replace(".", "p")
        helpers.plot_apodization_before_after(
            cm=cm,
            apod_before=apod_before_ch,
            apod_after=apod_after_ch,
            output_path=str(
                Path(sandbox_dir) / f"apodization_map_ch{ch_idx}_x_{x_token}_with_hanning.png"
            ),
            x_fixed=float(x_value),
            z_profiles=z_profiles_mm,
            cmap=str(plot_cfg.get("apod_cmap", "viridis")),
            hanning_apod=hanning_weights_np,
        )


# ============================================================================
# 10) Full validation-set evaluation (MAE + optional scatterer metrics)
# ============================================================================

validation_targets = targets[val_idx].astype(np.float32, copy=False)
validation_sample_weights = (
    train_loss_weights[val_idx].astype(np.float32, copy=False)
    if train_loss_weights is not None
    else None
)
scatterer_eval_cfg = cfg.get("scatterer_eval", {})
radius_mm_eval = float(scatterer_eval_cfg.get("radius_mm", 1.5))
hist_bins_eval = int(scatterer_eval_cfg.get("hist_bins", 50))
eval_batch_size = int(scatterer_eval_cfg.get("eval_batch_size", cfg["training"].get("batch_size", 1)))
n_val = int(len(val_idx))

predicted_val_abs_list = []
uniform_val_abs_list = []
hanning_val_abs_list = []
boxcar_val_abs_list = []

for start in range(0, n_val, eval_batch_size):
    end = min(start + eval_batch_size, n_val)
    idx_chunk = val_idx[start:end]

    if eval_noise_enabled:
        if noise is None:
            raise ValueError(
                "Eval noise requested but dataset does not contain precomputed noise. "
                "Provide delayed_samples_noise.npy or delayed_samples_combined.npy."
            )
        sig_chunk = delayed[idx_chunk].astype(np.complex64, copy=False)
        noise_chunk = noise[idx_chunk].astype(np.complex64, copy=False)
        val_delayed_chunk_np = sig_chunk + eval_noise_scale * noise_chunk
    else:
        val_delayed_chunk_np = delayed[idx_chunk].astype(np.complex64, copy=False)

    val_delayed_chunk = tf.convert_to_tensor(val_delayed_chunk_np)

    # reconstruct_image returns (combined_image, weights_grid); combined_image is (B, Z, X).
    predicted_chunk_image, _weights_chunk = trainer.reconstruct_image(val_delayed_chunk, training=False)
    predicted_val_abs_list.append(predicted_chunk_image.numpy())

    uniform_val_abs_list.append(
        helpers.compute_das_baseline_numpy(val_delayed_chunk_np)
    )
    hanning_val_abs_list.append(
        helpers.compute_das_baseline_numpy(
            delayed_samples=val_delayed_chunk_np,
            apodization=hanning_weights_np,
        )
    )
    boxcar_val_abs_list.append(
        helpers.compute_das_baseline_numpy(
            delayed_samples=val_delayed_chunk_np,
            apodization=boxcar_weights_np,
        )
    )

predicted_val_abs = np.concatenate(predicted_val_abs_list, axis=0)
uniform_val_abs = np.concatenate(uniform_val_abs_list, axis=0)
hanning_val_abs = np.concatenate(hanning_val_abs_list, axis=0)
boxcar_val_abs = np.concatenate(boxcar_val_abs_list, axis=0)

images_abs_eval = {
    "uniform": uniform_val_abs,
    "hanning": hanning_val_abs,
    "inr_after": predicted_val_abs,
    "boxcar": boxcar_val_abs,
}

validation_bundle = helpers.compute_validation_and_reference_metrics(
    images_abs=images_abs_eval,
    targets=validation_targets,
    scatterers_xy=None,
    cm=cm,
    sample_weights=validation_sample_weights,
    radius_mm=radius_mm_eval,
    hist_bins=hist_bins_eval,
)


# ============================================================================
# 11) Optional scatterer diagnostics and SNR-ratio plots
# ============================================================================

# --- Scatterer metrics evaluation (full validation set) ---
if bool(scatterer_eval_cfg.get("enabled", False)):
    try:
        scatterers_batch = load_validation_scatterers(dataset_folder, val_idx)

        baseline_reference_summary, baseline_reference_arrays = find_latest_baseline_reference(
            dataset_folder=dataset_folder,
            baseline_f_number=baseline_f_number,
            baseline_output_root=cfg["io"].get(
                "baseline_output",
                cfg["io"].get("scripts_output_root", "scripts/outputs/train_mixer"),
            ),
        )

        if baseline_reference_arrays is not None:
            print(
                "Loaded baseline scatterer references from: "
                f"{baseline_reference_summary['output_dir']}"
            )

        validation_bundle = helpers.compute_validation_and_reference_metrics(
            images_abs=images_abs_eval,
            targets=validation_targets,
            scatterers_xy=scatterers_batch,
            cm=cm,
            sample_weights=validation_sample_weights,
            radius_mm=radius_mm_eval,
            hist_bins=hist_bins_eval,
        )

        if baseline_reference_arrays is not None and "inr_after" in validation_bundle.get("scatterer_metrics", {}):
            inr_after_metrics = validation_bundle["scatterer_metrics"]["inr_after"]
            inr_after_snr = extract_scatterer_snr(inr_after_metrics)
            ratio_summary: dict[str, dict[str, float]] = {}
            ratio_rows: list[dict[str, object]] = []

            for ref_name in ("uniform", "hanning", "boxcar"):
                ref_key = f"snr_{ref_name}"
                if ref_key not in baseline_reference_arrays:
                    continue

                ref_snr = np.asarray(baseline_reference_arrays[ref_key], dtype=np.float64)
                if ref_snr.shape != inr_after_snr.shape:
                    print(
                        f"Warning: baseline SNR shape mismatch for {ref_name}: "
                        f"{ref_snr.shape} vs {inr_after_snr.shape}"
                    )
                    continue

                ratio = inr_after_snr / np.maximum(ref_snr, 1e-12)
                ratio_summary[ref_name] = {
                    "mean": float(ratio.mean()) if ratio.size > 0 else 0.0,
                    "std": float(ratio.std()) if ratio.size > 0 else 0.0,
                    "min": float(ratio.min()) if ratio.size > 0 else 0.0,
                    "max": float(ratio.max()) if ratio.size > 0 else 0.0,
                }

                peak_example_indices = np.asarray(
                    validation_bundle["scatterer_metrics"]["inr_after"]["aggregated"].get(
                        "peak_example_indices", np.arange(ratio.size, dtype=np.int32)
                    ),
                    dtype=np.int32,
                )
                peak_scatterer_indices = np.asarray(
                    validation_bundle["scatterer_metrics"]["inr_after"]["aggregated"].get(
                        "peak_scatterer_indices", np.arange(ratio.size, dtype=np.int32)
                    ),
                    dtype=np.int32,
                )

                ratio_rows.extend(
                    {
                        "reference": ref_name,
                        "point_index": int(point_idx),
                        "example_index": int(example_idx),
                        "scatterer_index": int(scatterer_idx),
                        "snr_inr": float(inr_after_snr[point_idx]),
                        "snr_ref": float(ref_snr[point_idx]),
                        "snr_ratio": float(ratio[point_idx]),
                    }
                    for point_idx, (example_idx, scatterer_idx) in enumerate(
                        zip(peak_example_indices, peak_scatterer_indices, strict=False)
                    )
                )

                fig_ratio_hist, ax_ratio_hist = plt.subplots(1, 1, figsize=(8, 5))
                ax_ratio_hist.hist(ratio, bins=50, color="tab:blue", alpha=0.75, edgecolor="black")
                ax_ratio_hist.set_xlabel(f"INR / {ref_name} SNR ratio")
                ax_ratio_hist.set_ylabel("Frequency")
                ax_ratio_hist.set_title(
                    f"SNR ratio histogram: INR / {ref_name} ({len(ratio)} points)"
                )
                ax_ratio_hist.grid(True, alpha=0.3)
                fig_ratio_hist.tight_layout()
                fig_ratio_hist.savefig(
                    str(Path(sandbox_dir) / f"snr_ratio_hist_inr_vs_{ref_name}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig_ratio_hist)

            if ratio_summary:
                with open(Path(sandbox_dir) / "snr_ratio_summary.json", "w", encoding="utf-8") as handle:
                    json.dump(ratio_summary, handle, indent=2)

            if ratio_rows:
                with open(Path(sandbox_dir) / "snr_ratio_points.csv", "w", encoding="utf-8", newline="") as csv_file:
                    writer = csv.DictWriter(
                        csv_file,
                        fieldnames=[
                            "reference",
                            "point_index",
                            "example_index",
                            "scatterer_index",
                            "snr_inr",
                            "snr_ref",
                            "snr_ratio",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(ratio_rows)

                np.savez(
                    Path(sandbox_dir) / "snr_ratio_points.npz",
                    snr_inr=inr_after_snr.astype(np.float32, copy=False),
                    **{
                        f"snr_ref_{name}": np.asarray(
                            baseline_reference_arrays[f"snr_{name}"], dtype=np.float32
                        )
                        for name in ("uniform", "hanning", "boxcar")
                        if f"snr_{name}" in baseline_reference_arrays
                    },
                )

        helpers.plot_scatterer_evaluation(
            images_abs=images_abs_eval,
            scatterers_xy=scatterers_batch,
            cm=cm,
            output_dir=sandbox_dir,
            radius_mm=radius_mm_eval,
            hist_bins=hist_bins_eval,
            compare_pairs=[("uniform", "inr_after"), ("hanning", "inr_after")],
            extent=kp.get_imshow_extent(),
            vmin_db=float(plot_cfg.get("vmin_db", -60.0)),
            vmax_db=float(plot_cfg.get("vmax_db", 0.0)),
            cmap=str(plot_cfg.get("cmap", "gray")),
            example_suffix=f"val_all_{len(val_idx)}",
            all_metrics=validation_bundle["scatterer_metrics"],
        )

        # Also produce SNR ratio scatter plots per-reflector for requested comparisons
        compare_pairs_snr = [("uniform", "inr_after"), ("hanning", "inr_after")]
        for ref_name, cmp_name in compare_pairs_snr:
            try:
                res = helpers.plot_scatterer_snr_ratio(
                    images_abs=images_abs_eval,
                    scatterers_xy=scatterers_batch,
                    cm=cm,
                    ref_method=ref_name,
                    cmp_method=cmp_name,
                    radius_mm=radius_mm_eval,
                    extent=kp.get_imshow_extent(),
                    cmap="RdBu_r",
                    scale="linear",
                    clip_percentiles=(1.0, 99.0),
                    point_size=15,
                    alpha=0.7,
                    return_fig=True,
                    all_metrics=validation_bundle["scatterer_metrics"],
                )
                fig = res.get("fig")
                label = res.get("ratio_label", f"{cmp_name}/{ref_name}")
                label_fname = label.replace('/', '_')
                if fig is not None:
                    out_path = str(Path(sandbox_dir) / f"scatt_snr_ratio_{label_fname}_val_{len(val_idx)}.png")
                    fig.savefig(out_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
            except FileNotFoundError as e:
                print(f"Warning: scatterer_snr_ratio skipped — {e}")
            except Exception as e:
                print(f"Warning: scatterer_snr_ratio failed — {e}")
        print(f"Scatterer evaluation figures saved to: {sandbox_dir}")
    except FileNotFoundError as e:
        print(f"Warning: scatterer_eval skipped — {e}")
    except Exception as e:
        print(f"Warning: scatterer_eval failed — {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# 12) End-of-run summaries and optional re-plot with references
# ============================================================================

history_val_mae = history.history.get("val_mae")
if history_val_mae is None:
    history_val_mae = history.history.get("val_mean_absolute_error")
if history_val_mae is None:
    history_val_mae = history.history.get("val_loss")
if history_val_mae is None:
    history_val_mae = history.history.get("val_pixel_weighted_mae")
if history_val_mae is None:
    history_val_mae = history.history.get("val_masked_mae")
if history_val_mae is None:
    raise ValueError(
        "history does not contain val_mae, val_mean_absolute_error, val_loss, "
        "val_pixel_weighted_mae, or val_masked_mae"
    )
history_val_mae = float(history_val_mae[-1])

method_order = ("uniform", "hanning", "boxcar", "inr_after")
relative_mae_y_pred_by_method = {}
relative_mae_y_true_by_method = {}
for method_name in method_order:
    pred_values = images_abs_eval.get(method_name)
    if pred_values is None:
        continue
    relative_mae_y_pred_by_method[method_name] = float(
        relative_mae(
            y_true=validation_targets,
            y_pred=pred_values,
            normalize_by="y_pred",
            sample_weights=validation_sample_weights,
        )
    )
    relative_mae_y_true_by_method[method_name] = float(
        relative_mae(
            y_true=validation_targets,
            y_pred=pred_values,
            normalize_by="y_true",
            sample_weights=validation_sample_weights,
        )
    )

reference_pixel_weighted_mae_for_plot = validation_bundle.get("masked_mae_by_method", {})
comparison_summary = {
    "history_val_mae": history_val_mae,
    "validation_mae": validation_bundle["mae_by_method"],
    "validation_pixel_weighted_mae": validation_bundle.get("masked_mae_by_method", {}),
    "validation_relative_mae": {
        "relative_mae_y_pred": relative_mae_y_pred_by_method,
        "relative_mae_y_true": relative_mae_y_true_by_method,
    },
    "delta_inr_after_vs_history": float(
        validation_bundle["mae_by_method"]["inr_after"] - history_val_mae
    ),
}

with open(Path(sandbox_dir) / "validation_mae_summary.json", "w", encoding="utf-8") as file:
    json.dump(comparison_summary, file, indent=2)

print("Validation MAE summary:")
for method_name in method_order:
    mae_value = validation_bundle["mae_by_method"].get(method_name)
    if mae_value is not None:
        print(f"  {method_name:>10}: {mae_value:.6g}")
if validation_bundle.get("masked_mae_by_method"):
    print("Validation PixelWeightedMAE summary:")
    for method_name in method_order:
        mae_value = validation_bundle["masked_mae_by_method"].get(method_name)
        if mae_value is not None:
            print(f"  {method_name:>10}: {mae_value:.6g}")
if relative_mae_y_pred_by_method:
    print("Validation RelativeMAE (normalize_by=y_pred) summary:")
    for method_name in method_order:
        rel_value = relative_mae_y_pred_by_method.get(method_name)
        if rel_value is not None:
            print(f"  {method_name:>10}: {rel_value:.6g}")
if relative_mae_y_true_by_method:
    print("Validation RelativeMAE (normalize_by=y_true) summary:")
    for method_name in method_order:
        rel_value = relative_mae_y_true_by_method.get(method_name)
        if rel_value is not None:
            print(f"  {method_name:>10}: {rel_value:.6g}")
if reference_pixel_weighted_mae_for_plot:
    print("Reference PixelWeightedMAE (derived, not persisted):")
    for ref_name, ref_val in reference_pixel_weighted_mae_for_plot.items():
        print(f"  ref_{ref_name:>7}: {ref_val:.6g}")
print(f"  {'history_val_mae':>10}: {history_val_mae:.6g}")
print(
    f"  {'delta_inr_after_vs_history':>10}: "
    f"{comparison_summary['delta_inr_after_vs_history']:.6g}"
)

print("Training finished.")
print("Sandbox artifacts:", sandbox_dir)

# Save training curves with hanning reference lines (absolute and relative metrics)
helpers.plot_training_curves(
    history.history,
    output_path=str(history_dir / "training_history.png"),
    reference_mae={"hanning": validation_bundle.get("masked_mae_by_method", {}).get("hanning")},
    reference_relative_y_pred={"hanning": relative_mae_y_pred_by_method.get("hanning")},
    reference_relative_y_true={"hanning": relative_mae_y_true_by_method.get("hanning")},
    weight_reg_lambda=weight_reg_lambda if weight_reg_enabled else None,
)
