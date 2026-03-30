"""Experimental training script for INR-based DAS apodization.

This sandbox script implements the physical forward requested for the first
experiment:

1. Build INR weights for every (element, z, x) position from geometry-only features.
2. Multiply those weights by the delayed samples.
3. Sum over the element axis to reconstruct a complex DAS image.
4. Compare ``abs(image)`` against the provided target with RMSE.

The script keeps logic direct and sandbox-oriented. Configuration lives in
``configs/train_config.yml``.
"""
from __future__ import annotations

import os
import pprint
import random
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

from inr_apodizations import config

import numpy as np
import tensorflow as tf

import helpers
from inr_apodizations.modeling.trainer import DasInrTrainer
from inr_apodizations.modeling.metrics import ssim_metric
from inr_apodizations.modeling.metrics import mae_db_factory
from inr_apodizations.apodizations import compute_dynamic_apodizations_tf


CONFIG_PATH = Path("configs/train_config.yml")
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
sigma_x_override, sigma_z_override = helpers.get_target_sigma_override(cfg)
print("Loading dataset from:", dataset_folder)
delayed, targets, gaussian_masks, info = helpers.load_delayed_samples_dataset(
    dataset_folder,
    sigma_x=sigma_x_override,
    sigma_z=sigma_z_override,
)
helpers.validate_dataset_shapes(delayed, targets, gaussian_masks)
kp, cm = helpers.build_coordinate_manager(dataset_folder)

delayed_dataset_bytes = int(delayed.nbytes)
delayed_example_bytes = int(np.prod(delayed.shape[1:], dtype=np.int64) * delayed.dtype.itemsize)

max_examples = cfg["training"].get("max_examples")
if max_examples is not None:
    delayed = delayed[: int(max_examples)]
    targets = targets[: int(max_examples)]

mask_weighting_cfg = dict(cfg["training"].get("mask_weighting", {}))
train_loss_weights = helpers.build_gaussian_loss_weights(gaussian_masks, mask_weighting_cfg)
use_pixelwise_weights = train_loss_weights is not None

train_delayed, train_targets, val_delayed, val_targets = helpers.split_train_validation_examples(
    delayed,
    targets,
    train_fraction=float(cfg["training"]["train_fraction"]),
    seed=int(cfg["training"]["seed"]),
)

configured_batch_size = int(cfg["training"]["batch_size"])
effective_batch_examples = min(configured_batch_size, int(train_delayed.shape[0]))
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
        "train_delayed.shape": train_delayed.shape,
        "val_delayed.shape": val_delayed.shape,
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

# Recover the same train indices used to split delayed/targets so weights
# can be sliced consistently. split_train_validation_indices uses the same
# seed-driven RNG as split_train_validation_examples.
if train_loss_weights is not None:
    _train_idx, _ = helpers.split_train_validation_indices(
        n_examples=delayed.shape[0],
        train_fraction=float(cfg["training"]["train_fraction"]),
        seed=int(cfg["training"]["seed"]),
    )
    train_weights_split = train_loss_weights[_train_idx]
else:
    train_weights_split = None

train_ds = helpers.build_tf_dataset_by_examples(
    train_delayed,
    train_targets,
    sample_weights=train_weights_split,
    batch_size=int(cfg["training"]["batch_size"]),
    shuffle=True,
    seed=int(cfg["training"]["seed"]),
)
val_ds = helpers.build_tf_dataset_by_examples(
    val_delayed,
    val_targets,
    batch_size=int(cfg["training"]["batch_size"]),
    shuffle=False,
    seed=int(cfg["training"]["seed"]),
)

features_grid = cm.get_features_grid(scaled=bool(cfg["model"]["scaled_features"]))
output_activation = cfg["model"].get("output_activation", "sigmoid")
apodization_model = helpers.build_mlp_inr(
    input_dim=3,
    hidden_units=int(cfg["model"]["hidden_units"]),
    n_hidden=int(cfg["model"]["n_hidden_layers"]),
    activation=cfg["model"]["activation"],
    output_activation=output_activation,
)
trainer = DasInrTrainer(
    apodization_model=apodization_model,
    features_grid=features_grid,
    feature_chunk_size=int(cfg["model"]["feature_chunk_size"]),
)
# Shared optional parameters for custom mae_db loss/metric.
mae_db_ref_cfg = cfg["training"].get("mae_db_ref", None)
mae_db_eps = float(cfg["training"].get("mae_db_eps", 1e-8))

if mae_db_ref_cfg is None:
    mae_db_ref = None
else:
    if not isinstance(mae_db_ref_cfg, (list, tuple)) or len(mae_db_ref_cfg) != 2:
        raise ValueError("training.mae_db_ref must be a list/tuple with two values")
    mae_db_ref = (float(mae_db_ref_cfg[0]), float(mae_db_ref_cfg[1]))


def _resolve_custom_mae_db(item, *, name: str):
    if isinstance(item, str) and item.lower() == "mae_db":
        return mae_db_factory(ref=mae_db_ref, eps=mae_db_eps, name=name)
    return item


def _pixelwise_mae(y_true, y_pred):
    """Return element-wise absolute error for per-pixel sample weighting."""
    return tf.abs(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32))


def _pixelwise_mse(y_true, y_pred):
    """Return element-wise squared error for per-pixel sample weighting."""
    diff = tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)
    return tf.square(diff)


_pixelwise_mae.__name__ = "mae"
_pixelwise_mse.__name__ = "mse"


# Resolve loss from config and instantiate a Keras loss object.
# The config can contain any valid identifier accepted by `tf.keras.losses.get`,
# fallback to MAE if resolution fails.
loss_name = cfg["training"].get("loss", "mae")
if isinstance(loss_name, str) and loss_name.lower() == "mae_db":
    loss_obj = mae_db_factory(ref=mae_db_ref, eps=mae_db_eps, name="mae_db")
else:
    try:
        loss_obj = tf.keras.losses.get(loss_name)
    except Exception:
        loss_str = str(loss_name).lower()
        if loss_str in ("mae", "mean_absolute_error"):
            loss_obj = tf.keras.losses.MeanAbsoluteError(name="mae")
        elif loss_str in ("mse", "mean_squared_error"):
            loss_obj = tf.keras.losses.MeanSquaredError(name="mse")
        else:
            loss_obj = tf.keras.losses.MeanAbsoluteError(name="mae")

if use_pixelwise_weights and isinstance(loss_name, str):
    loss_str = loss_name.lower()
    if loss_str in ("mae", "mean_absolute_error"):
        loss_obj = _pixelwise_mae
    elif loss_str in ("mse", "mean_squared_error"):
        loss_obj = _pixelwise_mse

# Resolve metrics from config with optional support for custom mae_db.
metrics_cfg = cfg["training"].get("metric", "mae")
def _resolve_metric(metric_item):
    if use_pixelwise_weights and isinstance(metric_item, str):
        metric_str = metric_item.lower()
        if metric_str in ("mae", "mean_absolute_error"):
            return _pixelwise_mae
        if metric_str in ("mse", "mean_squared_error"):
            return _pixelwise_mse
    return _resolve_custom_mae_db(metric_item, name="mae_db")


if isinstance(metrics_cfg, (list, tuple)):
    metrics_list = [_resolve_metric(m) for m in metrics_cfg]
else:
    metrics_list = [_resolve_metric(metrics_cfg)]

trainer.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg["training"]["learning_rate"])),
    loss=loss_obj,
    metrics=metrics_list,
)

# Keep a deterministic baseline prediction from random INR initialization.
sample_delayed = tf.convert_to_tensor(val_delayed[:1])
predicted_before_image, weights_before_grid = trainer.reconstruct_image(sample_delayed, training=False)

# Resolve sandbox output root and create a timestamped sandbox outputs folder.
sandbox_root_cfg = Path(cfg["io"]["sandbox_output_root"])
if not sandbox_root_cfg.is_absolute():
    sandbox_root = config.PROJ_ROOT / sandbox_root_cfg
else:
    sandbox_root = sandbox_root_cfg

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

print("Starting physical-forward training...")
history = trainer.fit(
    train_ds,
    validation_data=val_ds,
    epochs=int(cfg["training"]["epochs"]),
    callbacks=callbacks,
    verbose=1,
)

predicted_after_image, weights_after_grid = trainer.reconstruct_image(sample_delayed, training=False)
uniform_image = tf.abs(tf.reduce_sum(sample_delayed, axis=1))

# Determine baseline f_number: allow override from config `training.baseline_f_number`.
_cfg_f_number = cfg["training"].get("baseline_f_number", None)
if _cfg_f_number is None:
    baseline_f_number = kp.f_number
else:
    try:
        baseline_f_number = float(_cfg_f_number)
    except Exception:
        baseline_f_number = kp.f_number

# Compute Hanning baseline DAS image using library apodizations (single example batch)
apods_h = compute_dynamic_apodizations_tf(
    cm=cm, f_number=baseline_f_number, methods=("hanning",), scaled=bool(cfg["model"]["scaled_features"]))

hanning_weights = apods_h["hanning"]  # shape: (E, Z, X)
hanning_weights_b = tf.expand_dims(hanning_weights, axis=0)  # add batch dim -> (1, E, Z, X)
hanning_image_complex = tf.reduce_sum(sample_delayed * tf.cast(hanning_weights_b, sample_delayed.dtype), axis=1)
hanning_image = tf.abs(hanning_image_complex)

# Compute Boxcar baseline DAS image using library apodizations (single example batch)
apods_b = compute_dynamic_apodizations_tf(
    cm=cm, f_number=baseline_f_number, methods=("boxcar",), scaled=bool(cfg["model"]["scaled_features"]))

boxcar_weights = apods_b["boxcar"]  # shape: (E, Z, X)
boxcar_weights_b = tf.expand_dims(boxcar_weights, axis=0)  # add batch dim -> (1, E, Z, X)
boxcar_image_complex = tf.reduce_sum(sample_delayed * tf.cast(boxcar_weights_b, sample_delayed.dtype), axis=1)
boxcar_image = tf.abs(boxcar_image_complex)


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
}
helpers.save_artifacts(sandbox_dir, apodization_model, history.history, effective_cfg)

plot_cfg = cfg.get("plots", {})
normalize_each_image = bool(plot_cfg.get("normalize_each_image", False))
helpers.plot_training_curves(
    history.history,
    output_path=str(Path(sandbox_dir) / "training_loss.png"),
)
helpers.plot_das_comparison_db(
    uniform_image=uniform_image.numpy()[0],
    inr_before_image=predicted_before_image.numpy()[0],
    inr_after_image=predicted_after_image.numpy()[0],
    target_image=val_targets[0],
    output_path=str(Path(sandbox_dir) / "das_images_comparison_db.png"),
    extent=kp.get_imshow_extent(),
    cmap=str(plot_cfg.get("cmap", "gray")),
    vmin_db=float(plot_cfg.get("vmin_db", -60.0)),
    vmax_db=float(plot_cfg.get("vmax_db", 0.0)),
    normalize_each_image=normalize_each_image,
)
# Also save a comparison figure using Hanning as the baseline instead of Uniform
helpers.plot_das_comparison_db(
    uniform_image=hanning_image.numpy()[0],
    inr_before_image=predicted_before_image.numpy()[0],
    inr_after_image=predicted_after_image.numpy()[0],
    target_image=val_targets[0],
    output_path=str(Path(sandbox_dir) / "das_images_comparison_db_hanning.png"),
    extent=kp.get_imshow_extent(),
    cmap=str(plot_cfg.get("cmap", "gray")),
    vmin_db=float(plot_cfg.get("vmin_db", -60.0)),
    vmax_db=float(plot_cfg.get("vmax_db", 0.0)),
    normalize_each_image=normalize_each_image,
)

# Also save a comparison figure using Boxcar as the baseline
helpers.plot_das_comparison_db(
    uniform_image=boxcar_image.numpy()[0],
    inr_before_image=predicted_before_image.numpy()[0],
    inr_after_image=predicted_after_image.numpy()[0],
    target_image=val_targets[0],
    output_path=str(Path(sandbox_dir) / "das_images_comparison_db_boxcar.png"),
    extent=kp.get_imshow_extent(),
    cmap=str(plot_cfg.get("cmap", "gray")),
    vmin_db=float(plot_cfg.get("vmin_db", -60.0)),
    vmax_db=float(plot_cfg.get("vmax_db", 0.0)),
    normalize_each_image=normalize_each_image,
)

helpers.plot_apodization_energy_comparison(
    hanning_apod=hanning_weights.numpy(),
    inr_apod_after=weights_after_grid.numpy(),
    output_path=str(Path(sandbox_dir) / "apodization_energy_comparison_hanning_vs_inr_after.png"),
    extent=kp.get_imshow_extent(),
    cmap=str(plot_cfg.get("apod_cmap", "viridis")),
)

# Save one apodization figure per selected x, with multiple z profiles overlaid.
x_values_cfg = plot_cfg.get("x_values_apod", None)
if x_values_cfg is None:
    raise ValueError("plots.x_values_apod must be provided in configs/train_config.yml")
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

for x_value in x_values_apod:
    x_token = f"{x_value:.2f}".replace("-", "m").replace(".", "p")
    helpers.plot_apodization_before_after(
        cm=cm,
        apod_before=weights_before_grid.numpy(),
        apod_after=weights_after_grid.numpy(),
        output_path=str(Path(sandbox_dir) / f"apodization_map_x_{x_token}_with_hanning.png"),
        x_fixed=float(x_value),
        z_profiles=z_profiles_mm,
        cmap=str(plot_cfg.get("apod_cmap", "viridis")),
        hanning_apod=hanning_weights.numpy(),
    )

print("Training finished.")
print("Sandbox artifacts:", sandbox_dir)
