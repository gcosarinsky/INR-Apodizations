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
import matplotlib.pyplot as plt


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
physical_feature_set = str(
    cfg.get("model", {}).get("physical_feature_set", "distance_depth_edge")
)
kp, cm = helpers.build_coordinate_manager(
    dataset_folder,
    physical_feature_set=physical_feature_set,
)

delayed_dataset_bytes = int(delayed.nbytes)
delayed_example_bytes = int(np.prod(delayed.shape[1:], dtype=np.int64) * delayed.dtype.itemsize)

max_examples = cfg["training"].get("max_examples")
if max_examples is not None:
    delayed = delayed[: int(max_examples)]
    targets = targets[: int(max_examples)]

mask_weighting_cfg = dict(cfg["training"].get("mask_weighting", {}))
train_loss_weights = helpers.build_gaussian_loss_weights(gaussian_masks, mask_weighting_cfg)
use_pixelwise_weights = train_loss_weights is not None

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
    batch_size=int(cfg["training"]["batch_size"]),
    shuffle=False,
    seed=int(cfg["training"]["seed"]),
)

features_grid = cm.get_features_grid(scaled=bool(cfg["model"]["scaled_features"]))
output_activation = cfg["model"].get("output_activation", "sigmoid")
apodization_model = helpers.build_mlp_inr(
    input_dim=cm.n_physical_features,
    hidden_units=int(cfg["model"]["hidden_units"]),
    n_hidden=int(cfg["model"]["n_hidden_layers"]),
    activation=cfg["model"]["activation"],
    output_activation=output_activation,
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

if weight_reg_lambda < 0.0:
    raise ValueError("training.weight_regularization.lambda must be >= 0")
if weight_reg_tau < 0.0:
    raise ValueError("training.weight_regularization.tau must be >= 0")
if weight_reg_epsilon <= 0.0:
    raise ValueError("training.weight_regularization.epsilon must be > 0")

resolved_weight_reg_type = "hinge_low_norm" if weight_reg_type == "hinge" else weight_reg_type

trainer = DasInrTrainer(
    apodization_model=apodization_model,
    features_grid=features_grid,
    feature_chunk_size=int(cfg["model"]["feature_chunk_size"]),
    weight_regularization_enabled=weight_reg_enabled,
    weight_regularization_lambda=weight_reg_lambda,
    weight_regularization_tau=weight_reg_tau,
    weight_regularization_epsilon=weight_reg_epsilon,
    weight_regularization_normalize=weight_reg_normalize,
)
print("Weight regularization configuration:")
print(
    {
        "enabled": weight_reg_enabled,
        "type": resolved_weight_reg_type,
        "lambda": weight_reg_lambda,
        "tau": weight_reg_tau,
        "epsilon": weight_reg_epsilon,
        "normalize_norm": weight_reg_normalize,
    }
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
sample_delayed = tf.convert_to_tensor(delayed[val_idx[:1]].astype(np.complex64, copy=False))
sample_target = np.expand_dims(targets[val_idx[0]].astype(np.float32, copy=False), axis=0)
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
    "resolved_weight_regularization": {
        "enabled": weight_reg_enabled,
        "type": resolved_weight_reg_type,
        "lambda": weight_reg_lambda,
        "tau": weight_reg_tau,
        "epsilon": weight_reg_epsilon,
        "normalize_norm": weight_reg_normalize,
    },
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
    target_image=sample_target[0],
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
    target_image=sample_target[0],
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
    target_image=sample_target[0],
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

# --- Scatterer metrics evaluation (full validation set) ---
scatterer_eval_cfg = cfg.get("scatterer_eval", {})
if bool(scatterer_eval_cfg.get("enabled", False)):
    try:
        scatterers_all = helpers.load_saved_scatterers(dataset_folder)

        # Build per-example scatterer lists for the validation indices (convert m -> mm)
        scatterers_batch: list[np.ndarray] = []
        for idx in val_idx:
            s = np.asarray(scatterers_all[int(idx)], dtype=np.float32).copy()
            s[:, :2] *= 1000.0
            scatterers_batch.append(s[:, :2])

        radius_mm_eval = float(scatterer_eval_cfg.get("radius_mm", 1.5))
        hist_bins_eval = int(scatterer_eval_cfg.get("hist_bins", 50))

        # Reconstruct full validation set images in smaller chunks to avoid GPU OOM.
        eval_batch_size = int(scatterer_eval_cfg.get("eval_batch_size", cfg["training"].get("batch_size", 1)))
        n_val = int(len(val_idx))

        # Prepare lists to accumulate per-chunk results
        predicted_val_abs_list = []
        uniform_val_abs_list = []
        hanning_val_abs_list = []
        boxcar_val_abs_list = []

        # Precompute baseline apodization batches (will be cast per-chunk)
        hanning_weights_b = tf.expand_dims(hanning_weights, axis=0)
        boxcar_weights_b = tf.expand_dims(boxcar_weights, axis=0)

        for start in range(0, n_val, eval_batch_size):
            end = min(start + eval_batch_size, n_val)
            idx_chunk = val_idx[start:end]

            # Build tensor for this chunk and run reconstruction
            val_delayed_chunk = tf.convert_to_tensor(delayed[idx_chunk].astype(np.complex64, copy=False))

            predicted_chunk_complex, weights_chunk = trainer.reconstruct_image(
                val_delayed_chunk, training=False
            )
            predicted_val_abs_list.append(tf.abs(predicted_chunk_complex).numpy())

            uniform_val_abs_list.append(tf.abs(tf.reduce_sum(val_delayed_chunk, axis=1)).numpy())

            hanning_val_complex_chunk = tf.reduce_sum(
                val_delayed_chunk * tf.cast(hanning_weights_b, val_delayed_chunk.dtype), axis=1
            )
            hanning_val_abs_list.append(tf.abs(hanning_val_complex_chunk).numpy())

            boxcar_val_complex_chunk = tf.reduce_sum(
                val_delayed_chunk * tf.cast(boxcar_weights_b, val_delayed_chunk.dtype), axis=1
            )
            boxcar_val_abs_list.append(tf.abs(boxcar_val_complex_chunk).numpy())

        # Concatenate chunks back into full arrays
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

print("Training finished.")
print("Sandbox artifacts:", sandbox_dir)
