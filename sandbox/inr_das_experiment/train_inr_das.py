"""Experimental training script for INR-based DAS apodization.

This sandbox script implements the physical forward requested for the first
experiment:

1. Build INR weights for every (element, z, x) position from geometry-only features.
2. Multiply those weights by the delayed samples.
3. Sum over the element axis to reconstruct a complex DAS image.
4. Compare ``abs(image)`` against the provided target with RMSE.

The script keeps logic direct and sandbox-oriented. Configuration lives in
``sandbox/inr_das_experiment/config.yml``.
"""
from __future__ import annotations

import pprint
from pathlib import Path
from inr_apodizations import config

import os
from datetime import datetime

import numpy as np
import tensorflow as tf

import helpers
from model_defs import DasInrTrainer, ssim_metric


CONFIG_PATH = Path("sandbox/inr_das_experiment/config.yml")
cfg = helpers.load_experiment_config(str(CONFIG_PATH))
tf.random.set_seed(int(cfg["training"]["seed"]))
np.random.seed(int(cfg["training"]["seed"]))

# Resolve dataset folder relative to project root when given as a relative path
dataset_folder = Path(cfg["io"]["dataset_folder"])
if not dataset_folder.is_absolute():
    dataset_folder = config.PROJ_ROOT / dataset_folder
dataset_folder = str(dataset_folder)
print("Loading dataset from:", dataset_folder)
delayed, targets, info = helpers.load_delayed_samples_dataset(dataset_folder)
helpers.validate_dataset_shapes(delayed, targets)
kp, cm = helpers.build_coordinate_manager(dataset_folder)

max_examples = cfg["training"].get("max_examples")
if max_examples is not None:
    delayed = delayed[: int(max_examples)]
    targets = targets[: int(max_examples)]

train_delayed, train_targets, val_delayed, val_targets = helpers.split_train_validation_examples(
    delayed,
    targets,
    train_fraction=float(cfg["training"]["train_fraction"]),
    seed=int(cfg["training"]["seed"]),
)

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

train_ds = helpers.build_tf_dataset_by_examples(
    train_delayed,
    train_targets,
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
apodization_model = helpers.build_mlp_inr(
    input_dim=3,
    hidden_units=int(cfg["model"]["hidden_units"]),
    n_hidden=int(cfg["model"]["n_hidden_layers"]),
    activation=cfg["model"]["activation"],
    output_activation=cfg["model"]["output_activation"],
)
trainer = DasInrTrainer(
    apodization_model=apodization_model,
    features_grid=features_grid,
    feature_chunk_size=int(cfg["model"]["feature_chunk_size"]),
)
trainer.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg["training"]["learning_rate"])),
    loss=tf.keras.losses.MeanAbsoluteError(name="mae"),
    metrics=[ssim_metric],
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

effective_cfg = {
    "config_path": str(CONFIG_PATH),
    "dataset_folder": dataset_folder,
    "run_timestamp": timestamp,
    "beamforming": {
        "n_elements": kp.n_elements,
        "nz": kp.nz,
        "nx": kp.nx,
        "roi_effective": list(kp.roi_effective),
    },
    "experiment": cfg,
}
helpers.save_artifacts(sandbox_dir, apodization_model, history.history, effective_cfg)
helpers.save_debug_arrays(
    sandbox_dir,
    {
        "weights_grid": weights_after_grid.numpy(),
        "predicted_image": predicted_after_image.numpy(),
        "predicted_before_image": predicted_before_image.numpy(),
        "weights_before_grid": weights_before_grid.numpy(),
        "target_image": val_targets[:1],
        "uniform_image": uniform_image.numpy(),
    },
)

plot_cfg = cfg.get("plots", {})
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
)
helpers.plot_apodization_before_after(
    cm=cm,
    apod_before=weights_before_grid.numpy(),
    apod_after=weights_after_grid.numpy(),
    output_path=str(Path(sandbox_dir) / "apodization_map_before_after.png"),
    x_fixed=float(plot_cfg.get("x_fixed_apod", 0.0)),
    scaled=bool(cfg["model"]["scaled_features"]),
    cmap=str(plot_cfg.get("apod_cmap", "viridis")),
)

print("Training finished.")
print("Sandbox artifacts:", sandbox_dir)
