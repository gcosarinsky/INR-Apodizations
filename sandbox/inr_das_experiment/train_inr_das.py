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

import numpy as np
import tensorflow as tf

import helpers


def rmse(y_true, y_pred):
    """Compute RMSE between target and prediction images."""
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


class DasInrTrainer(tf.keras.Model):
    """Keras model that wraps the INR and the physical DAS forward."""

    def __init__(self, apodization_model: tf.keras.Model, features_grid: tf.Tensor, feature_chunk_size: int):
        super().__init__(name="das_inr_trainer")
        self.apodization_model = apodization_model
        self.features_flat = tf.reshape(tf.cast(features_grid, tf.float32), (-1, 3))
        self.n_elem, self.nz, self.nx = [int(dim) for dim in features_grid.shape[:3]]
        self.feature_chunk_size = int(feature_chunk_size)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.rmse_tracker = tf.keras.metrics.Mean(name="rmse")

    @property
    def metrics(self):
        """Expose tracked metrics to Keras."""
        return [self.loss_tracker, self.rmse_tracker]

    def predict_weights_grid(self, training: bool = False) -> tf.Tensor:
        """Run the INR on geometry features and reshape to ``(E, Z, X)``."""
        n_features = int(self.features_flat.shape[0])
        predictions = []
        for start_idx in range(0, n_features, self.feature_chunk_size):
            end_idx = min(start_idx + self.feature_chunk_size, n_features)
            chunk_pred = self.apodization_model(self.features_flat[start_idx:end_idx], training=training)
            predictions.append(chunk_pred)

        weights_flat = tf.concat(predictions, axis=0)
        weights_grid = tf.reshape(weights_flat, (self.n_elem, self.nz, self.nx))
        return tf.cast(weights_grid, tf.float32)

    def reconstruct_image(self, delayed_batch: tf.Tensor, training: bool = False) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply weights to delayed samples and reconstruct ``abs(DAS)`` images."""
        weights_grid = self.predict_weights_grid(training=training)
        weighted_delayed = delayed_batch * tf.cast(weights_grid[tf.newaxis, ...], delayed_batch.dtype)
        predicted_complex = tf.reduce_sum(weighted_delayed, axis=1)
        predicted_image = tf.abs(predicted_complex)
        return predicted_image, weights_grid

    def train_step(self, data):
        """Execute one optimization step on a batch of examples."""
        delayed_batch, target_batch = data
        with tf.GradientTape() as tape:
            predicted_image, _ = self.reconstruct_image(delayed_batch, training=True)
            loss_value = rmse(tf.cast(target_batch, tf.float32), predicted_image)
            if self.losses:
                loss_value += tf.add_n(self.losses)

        gradients = tape.gradient(loss_value, self.apodization_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.apodization_model.trainable_variables))

        self.loss_tracker.update_state(loss_value)
        self.rmse_tracker.update_state(loss_value)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        """Evaluate the model on a validation batch."""
        delayed_batch, target_batch = data
        predicted_image, _ = self.reconstruct_image(delayed_batch, training=False)
        loss_value = rmse(tf.cast(target_batch, tf.float32), predicted_image)
        self.loss_tracker.update_state(loss_value)
        self.rmse_tracker.update_state(loss_value)
        return {metric.name: metric.result() for metric in self.metrics}


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
trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg["training"]["learning_rate"])))

# Resolve output roots using project config and avoid writing into global data/ by
# default — prefer sandbox outputs for processed artifacts when processed_root
# points to the central `data/` folder.
proc_root_cfg = Path(cfg["io"]["processed_root"])
sandbox_root_cfg = Path(cfg["io"]["sandbox_output_root"])
if not proc_root_cfg.is_absolute():
    proc_root = config.PROJ_ROOT / proc_root_cfg
else:
    proc_root = proc_root_cfg
if not sandbox_root_cfg.is_absolute():
    sandbox_root = config.PROJ_ROOT / sandbox_root_cfg
else:
    sandbox_root = sandbox_root_cfg

# If processed root would write into the repository `data/` folder, redirect
# processed outputs to a sandbox location to avoid modifying `data/`.
if "data" in str(proc_root):
    proc_root = config.PROJ_ROOT / "sandbox" / "inr_das_experiment" / "outputs" / "processed"

timestamp, processed_dir, sandbox_dir = helpers.create_run_directories(
    str(proc_root),
    str(sandbox_root),
)
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

sample_delayed = tf.convert_to_tensor(val_delayed[:1])
predicted_image, weights_grid = trainer.reconstruct_image(sample_delayed, training=False)
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
helpers.save_artifacts(processed_dir, apodization_model, history.history, effective_cfg)
helpers.save_artifacts(sandbox_dir, apodization_model, history.history, effective_cfg)
helpers.save_debug_arrays(
    processed_dir,
    {
        "weights_grid": weights_grid.numpy(),
        "predicted_image": predicted_image.numpy(),
        "target_image": val_targets[:1],
        "uniform_image": uniform_image.numpy(),
    },
)
helpers.save_debug_arrays(
    sandbox_dir,
    {
        "weights_grid": weights_grid.numpy(),
        "predicted_image": predicted_image.numpy(),
        "target_image": val_targets[:1],
        "uniform_image": uniform_image.numpy(),
    },
)

print("Training finished.")
print("Processed artifacts:", processed_dir)
print("Sandbox artifacts:", sandbox_dir)
