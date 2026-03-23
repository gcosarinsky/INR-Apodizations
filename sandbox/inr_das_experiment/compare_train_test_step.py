"""Compare train_step and test_step computations for DAS INR trainer.

This sandbox script inspects:
1) Forward RMSE in training/inference modes on the same batch.
2) Extra loss term coming from ``self.losses``.
3) Actual outputs of ``train_step`` and ``test_step``.

Run:
    python sandbox/inr_das_experiment/compare_train_test_step.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

import helpers
from inr_apodizations import config
from model_defs import DasInrTrainer, rmse


# Load config and seeds.
CONFIG_PATH = Path("sandbox/inr_das_experiment/config.yml")
cfg = helpers.load_experiment_config(str(CONFIG_PATH))
tf.random.set_seed(int(cfg["training"]["seed"]))
np.random.seed(int(cfg["training"]["seed"]))

# Resolve dataset path.
dataset_folder = Path(cfg["io"]["dataset_folder"])
if not dataset_folder.is_absolute():
    dataset_folder = config.PROJ_ROOT / dataset_folder

# Load and validate dataset.
delayed, targets, _ = helpers.load_delayed_samples_dataset(str(dataset_folder))
helpers.validate_dataset_shapes(delayed, targets)
_, cm = helpers.build_coordinate_manager(str(dataset_folder))

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

train_ds = helpers.build_tf_dataset_by_examples(
    train_delayed,
    train_targets,
    batch_size=int(cfg["training"]["batch_size"]),
    shuffle=False,
    seed=int(cfg["training"]["seed"]),
)
val_ds = helpers.build_tf_dataset_by_examples(
    val_delayed,
    val_targets,
    batch_size=int(cfg["training"]["batch_size"]),
    shuffle=False,
    seed=int(cfg["training"]["seed"]),
)

# Build INR model and trainer exactly like train_inr_das.
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

train_batch = next(iter(train_ds))
val_batch = next(iter(val_ds))

# Compare pure compute on the same batch by only switching `training` mode.
delayed_batch, target_batch = train_batch
pred_train_mode, _ = trainer.reconstruct_image(delayed_batch, training=True)
pred_eval_mode, _ = trainer.reconstruct_image(delayed_batch, training=False)

rmse_train_mode = rmse(tf.cast(target_batch, tf.float32), pred_train_mode)
rmse_eval_mode = rmse(tf.cast(target_batch, tf.float32), pred_eval_mode)
extra_losses_same_batch = tf.add_n(trainer.losses) if trainer.losses else tf.constant(0.0, dtype=tf.float32)

print("=== Same batch, pure forward compute ===")
print(f"rmse(training=True) : {float(rmse_train_mode.numpy()):.6f}")
print(f"rmse(training=False): {float(rmse_eval_mode.numpy()):.6f}")
print(f"len(self.losses)    : {len(trainer.losses)}")
print(f"sum(self.losses)    : {float(extra_losses_same_batch.numpy()):.6f}")

# Compare actual step outputs.
print("\n=== Step outputs ===")
trainer.reset_metrics()
out_train_step = trainer.train_step(train_batch)
out_train_step = {k: float(v.numpy()) for k, v in out_train_step.items()}
print("train_step(train_batch):", out_train_step)

trainer.reset_metrics()
out_test_train = trainer.test_step(train_batch)
out_test_train = {k: float(v.numpy()) for k, v in out_test_train.items()}
print("test_step(train_batch) :", out_test_train)

trainer.reset_metrics()
out_test_val = trainer.test_step(val_batch)
out_test_val = {k: float(v.numpy()) for k, v in out_test_val.items()}
print("test_step(val_batch)   :", out_test_val)

print("\nDone.")
