"""Helpers for the sandbox INR DAS experiment.

This module provides lightweight utilities to load the delayed-samples
dataset, create the coordinate/features context required by the INR,
split examples into train/validation sets, build tf.data pipelines, and
persist minimal artifacts.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import yaml

from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.kernels import KernelParameters2D


def load_delayed_samples_dataset(folder: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load delayed samples dataset and targets from a dataset folder.

    Expects files inside `folder`: `delayed_samples_dataset.npy`,
    `targets_dataset.npy` and `delayed_samples_info.yaml` (optional).

    Returns:
        delayed: np.ndarray, shape (N, E, Z, X), dtype complex64
        targets: np.ndarray, shape (N, Z, X), dtype float32
        info: dict with parsed YAML metadata (empty dict if not present)
    """
    delayed_path = os.path.join(folder, "delayed_samples_dataset.npy")
    targets_path = os.path.join(folder, "targets_dataset.npy")
    info_path = os.path.join(folder, "delayed_samples_info.yaml")

    if not os.path.exists(delayed_path) or not os.path.exists(targets_path):
        raise FileNotFoundError("Delayed samples or targets .npy not found in %s" % folder)

    delayed = np.load(delayed_path, allow_pickle=False)
    targets = np.load(targets_path, allow_pickle=False)

    info = {}
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = yaml.safe_load(f) or {}

    return delayed, targets, info


def load_saved_beamforming_config(folder: str) -> dict:
    """Load the delayed-samples configuration saved next to the dataset.

    Args:
        folder: Dataset folder containing ``cfg_delayed_samples.npy``.

    Returns:
        Configuration dictionary used to generate the dataset.

    Raises:
        FileNotFoundError: If the config file is missing.
    """
    cfg_path = os.path.join(folder, "cfg_delayed_samples.npy")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    return np.load(cfg_path, allow_pickle=True).item()


def build_coordinate_manager(dataset_folder: str) -> tuple[KernelParameters2D, CoordinateManager]:
    """Build ``KernelParameters2D`` and ``CoordinateManager`` from a dataset folder."""
    cfg = load_saved_beamforming_config(dataset_folder)
    kp = KernelParameters2D(cfg)
    cm = CoordinateManager(kp)
    return kp, cm


def validate_dataset_shapes(delayed: np.ndarray, targets: np.ndarray) -> None:
    """
    Basic assertions ensuring the dataset contract we rely on.

    Raises AssertionError if mismatch or types unexpected.
    """
    assert delayed.ndim == 4, "delayed must be (N, E, Z, X)"
    assert targets.ndim == 3, "targets must be (N, Z, X)"
    assert delayed.shape[0] == targets.shape[0], "N mismatch between delayed and targets"
    assert np.iscomplexobj(delayed), "delayed_samples must be complex-valued"
    assert targets.dtype == np.float32 or targets.dtype == np.float64, "targets must be float"


def split_train_validation_examples(
    delayed: np.ndarray,
    targets: np.ndarray,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the dataset along the example axis.

    If the dataset contains a single example, the same example is used for
    both training and validation so the sandbox remains runnable.
    """
    if delayed.shape[0] != targets.shape[0]:
        raise ValueError("delayed and targets must have the same number of examples")

    n_examples = delayed.shape[0]
    if n_examples == 1:
        return delayed, targets, delayed.copy(), targets.copy()

    if train_fraction <= 0.0 or train_fraction >= 1.0:
        raise ValueError("train_fraction must be in the open interval (0, 1)")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_examples)
    rng.shuffle(indices)

    n_train = max(1, int(np.floor(n_examples * train_fraction)))
    n_train = min(n_train, n_examples - 1)
    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train:])

    return delayed[train_idx], targets[train_idx], delayed[val_idx], targets[val_idx]


def build_tf_dataset_by_examples(delayed: np.ndarray,
                                 targets: np.ndarray,
                                 batch_size: int = 1,
                                 shuffle: bool = True,
                                 seed: Optional[int] = 42) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset that yields (delayed_example, target_example).

    delayed_example: complex64 array (E, Z, X)
    target_example: float32 array (Z, X)

    The dataset yields numpy arrays and lets the training logic convert
    into tensors / compute features. This is simple and robust for
    an initial baseline that batches by examples (axis=0).
    """
    N = delayed.shape[0]
    ds = tf.data.Dataset.from_tensor_slices((delayed, targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=N, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_mlp_inr(input_dim: int = 3,
                  hidden_units: int = 8,
                  n_hidden: int = 3,
                  activation: str = "relu",
                  output_activation: str = "sigmoid") -> tf.keras.Model:
    """
    Build a small MLP that maps `input_dim` features -> scalar weight [0,1].

    Returns a compiled but un-trained `tf.keras.Model` (caller compiles with chosen optimizer/loss).
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="coords")
    x = inputs
    for i in range(n_hidden):
        x = tf.keras.layers.Dense(hidden_units, activation=activation,
                                  name=f"dense_{i+1}")(x)
    outputs = tf.keras.layers.Dense(1, activation=output_activation, name="weight_out")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="inr_mlp")
    return model


def load_experiment_config(config_path: str) -> dict:
    """Load the sandbox experiment YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_run_directories(processed_root: str, sandbox_root: str) -> tuple[str, str, str]:
    """Create timestamped run directories for processed and sandbox outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_dir = os.path.join(processed_root, timestamp)
    sandbox_dir = os.path.join(sandbox_root, timestamp)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(sandbox_dir, exist_ok=True)
    return timestamp, processed_dir, sandbox_dir


def save_artifacts(output_dir: str, model: tf.keras.Model, history: dict, config: dict) -> None:
    """Save model and minimal artifacts into ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.keras")
    model.save(model_path)

    def _to_serializable(obj):
        """Recursively convert numpy/TF types into Python built-ins for JSON."""
        # Scalars
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        # Tensors
        if isinstance(obj, tf.Tensor):
            try:
                val = obj.numpy()
            except Exception:
                return str(obj)
            return _to_serializable(val)
        # NumPy arrays
        if isinstance(obj, np.ndarray):
            return _to_serializable(obj.tolist())
        # Dicts, lists, tuples
        if isinstance(obj, dict):
            return {str(k): _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]
        # Fallback for built-ins
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

    serializable_history = _to_serializable(history)

    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as file:
        json.dump(serializable_history, file, indent=2)

    with open(os.path.join(output_dir, "config.yml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def save_debug_arrays(output_dir: str, arrays: dict[str, np.ndarray]) -> None:
    """Persist selected NumPy arrays for quick inspection."""
    os.makedirs(output_dir, exist_ok=True)
    for name, array in arrays.items():
        np.save(os.path.join(output_dir, f"{name}.npy"), array)
