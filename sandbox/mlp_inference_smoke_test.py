"""Minimal INR-style MLP inference smoke test.

This script:
1. Loads the latest delayed-samples configuration.
2. Builds KernelParameters2D and CoordinateManager.
3. Creates a small random-initialized MLP (3 hidden layers, 8 units each).
4. Runs inference over the full coordinate features grid.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

from inr_apodizations.config import DATA_DIR
from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.kernels import KernelParameters2D


def find_latest_delayed_folder(base_dir: Path) -> Path:
    """Return the most recently modified delayed-samples dataset folder.

    Args:
        base_dir: Base data directory that contains delayed_samples_dataset.

    Returns:
        Path to the latest delayed-samples subfolder.

    Raises:
        FileNotFoundError: If the delayed_samples_dataset folder or subfolders do not exist.
    """
    delayed_root = base_dir / "delayed_samples_dataset"
    if not delayed_root.exists():
        raise FileNotFoundError(f"No delayed_samples_dataset folder at {delayed_root}")

    subfolders = [folder for folder in delayed_root.iterdir() if folder.is_dir()]
    if not subfolders:
        raise FileNotFoundError(f"No delayed-samples subfolders found in {delayed_root}")

    return max(subfolders, key=lambda folder: folder.stat().st_mtime)


def build_random_mlp() -> tf.keras.Model:
    """Create a small MLP with random-initialized weights for INR inference.

    Returns:
        A TensorFlow Keras model with input shape (3,) and scalar output.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(3,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ],
        name="inr_mlp_smoke_test",
    )
    return model


latest_folder = find_latest_delayed_folder(DATA_DIR)
print(f"Using delayed-samples dataset: {latest_folder}")

cfg_path = latest_folder / "cfg_delayed_samples.npy"
if not cfg_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

cfg = np.load(cfg_path, allow_pickle=True).item()
print(f"Loaded configuration from: {cfg_path}")

kp = KernelParameters2D(cfg)
cm = CoordinateManager(kp)

print(f"Kernel params: n_elements={kp.n_elements}, nx={kp.nx}, nz={kp.nz}")
print(f"CoordinateManager shape (n_elem, nz, nx): {cm.shape}")

model = build_random_mlp()
print("Created random MLP model")
model.summary()

features_grid = tf.cast(cm.get_features_grid(scaled=False), tf.float32)
print(f"Features grid shape: {features_grid.shape}, dtype: {features_grid.dtype}")

n_elem, nz, nx, n_features = features_grid.shape
features_flat = tf.reshape(features_grid, [-1, n_features])
print(f"Flattened features shape: {features_flat.shape}")

pred_flat = model(features_flat, training=False)
print(f"Flat prediction shape: {pred_flat.shape}")

pred_grid = tf.reshape(pred_flat, [n_elem, nz, nx])
print(f"Grid prediction shape: {pred_grid.shape}")

pred_np = pred_grid.numpy()
print("Prediction stats:")
print(f"  min={pred_np.min():.6f}")
print(f"  max={pred_np.max():.6f}")
print(f"  mean={pred_np.mean():.6f}")
print(f"  has_nan={np.isnan(pred_np).any()}")
print(f"  has_inf={np.isinf(pred_np).any()}")

print("INR inference smoke test completed successfully")
