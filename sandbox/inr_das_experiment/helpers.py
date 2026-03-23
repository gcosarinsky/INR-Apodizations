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

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from inr_apodizations.apodizations import extract_map_for_x
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


def to_db(image: np.ndarray, ref: float, eps: float = 1e-8) -> np.ndarray:
    """Convert an image magnitude from linear domain to decibels.

    Args:
        image: Real or complex image in linear domain.
        ref: Positive reference magnitude used as 0 dB.
        eps: Small value to avoid numerical instability.

    Returns:
        NumPy array with image values in dB.
    """
    magnitude = np.abs(image)
    return 20.0 * np.log10((magnitude / (ref + eps)) + eps)


def plot_training_curves(history: dict, output_path: str) -> None:
    """Save a training curve figure from a Keras history dictionary.

    Args:
        history: Mapping with metric lists, typically ``history.history``.
        output_path: Path to the output PNG file.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    if "loss" in history:
        axes[0].plot(history["loss"], label="loss", color="black")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val_loss", color="tab:red")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    if axes[0].lines:
        axes[0].legend()

    # Plot MAE on the primary y-axis and SSIM on a secondary y-axis if present.
    has_metric = False
    mae_plotted = False
    ssim_plotted = False
    if "mae" in history or "val_mae" in history:
        if "mae" in history:
            axes[1].plot(history["mae"], label="mae", color="tab:blue")
            mae_plotted = True
        if "val_mae" in history:
            axes[1].plot(history["val_mae"], label="val_mae", color="tab:orange")
            mae_plotted = True
        has_metric = True

    ssim_ax = None
    if "ssim" in history or "val_ssim_metric" in history:
        # use a twin y-axis for SSIM (range ~[0,1]) to avoid mixing scales
        ssim_ax = axes[1].twinx()
        if "ssim" in history:
            ssim_ax.plot(history["ssim"], label="ssim", color="tab:green")
            ssim_plotted = True
        if "val_ssim" in history:
            ssim_ax.plot(history["val_ssim_metric"], label="val_ssim", color="tab:red")
            ssim_plotted = True
        has_metric = True

    if has_metric:
        axes[1].set_title("Metrics")
        axes[1].set_xlabel("Epoch")
        if mae_plotted:
            axes[1].set_ylabel("MAE")
            axes[1].grid(True, alpha=0.3)
        if ssim_plotted and ssim_ax is not None:
            ssim_ax.set_ylabel("SSIM")

        # build combined legend from both axes if needed
        lines, labels = axes[1].get_legend_handles_labels()
        if ssim_ax is not None:
            l2, lbl2 = ssim_ax.get_legend_handles_labels()
            lines += l2
            labels += lbl2
        if lines:
            axes[1].legend(lines, labels)
    else:
        axes[1].axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_das_comparison_db(
    uniform_image: np.ndarray,
    inr_before_image: np.ndarray,
    inr_after_image: np.ndarray,
    target_image: np.ndarray,
    output_path: str,
    extent: tuple[float, float, float, float],
    cmap: str = "gray",
    vmin_db: float = -60.0,
    vmax_db: float = 0.0,
) -> None:
    """Save a four-panel DAS comparison in decibels.

    Args:
        uniform_image: DAS image using uniform apodization.
        inr_before_image: INR reconstruction before training.
        inr_after_image: INR reconstruction after training.
        target_image: Reference target image.
        output_path: Path to the output PNG file.
        extent: Matplotlib imshow extent from beamforming geometry.
        cmap: Colormap used for all panels.
        vmin_db: Lower dB display bound.
        vmax_db: Upper dB display bound.
    """
    images_linear = {
        "Uniform": np.asarray(uniform_image),
        "INR before": np.asarray(inr_before_image),
        "INR after": np.asarray(inr_after_image),
        "Target": np.asarray(target_image),
    }
    shared_ref = max(float(np.max(np.abs(image))) for image in images_linear.values())
    images_db = {name: to_db(image, ref=shared_ref) for name, image in images_linear.items()}

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharex=True, sharey=True)
    first_im = None
    for idx, (title, image_db) in enumerate(images_db.items()):
        current_im = axes[idx].imshow(
            image_db,
            cmap=cmap,
            vmin=vmin_db,
            vmax=vmax_db,
            extent=extent,
            aspect="auto",
        )
        if first_im is None:
            first_im = current_im
        axes[idx].set_title(f"DAS {title} (dB)")
        axes[idx].set_xlabel("x (mm)")
        if idx == 0:
            axes[idx].set_ylabel("z (mm)")

    fig.suptitle("DAS comparison")
    fig.tight_layout(rect=[0, 0, 0.94, 1])
    cbar_ax = fig.add_axes([0.95, 0.13, 0.012, 0.74])
    fig.colorbar(first_im, cax=cbar_ax, label="dB")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_apodization_before_after(
    cm: CoordinateManager,
    apod_before: np.ndarray,
    apod_after: np.ndarray,
    output_path: str,
    x_fixed: float = 0.0,
    scaled: bool = False,
    cmap: str = "viridis",
) -> None:
    """Save a two-panel map comparing apodization before and after training.

    Args:
        cm: Coordinate manager used to extract geometry coordinates.
        apod_before: INR weights before training with shape (E, Z, X).
        apod_after: INR weights after training with shape (E, Z, X).
        output_path: Path to the output PNG file.
        x_fixed: Lateral x value used for map extraction.
        scaled: Whether CoordinateManager scaled coordinates are used.
        cmap: Colormap used for both maps.
    """
    coords = cm.get_coordinates_1d(scaled=scaled)
    x_elems = np.asarray(coords["x_elem"])
    z_coords = np.asarray(coords["z"])
    map_extent = (float(x_elems[0]), float(x_elems[-1]), float(z_coords[-1]), float(z_coords[0]))

    map_before = extract_map_for_x(tf.convert_to_tensor(apod_before), cm, x_fixed=x_fixed, scaled=scaled)
    map_after = extract_map_for_x(tf.convert_to_tensor(apod_after), cm, x_fixed=x_fixed, scaled=scaled)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    mb = map_before.numpy()
    ma = map_after.numpy()
    vmin = float(min(float(mb.min()), float(ma.min())))
    vmax = float(max(float(mb.max()), float(ma.max())))

    im0 = axes[0].imshow(
        mb,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=map_extent,
        aspect="auto",
    )
    axes[0].set_title("Apodization INR before")
    axes[0].set_xlabel("Element lateral coordinate (mm)")
    axes[0].set_ylabel("Depth z (mm)")

    im1 = axes[1].imshow(
        ma,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=map_extent,
        aspect="auto",
    )
    axes[1].set_title("Apodization INR after")
    axes[1].set_xlabel("Element lateral coordinate (mm)")
    fig.colorbar(im0, ax=axes[0], label="Weight")
    fig.colorbar(im1, ax=axes[1], label="Weight")

    fig.suptitle(f"Apodization maps at x={x_fixed:.2f}")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
