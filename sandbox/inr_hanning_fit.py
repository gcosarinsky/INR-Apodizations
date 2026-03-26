"""Fit a small INR-style MLP to a dynamic Hanning apodization.

This script:
1. Loads the latest delayed-samples configuration.
2. Builds KernelParameters2D and CoordinateManager.
3. Creates a small MLP matching the smoke-test architecture.
4. Fits the MLP to the dynamic Hanning apodization.
5. Plots maps and profiles before and after training.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from inr_apodizations.apodizations import compute_dynamic_apodizations_tf, extract_map_for_x
from inr_apodizations.config import DATA_DIR
from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.kernels import KernelParameters2D
from inr_apodizations.utils import find_latest_dataset_folder

HIDDEN_UNITS = 8
N_HIDDEN_LAYERS = 3
N_EPOCHS = 20
BATCH_SIZE = 8192
LEARNING_RATE = 1e-3
SCALED_FEATURES = True
TRAIN_SUBSET_FRACTION = 0.1  # Set to None to use the full dataset for training
FIGURE_PATH = Path("sandbox/figures/inr_hanning_fit.png")
RANDOM_SEED = 42


def build_mlp() -> tf.keras.Model:
    """Create the INR MLP used to fit the Hanning apodization.

    Returns:
        A TensorFlow Keras model with input shape (3,) and scalar output.
    """
    hidden_layers = [tf.keras.layers.Dense(HIDDEN_UNITS, activation="relu")
                     for _ in range(N_HIDDEN_LAYERS)]
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(3,)),
            *hidden_layers,
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="inr_hanning_fit",
    )
    return model


def load_coordinate_manager() -> tuple[KernelParameters2D, CoordinateManager, Path]:
    """Load the latest delayed-samples configuration and derived objects.

    Returns:
        Tuple containing kernel parameters, coordinate manager, and config path.

    Raises:
        FileNotFoundError: If the delayed-samples config file does not exist.
    """
    latest_folder = find_latest_dataset_folder(DATA_DIR, "delayed_samples_dataset")
    cfg_path = latest_folder / "cfg_delayed_samples.npy"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    cfg = np.load(cfg_path, allow_pickle=True).item()
    kp = KernelParameters2D(cfg)
    cm = CoordinateManager(kp)
    return kp, cm, cfg_path


def predict_grid(
    model: tf.keras.Model,
    features_flat: tf.Tensor,
    grid_shape: tuple[int, int, int],
) -> tf.Tensor:
    """Run MLP inference and reshape predictions to apodization grid format.

    Args:
        model: Keras model used for inference.
        features_flat: Flat feature tensor with shape (N, 3).
        grid_shape: Target grid shape as (n_elem, nz, nx).

    Returns:
        Tensor with shape (n_elem, nz, nx) and dtype tf.float32.
    """
    pred_flat = model(features_flat, training=False)
    return tf.reshape(tf.cast(pred_flat, tf.float32), grid_shape)


def extract_center_profile(apod_tensor: tf.Tensor, x_idx: int, z_idx: int) -> np.ndarray:
    """Extract an element-axis profile at a fixed x and z index.

    Args:
        apod_tensor: Tensor with shape (n_elem, nz, nx).
        x_idx: Lateral index.
        z_idx: Depth index.

    Returns:
        NumPy array with shape (n_elem,).
    """
    return tf.cast(apod_tensor[:, z_idx, x_idx], tf.float32).numpy()


def select_training_subset(
    features_flat: tf.Tensor,
    target_flat: tf.Tensor,
    subset_fraction: float | None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Select a uniform random subset of training samples without replacement.

    Args:
        features_flat: Feature tensor with shape (N, 3).
        target_flat: Target tensor with shape (N, 1).
        subset_fraction: Fraction in (0, 1] to sample, or None for full dataset.

    Returns:
        Tuple containing the selected training features and targets.

    Raises:
        ValueError: If subset_fraction is not None and is outside (0, 1].
    """
    if subset_fraction is None:
        return features_flat, target_flat

    if subset_fraction <= 0.0 or subset_fraction > 1.0:
        raise ValueError("TRAIN_SUBSET_FRACTION must be in (0, 1].")

    n_total = int(features_flat.shape[0])
    n_train = max(1, int(np.round(n_total * subset_fraction)))

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.choice(n_total, size=n_train, replace=False)
    indices = np.sort(indices)

    features_train = tf.gather(features_flat, indices)
    target_train = tf.gather(target_flat, indices)
    return features_train, target_train


def plot_results(
    cm: CoordinateManager,
    target_hanning: tf.Tensor,
    pred_before: tf.Tensor,
    pred_after: tf.Tensor,
    coords: dict[str, np.ndarray],
    x_fixed: float,
    x_idx: int,
    z_idx: int,
    history: tf.keras.callbacks.History,
    figure_path: Path,
) -> None:
    """Create the requested maps and profiles before and after training.

    Args:
        cm: Coordinate manager used to extract the x=0 map.
        target_hanning: Reference Hanning tensor with shape (n_elem, nz, nx).
        pred_before: Model prediction before training.
        pred_after: Model prediction after training.
        coords: Coordinate dictionary from CoordinateManager.
        x_fixed: Lateral position used for the 2D map extraction.
        x_idx: Lateral index used for the profile extraction.
        z_idx: Depth index used for the profile extraction.
        history: Keras training history.
        figure_path: Output path for the saved figure.
    """
    x_elem = np.asarray(coords["x_elem"])
    z = np.asarray(coords["z"])

    map_target = extract_map_for_x(target_hanning, cm, x_fixed=x_fixed).numpy()
    map_before = extract_map_for_x(pred_before, cm, x_fixed=x_fixed).numpy()
    map_after = extract_map_for_x(pred_after, cm, x_fixed=x_fixed).numpy()
    diff_before = np.abs(map_before - map_target)
    diff_after = np.abs(map_after - map_target)

    profile_target = extract_center_profile(target_hanning, x_idx=x_idx, z_idx=z_idx)
    profile_before = extract_center_profile(pred_before, x_idx=x_idx, z_idx=z_idx)
    profile_after = extract_center_profile(pred_after, x_idx=x_idx, z_idx=z_idx)

    extent = [x_elem.min(), x_elem.max(), z.max(), z.min()]
    map_vmin = 0.0
    map_vmax = 1.0
    diff_vmax = max(float(diff_before.max()), float(diff_after.max()), 1e-8)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)

    map_specs = [
        (axes[0, 0], map_before, "MLP before training"),
        (axes[0, 1], map_target, "Hanning target"),
        (axes[0, 2], diff_before, "|difference| before"),
        (axes[1, 0], map_after, "MLP after training"),
        (axes[1, 1], map_target, "Hanning target"),
        (axes[1, 2], diff_after, "|difference| after"),
    ]
    for axis, image_data, title in map_specs:
        is_diff = "difference" in title
        im = axis.imshow(
            image_data,
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=0.0 if is_diff else map_vmin,
            vmax=diff_vmax if is_diff else map_vmax,
        )
        axis.set_title(title)
        axis.set_xlabel("Element lateral position [mm]")
        axis.set_ylabel("Depth [mm]")
        fig.colorbar(im, ax=axis)

    z_value = float(z[z_idx])
    axes[2, 0].plot(x_elem, profile_target, label="Hanning", linewidth=2)
    axes[2, 0].plot(x_elem, profile_before, label="MLP before", linestyle="--")
    axes[2, 0].set_title(f"Profile before at z={z_value:.2f} mm, x=0")
    axes[2, 0].set_xlabel("Element lateral position [mm]")
    axes[2, 0].set_ylabel("Apodization weight")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()

    axes[2, 1].plot(x_elem, profile_target, label="Hanning", linewidth=2)
    axes[2, 1].plot(x_elem, profile_after, label="MLP after", linestyle="--")
    axes[2, 1].set_title(f"Profile after at z={z_value:.2f} mm, x=0")
    axes[2, 1].set_xlabel("Element lateral position [mm]")
    axes[2, 1].set_ylabel("Apodization weight")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()

    axes[2, 2].plot(history.history["loss"], color="black")
    axes[2, 2].set_title("Training loss")
    axes[2, 2].set_xlabel("Epoch")
    axes[2, 2].set_ylabel("MSE")
    axes[2, 2].grid(True, alpha=0.3)

    fig.suptitle("INR fit to dynamic Hanning apodization", fontsize=14)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.show()


tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

kp, cm, cfg_path = load_coordinate_manager()
print(f"Loaded configuration from: {cfg_path}")
print(f"Kernel params: n_elements={kp.n_elements}, nx={kp.nx}, nz={kp.nz}, f_number={kp.f_number}")
print(f"CoordinateManager shape (n_elem, nz, nx): {cm.shape}")

coords = cm.get_coordinates_1d(scaled=False)
x_coords = np.asarray(coords["x"])
z_coords = np.asarray(coords["z"])
x_fixed = 0.0
x_idx = int(np.argmin(np.abs(x_coords - x_fixed)))
z_idx = cm.nz // 2

target_hanning = compute_dynamic_apodizations_tf(
    cm,
    kp.f_number,
    methods=("hanning",),
    scaled=False,
)["hanning"]
features_flat = tf.cast(cm.get_features_flat(scaled=SCALED_FEATURES), tf.float32)
target_flat = tf.reshape(tf.cast(target_hanning, tf.float32), [-1, 1])
features_train, target_train = select_training_subset(
    features_flat=features_flat,
    target_flat=target_flat,
    subset_fraction=TRAIN_SUBSET_FRACTION,
)

print(f"Features flat shape: {features_flat.shape}, scaled={SCALED_FEATURES}")
print(f"Target flat shape: {target_flat.shape}")
print(f"Training features shape: {features_train.shape}")
print(f"Using {features_train.shape[0]} of {features_flat.shape[0]} points for training")
print(f"Map extraction at x={x_coords[x_idx]:.4f} mm")
print(f"Profile extraction at z={z_coords[z_idx]:.4f} mm, x={x_coords[x_idx]:.4f} mm")

model = build_mlp()
model.summary()

pred_before = predict_grid(model, features_flat, cm.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_train, target_train))
dataset = dataset.shuffle(buffer_size=int(features_train.shape[0]), reshuffle_each_iteration=True)
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mse",
)
history = model.fit(dataset, epochs=N_EPOCHS, verbose=1)

pred_after = predict_grid(model, features_flat, cm.shape)

plot_results(
    cm=cm,
    target_hanning=target_hanning,
    pred_before=pred_before,
    pred_after=pred_after,
    coords=coords,
    x_fixed=x_fixed,
    x_idx=x_idx,
    z_idx=z_idx,
    history=history,
    figure_path=FIGURE_PATH,
)

print(f"Saved figure to: {FIGURE_PATH}")