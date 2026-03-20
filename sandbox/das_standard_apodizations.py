"""Sandbox script: DAS reconstruction with standard apodizations on latest delayed samples.

This script:
1. Loads the latest delayed-samples dataset (delayed samples + targets).
2. Computes standard dynamic apodizations (boxcar, hanning).
3. Performs DAS on TensorFlow tensors (GPU when available).
4. Plots four images in dB: uniform, boxcar, hanning, and target.
"""

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from inr_apodizations.config import DATA_DIR
from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.kernels import KernelParameters2D
from inr_apodizations.apodizations import compute_dynamic_apodizations_tf
from inr_apodizations.utils import find_latest_dataset_folder


def to_db(image: np.ndarray, ref: float, eps: float = 1e-8) -> np.ndarray:
    """Convert a linear-magnitude image to dB using a shared reference level.

    Args:
        image: Complex or real image in linear scale.
        ref: Reference magnitude used for normalization.
        eps: Small constant to avoid log(0).

    Returns:
        Image in dB.
    """
    magnitude = np.abs(image)
    return 20.0 * np.log10((magnitude / (ref + eps)) + eps)


# User-editable parameters
example_idx = 0
vmin_db = -60.0
vmax_db = 0.0
# If True, each DAS image is normalized by its own maximum.
# If False, all DAS images share the same reference (global max).
normalize_per_image = True

latest_folder = find_latest_dataset_folder(DATA_DIR, "delayed_samples_dataset")
print(f"Using delayed-samples dataset: {latest_folder}")

cfg_path = latest_folder / "cfg_delayed_samples.npy"
delayed_path = latest_folder / "delayed_samples_dataset.npy"
targets_path = latest_folder / "targets_dataset.npy"

if not cfg_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
if not delayed_path.exists():
    raise FileNotFoundError(f"Delayed samples file not found: {delayed_path}")

cfg = np.load(cfg_path, allow_pickle=True).item()
delayed_samples_all = np.load(delayed_path)
targets_all = np.load(targets_path) if targets_path.exists() else None
if targets_all is None:
    print("Warning: targets_dataset.npy not found, target panel will be skipped.")

if delayed_samples_all.ndim != 4:
    raise ValueError(
        "Expected delayed_samples_dataset.npy shape (n_examples, n_elem, nz, nx), "
        f"got {delayed_samples_all.shape}"
    )

n_examples = delayed_samples_all.shape[0]
if example_idx < 0 or example_idx >= n_examples:
    raise ValueError(f"example_idx={example_idx} is out of range [0, {n_examples - 1}]")

print(f"Loaded delayed samples with shape={delayed_samples_all.shape}, dtype={delayed_samples_all.dtype}")
print(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')}")

kp = KernelParameters2D(cfg)
cm = CoordinateManager(kp)

# delayed_samples: (n_elem, nz, nx)
delayed_samples_np = delayed_samples_all[example_idx]
delayed_samples_tf = tf.convert_to_tensor(delayed_samples_np, dtype=tf.complex64)

apodizations = compute_dynamic_apodizations_tf(
    cm=cm,
    bfd=kp.bfd,
    methods=("boxcar", "hanning"),
    scaled=False,
)

apod_boxcar_tf = tf.cast(apodizations["boxcar"], tf.complex64)
apod_hanning_tf = tf.cast(apodizations["hanning"], tf.complex64)

# DAS: sum over receive-element axis
img_uniform_tf = tf.reduce_sum(delayed_samples_tf, axis=0)
img_boxcar_tf = tf.reduce_sum(delayed_samples_tf * apod_boxcar_tf, axis=0)
img_hanning_tf = tf.reduce_sum(delayed_samples_tf * apod_hanning_tf, axis=0)

img_uniform = img_uniform_tf.numpy()
img_boxcar = img_boxcar_tf.numpy()
img_hanning = img_hanning_tf.numpy()

if normalize_per_image:
    img_uniform_db = to_db(img_uniform, ref=np.max(np.abs(img_uniform)))
    img_boxcar_db = to_db(img_boxcar, ref=np.max(np.abs(img_boxcar)))
    img_hanning_db = to_db(img_hanning, ref=np.max(np.abs(img_hanning)))
else:
    shared_ref = max(
        np.max(np.abs(img_uniform)),
        np.max(np.abs(img_boxcar)),
        np.max(np.abs(img_hanning)),
    )
    img_uniform_db = to_db(img_uniform, ref=shared_ref)
    img_boxcar_db = to_db(img_boxcar, ref=shared_ref)
    img_hanning_db = to_db(img_hanning, ref=shared_ref)

target_np = targets_all[example_idx] if targets_all is not None else None
if target_np is not None:
    target_db = to_db(target_np, ref=np.max(np.abs(target_np)))

extent = kp.get_imshow_extent()
n_panels = 4 if target_np is not None else 3
fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels + 1, 5), sharex=True, sharey=True)

im0 = axes[0].imshow(
    img_uniform_db,
    cmap="gray",
    vmin=vmin_db,
    vmax=vmax_db,
    extent=extent,
    aspect="auto",
)
axes[0].set_title("DAS Uniform (dB)")
axes[0].set_xlabel("x (mm)")
axes[0].set_ylabel("z (mm)")

axes[1].imshow(
    img_boxcar_db,
    cmap="gray",
    vmin=vmin_db,
    vmax=vmax_db,
    extent=extent,
    aspect="auto",
)
axes[1].set_title("DAS Boxcar (dB)")
axes[1].set_xlabel("x (mm)")

axes[2].imshow(
    img_hanning_db,
    cmap="gray",
    vmin=vmin_db,
    vmax=vmax_db,
    extent=extent,
    aspect="auto",
)
axes[2].set_title("DAS Hanning (dB)")
axes[2].set_xlabel("x (mm)")

if target_np is not None:
    axes[3].imshow(
        target_db,
        cmap="gray",
        vmin=vmin_db,
        vmax=vmax_db,
        extent=extent,
        aspect="auto",
    )
    axes[3].set_title("Target (dB)")
    axes[3].set_xlabel("x (mm)")

fig.suptitle(f"Example {example_idx} - Standard apodizations")
fig.tight_layout(rect=[0, 0, 0.92, 1])
cbar_ax = fig.add_axes([0.93, 0.1, 0.013, 0.78])
fig.colorbar(im0, cax=cbar_ax, label="dB")
plt.show()
