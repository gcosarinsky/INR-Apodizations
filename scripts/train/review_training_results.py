"""Post-training reviewer: Interactive visualization of INR trained results.

This script loads a trained INR model and generated predictions over the
validation or test set, allowing interactive navigation and analysis.

Configuration: Uses the same train_config.yml as the training script.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import tensorflow as tf
import yaml

# Add script folders to path for local imports when run directly
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from inr_apodizations import config
import inr_apodizations.experiment_helpers as helpers
from inr_apodizations.config import PROJ_ROOT
from inr_apodizations.interactive_navigator import InteractiveImageNavigator
from inr_apodizations.apodizations import compute_dynamic_apodizations_tf
from inr_apodizations.utils import to_db
from inr_apodizations.kernels import KernelParameters2D
from inr_apodizations.plots import _prepare_comparison_images_db
import matplotlib.pyplot as plt
import math


def generate_inr_comparison_figure(
    example_idx: int,
    delayed_samples: tf.Tensor,
    targets: np.ndarray,
    uniform_image: np.ndarray,
    inr_before_image: np.ndarray,
    inr_after_image: np.ndarray,
    kp: KernelParameters2D,
    cmap: str = "gray",
    vmin_db: float = -60.0,
    vmax_db: float = 0.0,
    normalize_each_image: bool = False,
) -> tuple[plt.Figure, dict]:
    """Generate an INR training comparison figure for a single validation example.

    Args:
        example_idx: Index of the validation example.
        delayed_samples: Delayed samples tensor (n_examples, n_elem, nz, nx) as complex64.
        targets: Targets array (n_examples, nz, nx).
        uniform_image: Uniform apodized DAS image output.
        inr_before_image: INR prediction before training.
        inr_after_image: INR prediction after training.
        kp: KernelParameters2D instance.
        cmap: Matplotlib colormap.
        vmin_db: Lower dB display bound.
        vmax_db: Upper dB display bound.
        normalize_each_image: If True, each panel uses its own max for dB ref.

    Returns:
        Tuple of (matplotlib.figure.Figure, metadata_dict).
    """
    # Prepare images for dB conversion
    images_linear = {
        "Uniform": np.abs(uniform_image),
        "INR Before": np.abs(inr_before_image),
        "INR After": np.abs(inr_after_image),
        "Target": np.abs(np.asarray(targets)),
    }

    # Convert to dB
    images_db = _prepare_comparison_images_db(images_linear, normalize_each=normalize_each_image)

    # Create figure with 4 panels (or dynamic layout)
    n_panels = len(images_db)
    ncols = 2 if n_panels > 1 else 1
    nrows = math.ceil(n_panels / ncols)
    width_per_panel = 5
    height_per_panel = 5

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width_per_panel * ncols, height_per_panel * nrows),
        sharex=True,
        sharey=True,
    )

    if n_panels == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    extent = kp.get_imshow_extent()
    first_im = None
    panel_names = list(images_db.keys())

    for idx, name in enumerate(panel_names):
        ax = axes[idx]
        im = ax.imshow(
            images_db[name],
            cmap=cmap,
            vmin=vmin_db,
            vmax=vmax_db,
            extent=extent,
            aspect="auto",
        )
        if first_im is None:
            first_im = im

        ax.set_title(f"{name} (dB)")
        ax.set_xlabel("x (mm)")
        if idx % ncols == 0:
            ax.set_ylabel("z (mm)")

    # Hide unused subplots
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"INR Training Results - Validation Example {example_idx}", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    if first_im is not None:
        cbar_ax = fig.add_axes([0.93, 0.1, 0.013, 0.8])
        fig.colorbar(first_im, cax=cbar_ax, label="dB")

    metadata = {
        "example_idx": example_idx,
        "images_db": images_db,
        "images_linear": images_linear,
    }

    return fig, metadata


# ============================================================================
# MAIN: Load trained model and enable interactive review
# ============================================================================

CONFIG_PATH = config.CONFIGS_DIR / "train_config.yml"
if not CONFIG_PATH.exists():
    print(f"Error: config file not found at {CONFIG_PATH}")
    sys.exit(1)

cfg = helpers.load_experiment_config(str(CONFIG_PATH))
review_cfg = cfg.get("results_review", {})
run_timestamp = str(review_cfg.get("run_timestamp", "")).strip()
if not run_timestamp:
    raise ValueError("results_review.run_timestamp must be set to the training run timestamp.")

# Resolve output root (where training outputs are stored)
scripts_output_root_cfg = Path(
    cfg["io"].get(
        "scripts_output_root",
        cfg["io"].get("sandbox_output_root", "scripts/outputs/train"),
    )
)
if not scripts_output_root_cfg.is_absolute():
    sandbox_root = config.PROJ_ROOT / scripts_output_root_cfg
else:
    sandbox_root = scripts_output_root_cfg

if not sandbox_root.exists():
    print(f"Error: training output root not found at {sandbox_root}")
    sys.exit(1)

artifacts_dir = sandbox_root / run_timestamp
if not artifacts_dir.exists():
    print(f"Error: artifacts directory not found at {artifacts_dir}")
    sys.exit(1)

train_config_info_path = artifacts_dir / "train_config_info.yml"
if not train_config_info_path.exists():
    print(f"Error: train_config_info.yml missing in {artifacts_dir}")
    sys.exit(1)

with train_config_info_path.open("r", encoding="utf-8") as handle:
    run_cfg = yaml.safe_load(handle) or {}
if not isinstance(run_cfg, dict):
    raise ValueError("Saved run configuration must be a mapping")

run_experiment_cfg = run_cfg.get("experiment", run_cfg)
training_cfg = run_experiment_cfg.get("training")
if training_cfg is None or "seed" not in training_cfg:
    raise ValueError("Saved run experiment config lacks training.seed")

seed = int(training_cfg["seed"])

dataset_folder_entry = run_cfg.get("dataset_folder") or run_experiment_cfg.get("io", {}).get("dataset_folder")
if dataset_folder_entry is None:
    raise ValueError("Saved run config does not expose a dataset_folder entry")

dataset_folder = Path(dataset_folder_entry)
if not dataset_folder.is_absolute():
    dataset_folder = config.PROJ_ROOT / dataset_folder
dataset_folder = str(dataset_folder)

print(f"Loading configuration from: {CONFIG_PATH}")
print(f"Reviewing training run: {run_timestamp}")
print(f"Dataset folder (from run config): {dataset_folder}")
print(f"Using training artifacts from: {artifacts_dir}")

sigma_x_override, sigma_z_override, alpha_override = helpers.get_target_regeneration_override(run_experiment_cfg)
delayed, noise, targets, gaussian_masks, info = helpers.load_delayed_samples_dataset(
    dataset_folder,
    sigma_x=sigma_x_override,
    sigma_z=sigma_z_override,
    alpha_override=alpha_override,
    load_noise=False,
)
physical_feature_set = str(
    run_experiment_cfg.get("model", {}).get("physical_feature_set", "distance_depth_edge")
)
kp, cm = helpers.build_coordinate_manager(
    dataset_folder,
    physical_feature_set=physical_feature_set,
)

train_fraction = float(training_cfg.get("train_fraction", 0.7))
train_idx, val_idx = helpers.split_train_validation_indices(
    n_examples=delayed.shape[0],
    train_fraction=train_fraction,
    seed=seed,
)

print(f"Dataset: {delayed.shape[0]} examples")
print(f"Validation set: {val_idx.shape[0]} examples (indices: {val_idx[:5]}...)")

model_path = artifacts_dir / "model.keras"
if not model_path.exists():
    alternative = artifacts_dir / "inr_model.h5"
    if alternative.exists():
        model_path = alternative
    else:
        print(f"Error: trained model not found at {artifacts_dir}")
        sys.exit(1)

print(f"\nLoading trained model from: {model_path}")
try:
    inr_model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Plot configuration (use the saved training run settings)
plot_cfg = run_experiment_cfg.get("plots", {})
normalize_each_image = bool(plot_cfg.get("normalize_each_image", False))
cmap = str(plot_cfg.get("cmap", "gray"))
vmin_db = float(plot_cfg.get("vmin_db", -60.0))
vmax_db = float(plot_cfg.get("vmax_db", 0.0))

# Determine which validation examples to review (configurable)
max_examples = review_cfg.get("max_validation_examples")
example_indices = val_idx if max_examples is None else val_idx[:int(max_examples)]

print(f"\nPrepared {len(example_indices)} validation examples for interactive review...")

review_output_root_cfg = cfg["io"].get(
    "review_output_root", "scripts/outputs/review"
)
review_output_root = Path(review_output_root_cfg)
if not review_output_root.is_absolute():
    review_output_root = config.PROJ_ROOT / review_output_root

review_output_dir = review_output_root / artifacts_dir.name
review_output_dir.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {review_output_dir}")

# Precompute INR apodization weights (constant across all examples)
print("Precomputing INR apodization weights...")
scaled_features = bool(run_experiment_cfg.get("model", {}).get("scaled_features", True))
features_grid = cm.get_features_grid(scaled=scaled_features)
features_flat = tf.reshape(features_grid, (-1, cm.n_physical_features))
coords_input = tf.cast(features_flat, tf.float32)

baseline_f_number = float(training_cfg.get("baseline_f_number", 0.75))
hanning_weights = compute_dynamic_apodizations_tf(
    cm, f_number=baseline_f_number, methods=("hanning",), scaled=False
)["hanning"]

inr_weights_pred = inr_model(coords_input, training=False)
inr_after_weights = tf.reshape(inr_weights_pred, (kp.n_elements, kp.nz, kp.nx))
print("Apodization weights computed.")


# Define content generator (closure capturing precomputed weights)
def generate_inr_review_content(fig: plt.Figure, ax: plt.Axes, example_idx: int) -> None:
    """Generate INR training review panel for a single validation example.
    
    This is a closure capturing the outer scope variables.
    """
    # Clear previous content
    fig.clear()

    # Create subplots
    n_panels = 4
    ncols = 2
    nrows = 2

    axes_list = []
    for idx in range(n_panels):
        ax_panel = fig.add_subplot(nrows, ncols, idx + 1)
        axes_list.append(ax_panel)

    # Extract single validation example
    val_delayed = tf.convert_to_tensor(
        delayed[example_idx : example_idx + 1].astype(np.complex64, copy=False)
    )
    val_target = np.expand_dims(targets[example_idx].astype(np.float32, copy=False), axis=0)

    # Compute reference images
    uniform_weights = tf.ones((kp.n_elements, kp.nz, kp.nx), dtype=tf.float32)
    uniform_image = tf.reduce_sum(
        val_delayed * tf.cast(uniform_weights, tf.complex64), axis=1
    )[0].numpy()

    # Apply precomputed apodization weights
    hanning_image = tf.reduce_sum(
        val_delayed * tf.cast(hanning_weights, tf.complex64), axis=1
    )[0].numpy()

    inr_after_image = tf.reduce_sum(
        val_delayed * tf.cast(inr_after_weights, tf.complex64), axis=1
    )[0].numpy()

    # Convert to dB
    images_linear = {
        "Uniform": np.abs(uniform_image),
        f"Hanning f/{baseline_f_number}": np.abs(hanning_image),
        "INR After": np.abs(inr_after_image),
        "Target": np.abs(val_target[0]),
    }

    if normalize_each_image:
        images_db = {
            name: to_db(image, ref=float(np.max(image)))
            for name, image in images_linear.items()
        }
    else:
        shared_ref = max(float(np.max(image)) for image in images_linear.values())
        images_db = {name: to_db(image, ref=shared_ref) for name, image in images_linear.items()}

    # Plot
    extent = kp.get_imshow_extent()
    panel_names = ["Uniform", f"Hanning f/{baseline_f_number}", "INR After", "Target"]

    first_im = None
    for idx, name in enumerate(panel_names):
        ax_panel = axes_list[idx]
        im = ax_panel.imshow(
            images_db[name],
            cmap=cmap,
            vmin=vmin_db,
            vmax=vmax_db,
            extent=extent,
            aspect="auto",
        )
        if first_im is None:
            first_im = im
        ax_panel.set_title(f"{name} (dB)")
        ax_panel.set_xlabel("x (mm)")
        if idx % ncols == 0:
            ax_panel.set_ylabel("z (mm)")

    # Add colorbar
    if first_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(first_im, cax=cbar_ax, label="dB")

    fig.suptitle(f"INR Training Results - Validation Example {example_idx}", 
                fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.91, 0.96])


print(f"\nLaunching interactive reviewer with {len(example_indices)} validation examples...")

nav = InteractiveImageNavigator(
    example_indices=list(example_indices),
    generate_figure_content=generate_inr_review_content,
    output_dir=review_output_dir,
    prefix="inr_training_result_val",
    dpi=150,
)
nav.show()
