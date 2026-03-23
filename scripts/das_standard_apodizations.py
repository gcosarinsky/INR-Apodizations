"""Sandbox script: unified apodization visualization and DAS reconstruction.

This script reads all runtime options from `configs/das_apodizations.yml`.
No CLI arguments are used.

Workflow:
1. Load latest delayed-samples dataset.
2. Compute dynamic apodizations for selected methods.
3. Plot apodization maps/profiles.
4. Compute DAS images (uniform + apodized methods + optional target).
5. Save and/or display figures according to YAML configuration.
"""

from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf
import yaml

from inr_apodizations.apodizations import (
    compute_dynamic_apodizations_tf,
    extract_map_for_x,
    extract_profile_for_z,
)
from inr_apodizations.config import PROJ_ROOT, DATA_DIR, CONFIGS_DIR
from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.kernels import KernelParameters2D
from inr_apodizations.utils import find_latest_dataset_folder

CONFIG_PATH = CONFIGS_DIR / "das_standard_apodizations.yml"


def load_yaml_config(config_path: Path) -> dict:
    """Load and validate YAML configuration for the combined sandbox script.

    Args:
        config_path: Absolute path to YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If YAML config file does not exist.
        ValueError: If YAML is empty or does not contain expected fields.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Create it from configs/das_standard_apodizations_example.yml"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("YAML config must be a dictionary.")

    required = ["dataset_subdir", "example_idx", "methods", "save", "show", "output_dir"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing YAML keys: {missing}")

    if not isinstance(config["methods"], list) or len(config["methods"]) == 0:
        raise ValueError("`methods` must be a non-empty YAML list.")

    if not config["save"] and not config["show"]:
        raise ValueError("At least one output mode must be enabled: save=true or show=true.")

    return config


def to_db(image: np.ndarray, ref: float, eps: float = 1e-8) -> np.ndarray:
    """Convert linear magnitude image to dB.

    Args:
        image: Complex or real image in linear domain.
        ref: Positive reference magnitude.
        eps: Small epsilon to avoid numerical issues.

    Returns:
        Magnitude image in dB.
    """
    magnitude = np.abs(image)
    return 20.0 * np.log10((magnitude / (ref + eps)) + eps)


cfg_user = load_yaml_config(CONFIG_PATH)

save_outputs = bool(cfg_user["save"])
show_outputs = bool(cfg_user["show"])
if save_outputs and not show_outputs:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

print(f"Using YAML config: {CONFIG_PATH}")
print(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')}")

latest_folder = find_latest_dataset_folder(DATA_DIR, cfg_user["dataset_subdir"])
print(f"Using delayed-samples dataset: {latest_folder}")

cfg_path = latest_folder / "cfg_delayed_samples.npy"
delayed_path = latest_folder / "delayed_samples_dataset.npy"
targets_path = latest_folder / "targets_dataset.npy"

if not cfg_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
if not delayed_path.exists():
    raise FileNotFoundError(f"Delayed samples file not found: {delayed_path}")

cfg = np.load(cfg_path, allow_pickle=True).item()
use_mmap = bool(cfg_user.get("use_mmap", True))
if use_mmap:
    delayed_samples_all = np.load(delayed_path, mmap_mode="r")
    targets_all = np.load(targets_path, mmap_mode="r") if targets_path.exists() else None
else:
    delayed_samples_all = np.load(delayed_path)
    targets_all = np.load(targets_path) if targets_path.exists() else None

if targets_all is None:
    print("Warning: targets_dataset.npy not found, target panel will be skipped.")

if delayed_samples_all.ndim != 4:
    raise ValueError(
        "Expected delayed_samples_dataset.npy shape (n_examples, n_elem, nz, nx), "
        f"got {delayed_samples_all.shape}"
    )

example_idx = int(cfg_user["example_idx"])
n_examples = delayed_samples_all.shape[0]
if example_idx < 0 or example_idx >= n_examples:
    raise ValueError(f"example_idx={example_idx} is out of range [0, {n_examples - 1}]")

kp = KernelParameters2D(cfg)
cm = CoordinateManager(kp)

methods = [str(method).strip().lower() for method in cfg_user["methods"]]
scaled = bool(cfg_user.get("scaled", False))
bfd = kp.bfd

apods = compute_dynamic_apodizations_tf(cm=cm, bfd=bfd, methods=methods, scaled=scaled)
if len(apods) == 0:
    raise ValueError(f"No valid apodization methods computed from methods={methods}")

output_dir = PROJ_ROOT / str(cfg_user["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

dpi = int(cfg_user.get("dpi", 150))
cmap = str(cfg_user.get("cmap", "gray"))
x_fixed = float(cfg_user.get("x_fixed", 0.0))
z_fixed = float(cfg_user.get("z_fixed", 20.0))

coords = cm.get_coordinates_1d(scaled=scaled)
x_elems = np.asarray(coords["x_elem"])
z_coords = np.asarray(coords["z"])
map_extent = (float(x_elems[0]), float(x_elems[-1]), float(z_coords[-1]), float(z_coords[0]))

for method_name, apod_tensor in apods.items():
    map_z_elem = extract_map_for_x(apod_tensor, cm, x_fixed=x_fixed, scaled=scaled).numpy()
    profile = extract_profile_for_z(
        apod_tensor,
        cm,
        z_fixed=z_fixed,
        x_fixed=x_fixed,
        scaled=scaled,
    ).numpy()

    if not (np.all(profile >= -1e-6) and np.all(profile <= 1.0 + 1e-6)):
        raise ValueError(f"Apodization profile out of expected range [0, 1] for {method_name}")

    fig_map, ax_map = plt.subplots(1, 1, figsize=(8, 5))
    im_map = ax_map.imshow(
        map_z_elem,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        extent=map_extent,
    )
    ax_map.set_xlabel("Element lateral coordinate (mm)")
    ax_map.set_ylabel("Depth z (mm)")
    ax_map.set_title(f"Apodization map ({method_name}) at x={x_fixed:.2f} mm")
    fig_map.colorbar(im_map, ax=ax_map, label="Apodization")
    fig_map.tight_layout()

    # collect profiles to plot them together later (single axis with labels)
    if "_collected_profiles" not in locals():
        _collected_profiles = {}
    _collected_profiles[method_name] = profile

    if save_outputs:
        out_map = output_dir / f"map_{method_name}_x{x_fixed:.2f}_example{example_idx}.png"
        fig_map.savefig(out_map, dpi=dpi, bbox_inches="tight")
        print(f"Saved {out_map}")

    if not show_outputs:
        plt.close(fig_map)

delayed_samples_np = np.asarray(delayed_samples_all[example_idx])
delayed_samples_tf = tf.convert_to_tensor(delayed_samples_np, dtype=tf.complex64)

das_images_linear = {"uniform": tf.reduce_sum(delayed_samples_tf, axis=0).numpy()}
for method_name, apod_tensor in apods.items():
    weighted = delayed_samples_tf * tf.cast(apod_tensor, tf.complex64)
    das_images_linear[method_name] = tf.reduce_sum(weighted, axis=0).numpy()

normalize_per_image = bool(cfg_user.get("normalize_per_image", True))
if normalize_per_image:
    das_images_db = {
        name: to_db(image, ref=float(np.max(np.abs(image))))
        for name, image in das_images_linear.items()
    }
else:
    shared_ref = max(float(np.max(np.abs(image))) for image in das_images_linear.values())
    das_images_db = {name: to_db(image, ref=shared_ref) for name, image in das_images_linear.items()}

target_np = np.asarray(targets_all[example_idx]) if targets_all is not None else None
if target_np is not None:
    target_db = to_db(target_np, ref=float(np.max(np.abs(target_np))))

if bool(cfg_user.get("save_npy", False)):
    arrays_path = output_dir / f"das_arrays_example{example_idx}.npz"
    np.savez(
        arrays_path,
        **{f"das_db_{name}": image for name, image in das_images_db.items()},
        target_db=target_db if target_np is not None else np.array([]),
    )
    print(f"Saved {arrays_path}")

# --- Combined profiles plot (single axis with labels/legend) ---
if "_collected_profiles" in locals() and len(_collected_profiles) > 0:
    fig_profile, ax_profile = plt.subplots(1, 1, figsize=(8, 4))
    for method_name, profile in _collected_profiles.items():
        ax_profile.plot(x_elems, profile, marker="o", label=method_name)
    ax_profile.set_xlabel("Element lateral coordinate (mm)")
    ax_profile.set_ylabel("Apodization")
    ax_profile.set_title(f"Apodization profiles at z={z_fixed:.2f} mm, x={x_fixed:.2f} mm")
    ax_profile.grid(True)
    ax_profile.legend(title="Method")
    fig_profile.tight_layout()

    if save_outputs:
        out_profiles = output_dir / f"profiles_combined_z{z_fixed:.2f}_x{x_fixed:.2f}_example{example_idx}.png"
        fig_profile.savefig(out_profiles, dpi=dpi, bbox_inches="tight")
        print(f"Saved {out_profiles}")

    if show_outputs:
        plt.show()
    else:
        plt.close(fig_profile)

extent = kp.get_imshow_extent()
vmin_db = float(cfg_user.get("vmin_db", -60.0))
vmax_db = float(cfg_user.get("vmax_db", 0.0))

ordered_names = ["uniform"] + [name for name in methods if name in das_images_db and name != "uniform"]
if len(ordered_names) != len(das_images_db):
    for extra_name in das_images_db:
        if extra_name not in ordered_names:
            ordered_names.append(extra_name)

n_panels = len(ordered_names) + (1 if target_np is not None else 0)
fig_das, axes = plt.subplots(
    1,
    n_panels,
    figsize=(5 * n_panels + 1, 5),
    sharex=True,
    sharey=True,
)
if n_panels == 1:
    axes = [axes]

first_im = None
for idx, image_name in enumerate(ordered_names):
    title_name = "Uniform" if image_name == "uniform" else image_name.capitalize()
    current_im = axes[idx].imshow(
        das_images_db[image_name],
        cmap=cmap,
        vmin=vmin_db,
        vmax=vmax_db,
        extent=extent,
        aspect="auto",
    )
    if first_im is None:
        first_im = current_im
    axes[idx].set_title(f"DAS {title_name} (dB)")
    axes[idx].set_xlabel("x (mm)")
    if idx == 0:
        axes[idx].set_ylabel("z (mm)")

if target_np is not None:
    axes[-1].imshow(
        target_db,
        cmap=cmap,
        vmin=vmin_db,
        vmax=vmax_db,
        extent=extent,
        aspect="auto",
    )
    axes[-1].set_title("Target (dB)")
    axes[-1].set_xlabel("x (mm)")

fig_das.suptitle(f"Example {example_idx} - Standard apodizations")
fig_das.tight_layout(rect=[0, 0, 0.92, 1])
cbar_ax = fig_das.add_axes([0.93, 0.1, 0.013, 0.78])
fig_das.colorbar(first_im, cax=cbar_ax, label="dB")

if save_outputs:
    out_das = output_dir / f"das_panel_example{example_idx}.png"
    fig_das.savefig(out_das, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out_das}")

if show_outputs:
    plt.show()
else:
    plt.close(fig_das)
