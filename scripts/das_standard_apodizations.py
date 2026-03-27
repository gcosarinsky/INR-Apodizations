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
import matplotlib.pyplot as plt
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
import sys

# Import compute_scatterer_metrics from sandbox helpers
sys.path.insert(0, str(PROJ_ROOT / "sandbox"))
try:
    from inr_das_experiment.helpers import compute_scatterer_metrics
except ImportError:
    compute_scatterer_metrics = None
plt.ion()  # Enable interactive mode for better display control (can be turned off if not desired)
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

    required = [
        "delayed_samples_dataset",
        "example_idx",
        "methods",
        "f_number",
        "save",
        "show",
        "output_dir",
    ]
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
# show_outputs = bool(cfg_user["show"])
# if save_outputs and not show_outputs:
#     matplotlib.use("Agg")

print(f"Using YAML config: {CONFIG_PATH}")
print(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')}")

dataset_subdir = str(cfg_user["delayed_samples_dataset"]).strip()
dataset_folder = DATA_DIR / "delayed_samples_dataset" / dataset_subdir
if not dataset_folder.exists():
    raise FileNotFoundError(f"Configured dataset folder does not exist: {dataset_folder}")

print(f"Using delayed-samples dataset: {dataset_folder}")

cfg_path = dataset_folder / "cfg_delayed_samples.npy"
delayed_path = dataset_folder / "delayed_samples_dataset.npy"
targets_path = dataset_folder / "targets_dataset.npy"

if not cfg_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
if not delayed_path.exists():
    raise FileNotFoundError(f"Delayed samples file not found: {delayed_path}")

cfg = np.load(cfg_path, allow_pickle=True).item()
if "f_number" not in cfg and "bfd" in cfg:
    cfg["f_number"] = cfg["bfd"]/2.0
    print(
        "Legacy dataset cfg detected: using cfg['bfd'] to populate cfg['f_number'] "
        "before KernelParameters2D initialization."
    )

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
f_number = float(cfg_user["f_number"])
if f_number <= 0.0:
    raise ValueError(f"`f_number` must be > 0, got {f_number}")
print(
    "Using f_number from YAML: "
    f"{f_number} (kp.f_number={kp.f_number} from dataset config is ignored in this script)."
)

apods = compute_dynamic_apodizations_tf(cm=cm, f_number=f_number, methods=methods, scaled=scaled)
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

# --- Scatterer evaluation (if enabled) ---
scatterer_eval_cfg = cfg_user.get("scatterer_eval", {})
scatterer_eval_enabled = bool(scatterer_eval_cfg.get("enabled", False))
scatterer_metrics = None
scatterer_union_mask = None

if scatterer_eval_enabled:
    if compute_scatterer_metrics is None:
        print("Warning: compute_scatterer_metrics could not be imported, skipping scatterer evaluation.")
    else:
        try:
            # Require explicit RF dataset reference in config
            if "rf_dataset_name" not in cfg:
                raise ValueError(
                    "scatterer_eval.enabled=true requires 'rf_dataset_name' in dataset config. "
                    "This should be set in cfg_delayed_samples.npy from the generate process."
                )
            
            rf_dataset_name = cfg["rf_dataset_name"]
            scatterers_path = DATA_DIR / "rf_dataset_simus" / rf_dataset_name / "scatterers.npy"
            
            if not scatterers_path.exists():
                raise FileNotFoundError(
                    f"scatterers.npy not found at: {scatterers_path}\n"
                    f"Expected RF dataset: {rf_dataset_name}\n"
                    f"Ensure the RF dataset was generated and 'rf_dataset_name' in config is correct."
                )
            
            print(f"Loading scatterers from: {scatterers_path}")
            scatterers_list = np.load(scatterers_path, allow_pickle=True)
            scatterers_example = scatterers_list[example_idx]  # shape (n_scatt, 3), columns [x, z, reflectivity]
            scatterers_xy = scatterers_example[:, :2] * 1000.0  # Convert to mm
            
            radius_mm = float(scatterer_eval_cfg.get("radius_mm", 1.5))
            hist_bins = int(scatterer_eval_cfg.get("hist_bins", 50))
            
            # Evaluate each apodization method
            scatterer_metrics = {}
            methods_to_eval = ["uniform"] + [m for m in methods if m in das_images_linear]
            
            print(f"\n=== Scatterer Metrics (radius={radius_mm:.2f} mm, {len(scatterers_xy)} scatterers) ===")
            
            for method_name in methods_to_eval:
                das_image = np.abs(das_images_linear[method_name])
                metrics = compute_scatterer_metrics(
                    das_image,
                    scatterers_xy,
                    cm,
                    radius_mm=radius_mm,
                    return_masks=(method_name == "uniform"),
                    return_background_hist=True,
                    hist_bins=hist_bins,
                )
                scatterer_metrics[method_name] = metrics

                if method_name == "uniform" and "background_mask" in metrics:
                    scatterer_union_mask = ~metrics["background_mask"]
                
                peak_amps = metrics["peak_amplitudes"]
                bg_rms = metrics["background_rms"]
                print(f"  {method_name:12s}: peak_mean={peak_amps.mean():.4f} "
                      f"peak_std={peak_amps.std():.4f} peak_max={peak_amps.max():.4f} "
                      f"bg_rms={bg_rms:.4f}")
            
            print()
        except Exception as e:
            print(f"Error during scatterer evaluation: {e}")
            import traceback
            traceback.print_exc()

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

# --- Scatterer background noise histograms ---
if scatterer_metrics is not None and len(scatterer_metrics) > 0:
    if scatterer_union_mask is not None:
        fig_uniform_mask, axes_uniform_mask = plt.subplots(
            1,
            3,
            figsize=(18, 5),
            sharex=True,
            sharey=True,
        )
        im_uniform = axes_uniform_mask[0].imshow(
            das_images_db["uniform"],
            cmap=cmap,
            vmin=vmin_db,
            vmax=vmax_db,
            extent=extent,
            aspect="auto",
        )
        axes_uniform_mask[0].set_title("DAS Uniform (dB)")
        axes_uniform_mask[0].set_xlabel("x (mm)")
        axes_uniform_mask[0].set_ylabel("z (mm)")

        axes_uniform_mask[1].imshow(
            scatterer_union_mask.astype(np.float32),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            extent=extent,
            aspect="auto",
        )
        axes_uniform_mask[1].set_title("Total Scatterer Mask")
        axes_uniform_mask[1].set_xlabel("x (mm)")

        axes_uniform_mask[2].imshow(
            das_images_db["uniform"],
            cmap=cmap,
            vmin=vmin_db,
            vmax=vmax_db,
            extent=extent,
            aspect="auto",
        )
        overlay_mask = np.ma.masked_where(
            ~scatterer_union_mask,
            scatterer_union_mask.astype(np.float32),
        )
        axes_uniform_mask[2].imshow(
            overlay_mask,
            cmap="autumn",
            vmin=0.0,
            vmax=1.0,
            extent=extent,
            aspect="auto",
            alpha=0.35,
        )
        axes_uniform_mask[2].set_title("Uniform DAS + Scatterer Mask Overlay")
        axes_uniform_mask[2].set_xlabel("x (mm)")

        fig_uniform_mask.suptitle(
            f"Uniform DAS, scatterer mask, and overlay (example {example_idx})"
        )
        fig_uniform_mask.tight_layout()
        cbar_uniform = fig_uniform_mask.colorbar(
            im_uniform,
            ax=axes_uniform_mask[0],
            fraction=0.046,
            pad=0.04,
        )
        cbar_uniform.set_label("dB")

        if save_outputs:
            out_uniform_mask = output_dir / f"uniform_and_scatterer_mask_example{example_idx}.png"
            fig_uniform_mask.savefig(out_uniform_mask, dpi=dpi, bbox_inches="tight")
            print(f"Saved {out_uniform_mask}")

    fig_hist, axes_hist = plt.subplots(1, 1, figsize=(10, 5))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    
    for idx, method_name in enumerate(sorted(scatterer_metrics.keys())):
        metrics = scatterer_metrics[method_name]
        hist_counts = metrics.get("background_hist_counts")
        hist_edges = metrics.get("background_hist_edges")
        
        if hist_counts is not None and hist_edges is not None:
            bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2.0
            color = colors[idx % len(colors)]
            axes_hist.plot(bin_centers, hist_counts, marker="o", label=method_name, 
                          color=color, linewidth=2, markersize=4, alpha=0.7)
            axes_hist.fill_between(bin_centers, hist_counts, alpha=0.2, color=color)
    
    axes_hist.set_xlabel("Amplitude (linear)")
    axes_hist.set_ylabel("Frequency")
    axes_hist.set_title(f"Background noise histograms (example {example_idx})")
    axes_hist.grid(True, alpha=0.3)
    axes_hist.legend()
    fig_hist.tight_layout()
    
    if save_outputs:
        out_hist = output_dir / f"background_noise_hist_example{example_idx}.png"
        fig_hist.savefig(out_hist, dpi=dpi, bbox_inches="tight")
        print(f"Saved {out_hist}")

    # --- Scatterer-wise peak amplitude scatter plot (each method vs uniform) ---
    methods_to_compare = [m for m in scatterer_metrics if m != "uniform"]
    n_scatter_cols = len(methods_to_compare)
    if n_scatter_cols > 0 and "uniform" in scatterer_metrics:
        uniform_peaks = scatterer_metrics["uniform"]["peak_amplitudes"]
        uniform_bg_rms = float(scatterer_metrics["uniform"]["background_rms"])
        uniform_image_max = max(float(np.max(np.abs(das_images_linear["uniform"]))), 1e-12)
        fig_scatter, axes_scatter = plt.subplots(
            1, n_scatter_cols, figsize=(5 * n_scatter_cols, 5), squeeze=False
        )
        for col_idx, method_name in enumerate(methods_to_compare):
            ax = axes_scatter[0, col_idx]
            method_peaks = scatterer_metrics[method_name]["peak_amplitudes"]
            method_bg_rms = float(scatterer_metrics[method_name]["background_rms"])
            ax.scatter(uniform_peaks, method_peaks, s=30, alpha=0.7)
            ax_max = max(float(uniform_peaks.max()), float(method_peaks.max()))
            ax_max = max(ax_max, uniform_bg_rms, method_bg_rms)
            ax.plot([0, ax_max], [0, ax_max], color="red", linewidth=1, linestyle="--", label="y = x")
            ax.scatter(
                [uniform_bg_rms],
                [method_bg_rms],
                s=110,
                marker="D",
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="bg_rms",
                zorder=5,
            )
            ax.set_xlabel("Uniform peak amplitude")
            ax.set_ylabel(f"{method_name} peak amplitude")
            ax.set_title(
                f"{method_name} vs uniform ({len(uniform_peaks)} scatterers)"
            )
            ax.set_aspect("equal")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig_scatter.suptitle(f"Peak amplitude comparison (example {example_idx})")
        fig_scatter.tight_layout()

        if save_outputs:
            out_scatter = output_dir / f"scatter_peak_amp_example{example_idx}.png"
            fig_scatter.savefig(out_scatter, dpi=dpi, bbox_inches="tight")
            print(f"Saved {out_scatter}")

        fig_scatter_norm, axes_scatter_norm = plt.subplots(
            1, n_scatter_cols, figsize=(5 * n_scatter_cols, 5), squeeze=False
        )
        uniform_peaks_norm = uniform_peaks / uniform_image_max
        uniform_bg_rms_norm = uniform_bg_rms / uniform_image_max
        for col_idx, method_name in enumerate(methods_to_compare):
            ax = axes_scatter_norm[0, col_idx]
            method_peaks = scatterer_metrics[method_name]["peak_amplitudes"]
            method_bg_rms = float(scatterer_metrics[method_name]["background_rms"])
            method_image_max = max(float(np.max(np.abs(das_images_linear[method_name]))), 1e-12)
            method_peaks_norm = method_peaks / method_image_max
            method_bg_rms_norm = method_bg_rms / method_image_max
            ax.scatter(uniform_peaks_norm, method_peaks_norm, s=30, alpha=0.7)
            ax.plot([0, 1], [0, 1], color="red", linewidth=1, linestyle="--", label="y = x")
            ax.scatter(
                [uniform_bg_rms_norm],
                [method_bg_rms_norm],
                s=110,
                marker="D",
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="bg_rms",
                zorder=5,
            )
            ax.set_xlabel("Uniform peak amplitude / image max")
            ax.set_ylabel(f"{method_name} peak amplitude / image max")
            ax.set_title(
                f"{method_name} vs uniform normalized ({len(uniform_peaks_norm)} scatterers)"
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_aspect("equal")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig_scatter_norm.suptitle(
            f"Normalized peak amplitude comparison (example {example_idx})"
        )
        fig_scatter_norm.tight_layout()

        if save_outputs:
            out_scatter_norm = output_dir / f"scatter_peak_amp_normalized_example{example_idx}.png"
            fig_scatter_norm.savefig(out_scatter_norm, dpi=dpi, bbox_inches="tight")
            print(f"Saved {out_scatter_norm}")


