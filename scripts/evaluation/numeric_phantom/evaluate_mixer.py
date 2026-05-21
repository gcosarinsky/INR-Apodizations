"""Evaluate numeric phantom with a trained mixer model against baseline apodizations.

This script mirrors the numeric phantom INR evaluation flow but loads split
mixer artifacts from ``scripts/outputs/train_mixer``:
- ``model.keras`` for the INR apodization MLP.
- ``mixer_combiner_weights.npz`` for the pixel-wise mixer combiner.

The evaluator reconstructs ``DasInrApodMixer`` from those artifacts and compares
the final combined output against selected classical references.

Design decisions:
- Keep script simple (no argparse/main), driven by
  ``configs/numeric_phantom_evaluation_mixer_config.yml``.
- Reuse existing profile/FWHM/SNR utilities for consistency with INR reports.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import csv
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from inr_apodizations.apodizations import (
    compute_dynamic_apodizations_tf,
    compute_nsi_from_delayed_samples_numpy,
)
from inr_apodizations.config import CONFIGS_DIR
from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.evaluation.config_utils import (
    build_grid_reflector_points,
    resolve_reflector_indices,
)
from inr_apodizations.evaluation import compute_reflector_snr, compute_scatterer_metrics
from inr_apodizations.evaluation.io_utils import (
    extract_mixer_train_metadata,
    load_config_yaml,
    resolve_latest_delayed_samples_path,
    resolve_mixer_artifacts,
    resolve_project_path,
)
from inr_apodizations.evaluation.profiles import compute_fwhm_batch, extract_reflector_profiles
from inr_apodizations.kernels import KernelParameters2D
from inr_apodizations.modeling.das_models import DasInrApodMixer
import inr_apodizations.experiment_helpers as helpers

plt.ion()


def _build_coordinate_manager(
    dataset_folder: Path,
    sim_cfg: dict[str, Any],
    bf_cfg: dict[str, Any],
    n_elements: int,
    physical_feature_set: str,
    physical_feature_components: list[str] | None,
) -> tuple[KernelParameters2D, CoordinateManager]:
    try:
        kp, cm = helpers.build_coordinate_manager(
            str(dataset_folder),
            physical_feature_set=physical_feature_set,
            physical_feature_components=physical_feature_components,
        )
        return kp, cm
    except FileNotFoundError:
        kp_cfg = {
            "fs": sim_cfg.get("fs", 20.832),
            "c1": sim_cfg.get("c1", 1.54),
            "pitch": sim_cfg.get("pitch", 0.3),
            "f1": bf_cfg.get("f1", 1.0),
            "f2": bf_cfg.get("f2", 10.0),
            "f_number": bf_cfg.get("f_number", 1.0),
            "x_step": bf_cfg.get("x_step", 0.1),
            "z_step": bf_cfg.get("z_step", 0.1),
            "t_start": sim_cfg.get("t_start", 0.0),
            "taps": bf_cfg.get("taps", 62),
            "n_batch": bf_cfg.get("n_batch", 0),
            "n_elements": int(sim_cfg.get("n_elements", n_elements)),
            "n_ch": bf_cfg.get("n_ch", int(sim_cfg.get("n_elements", n_elements))),
            "n_angles": int(len(np.arange(*sim_cfg.get("angles", [-5, 6, 1])))),
            "n_samples": int(sim_cfg.get("n_samples") or 1000),
            "roi_user": bf_cfg.get("roi_user"),
            "blocksize_img": tuple(bf_cfg.get("blocksize_img", [32, 8])),
        }
        kp = KernelParameters2D(kp_cfg)
        cm = CoordinateManager(
            kp,
            physical_feature_set=physical_feature_set,
            physical_feature_components=physical_feature_components,
        )
        return kp, cm


def _db_image(image: np.ndarray) -> np.ndarray:
    ref = float(np.max(np.abs(image)))
    return 20.0 * np.log10(np.abs(image) / (ref + 1e-12) + 1e-12)


def _subplot_grid(n_items: int, max_cols: int = 2) -> tuple[int, int]:
    cols = min(max_cols, max(1, n_items))
    rows = int(math.ceil(n_items / cols))
    return rows, cols


cfg = load_config_yaml(CONFIGS_DIR / "numeric_phantom_evaluation_mixer_config.yml")
io_cfg = cfg.get("io", {})
sim_cfg = cfg.get("simulation", {})
bf_cfg = cfg.get("beamforming", {})
mixer_cfg = cfg.get("mixer_model", {})
nsi_cfg = cfg.get("nsi", {})
nsi_enabled = bool(nsi_cfg.get("enabled", True))
nsi_f_number = float(nsi_cfg.get("f_number", bf_cfg.get("f_number", 1.0)))
nsi_dc = float(nsi_cfg.get("dc", 0.05))
nsi_normalize = bool(nsi_cfg.get("normalize_by_n_subap", False))

plot_cfg = cfg.get("plots", {})
fontsize_legend = int(plot_cfg.get("legend_fontsize", 8))
fontsize_title = int(plot_cfg.get("title_fontsize", 10))
fontsize_axis = int(plot_cfg.get("axis_fontsize", 10))

if not bool(mixer_cfg.get("eval_combined", True)):
    sys.exit("Error: `mixer_model.eval_combined` must be true for this evaluator.")

model_file, combiner_weights_file, train_info_file, model_run_dir = resolve_mixer_artifacts(io_cfg.get("model_path", ""))
scaled_features, n_apodizations, physical_feature_set, physical_feature_components = extract_mixer_train_metadata(train_info_file)
train_info = load_config_yaml(train_info_file)
resolved_mixer_cfg = train_info.get("resolved_mixer", {}) if isinstance(train_info, dict) else {}
forced_boxcar_cfg = resolved_mixer_cfg.get("forced_boxcar", {}) if isinstance(resolved_mixer_cfg, dict) else {}
print("Using mixer run:", model_run_dir)
print("Using model:", model_file)
print("Using combiner weights:", combiner_weights_file)
print("Using train metadata:", train_info_file)

apodization_model = tf.keras.models.load_model(str(model_file), compile=False)
out_shape = apodization_model.output_shape
if isinstance(out_shape, tuple):
    output_last_dim = int(out_shape[-1])
else:
    output_last_dim = int(out_shape[0][-1])
if output_last_dim != n_apodizations:
    raise ValueError(
        "Mixer MLP output shape does not match n_apodizations from train metadata: "
        f"model output last dim={output_last_dim}, n_apodizations={n_apodizations}"
    )

delayed_samples_path, latest_simulation_run = resolve_latest_delayed_samples_path(io_cfg)
print("Selected simulation run:", latest_simulation_run)
print("Using delayed samples:", delayed_samples_path)

delayed = np.load(delayed_samples_path)
delayed0 = delayed[0]
n_elements = int(delayed0.shape[0])
dataset_folder = delayed_samples_path.parent

kp, cm = _build_coordinate_manager(
    dataset_folder,
    sim_cfg,
    bf_cfg,
    n_elements=n_elements,
    physical_feature_set=physical_feature_set,
    physical_feature_components=physical_feature_components,
)

feature_chunk_size = int(mixer_cfg.get("feature_chunk_size", 65536))
features_grid = cm.get_features_grid(scaled=scaled_features)
mixer_model = DasInrApodMixer(
    apodization_model=apodization_model,
    features_grid=features_grid,
    feature_chunk_size=feature_chunk_size,
    n_apodizations=n_apodizations,
    weight_regularization_enabled=False,
)

delayed_batch = tf.convert_to_tensor(np.expand_dims(delayed0, axis=0).astype(np.complex64))
_warmup_image, _warmup_weights_grid = mixer_model.reconstruct_image(delayed_batch, training=False)

combiner_weights_npz = np.load(combiner_weights_file)
if "kernel" not in combiner_weights_npz or "bias" not in combiner_weights_npz:
    raise ValueError(
        "Invalid mixer combiner file. Expected `kernel` and `bias` arrays in: "
        f"{combiner_weights_file}"
    )

combiner_kernel = np.asarray(combiner_weights_npz["kernel"], dtype=np.float32)
combiner_bias = np.asarray(combiner_weights_npz["bias"], dtype=np.float32)
expected_kernel_shape = (n_apodizations, 1)
expected_bias_shape = (1,)
if combiner_kernel.shape != expected_kernel_shape:
    raise ValueError(
        "Mixer combiner kernel shape mismatch: "
        f"got {combiner_kernel.shape}, expected {expected_kernel_shape}"
    )
if combiner_bias.shape != expected_bias_shape:
    raise ValueError(
        "Mixer combiner bias shape mismatch: "
        f"got {combiner_bias.shape}, expected {expected_bias_shape}"
    )
mixer_model.pixel_combiner.set_weights([combiner_kernel, combiner_bias])

dyn_apods = compute_dynamic_apodizations_tf(
    cm,
    float(bf_cfg.get("f_number", 1.0)),
    methods=("hanning",),
    scaled=False,
)
if "hanning" not in dyn_apods:
    raise RuntimeError("Could not compute hanning apodization from CoordinateManager.")

hanning_w = dyn_apods["hanning"].numpy()
hanning_img = (delayed0 * hanning_w).sum(axis=0)

mixer_combined_batch, _weights_grid = mixer_model.reconstruct_image(delayed_batch, training=False)
mixer_combined_img = np.asarray(mixer_combined_batch.numpy()[0], dtype=np.float64)

images = {
    "hanning": np.asarray(hanning_img, dtype=np.float64),
    "mixer_combined": mixer_combined_img,
}

if nsi_enabled:
    coords = cm.get_coordinates_1d(scaled=False)
    x_coords = np.asarray(coords["x"], dtype=np.float32)
    z_coords = np.asarray(coords["z"], dtype=np.float32)
    x_elem_coords = np.asarray(coords["x_elem"], dtype=np.float32)

    _img_sum, _img_diff, nsi_img = compute_nsi_from_delayed_samples_numpy(
        delayed_samples=np.asarray(delayed0, dtype=np.complex64),
        x_coords=x_coords,
        z_coords=z_coords,
        x_elem_coords=x_elem_coords,
        f_number=nsi_f_number,
        dc=nsi_dc,
        normalize_by_n_subap=nsi_normalize,
    )
    if np.isnan(nsi_img).any() or np.isinf(nsi_img).any():
        print("Warning: NSI image contains NaN/Inf values.")
    images["nsi"] = np.asarray(nsi_img, dtype=np.float64)
    print(
        "Computed NSI image with "
        f"f_number={nsi_f_number}, dc={nsi_dc}, "
        f"normalize_by_n_subap={nsi_normalize}, shape={images['nsi'].shape}"
    )

mixer_coefficients = mixer_model.mixer_coefficients
mixer_coefficients_serializable = {
    "weights": [float(v) for v in np.asarray(mixer_coefficients["weights"]).ravel().tolist()],
    "bias": [float(v) for v in np.asarray(mixer_coefficients["bias"]).ravel().tolist()],
}

profile_cfg = cfg.get("reflector_lateral_profiles", {})
methods_cfg = profile_cfg.get("methods", list(images.keys()))
if not isinstance(methods_cfg, list) or len(methods_cfg) == 0:
    raise ValueError("`reflector_lateral_profiles.methods` must be a non-empty list.")
selected_method_names = [str(name).strip().lower() for name in methods_cfg]
selected_method_names = list(dict.fromkeys(selected_method_names))

missing_methods = [name for name in selected_method_names if name not in images]
if missing_methods:
    raise RuntimeError(
        "Requested methods are not available in this mixer evaluator: "
        f"{missing_methods}. Available: {list(images.keys())}"
    )

images_db = {name: _db_image(image) for name, image in images.items()}
display_labels = [name for name in selected_method_names if name in images_db]

extent = kp.get_imshow_extent()
vmin_db = -60.0
vmax_db = 0.0

rows, cols = _subplot_grid(len(display_labels), max_cols=2)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
axes_flat = np.atleast_1d(np.asarray(axes).ravel())
first_im = None
for idx, label in enumerate(display_labels):
    ax = axes_flat[idx]
    im = ax.imshow(
        images_db[label],
        aspect="auto",
        cmap="gray",
        vmin=vmin_db,
        vmax=vmax_db,
        extent=extent,
    )
    if first_im is None:
        first_im = im
    ax.set_title(label, fontsize=fontsize_title)
    ax.set_xlabel("x (mm)", fontsize=fontsize_axis)
    if idx % cols == 0:
        ax.set_ylabel("z (mm)", fontsize=fontsize_axis)

for j in range(len(display_labels), rows * cols):
    axes_flat[j].axis("off")

fig.suptitle("DAS comparison (Mixer)", fontsize=fontsize_title)
fig.tight_layout(rect=[0, 0, 0.92, 1])
cbar_ax = fig.add_axes([0.93, 0.1, 0.013, 0.78])
if first_im is not None:
    fig.colorbar(first_im, cax=cbar_ax, label="dB")

out_root = resolve_project_path(io_cfg.get("evaluation_output_root", "scripts/outputs/evaluation/numeric_phantom/mixer"))
out_root.mkdir(parents=True, exist_ok=True)
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_out_dir = out_root / run_timestamp
run_out_dir.mkdir(parents=True, exist_ok=False)

with (run_out_dir / "numeric_phantom_evaluation_mixer_config_info.yml").open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)

metadata = {
    "model_file": str(model_file),
    "combiner_weights_file": str(combiner_weights_file),
    "train_info_file": str(train_info_file),
    "model_run_dir": str(model_run_dir),
    "simulation_run_dir": str(latest_simulation_run),
    "delayed_samples_path": str(delayed_samples_path),
    "scaled_features": bool(scaled_features),
    "n_apodizations": int(n_apodizations),
    "feature_chunk_size": int(feature_chunk_size),
    "mixer_coefficients": mixer_coefficients_serializable,
}
if nsi_enabled:
    metadata["nsi"] = {
        "enabled": True,
        "f_number": float(nsi_f_number),
        "dc": float(nsi_dc),
        "normalize_by_n_subap": bool(nsi_normalize),
    }
with (run_out_dir / "mixer_model_metadata.json").open("w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2)

plot_path = run_out_dir / "evaluate_mixer_quicklook.png"
fig.savefig(plot_path, dpi=150)
plt.show()

profiles_enabled = bool(profile_cfg.get("enabled", True))
if profiles_enabled:
    selected_images_abs = {
        name: np.abs(np.asarray(images[name])).astype(np.float64) for name in selected_method_names
    }

    half_width_lateral_mm = float(profile_cfg.get("half_width_lateral_mm", 1.5))
    half_width_axial_mm = float(profile_cfg.get("half_width_axial_mm", 1.0))
    profile_vmin_db = float(profile_cfg.get("vmin_db", -60.0))
    snr_radius_mm = float(profile_cfg.get("snr_radius_mm", 1.5))
    snr_y_lim_cfg = profile_cfg.get("snr_y_lim_db", None)
    snr_y_lim_db: tuple[float, float] | None = None
    if snr_y_lim_cfg is not None:
        if not isinstance(snr_y_lim_cfg, (list, tuple)) or len(snr_y_lim_cfg) != 2:
            raise ValueError("`reflector_lateral_profiles.snr_y_lim_db` must be null or [ymin, ymax].")
        snr_y_min = float(snr_y_lim_cfg[0])
        snr_y_max = float(snr_y_lim_cfg[1])
        if snr_y_max <= snr_y_min:
            raise ValueError("`snr_y_lim_db` must satisfy ymax > ymin.")
        snr_y_lim_db = (snr_y_min, snr_y_max)

    reflector_points = build_grid_reflector_points(cfg)
    selected_indices = resolve_reflector_indices(
        profile_cfg.get("reflector_indices", "all"),
        n_reflectors=int(reflector_points.shape[0]),
    )
    selected_points = reflector_points[selected_indices]

    method_profiles: dict[str, dict[str, np.ndarray | int]] = {}
    fwhm_rows: list[dict[str, float | int | str]] = []
    snr_by_method: dict[str, dict[str, np.ndarray | float]] = {}

    for method_name in selected_method_names:
        extracted = extract_reflector_profiles(
            image=selected_images_abs[method_name],
            scatterers=selected_points,
            cm=cm,
            half_width_lateral_mm=half_width_lateral_mm,
            half_width_axial_mm=half_width_axial_mm,
        )
        lateral_profiles = np.asarray(extracted["lateral_profiles"], dtype=np.float64)
        axial_profiles = np.asarray(extracted["axial_profiles"], dtype=np.float64)
        lateral_offsets_mm = np.asarray(extracted["lateral_offsets_mm"], dtype=np.float64)
        axial_offsets_mm = np.asarray(extracted["axial_offsets_mm"], dtype=np.float64)

        fwhm_lat = compute_fwhm_batch(lateral_profiles, lateral_offsets_mm)
        fwhm_ax = compute_fwhm_batch(axial_profiles, axial_offsets_mm)

        for local_idx, refl_idx in enumerate(selected_indices.tolist()):
            fwhm_rows.append(
                {
                    "reflector_index": int(refl_idx),
                    "method": method_name,
                    "fwhm_lateral_mm": float(fwhm_lat["fwhm_mm"][local_idx]),
                    "fwhm_axial_mm": float(fwhm_ax["fwhm_mm"][local_idx]),
                }
            )

        scatterer_metrics = compute_scatterer_metrics(
            image=selected_images_abs[method_name],
            scatterers=selected_points,
            cm=cm,
            radius_mm=snr_radius_mm,
            profile_half_lateral_mm=None,
            profile_half_axial_mm=None,
            return_masks=False,
            return_background_hist=False,
        )
        peak_amplitudes = np.asarray(scatterer_metrics["peak_amplitudes"], dtype=np.float64)
        background_rms = float(scatterer_metrics["background_rms"])
        snr_values = compute_reflector_snr(peak_amplitudes, background_rms)
        snr_values_db = compute_reflector_snr(peak_amplitudes, background_rms, return_db=True)
        snr_by_method[method_name] = {
            "snr": snr_values,
            "snr_db": snr_values_db,
            "peak_amplitudes": peak_amplitudes,
            "background_rms": background_rms,
        }

        method_profiles[method_name] = {
            "lateral_profiles": lateral_profiles,
            "axial_profiles": axial_profiles,
            "lateral_offsets_mm": lateral_offsets_mm,
            "axial_offsets_mm": axial_offsets_mm,
        }

    profile_type = str(profile_cfg.get("profile_type", "both")).strip().lower()
    if profile_type not in ("lateral", "axial", "both"):
        raise ValueError(
            "`reflector_lateral_profiles.profile_type` must be 'lateral', 'axial', or 'both'. "
            f"Got: '{profile_type}'"
        )
    show_lateral = profile_type in ("lateral", "both")
    show_axial = profile_type in ("axial", "both")

    profiles_dir = run_out_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    for local_idx, refl_idx in enumerate(selected_indices.tolist()):
        x_mm = float(reflector_points[refl_idx, 0])
        z_mm = float(reflector_points[refl_idx, 1])

        n_subplots = int(show_lateral) + int(show_axial)
        fig_ref, axes_ref_raw = plt.subplots(
            1, n_subplots, figsize=(6 * n_subplots, 4), constrained_layout=True
        )
        axes_ref_arr = np.atleast_1d(axes_ref_raw)

        ax_lat = axes_ref_arr[0] if show_lateral else None
        ax_ax = axes_ref_arr[1] if (show_lateral and show_axial) else (axes_ref_arr[0] if show_axial else None)

        for method_name in selected_method_names:
            lat_offsets = np.asarray(method_profiles[method_name]["lateral_offsets_mm"], dtype=np.float64)
            ax_offsets = np.asarray(method_profiles[method_name]["axial_offsets_mm"], dtype=np.float64)
            lat_profile = np.asarray(method_profiles[method_name]["lateral_profiles"], dtype=np.float64)[local_idx]
            ax_profile = np.asarray(method_profiles[method_name]["axial_profiles"], dtype=np.float64)[local_idx]

            lat_peak = float(np.nanmax(np.abs(lat_profile))) if lat_profile.size > 0 else 1.0
            ax_peak = float(np.nanmax(np.abs(ax_profile))) if ax_profile.size > 0 else 1.0
            lat_db_arr = 20.0 * np.log10(np.maximum(np.abs(lat_profile), 1e-12) / max(lat_peak, 1e-12))
            ax_db_arr = 20.0 * np.log10(np.maximum(np.abs(ax_profile), 1e-12) / max(ax_peak, 1e-12))

            if show_lateral:
                ax_lat.plot(lat_offsets, lat_db_arr, linewidth=2, label=method_name)
            if show_axial:
                ax_ax.plot(ax_offsets, ax_db_arr, linewidth=2, label=method_name)

        if show_lateral:
            ax_lat.set_ylim(profile_vmin_db, 0.0)
            ax_lat.grid(True, alpha=0.3)
            ax_lat.set_xlabel("Lateral offset (mm)")
            ax_lat.set_ylabel("Amplitude (dB)")
            ax_lat.set_title("Lateral profile")
            ax_lat.legend(ncol=min(4, len(selected_method_names)), fontsize=fontsize_legend)
        if show_axial:
            ax_ax.set_ylim(profile_vmin_db, 0.0)
            ax_ax.grid(True, alpha=0.3)
            ax_ax.set_xlabel("Axial offset (mm)")
            ax_ax.set_ylabel("Amplitude (dB)")
            ax_ax.set_title("Axial profile")
            if not show_lateral:
                ax_ax.legend(ncol=min(4, len(selected_method_names)), fontsize=fontsize_legend)

        fig_ref.suptitle(f"Reflector {refl_idx} at x={x_mm:.2f} mm, z={z_mm:.2f} mm")
        fig_ref.savefig(profiles_dir / f"reflector_{refl_idx:03d}_profiles_db.png", dpi=150)
        plt.close(fig_ref)

    fwhm_csv_path = run_out_dir / "reflector_fwhm_summary.csv"
    per_reflector: dict[int, dict[str, float]] = {}
    for row in fwhm_rows:
        rid = int(row["reflector_index"])
        method = str(row["method"])
        lat_key = f"{method}_fwhm_lateral_mm"
        ax_key = f"{method}_fwhm_axial_mm"
        if rid not in per_reflector:
            per_reflector[rid] = {}
        per_reflector[rid][lat_key] = float(row.get("fwhm_lateral_mm", float("nan")))
        per_reflector[rid][ax_key] = float(row.get("fwhm_axial_mm", float("nan")))

    for method_name, snr_data in snr_by_method.items():
        snr_vals = np.asarray(snr_data["snr"], dtype=np.float64)
        snr_vals_db = np.asarray(snr_data["snr_db"], dtype=np.float64)
        peak_vals = np.asarray(snr_data["peak_amplitudes"], dtype=np.float64)
        bg_rms = float(snr_data["background_rms"])
        for local_idx, rid in enumerate(selected_indices.tolist()):
            if int(rid) not in per_reflector:
                per_reflector[int(rid)] = {}
            per_reflector[int(rid)][f"{method_name}_snr"] = float(snr_vals[local_idx])
            per_reflector[int(rid)][f"{method_name}_snr_db"] = float(snr_vals_db[local_idx])
            per_reflector[int(rid)][f"{method_name}_peak_amplitude"] = float(peak_vals[local_idx])
            per_reflector[int(rid)][f"{method_name}_background_rms"] = bg_rms

    header = ["reflector_index"]
    for method_name in selected_method_names:
        header.append(f"{method_name}_fwhm_lateral_mm")
        header.append(f"{method_name}_fwhm_axial_mm")
        header.append(f"{method_name}_snr")
        header.append(f"{method_name}_snr_db")
        header.append(f"{method_name}_peak_amplitude")
        header.append(f"{method_name}_background_rms")

    with open(fwhm_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for rid in selected_indices.tolist():
            row_values = [int(rid)]
            values = per_reflector.get(int(rid), {})

            def _format_or_empty(value: object) -> str:
                if value is None:
                    return ""
                if isinstance(value, (float, np.floating)):
                    return "" if np.isnan(float(value)) else f"{float(value):.6f}"
                return ""

            for method_name in selected_method_names:
                row_values.append(_format_or_empty(values.get(f"{method_name}_fwhm_lateral_mm", None)))
                row_values.append(_format_or_empty(values.get(f"{method_name}_fwhm_axial_mm", None)))
                row_values.append(_format_or_empty(values.get(f"{method_name}_snr", None)))
                row_values.append(_format_or_empty(values.get(f"{method_name}_snr_db", None)))
                row_values.append(_format_or_empty(values.get(f"{method_name}_peak_amplitude", None)))
                row_values.append(_format_or_empty(values.get(f"{method_name}_background_rms", None)))
            writer.writerow(row_values)

    summary_fig, (ax_fwhm, ax_snr) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    refl_ids_ordered = selected_indices.tolist()

    for method_name in selected_method_names:
        fwhm_lat_vals = [
            float(per_reflector.get(int(rid), {}).get(f"{method_name}_fwhm_lateral_mm", float("nan")))
            for rid in refl_ids_ordered
        ]
        snr_db_vals = [
            float(per_reflector.get(int(rid), {}).get(f"{method_name}_snr_db", float("nan")))
            for rid in refl_ids_ordered
        ]
        ax_fwhm.plot(refl_ids_ordered, fwhm_lat_vals, marker="o", linewidth=2, label=method_name)
        ax_snr.plot(refl_ids_ordered, snr_db_vals, marker="o", linewidth=2, label=method_name)

    ax_fwhm.set_xlabel("Reflector index")
    ax_fwhm.set_ylabel("FWHM lateral (mm)")
    ax_fwhm.set_title("Lateral resolution vs reflector index")
    ax_fwhm.legend(fontsize=fontsize_legend)
    ax_fwhm.grid(True, alpha=0.3)

    ax_snr.set_xlabel("Reflector index")
    ax_snr.set_ylabel("SNR (dB)")
    ax_snr.set_title("SNR vs reflector index")
    if snr_y_lim_db is not None:
        ax_snr.set_ylim(*snr_y_lim_db)
    ax_snr.legend(fontsize=fontsize_legend)
    ax_snr.grid(True, alpha=0.3)

    summary_fig_path = run_out_dir / "resolution_and_snr_summary.png"
    summary_fig.savefig(summary_fig_path, dpi=150)
    plt.close(summary_fig)

print("Mixer evaluation complete. Quicklook saved to:", plot_path)
if bool(mixer_cfg.get("log_mixer_weights", True)):
    print("Mixer metadata saved to:", run_out_dir / "mixer_model_metadata.json")
if profiles_enabled:
    print("Per-reflector profiles saved in:", run_out_dir / "profiles")
    print("FWHM + SNR summary saved to:", run_out_dir / "reflector_fwhm_summary.csv")
    print("Resolution and SNR summary plot saved to:", run_out_dir / "resolution_and_snr_summary.png")
