"""Standalone baseline apodization evaluation script.

This script evaluates the same delayed-samples dataset used during training,
but without running the INR model. It computes baseline DAS references,
validation metrics, and scatterer diagnostics so the resulting outputs can be
used later by training runs for SNR comparisons.

Configuration is read directly from ``configs/train_config.yml``.
"""

from __future__ import annotations

import csv
import json
import yaml
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf

from inr_apodizations.apodizations import (
    compute_dynamic_apodizations_tf,
    extract_map_for_x,
    extract_profile_for_z,
)
from inr_apodizations.config import CONFIGS_DIR, PROJ_ROOT
from inr_apodizations.evaluation import (
    build_scatterer_reference_bundle,
    build_validation_weights,
    compute_validation_baseline_metrics,
    load_validation_scatterers,
    resolve_baseline_f_number,
    select_reflector_scatterer,
    select_reflector_scatterer_index,
)
from inr_apodizations.plots import plot_lateral_reflector_profiles, to_db
from inr_apodizations.experiment_helpers import build_reflector_profile_context
import inr_apodizations.experiment_helpers as helpers
from inr_apodizations.utils import relative_mae


CONFIG_PATH = CONFIGS_DIR / "train_config.yml"


def load_yaml_config(config_path: Path) -> dict:
    """Load the baseline evaluation YAML configuration.

    Args:
        config_path: Absolute path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is missing.
        ValueError: If the parsed content is not a dictionary.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("YAML config must be a dictionary.")

    return config


def resolve_dataset_folder(cfg_user: dict) -> Path:
    """Resolve the delayed-samples dataset folder from the user config."""
    dataset_subdir = str(cfg_user["io"]["dataset_folder"]).strip()
    dataset_folder = Path(dataset_subdir)
    if not dataset_folder.is_absolute():
        dataset_folder = PROJ_ROOT / dataset_folder
    return dataset_folder


def resolve_baseline_output_root(cfg_user: dict) -> Path:
    """Resolve the baseline output root folder from config.

    Preferred key is ``io.baseline_output``. Legacy typo ``io.baseline_ouput``
    is accepted for compatibility.
    """
    io_cfg = cfg_user.get("io", {})
    baseline_output_cfg = io_cfg.get("baseline_output", io_cfg.get("baseline_ouput"))
    if baseline_output_cfg is None:
        baseline_output_cfg = "scripts/outputs/evaluation/baseline"

    baseline_output = Path(str(baseline_output_cfg).strip())
    if not baseline_output.is_absolute():
        baseline_output = PROJ_ROOT / baseline_output
    return baseline_output


def load_validation_scatterers_full(
    dataset_folder: str,
    validation_indices: np.ndarray,
) -> list[np.ndarray]:
    """Load validation scatterers preserving reflectivity when available."""
    scatterers_all = helpers.load_saved_scatterers(dataset_folder)
    scatterers_batch: list[np.ndarray] = []
    for idx in validation_indices:
        scatterers_example = np.asarray(scatterers_all[int(idx)], dtype=np.float32).copy()
        scatterers_example[:, :2] *= 1000.0
        scatterers_batch.append(scatterers_example)
    return scatterers_batch


def save_json(path: Path, payload: dict) -> None:
    """Save a JSON file with deterministic formatting."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_csv(path: Path, rows: list[dict[str, object]], header: list[str]) -> None:
    """Save a CSV file from a list of dictionaries."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_scatterer_profile_row(metrics: dict, example_idx: int, scatterer_idx: int) -> int:
    """Find the aggregated row matching one reflector in one example."""
    view = metrics.get("aggregated", metrics)
    example_indices = np.asarray(view.get("peak_example_indices", np.empty(0)), dtype=np.int32)
    scatterer_indices = np.asarray(view.get("peak_scatterer_indices", np.empty(0)), dtype=np.int32)
    matches = np.flatnonzero(
        (example_indices == int(example_idx)) & (scatterer_indices == int(scatterer_idx))
    )
    if matches.size == 0:
        raise ValueError(
            f"Could not find profile row for example_idx={example_idx}, scatterer_idx={scatterer_idx}."
        )
    return int(matches[0])


def plot_selected_scatterer_profiles(
    profile_metrics: dict[str, dict],
    example_idx: int,
    scatterer_idx: int,
    output_path: Path,
    dpi: int,
    use_db: bool,
    selected_scatterer: np.ndarray,
    db_min: float = -60.0,
) -> None:
    """Plot lateral and axial profiles for one reflector across all methods.

    Args:
        profile_metrics: Mapping from method name to scatterer-metric bundles.
        example_idx: Validation-example index inside the local validation batch.
        scatterer_idx: Reflector index inside the selected validation example.
        output_path: Destination figure path.
        dpi: Figure resolution.
        use_db: If ``True``, normalize each profile by its own local maximum and
            display it in dB.
        selected_scatterer: Scatterer row with at least ``[x_mm, z_mm]``.
    """
    axis_specs = {
        "lateral": {
            "profiles_key": "lateral_profiles",
            "offsets_key": "lateral_profile_offsets_mm",
            "xlabel": "Relative x (mm)",
            "title": "Lateral profile (max over z in reflector box)",
        },
        "axial": {
            "profiles_key": "axial_profiles",
            "offsets_key": "axial_profile_offsets_mm",
            "xlabel": "Relative z (mm)",
            "title": "Axial profile (max over x in reflector box)",
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=use_db)

    for axis_idx, (axis_name, axis_cfg) in enumerate(axis_specs.items()):
        ax = axes[axis_idx]
        for method_name, metrics in profile_metrics.items():
            view = metrics.get("aggregated", metrics)
            profiles = np.asarray(view.get(axis_cfg["profiles_key"], np.empty((0, 0))), dtype=np.float64)
            offsets = np.asarray(view.get(axis_cfg["offsets_key"], np.empty(0)), dtype=np.float64)
            try:
                row_idx = find_scatterer_profile_row(metrics, example_idx=example_idx, scatterer_idx=scatterer_idx)
            except ValueError:
                continue

            if profiles.ndim != 2 or row_idx >= profiles.shape[0] or offsets.size != profiles.shape[1]:
                continue

            raw_profile = np.abs(profiles[row_idx])
            valid = np.isfinite(raw_profile)
            if not np.any(valid):
                continue

            plot_x = offsets[valid]
            if use_db:
                ref_value = float(np.nanmax(raw_profile[valid]))
                plot_y = to_db(raw_profile[valid], ref=max(ref_value, 1e-12))
            else:
                plot_y = raw_profile[valid]

            ax.plot(plot_x, plot_y, linewidth=2, label=method_name.capitalize())

        ax.set_xlabel(axis_cfg["xlabel"])
        ax.set_title(axis_cfg["title"])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Amplitude (dB re. local peak)" if use_db else "Amplitude")
    axes[0].legend(title="Method")
    fig.suptitle(
        "Reflector profiles at "
        f"x={float(selected_scatterer[0]):.2f} mm, z={float(selected_scatterer[1]):.2f} mm"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if use_db:
        for ax in axes:
            ax.set_ylim(db_min, 0.0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run baseline evaluation end to end."""
    cfg_user = load_yaml_config(CONFIG_PATH)
    save_outputs = bool(cfg_user.get("save", True))
    show_outputs = bool(cfg_user.get("show", False))

    print(f"Using YAML config: {CONFIG_PATH}")
    print(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')}")

    dataset_folder = resolve_dataset_folder(cfg_user)
    if not dataset_folder.exists():
        raise FileNotFoundError(f"Configured dataset folder does not exist: {dataset_folder}")

    cfg_path = dataset_folder / "cfg_delayed_samples.npy"
    # The delayed-samples files (signal/combined/noise) are resolved by
    # `helpers.load_delayed_samples_dataset` later; here we only ensure the
    # beamforming config exists so we can inspect f-number and geometry.
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    cfg = np.load(cfg_path, allow_pickle=True).item()
    if "f_number" not in cfg and "bfd" in cfg:
        cfg["f_number"] = cfg["bfd"] / 2.0

    sigma_x_override, sigma_z_override, alpha_override = helpers.get_target_regeneration_override(
        cfg_user
    )
    delayed, noise, targets, gaussian_masks, info = helpers.load_delayed_samples_dataset(
        str(dataset_folder),
        sigma_x=sigma_x_override,
        sigma_z=sigma_z_override,
        alpha_override=alpha_override,
    )

    if info.get("precomputed_noise_source", {}):
        pinfo = info.get("precomputed_noise_source", {})
        print("Dataset contains precomputed noise information:")
        print(f"  source={pinfo.get('source')}")

    helpers.validate_dataset_shapes(delayed, targets, gaussian_masks)
    physical_feature_set = str(
        cfg_user.get("model", {}).get("physical_feature_set", "distance_depth_edge")
    )
    kp, cm = helpers.build_coordinate_manager(
        str(dataset_folder), physical_feature_set=physical_feature_set
    )

    max_examples = cfg_user.get("training", {}).get("max_examples")
    if max_examples is not None:
        max_examples = int(max_examples)
        delayed = delayed[:max_examples]
        targets = targets[:max_examples]
        gaussian_masks = gaussian_masks[:max_examples]
        if noise is not None:
            noise = noise[:max_examples]

    train_idx, val_idx = helpers.split_train_validation_indices(
        n_examples=delayed.shape[0],
        train_fraction=float(cfg_user["training"]["train_fraction"]),
        seed=int(cfg_user["training"]["seed"]),
    )

    train_loss_weights = build_validation_weights(gaussian_masks, cfg_user)

    eval_noise_cfg = dict(cfg_user.get("eval_noise", {}))
    eval_noise_enabled = bool(eval_noise_cfg.get("enabled", False))
    eval_noise_scale = float(eval_noise_cfg.get("scale", 1.0)) if eval_noise_enabled else 1.0

    baseline_f_number = resolve_baseline_f_number(cfg_user, kp)

    scaled_features = bool(cfg_user.get("model", {}).get("scaled_features", False))
    apods = compute_dynamic_apodizations_tf(
        cm=cm,
        f_number=baseline_f_number,
        methods=("hanning", "boxcar"),
        scaled=scaled_features,
    )

    hanning_weights_np = apods["hanning"].numpy()
    boxcar_weights_np = apods["boxcar"].numpy()

    output_root = resolve_baseline_output_root(cfg_user)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "train_config_info.yml").open("w", encoding="utf-8") as _f:
        yaml.safe_dump(cfg_user, _f, sort_keys=False)

    dpi = int(cfg_user.get("dpi", 150))
    cmap = str(cfg_user.get("cmap", "gray"))
    x_fixed = float(cfg_user.get("x_fixed", 0.0))
    z_fixed = float(cfg_user.get("z_fixed", 20.0))
    extent = kp.get_imshow_extent()
    coords = cm.get_coordinates_1d(scaled=scaled_features)
    x_elems = np.asarray(coords["x_elem"])
    z_coords = np.asarray(coords["z"])

    validation_indices = val_idx
    validation_targets = targets[validation_indices].astype(np.float32, copy=False)
    validation_sample_weights = train_loss_weights[validation_indices].astype(
        np.float32, copy=False
    )
    eval_batch_size = int(
        cfg_user.get("scatterer_eval", {}).get("eval_batch_size", cfg_user["training"]["batch_size"])
    )
    radius_mm = float(cfg_user.get("scatterer_eval", {}).get("radius_mm", 1.5))
    hist_bins = int(cfg_user.get("scatterer_eval", {}).get("hist_bins", 50))

    scatterers_batch = load_validation_scatterers(str(dataset_folder), validation_indices)
    validation_scatterers_full = load_validation_scatterers_full(
        str(dataset_folder),
        validation_indices,
    )

    images_abs_eval, validation_bundle, pre_training_reference_mae = (
        compute_validation_baseline_metrics(
            delayed=delayed,
            noise=noise,
            validation_indices=validation_indices,
            targets=targets,
            validation_sample_weights=validation_sample_weights,
            hanning_weights_np=hanning_weights_np,
            boxcar_weights_np=boxcar_weights_np,
            eval_noise_enabled=eval_noise_enabled,
            eval_noise_scale=eval_noise_scale,
            eval_batch_size=eval_batch_size,
            cm=cm,
            scatterers_batch=scatterers_batch,
            radius_mm=radius_mm,
            hist_bins=hist_bins,
        )
    )

    weighted_mae_summary = {method_name: pre_training_reference_mae[method_name] for method_name in images_abs_eval}
    zero_reference_mae = float(validation_bundle.get("reference_mae", {}).get("zero", 0.0))
    zero_weighted_mae = float(pre_training_reference_mae.get("zero", 0.0))

    mae_by_method = validation_bundle.get("mae_by_method", {})
    relative_mae_y_pred = {
        method_name: float(
            relative_mae(
                y_true=validation_targets,
                y_pred=images_abs_eval[method_name],
                normalize_by="y_pred",
                sample_weights=validation_sample_weights,
            )
        )
        for method_name in images_abs_eval
    }
    relative_mae_y_true = {
        method_name: float(
            relative_mae(
                y_true=validation_targets,
                y_pred=images_abs_eval[method_name],
                normalize_by="y_true",
                sample_weights=validation_sample_weights,
            )
        )
        for method_name in images_abs_eval
    }

    scatterer_summary, scatterer_arrays = build_scatterer_reference_bundle(
        validation_bundle.get("scatterer_metrics", {}),
        scatterers_batch,
    )

    target_scatterer_metrics = helpers.compute_scatterer_metrics(
        validation_targets,
        scatterers_batch,
        cm,
        radius_mm=radius_mm,
        return_masks=False,
        return_background_hist=False,
    )
    target_view = target_scatterer_metrics.get("aggregated", target_scatterer_metrics)
    scatterer_arrays["lateral_profiles_target"] = np.asarray(
        target_view.get("lateral_profiles", np.empty((0, 0))),
        dtype=np.float32,
    )
    scatterer_arrays["axial_profiles_target"] = np.asarray(
        target_view.get("axial_profiles", np.empty((0, 0))),
        dtype=np.float32,
    )

    summary_payload = {
        "config_path": str(CONFIG_PATH),
        "dataset_folder": str(dataset_folder),
        "output_dir": str(output_dir),
        "timestamp": timestamp,
        "example_idx": int(cfg_user.get("example_idx", 0)),
        "train_indices": train_idx.tolist(),
        "validation_indices": validation_indices.tolist(),
        "baseline_f_number": baseline_f_number,
        "eval_noise": {
            "enabled": eval_noise_enabled,
            "scale": eval_noise_scale,
        },
        "reference_mae": {
            "zero": zero_reference_mae,
        },
        "zero_weighted_mae": zero_weighted_mae,
        "weighted_mae_by_method": weighted_mae_summary,
        "mae_by_method": mae_by_method,
        "relative_mae": {
            "relative_mae_y_pred": relative_mae_y_pred,
            "relative_mae_y_true": relative_mae_y_true,
        },
        "scatterer_summary": scatterer_summary,
    }

    save_json(output_dir / "baseline_summary.json", summary_payload)

    csv_rows = []
    for method_name in ["uniform", "hanning", "boxcar"]:
        csv_rows.append(
            {
                "method": method_name,
                "mae": mae_by_method.get(method_name, ""),
                "weighted_mae": weighted_mae_summary.get(method_name, ""),
                "relative_mae_y_pred": relative_mae_y_pred.get(method_name, ""),
                "relative_mae_y_true": relative_mae_y_true.get(method_name, ""),
            }
        )
    save_csv(
        output_dir / "baseline_metrics.csv",
        csv_rows,
        [
            "method",
            "mae",
            "weighted_mae",
            "relative_mae_y_pred",
            "relative_mae_y_true",
        ],
    )

    np.savez(
        output_dir / "scatterer_reference_metrics.npz",
        **scatterer_arrays,
        validation_indices=validation_indices.astype(np.int32, copy=False),
        train_indices=train_idx.astype(np.int32, copy=False),
    )

    profile_cfg = dict(cfg_user.get("reflector_lateral_profile", {}))
    profile_enabled = bool(profile_cfg.get("enabled", False))
    if profile_enabled:
        profile_context = build_reflector_profile_context(
            scatterers_mm=validation_scatterers_full[0],
            profile_cfg=profile_cfg,
            images_linear={
                "uniform": images_abs_eval["uniform"][0],
                "hanning": images_abs_eval["hanning"][0],
                "boxcar": images_abs_eval["boxcar"][0],
            },
            images_db={
                "uniform": to_db(images_abs_eval["uniform"][0], ref=float(np.max(images_abs_eval["uniform"][0]))),
                "hanning": to_db(images_abs_eval["hanning"][0], ref=float(np.max(images_abs_eval["hanning"][0]))),
                "boxcar": to_db(images_abs_eval["boxcar"][0], ref=float(np.max(images_abs_eval["boxcar"][0]))),
            },
            target_image=validation_targets[0] if bool(profile_cfg.get("include_target", True)) else None,
        )
        selected_scatterer = np.asarray(profile_context["selected_scatterer"], dtype=np.float64)
        line_length_mm = float(profile_context["line_length_mm"])
        overlay_profiles = bool(profile_context["overlay_profiles"])
        use_db_profiles = bool(profile_context["use_db_profiles"])
        selected_profile_images = dict(profile_context["selected_profile_images"])
        x_center_mm = float(selected_scatterer[0])
        z_center_mm = float(selected_scatterer[1])

        out_reflector_profile = output_dir / (
            f"reflector_lateral_profile_z{z_center_mm:.2f}_x{x_center_mm:.2f}_"
            f"len{line_length_mm:.2f}_{'db' if use_db_profiles else 'linear'}_example{validation_indices[0]}.png"
        )
        plot_lateral_reflector_profiles(
            images=selected_profile_images,
            output_path=str(out_reflector_profile),
            extent=extent,
            x_center=x_center_mm,
            z_center=z_center_mm,
            line_length=line_length_mm,
            overlay_profiles=overlay_profiles,
            cm=cm,
            vmin_db=float(cfg_user.get("vmin_db", -60.0)),
        )

    # indices to overlay on the uniform image (default: none)
    overlay_indices: list[int] = []
    profile_compare_cfg = dict(cfg_user.get("reflector_profile_comparison", {}))
    profile_compare_enabled = bool(profile_compare_cfg.get("enabled", True))
    if profile_compare_enabled and validation_scatterers_full:
        scatterer_selection = str(profile_compare_cfg.get("scatterer_selection", "strongest"))
        scatterer_idx_raw = profile_compare_cfg.get("scatterer_idx")
        # Allow list of indices or single int in config
        if isinstance(scatterer_idx_raw, list):
            indices_to_plot = [int(i) for i in scatterer_idx_raw]
        elif scatterer_idx_raw is None:
            # resolve single index by selection
            indices_to_plot = [
                select_reflector_scatterer_index(
                    validation_scatterers_full[0], selection=scatterer_selection, scatterer_idx=None
                )
            ]
        else:
            indices_to_plot = [int(scatterer_idx_raw)]

        use_db_profile_compare = bool(profile_compare_cfg.get("use_db", True))
        db_min = float(profile_compare_cfg.get("db_min", -80.0))
        profile_metrics = {
            **validation_bundle.get("scatterer_metrics", {}),
            "target": target_scatterer_metrics,
        }

        # expose indices for overlay use later when drawing DAS panel
        overlay_indices = list(indices_to_plot)

        for sel_idx in indices_to_plot:
            try:
                selected_scatterer = validation_scatterers_full[0][int(sel_idx)]
            except Exception:
                print(f"Warning: scatterer index {sel_idx} out of range, skipping")
                continue
            out_profile_compare = output_dir / (
                f"reflector_profiles_compare_example{validation_indices[0]}_"
                f"scatterer{sel_idx}_{'db' if use_db_profile_compare else 'linear'}.png"
            )
            plot_selected_scatterer_profiles(
                profile_metrics=profile_metrics,
                example_idx=0,
                scatterer_idx=int(sel_idx),
                output_path=out_profile_compare,
                dpi=dpi,
                use_db=use_db_profile_compare,
                selected_scatterer=selected_scatterer,
                db_min=db_min,
            )

    for method_name, apod_tensor in apods.items():
        map_z_elem = extract_map_for_x(apod_tensor, cm, x_fixed=x_fixed, scaled=scaled_features).numpy()
        profile = extract_profile_for_z(
            apod_tensor,
            cm,
            z_fixed=z_fixed,
            x_fixed=x_fixed,
            scaled=scaled_features,
        ).numpy()

        fig_map, ax_map = plt.subplots(1, 1, figsize=(8, 5))
        im_map = ax_map.imshow(
            map_z_elem,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            extent=(float(x_elems[0]), float(x_elems[-1]), float(z_coords[-1]), float(z_coords[0])),
        )
        ax_map.set_xlabel("Element lateral coordinate (mm)")
        ax_map.set_ylabel("Depth z (mm)")
        ax_map.set_title(f"Apodization map ({method_name}) at x={x_fixed:.2f} mm")
        fig_map.colorbar(im_map, ax=ax_map, label="Apodization")
        fig_map.tight_layout()
        if save_outputs:
            out_map = output_dir / f"map_{method_name}_x{x_fixed:.2f}_baseline.png"
            fig_map.savefig(out_map, dpi=dpi, bbox_inches="tight")
        plt.close(fig_map)

        fig_profile, ax_profile = plt.subplots(1, 1, figsize=(8, 4))
        ax_profile.plot(x_elems, profile, marker="o", label=method_name)
        ax_profile.set_xlabel("Element lateral coordinate (mm)")
        ax_profile.set_ylabel("Apodization")
        ax_profile.set_title(f"Apodization profile at z={z_fixed:.2f} mm")
        ax_profile.grid(True)
        ax_profile.legend(title="Method")
        fig_profile.tight_layout()
        if save_outputs:
            out_profile = output_dir / f"profiles_{method_name}_z{z_fixed:.2f}_baseline.png"
            fig_profile.savefig(out_profile, dpi=dpi, bbox_inches="tight")
        plt.close(fig_profile)

    fig_das, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    vmin_db = float(cfg_user.get("vmin_db", -60.0))
    vmax_db = float(cfg_user.get("vmax_db", 0.0))
    ordered = ["uniform", "hanning", "boxcar", "target"]

    first_im = None
    for idx, method_name in enumerate(ordered):
        ax = axes[idx // 2, idx % 2]
        if method_name == "target":
            image = validation_targets[0]
            image_db = to_db(image, ref=float(np.max(np.abs(image))))
        else:
            image_db = to_db(images_abs_eval[method_name][0], ref=float(np.max(images_abs_eval[method_name][0])))
        current_im = ax.imshow(
            image_db,
            cmap=cmap,
            vmin=vmin_db,
            vmax=vmax_db,
            extent=extent,
            aspect="auto",
        )
        # If this is the 'uniform' panel, overlay circles for selected scatterers
        if method_name == "uniform" and overlay_indices:
            for sel_idx in overlay_indices:
                try:
                    sc = validation_scatterers_full[0][int(sel_idx)]
                    x_center = float(sc[0])
                    z_center = float(sc[1])
                    circ = mpatches.Circle(
                        (x_center, z_center), radius=radius_mm, edgecolor="red", facecolor="none", linewidth=0.8, zorder=5
                    )
                    ax.add_patch(circ)
                except Exception:
                    print(f"Warning: could not overlay scatterer {sel_idx} on uniform image")
        if first_im is None:
            first_im = current_im
        title = method_name.capitalize() if method_name != "target" else "Target"
        ax.set_title(f"{title} (dB)")
        ax.set_xlabel("x (mm)")
        if idx % 2 == 0:
            ax.set_ylabel("z (mm)")

    fig_das.suptitle(f"Baseline apodizations - Example {validation_indices[0]}")
    fig_das.tight_layout(rect=[0, 0, 0.9, 1])
    cbar_ax = fig_das.add_axes([0.92, 0.12, 0.018, 0.76])
    fig_das.colorbar(first_im, cax=cbar_ax, label="dB")
    if save_outputs:
        fig_das.savefig(output_dir / f"baseline_panel_example{validation_indices[0]}.png", dpi=dpi, bbox_inches="tight")

    scatterer_metrics = validation_bundle.get("scatterer_metrics", {})
    if scatterer_metrics:
        try:
            compare_pairs = [("uniform", "hanning"), ("uniform", "boxcar")]
            helpers.plot_scatterer_evaluation(
                images_abs=images_abs_eval,
                scatterers_xy=scatterers_batch,
                cm=cm,
                output_dir=str(output_dir),
                radius_mm=radius_mm,
                hist_bins=hist_bins,
                compare_pairs=compare_pairs,
                extent=extent,
                vmin_db=float(cfg_user.get("vmin_db", -60.0)),
                vmax_db=float(cfg_user.get("vmax_db", 0.0)),
                cmap=cmap,
                example_suffix=f"val_{len(validation_indices)}",
                all_metrics=scatterer_metrics,
            )

            for ref_name, cmp_name in compare_pairs:
                res = helpers.plot_scatterer_snr_ratio(
                    images_abs=images_abs_eval,
                    scatterers_xy=scatterers_batch,
                    cm=cm,
                    ref_method=ref_name,
                    cmp_method=cmp_name,
                    radius_mm=radius_mm,
                    extent=extent,
                    cmap="RdBu_r",
                    scale="linear",
                    clip_percentiles=(1.0, 99.0),
                    point_size=15,
                    alpha=0.7,
                    return_fig=True,
                    all_metrics=scatterer_metrics,
                )
                fig = res.get("fig")
                label = res.get("ratio_label", f"{cmp_name}/{ref_name}")
                label_fname = label.replace("/", "_")
                if fig is not None:
                    out_path = output_dir / f"scatt_snr_ratio_{label_fname}_baseline.png"
                    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
                    plt.close(fig)
        except Exception as exc:
            print(f"Warning: scatterer evaluation plotting failed: {exc}")

    if save_outputs:
        print(f"Saved baseline outputs to: {output_dir}")
        print(f"Saved scatterer reference metrics to: {output_dir / 'scatterer_reference_metrics.npz'}")

    if show_outputs:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()