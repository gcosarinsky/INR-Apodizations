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
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from inr_apodizations.apodizations import (
    compute_dynamic_apodizations_tf,
    extract_map_for_x,
    extract_profile_for_z,
)
from inr_apodizations.config import CONFIGS_DIR, PROJ_ROOT
from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.modeling.metrics import PixelWeightedMAE
from inr_apodizations.plots import plot_lateral_reflector_profiles, to_db
from inr_apodizations.utils import relative_mae

sys.path.insert(0, str(PROJ_ROOT / "sandbox" / "inr_das_experiment"))
from baseline_evaluation import (  # noqa: E402
    build_scatterer_reference_bundle,
    build_validation_weights,
    compute_reference_apodizations,
    compute_validation_baseline_metrics,
    load_validation_scatterers,
    resolve_baseline_f_number,
)
import helpers


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


def select_reflector_scatterer(
    scatterers_mm: np.ndarray,
    selection: str = "strongest",
    scatterer_idx: int | None = None,
) -> np.ndarray:
    """Select a single scatterer to center reflector profile plots.

    Args:
        scatterers_mm: Scatterer coordinates with columns [x_mm, z_mm, reflectivity].
        selection: Selection mode. Supported values are ``strongest`` and ``first``.
        scatterer_idx: Optional explicit index with priority over ``selection``.

    Returns:
        Selected scatterer row.

    Raises:
        ValueError: If the array is empty, the index is invalid, or the mode is unsupported.
    """
    if scatterers_mm.shape[0] == 0:
        raise ValueError("No scatterers available for reflector profile selection.")

    if scatterer_idx is not None:
        if scatterer_idx < 0 or scatterer_idx >= scatterers_mm.shape[0]:
            raise ValueError(
                f"scatterer_idx={scatterer_idx} is out of range [0, {scatterers_mm.shape[0] - 1}]."
            )
        return scatterers_mm[scatterer_idx]

    selection_normalized = selection.strip().lower()
    if selection_normalized == "strongest":
        selected_idx = int(np.argmax(np.abs(scatterers_mm[:, 2])))
    elif selection_normalized == "first":
        selected_idx = 0
    else:
        raise ValueError("scatterer_selection must be 'strongest' or 'first'.")
    return scatterers_mm[selected_idx]


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

    hanning_weights_np, boxcar_weights_np = compute_reference_apodizations(
        cm=cm,
        baseline_f_number=baseline_f_number,
        scaled_features=scaled_features,
    )

    output_root_cfg = Path(cfg_user["io"]["sandbox_output_root"])
    if not output_root_cfg.is_absolute():
        output_root = PROJ_ROOT / output_root_cfg
    else:
        output_root = output_root_cfg
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / "baseline_apodizations" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

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

    weighted_mae_summary = {
        "zero": pre_training_reference_mae["zero"],
        "uniform": pre_training_reference_mae["uniform"],
        "hanning": pre_training_reference_mae["hanning"],
        "boxcar": pre_training_reference_mae["boxcar"],
    }

    reference_mae = validation_bundle.get("reference_mae", {})
    mae_by_method = validation_bundle.get("mae_by_method", {})
    masked_mae_by_method = validation_bundle.get("masked_mae_by_method", {})
    relative_mae_y_pred = {
        method_name: float(
            relative_mae(
                y_true=validation_targets,
                y_pred=images_abs_eval[method_name],
                normalize_by="y_pred",
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
            )
        )
        for method_name in images_abs_eval
    }

    scatterer_summary, scatterer_arrays = build_scatterer_reference_bundle(
        validation_bundle.get("scatterer_metrics", {}),
        scatterers_batch,
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
        "reference_mae": reference_mae,
        "weighted_mae_by_method": weighted_mae_summary,
        "mae_by_method": mae_by_method,
        "masked_mae_by_method": masked_mae_by_method,
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
                "masked_mae": masked_mae_by_method.get(method_name, ""),
                "reference_mae": reference_mae.get(method_name, ""),
                "weighted_reference_mae": weighted_mae_summary.get(method_name, ""),
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
            "masked_mae",
            "reference_mae",
            "weighted_reference_mae",
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
        scatterer_selection = str(profile_cfg.get("scatterer_selection", "strongest"))
        scatterer_idx_raw = profile_cfg.get("scatterer_idx")
        scatterer_idx = None if scatterer_idx_raw is None else int(scatterer_idx_raw)
        line_length_mm = float(profile_cfg.get("line_length_mm", 5.0))
        overlay_profiles = bool(profile_cfg.get("overlay_profiles", True))
        use_db_profiles = bool(profile_cfg.get("use_db", True))
        include_target_profile = bool(profile_cfg.get("include_target", True))

        selected_scatterer = select_reflector_scatterer(
            scatterers_batch[0], selection=scatterer_selection, scatterer_idx=scatterer_idx
        )
        x_center_mm = float(selected_scatterer[0])
        z_center_mm = float(selected_scatterer[1])

        profile_image_source = {
            "uniform": to_db(images_abs_eval["uniform"][0], ref=float(np.max(images_abs_eval["uniform"][0])))
            if use_db_profiles
            else images_abs_eval["uniform"][0],
            "hanning": to_db(images_abs_eval["hanning"][0], ref=float(np.max(images_abs_eval["hanning"][0])))
            if use_db_profiles
            else images_abs_eval["hanning"][0],
            "boxcar": to_db(images_abs_eval["boxcar"][0], ref=float(np.max(images_abs_eval["boxcar"][0])))
            if use_db_profiles
            else images_abs_eval["boxcar"][0],
        }
        if include_target_profile:
            target_img = validation_targets[0]
            profile_image_source["target"] = (
                to_db(target_img, ref=float(np.max(np.abs(target_img))))
                if use_db_profiles
                else np.abs(target_img)
            )

        requested_images = profile_cfg.get("images")
        if requested_images is not None:
            requested = [str(item).strip().lower() for item in requested_images]
            profile_image_source = {name: profile_image_source[name] for name in requested}

        out_reflector_profile = output_dir / (
            f"reflector_lateral_profile_z{z_center_mm:.2f}_x{x_center_mm:.2f}_"
            f"len{line_length_mm:.2f}_{'db' if use_db_profiles else 'linear'}_example{validation_indices[0]}.png"
        )
        plot_lateral_reflector_profiles(
            images=profile_image_source,
            output_path=str(out_reflector_profile),
            extent=extent,
            x_center=x_center_mm,
            z_center=z_center_mm,
            line_length=line_length_mm,
            overlay_profiles=overlay_profiles,
            cm=cm,
            vmin_db=float(cfg_user.get("vmin_db", -60.0)),
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

    fig_das, axes = plt.subplots(1, 4, figsize=(21, 5), sharex=True, sharey=True)
    vmin_db = float(cfg_user.get("vmin_db", -60.0))
    vmax_db = float(cfg_user.get("vmax_db", 0.0))
    ordered = ["uniform", "hanning", "boxcar", "target"]

    first_im = None
    for idx, method_name in enumerate(ordered):
        ax = axes[idx]
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
        if first_im is None:
            first_im = current_im
        title = method_name.capitalize() if method_name != "target" else "Target"
        ax.set_title(f"{title} (dB)")
        ax.set_xlabel("x (mm)")
        if idx == 0:
            ax.set_ylabel("z (mm)")

    fig_das.suptitle(f"Baseline apodizations - Example {validation_indices[0]}")
    fig_das.tight_layout(rect=[0, 0, 0.92, 1])
    cbar_ax = fig_das.add_axes([0.93, 0.1, 0.013, 0.78])
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