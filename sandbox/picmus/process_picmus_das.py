from inr_apodizations.config import CONFIGS_DIR
from inr_apodizations.picmus import (
    build_coordinate_manager,
    build_kernel_parameters,
    build_picmus_kernel_config,
    compute_hanning_das_image_chunked,
    compute_inr_das_image_chunked,
    compute_delayed_samples,
    compute_uniform_das_image,
    load_inr_model,
    load_picmus_hdf5,
    load_picmus_pipeline_config,
    select_angle_subset,
)

import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
plt.ion()

"""Run PICMUS delayed-sample pipeline and compute uniform DAS image.

Args:
    config_path: Path to a PICMUS YAML config.

Returns:
    Tuple ``(das_uniform, delayed_samples, cfg, kp)``.

Raises:
    FileNotFoundError: If config or PICMUS files are missing.
    ValueError: If any configuration or shape validation fails.
"""

config_path = CONFIGS_DIR / "picmus_beamforming.yml"
pipeline_cfg = load_picmus_pipeline_config(config_path)

io_cfg = pipeline_cfg["io"]
picmus_data_path = io_cfg["picmus_data_path"]
print(f"Loading PICMUS data from: {picmus_data_path}")
angles, rf_real, x_axis_mm, z_axis_mm = load_picmus_hdf5(
    picmus_data_path=picmus_data_path,
    rf_file=io_cfg["rf_file"],
    scan_file=io_cfg["scan_file"],
)

angles_sel, rf_sel = select_angle_subset(angles, rf_real, pipeline_cfg["angle_subset"])
cfg = build_picmus_kernel_config(pipeline_cfg, angles_sel, rf_sel, x_axis_mm, z_axis_mm)
kp = build_kernel_parameters(cfg)

print("Computing delayed samples...")
delayed_samples = compute_delayed_samples(rf_sel, angles_sel, kp, cfg)
print("Delayed samples computed.")
das_uniform = compute_uniform_das_image(delayed_samples)

# Release cached CuPy GPU memory before running apodization on TensorFlow.
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

print("PICMUS pipeline completed.")
print(f"Angles used: {cfg['n_angles']}")
print(f"RF shape used: {rf_sel.shape}")
print(f"Delayed samples shape: {delayed_samples.shape}")
print(f"Uniform DAS image shape: {das_uniform.shape}")
print(f"Effective ROI [xmin, xmax, zmin, zmax] (mm): {kp.roi_effective}")

# Compute apodizations if enabled
print("\nProcessing apodizations...")
cm = build_coordinate_manager(kp)
apod_dict = {}  # Store computed apodizations

# Hanning apodization
if pipeline_cfg["apodizations"]["hanning"]["enabled"]:
    f_number = float(pipeline_cfg["apodizations"]["hanning"]["f_number"])
    z_chunk_size = int(pipeline_cfg["apodizations"]["hanning"].get("z_chunk_size", 16))
    angle_chunk_size = int(pipeline_cfg["apodizations"]["hanning"].get("angle_chunk_size", 1))
    print(
        "Computing Hanning DAS in chunks "
        f"(f_number={f_number}, z_chunk_size={z_chunk_size}, angle_chunk_size={angle_chunk_size})..."
    )
    das_hanning = compute_hanning_das_image_chunked(
        delayed_samples=delayed_samples,
        cm=cm,
        f_number=f_number,
        z_chunk_size=z_chunk_size,
        angle_chunk_size=angle_chunk_size,
    )
    apod_dict["Hanning"] = das_hanning
    print(f"  Hanning DAS computed: shape={das_hanning.shape}")

# INR apodization
if pipeline_cfg["apodizations"]["inr"]["enabled"]:
    model_path = pipeline_cfg["apodizations"]["inr"]["model_path"]
    feature_batch_size = int(pipeline_cfg["apodizations"]["inr"].get("feature_batch_size", 32768))
    z_chunk_size = int(pipeline_cfg["apodizations"]["inr"].get("z_chunk_size", 16))
    angle_chunk_size = int(pipeline_cfg["apodizations"]["inr"].get("angle_chunk_size", 1))
    print(f"Loading INR model from: {model_path}")
    try:
        inr_model = load_inr_model(model_path)
        print(
            "Computing INR DAS in chunks "
            f"(feature_batch_size={feature_batch_size}, z_chunk_size={z_chunk_size}, "
            f"angle_chunk_size={angle_chunk_size})..."
        )
        das_inr = compute_inr_das_image_chunked(
            delayed_samples=delayed_samples,
            model=inr_model,
            cm=cm,
            feature_batch_size=feature_batch_size,
            z_chunk_size=z_chunk_size,
            angle_chunk_size=angle_chunk_size,
            scaled_features=True,
        )
        apod_dict["INR"] = das_inr
        print(f"  INR DAS computed: shape={das_inr.shape}")
    except FileNotFoundError as e:
        print(f"  Warning: {e}")

# Plot all apodizations side-by-side if any were computed
if apod_dict:
    n_apod = len(apod_dict) + 1  # +1 for uniform
    fig, axes = plt.subplots(1, n_apod, figsize=(5*n_apod, 5))
    if n_apod == 1:
        axes = [axes]  # Make iterable
    extent = kp.get_imshow_extent()
    
    # Uniform DAS
    das_uniform_db = 20.0 * np.log10(das_uniform / (np.max(das_uniform) + 1e-6) + 1e-6)
    im0 = axes[0].imshow(das_uniform_db, aspect="auto", cmap="gray", extent=extent, vmin=-60, vmax=0)
    axes[0].set_title("Uniform DAS (dB)")
    axes[0].set_xlabel("Lateral [mm]")
    axes[0].set_ylabel("Axial [mm]")
    plt.colorbar(im0, ax=axes[0], label="dB")

    # Apodized images
    for idx, (apod_name, das_img) in enumerate(apod_dict.items(), start=1):
        das_apod_db = 20.0 * np.log10(das_img / (np.max(das_img) + 1e-6) + 1e-6)
        im = axes[idx].imshow(das_apod_db, aspect="auto", cmap="gray", extent=extent, vmin=-60, vmax=0)
        axes[idx].set_title(f"{apod_name} DAS (dB)")
        axes[idx].set_xlabel("Lateral [mm]")
        axes[idx].set_ylabel("Axial [mm]")
        plt.colorbar(im, ax=axes[idx], label="dB")

    plt.tight_layout()
    plt.show()

    print("\nApodization images plotted.")


