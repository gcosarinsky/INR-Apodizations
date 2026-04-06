#%%
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cupy as cp
import inr_apodizations.hilbert_coef as hilb
import matplotlib.pyplot as plt
import numpy as np
import pymust
import yaml
from scipy import signal

from inr_apodizations.config import CUDA_DIR
from inr_apodizations.kernels import KernelParameters2D
from inr_apodizations.utils import cfg_to_must_param, save_config_yaml, to_db

plt.ion()


def _require_keys(cfg: dict[str, Any], section_name: str, keys: list[str]) -> None:
    """Validate that a configuration section includes required keys.

    Args:
        cfg: Section dictionary to validate.
        section_name: Human-readable section name for error messages.
        keys: Required keys for the section.

    Raises:
        KeyError: If any required key is missing.
    """
    missing = [key for key in keys if key not in cfg]
    if missing:
        raise KeyError(f"Missing keys in section '{section_name}': {missing}")


def _build_grid_scatterers(phantom_cfg: dict[str, Any]) -> np.ndarray:
    """Build a regular scatterer grid from phantom settings.

    Args:
        phantom_cfg: Phantom configuration with the ``grid`` subsection.

    Returns:
        Array with shape ``(n_scatterers, 3)`` and columns ``[x_m, z_m, amplitude]``.

    Raises:
        KeyError: If required grid keys are missing.
        ValueError: If spacing or counts are invalid.
    """
    grid_cfg = phantom_cfg["grid"]
    _require_keys(
        grid_cfg,
        "phantom.grid",
        [
            "x_count",
            "z_count",
            "x_center_mm",
            "z_start_mm",
            "x_spacing_mm",
            "z_spacing_mm",
            "amplitude_range",
        ],
    )

    x_count = int(grid_cfg["x_count"])
    z_count = int(grid_cfg["z_count"])
    x_spacing_mm = float(grid_cfg["x_spacing_mm"])
    z_spacing_mm = float(grid_cfg["z_spacing_mm"])

    if x_count <= 0 or z_count <= 0:
        raise ValueError("Grid counts must be positive integers.")
    if x_spacing_mm <= 0 or z_spacing_mm <= 0:
        raise ValueError("Grid spacing must be strictly positive.")

    x_center_mm = float(grid_cfg["x_center_mm"])
    z_start_mm = float(grid_cfg["z_start_mm"])

    x_offsets = (np.arange(x_count) - 0.5 * (x_count - 1)) * x_spacing_mm
    z_offsets = np.arange(z_count) * z_spacing_mm

    x_mm = x_center_mm + x_offsets
    z_mm = z_start_mm + z_offsets
    xx_mm, zz_mm = np.meshgrid(x_mm, z_mm)

    n_points = xx_mm.size
    unit_amplitude = bool(phantom_cfg.get("unit_amplitude", True))

    if unit_amplitude:
        amplitudes = np.ones(n_points, dtype=np.float32)
    else:
        amp_min, amp_max = map(float, grid_cfg["amplitude_range"])
        rng = np.random.default_rng(int(phantom_cfg.get("seed", 42)))
        amplitudes = rng.uniform(amp_min, amp_max, size=n_points).astype(np.float32)

    scat = np.stack(
        [
            (xx_mm.reshape(-1) / 1000.0).astype(np.float64),
            (zz_mm.reshape(-1) / 1000.0).astype(np.float64),
            amplitudes.astype(np.float64),
        ],
        axis=1,
    )
    return scat


def _build_list_scatterers(phantom_cfg: dict[str, Any]) -> np.ndarray:
    """Build scatterers from an explicit list of ``[x_mm, z_mm, amp]`` points.

    Args:
        phantom_cfg: Phantom configuration with the ``list`` subsection.

    Returns:
        Array with shape ``(n_scatterers, 3)`` and columns ``[x_m, z_m, amplitude]``.

    Raises:
        KeyError: If the point list is missing.
        ValueError: If any point has an invalid format.
    """
    list_cfg = phantom_cfg["list"]
    _require_keys(list_cfg, "phantom.list", ["points_mm"])

    points_mm = list_cfg["points_mm"]
    if not points_mm:
        raise ValueError("phantom.list.points_mm is empty. Add at least one scatterer point.")

    unit_amplitude = bool(phantom_cfg.get("unit_amplitude", True))
    parsed_points = []
    for idx, point in enumerate(points_mm):
        if len(point) not in (2, 3):
            raise ValueError(
                f"Point at index {idx} must have 2 or 3 values: [x_mm, z_mm, amp?]."
            )
        x_mm = float(point[0])
        z_mm = float(point[1])
        amp = 1.0 if unit_amplitude else float(point[2] if len(point) == 3 else 1.0)
        parsed_points.append([x_mm / 1000.0, z_mm / 1000.0, amp])

    return np.asarray(parsed_points, dtype=np.float64)


def build_scatterers(phantom_cfg: dict[str, Any]) -> np.ndarray:
    """Create scatterers according to the selected phantom mode.

    Args:
        phantom_cfg: Full phantom section from the evaluation config.

    Returns:
        Array of scatterers with columns ``[x_m, z_m, amplitude]``.

    Raises:
        ValueError: If the requested mode is not supported.
    """
    mode = str(phantom_cfg.get("mode", "grid")).lower()
    if mode == "grid":
        return _build_grid_scatterers(phantom_cfg)
    if mode == "list":
        return _build_list_scatterers(phantom_cfg)
    raise ValueError(f"Unsupported phantom.mode '{mode}'. Use 'grid' or 'list'.")


def estimate_n_samples(roi_mm: list[float], c1_mm_per_us: float, fs_mhz: float) -> int:
    """Estimate RF sample count from a rectangular ROI.

    Args:
        roi_mm: ROI in millimeters as ``[x_min, x_max, z_min, z_max]``.
        c1_mm_per_us: Sound speed in mm/us.
        fs_mhz: Sampling frequency in MHz.

    Returns:
        Estimated number of RF samples.
    """
    d1 = 0.5 * (float(roi_mm[1]) - float(roi_mm[0]))
    d2 = float(roi_mm[3]) - float(roi_mm[2])
    diagonal_mm = np.hypot(d1, d2)
    t_total_us = 2.0 * diagonal_mm / float(c1_mm_per_us)
    return int(np.ceil(t_total_us * float(fs_mhz)))


#%% ===== Load Config =====
script_dir = Path(__file__).resolve().parent
config_path = script_dir / "evaluation_config.yml"

with open(config_path, encoding="utf-8") as file:
    cfg = yaml.safe_load(file)

_require_keys(cfg, "root", ["io", "phantom", "simulation", "beamforming", "visualization"])

io_cfg = cfg["io"]
phantom_cfg = cfg["phantom"]
sim_cfg = cfg["simulation"]
bf_cfg = cfg["beamforming"]
vis_cfg = cfg["visualization"]

_require_keys(
    sim_cfg,
    "simulation",
    [
        "fs",
        "fc",
        "pitch",
        "element_width",
        "element_height",
        "n_elements",
        "bandwidth",
        "c1",
        "angles",
        "t_start",
    ],
)
_require_keys(
    bf_cfg,
    "beamforming",
    ["f1", "f2", "taps", "f_number", "x_step", "z_step", "roi_user", "blocksize_img", "n_batch", "n_ch"],
)

#%% ===== Output Folder =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_root = Path(io_cfg.get("output_root", script_dir / "outputs"))
if not output_root.is_absolute():
    output_root = (Path.cwd() / output_root).resolve()

run_folder = output_root / timestamp
run_folder.mkdir(parents=True, exist_ok=True)
print(f"Output folder: {run_folder}")

#%% ===== Build Phantom and RF =====
scatterers = build_scatterers(phantom_cfg)
angles_deg = np.arange(*sim_cfg["angles"])
angles_rad = np.deg2rad(angles_deg)
n_angles = int(len(angles_rad))

if n_angles == 0:
    raise ValueError("No plane-wave angles generated from simulation.angles.")

roi_for_samples = sim_cfg.get("roi_simulation", bf_cfg["roi_user"])
n_samples = int(sim_cfg.get("n_samples", 0) or 0)
if n_samples <= 0:
    n_samples = estimate_n_samples(roi_for_samples, sim_cfg["c1"], sim_cfg["fs"])

n_elements = int(sim_cfg["n_elements"])
rf_array = np.zeros((1, n_angles, n_elements, n_samples), dtype=np.int16)
param_must = cfg_to_must_param(sim_cfg)
tx_delays = [pymust.txdelay(param_must, angle).reshape(1, -1) for angle in angles_rad]

for angle_idx, tx_delay in enumerate(tx_delays):
    rf, _ = pymust.simus(
        scatterers[:, 0],
        scatterers[:, 1],
        scatterers[:, 2],
        tx_delay,
        param_must,
    )
    rf_max = np.max(np.abs(rf))
    if rf_max > 0:
        rf = (rf / rf_max) * 32767.0
    rf = rf.astype(np.int16)

    n_samples_actual = int(rf.shape[0])
    if n_samples_actual < n_samples:
        rf_fixed = np.pad(rf, ((0, n_samples - n_samples_actual), (0, 0)), mode="constant")
    else:
        rf_fixed = rf[:n_samples, :]

    rf_array[0, angle_idx, :, :] = rf_fixed.T

np.save(run_folder / "rf.npy", rf_array)
np.save(run_folder / "scatterers.npy", scatterers)

#%% ===== Build Delayed Samples =====
kp_cfg = {
    "fs": sim_cfg["fs"],
    "c1": sim_cfg["c1"],
    "pitch": sim_cfg["pitch"],
    "f1": bf_cfg["f1"],
    "f2": bf_cfg["f2"],
    "f_number": bf_cfg["f_number"],
    "x_step": bf_cfg["x_step"],
    "z_step": bf_cfg["z_step"],
    "t_start": sim_cfg["t_start"],
    "taps": bf_cfg["taps"],
    "n_batch": bf_cfg["n_batch"],
    "n_elements": sim_cfg["n_elements"],
    "n_ch": bf_cfg["n_ch"],
    "n_angles": n_angles,
    "n_samples": n_samples,
    "roi_user": bf_cfg["roi_user"],
    "blocksize_img": bf_cfg["blocksize_img"],
}
kp = KernelParameters2D(kp_cfg)

codefiles = [
    "constants.h",
    "enum_parameters.c",
    "fir_filter.cu",
    "pwi_1pix_per_thread.cu",
]
code = ""
for codefile in codefiles:
    with open(CUDA_DIR / codefile, encoding="utf-8") as file:
        code += file.read() + "\n"

kp.check_enum_consistency(code)
module = cp.RawModule(code=code, options=("--use_fast_math",))
filt_kernel = module.get_function("fir_filter")
pwi_gather_kernel = module.get_function("pwi_gather_delayed_samples")

bandpass_coef = signal.firwin(
    bf_cfg["taps"] + 1,
    [2 * bf_cfg["f1"] / sim_cfg["fs"], 2 * bf_cfg["f2"] / sim_cfg["fs"]],
    pass_zero=False,
)
bandpass_coef_gpu = cp.asarray(bandpass_coef, dtype=cp.float32)
hilb_coef_gpu = cp.asarray(hilb.coef, dtype=cp.float32)

int_params = cp.asarray(kp.get_int_array(), dtype=cp.int32)
float_params = cp.asarray(kp.get_float_array(), dtype=cp.float32)
angles_gpu = cp.asarray(angles_rad, dtype=cp.float32)

nblock = 128
n_ascans = n_angles * n_elements
grid_size = ((n_ascans + nblock - 1) // nblock,)
block_size = (nblock,)

rf_gpu = cp.asarray(rf_array[0])
rf_filt_gpu = cp.zeros_like(rf_gpu)
rf_imag_gpu = cp.zeros_like(rf_gpu)

filt_kernel(grid_size, block_size, (int_params, rf_gpu, bandpass_coef_gpu, rf_filt_gpu))
filt_kernel(grid_size, block_size, (int_params, rf_filt_gpu, hilb_coef_gpu, rf_imag_gpu))

delayed_samples_gpu = cp.zeros((n_angles, n_elements, kp.nz, kp.nx), dtype=cp.complex64)
pwi_gather_kernel(
    kp.gridsize_img,
    kp.blocksize_img,
    (int_params, float_params, angles_gpu, rf_filt_gpu, rf_imag_gpu, delayed_samples_gpu),
)

delayed_samples = cp.asnumpy(delayed_samples_gpu.sum(axis=0))
delayed_samples_all = delayed_samples[np.newaxis, ...]

cp.cuda.Device().synchronize()

np.save(run_folder / "delayed_samples_signal.npy", delayed_samples_all)

#%% ===== Metadata =====
das_uniform = delayed_samples.sum(axis=0)
meta = {
    "generated": timestamp,
    "phantom_mode": phantom_cfg.get("mode", "grid"),
    "n_scatterers": int(scatterers.shape[0]),
    "rf_shape": list(rf_array.shape),
    "delayed_samples_shape": list(delayed_samples_all.shape),
    "roi_effective_mm": list(kp.roi_effective),
    "angles_deg": angles_deg.tolist(),
    "config": cfg,
}
save_config_yaml(run_folder / "simulation_info.yml", meta, {})

#%% ===== Quick Visualization =====
if bool(vis_cfg.get("enabled", True)):
    dyn_range_db = float(vis_cfg.get("dyn_range_db", 60.0))
    rf_view = np.abs(rf_array[0, 0].T)
    das_db = to_db(das_uniform, ref=np.max(np.abs(das_uniform)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(rf_view, aspect="auto", cmap="gray")
    axes[0].set_title("RF example (first angle)")
    axes[0].set_xlabel("Element index")
    axes[0].set_ylabel("Sample index")

    axes[1].imshow(
        das_db,
        aspect="auto",
        cmap="gray",
        extent=kp.get_imshow_extent(),
        vmin=-dyn_range_db,
        vmax=0,
    )
    axes[1].set_title("Uniform DAS (log scale)")
    axes[1].set_xlabel("Lateral [mm]")
    axes[1].set_ylabel("Axial [mm]")

    plt.tight_layout()
    if bool(io_cfg.get("save_plot", True)):
        fig.savefig(run_folder / "quicklook.png", dpi=140)
    plt.show()

print("Simulation generation complete.")
print(f"Artifacts saved in: {run_folder}")
