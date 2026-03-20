"""Test script: compute TF apodizations from CoordinateManager and plot results.

Saves PNG figures to `sandbox/figures/` and prints basic info.
"""
from pathlib import Path
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from inr_apodizations.config import DATA_DIR
from inr_apodizations.kernels import KernelParameters2D
from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.utils import find_latest_dataset_folder

from sandbox.apodizations_tf import (
    compute_dynamic_apodizations_tf,
    extract_map_for_x,
    extract_profile_for_z,
)

plt.ion()  # Interactive mode for plotting


def ensure_fig_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


latest = find_latest_dataset_folder(DATA_DIR, "delayed_samples_dataset")
print(f"Using dataset: {latest}")

cfg_path = latest / 'cfg_delayed_samples.npy'
if not cfg_path.exists():
    raise FileNotFoundError(f"Config file not found in {latest}")

cfg = np.load(cfg_path, allow_pickle=True).item()

kp = KernelParameters2D(cfg)
print("KernelParameters2D created", kp)

cm = CoordinateManager(kp)
print("CoordinateManager instantiated")

# Parameters (edit directly here)
methods = ['boxcar', 'hanning']
x_fixed = 0.0
z_fixed = 20.0

apods = compute_dynamic_apodizations_tf(cm, kp.bfd, methods=methods, scaled=False)

figs_dir = Path('sandbox/figures')
ensure_fig_dir(figs_dir)

coords = cm.get_coordinates_1d(scaled=False)
x_elems = np.asarray(coords['x_elem'])
z_coords = np.asarray(coords['z'])

for method, tensor in apods.items():
    # tensor: tf.Tensor (n_elem, nz, nx)
    assert isinstance(tensor, tf.Tensor)
    n_elem, nz, nx = tensor.shape
    print(f"Method {method}: shape=(n_elem={n_elem}, nz={nz}, nx={nx})")

    # Map (z vs element) at x_fixed
    map_z_elem_tf = extract_map_for_x(tensor, cm, x_fixed, scaled=False)
    map_z_elem = map_z_elem_tf.numpy()

    plt.figure(figsize=(8, 5))
    extent = kp.get_imshow_extent()
    plt.imshow(map_z_elem, aspect='auto', extent=extent)
    plt.colorbar(label='Apodization')
    plt.xlabel('Element lateral coordinate (mm)')
    plt.ylabel('Depth z (mm)')
    plt.title(f'Apodization map ({method}) at x={x_fixed:.2f} mm')
    out_map = figs_dir / f"map_{method}_x{float(x_fixed):.2f}.png"
    plt.savefig(out_map, dpi=150, bbox_inches='tight')
    print(f"Saved {out_map}")


    # Profile at z_fixed (for given x_fixed)
    profile_tf = extract_profile_for_z(tensor, cm, z_fixed, x_fixed=x_fixed, scaled=False)
    profile = profile_tf.numpy()

    plt.figure(figsize=(8, 3))
    plt.plot(x_elems, profile, marker='o')
    plt.xlabel('Element lateral coordinate (mm)')
    plt.ylabel('Apodization')
    plt.title(f'Apodization profile ({method}) at z={z_fixed:.2f} mm, x={x_fixed:.2f} mm')
    plt.grid(True)
    out_profile = figs_dir / f"profile_{method}_z{float(z_fixed):.2f}_x{float(x_fixed):.2f}.png"
    plt.savefig(out_profile, dpi=150, bbox_inches='tight')
    print(f"Saved {out_profile}")

    # Basic checks
    assert np.all(profile >= -1e-6) and np.all(profile <= 1.0 + 1e-6)
    print(f"Values in range [0,1] for method {method}: OK")
