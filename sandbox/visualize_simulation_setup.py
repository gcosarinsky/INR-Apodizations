"""Visualize the simulated ultrasound setup: array, ROI, scatterers, and plane wave.

This script creates a schematic 2D view of the simulation geometry including:
- Transducer array (as a thin rectangle)
- Region of Interest (ROI) for simulation
- Example synthetic scatterers (grid pattern)
- Plane wave front at a configurable angle
- Information box with pitch, center frequency, and sound velocity

The script reads configuration from a YAML file and provides interactive
visualization with optional PNG export. Designed for exploration and quick
validation of simulation parameters.

Design: Simple cell-based script (no argparse/main). Docstrings in English,
user messages in Spanish.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Configuration defaults (editable)
CONFIG_PATH = Path("configs/rf_dataset_simus.yml")
OUTPUT_PATH = Path("sandbox/figures")
SAVE_PNG = True
WAVEFRONT_ANGLE_DEG = 5.0  # Angle of plane wave in degrees (positive = tilted right)
LANGUAGE = "es"  # 'es' Spanish, 'en' English
AXIS_MARGIN_MM = 8  # margin in mm around ROI for axis limits
FONT_SIZE = 14
INFO_FONT_SIZE = 11
TITLE_FONT_SIZE = FONT_SIZE + 2
LEGEND_FONT_SIZE = FONT_SIZE - 2

plt.ion()


# ============================================================================
# UTILITY: Load and validate configuration
# ============================================================================

def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration file and validate required keys.
    
    Args:
        path: Path to YAML config file.
    
    Returns:
        Dictionary with configuration loaded from YAML.
    
    Raises:
        FileNotFoundError: If config file does not exist.
        KeyError: If required keys are missing.
        yaml.YAMLError: If YAML parsing fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path) as f:
        cfg = yaml.safe_load(f)
    
    # Validate required keys
    required_keys = ["pitch", "fc", "c1", "n_elements", "dataset_generation"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys in config: {missing}")
    
    if "roi_simulation" not in cfg.get("dataset_generation", {}):
        raise KeyError("Missing 'roi_simulation' in dataset_generation section")
    
    return cfg


# ============================================================================
# PHASE 1: Extract parameters and prepare geometry
# ============================================================================

def build_geometry(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract and compute geometry parameters from configuration.
    
    Args:
        cfg: Configuration dictionary from YAML.
    
    Returns:
        Dictionary with geometry info: array bounds, ROI, pitch, fc, c1, etc.
    """
    pitch = cfg["pitch"]  # mm
    n_elements = cfg["n_elements"]
    fc = cfg["fc"]  # MHz
    c1 = cfg["c1"]  # mm/us
    roi_sim = cfg["dataset_generation"]["roi_simulation"]  # [xmin, xmax, zmin, zmax]
    
    # Array bounds (centered at x=0, z=0)
    array_width = (n_elements - 1) * pitch
    array_x_min = -array_width / 2
    array_x_max = array_width / 2
    array_z = 0  # Array is placed at z=0 (top surface)
    array_height = 0.5  # Thin rectangle for visualization (mm)
    
    geometry = {
        "pitch": pitch,
        "n_elements": n_elements,
        "fc": fc,
        "c1": c1,
        "roi_sim": roi_sim,  # [xmin, xmax, zmin, zmax]
        "array_x_min": array_x_min,
        "array_x_max": array_x_max,
        "array_z": array_z,
        "array_height": array_height,
        "array_width": array_width,
    }
    
    return geometry


# ============================================================================
# PHASE 2: Generate synthetic scatterers
# ============================================================================

def generate_synthetic_scatterers(
    roi: list[float],
    n_rows: int = 5,
    n_per_row: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic scatterer positions in a grid pattern with jitter.
    
    Creates a regular grid of scatterers within the ROI and adds small
    random offsets for realistic appearance.
    
    Args:
        roi: ROI bounds [xmin, xmax, zmin, zmax] in mm.
        n_rows: Number of rows of scatterers.
        n_per_row: Number of scatterers per row.
        seed: Random seed for reproducibility.
    
    Returns:
        Array of shape (n_scatterers, 2) with columns [x, z] in mm.
    """
    np.random.seed(seed)
    
    xmin, xmax, zmin, zmax = roi
    
    # Create regular grid
    x_grid = np.linspace(xmin + 2, xmax - 2, n_per_row)
    z_grid = np.linspace(zmin + 1, zmax - 1, n_rows)
    
    scatterers = []
    jitter_x = (xmax - xmin) * 0.02  # 2% jitter
    jitter_z = (zmax - zmin) * 0.02
    
    for z in z_grid:
        for x in x_grid:
            x_jit = x + np.random.uniform(-jitter_x, jitter_x)
            z_jit = z + np.random.uniform(-jitter_z, jitter_z)
            scatterers.append([x_jit, z_jit])
    
    return np.array(scatterers)


# ============================================================================
# PHASE 3: Plotting functions
# ============================================================================

def plot_simulation_setup(
    geometry: dict[str, Any],
    scatterers: np.ndarray,
    wavefront_angle_deg: float,
    figsize: tuple[float, float] = (12, 8),
    language: str = "es",
) -> tuple[plt.Figure, plt.Axes]:
    """Create main schematic figure with array, ROI, scatterers, and wavefront.
    
    Args:
        geometry: Geometry dictionary from build_geometry().
        scatterers: Scatterer positions (n, 2) with [x, z] in mm.
        wavefront_angle_deg: Angle of plane wave in degrees (positive = right tilt).
        figsize: Figure size in inches.
    
    Returns:
        Tuple (fig, ax) for further customization if needed.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract geometry
    roi = geometry["roi_sim"]
    xmin, xmax, zmin, zmax = roi
    
    # ---- Localized labels ----
    if language == "es":
        lbl_roi = "ROI (simulación)"
        lbl_array = "Array"
        lbl_scatterers = "Reflectores (ejemplo)"
        lbl_wave = f"Frente plano (θ={wavefront_angle_deg}°)"
        xlabel = "x [mm]"
        ylabel = "z [mm]"
        title = "Esquema del setup de simulación"
    else:
        lbl_roi = "ROI (simulation)"
        lbl_array = "Transducer array"
        lbl_scatterers = "Scatterers (example)"
        lbl_wave = f"Plane wave (θ={wavefront_angle_deg}°)"
        xlabel = "x [mm]"
        ylabel = "z [mm]"
        title = "Simulation Setup Schematic"

    # ---- Draw ROI rectangle ----
    roi_rect = plt.Rectangle(
        (xmin, zmin),
        xmax - xmin,
        zmax - zmin,
        fill=False,
        edgecolor="blue",
        linewidth=2,
        linestyle="--",
        label=lbl_roi,
    )
    ax.add_patch(roi_rect)
    
    # ---- Draw array as thin rectangle ----
    array_rect = plt.Rectangle(
        (geometry["array_x_min"], geometry["array_z"] - geometry["array_height"] / 2),
        geometry["array_width"],
        geometry["array_height"],
        fill=True,
        facecolor="lightgray",
        edgecolor="black",
        linewidth=2,
        label=lbl_array,
    )
    ax.add_patch(array_rect)
    
    # ---- Draw scatterers ----
    if len(scatterers) > 0:
        ax.scatter(
            scatterers[:, 0],
            scatterers[:, 1],
            s=40,
            marker="o",
            color="red",
            alpha=0.6,
            label=lbl_scatterers,
        )
    
    # ---- Draw plane wave front (inclined line) ----
    angle_rad = np.radians(wavefront_angle_deg)
    # Line passes through the center of the ROI at an angle
    roi_center_x = (xmin + xmax) / 2
    roi_center_z = (zmin + zmax) / 2
    
    # Extend line beyond ROI for visibility
    extend = max(abs(xmax - xmin), abs(zmax - zmin)) * 0.5
    
    # Parametric line: (x, z) = (roi_center_x, roi_center_z) + t * (cos(angle), sin(angle))
    t_vals = np.array([-extend, extend])
    x_wave = roi_center_x + t_vals * np.cos(angle_rad)
    z_wave = roi_center_z + t_vals * np.sin(angle_rad)
    
    ax.plot(
        x_wave,
        z_wave,
        color="green",
        linewidth=2.5,
        linestyle="-",
        label=lbl_wave,
    )
    
    # ---- Information box ----
    if language == "es":
        info_text = (
            f"Pitch: {geometry['pitch']:.2f} mm\n"
            f"f_c: {geometry['fc']:.3f} MHz\n"
            f"c: {geometry['c1']:.2f} mm/µs"
        )
    else:
        info_text = (
            f"Pitch: {geometry['pitch']:.2f} mm\n"
            f"f_c: {geometry['fc']:.3f} MHz\n"
            f"c: {geometry['c1']:.2f} mm/µs"
        )
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=INFO_FONT_SIZE,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        family="monospace",
    )
    
    # ---- Formatting ----
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE)
    
    # Set axis limits with padding and make axial axis point downwards
    padding = AXIS_MARGIN_MM
    ax.set_xlim(xmin - padding, xmax + padding)
    # For axial downwards, set ylim so top is zmin and bottom is zmax
    ax.set_ylim(zmax + padding, zmin - padding)
    
    plt.tight_layout()
    
    return fig, ax


# ============================================================================
# MAIN EXECUTION
# ============================================================================

#%% Load configuration and build geometry
cfg = load_config(CONFIG_PATH)
geometry = build_geometry(cfg)
print(f"✓ Loaded config from {CONFIG_PATH}")
print(f"  Probe: {geometry['n_elements']} elements, pitch={geometry['pitch']:.2f} mm")
print(f"  ROI: {geometry['roi_sim']}")
print(f"  Sound velocity: {geometry['c1']:.2f} mm/µs")

#%% Generate synthetic scatterers
scatterers = generate_synthetic_scatterers(
    geometry["roi_sim"],
    n_rows=5,
    n_per_row=8,
    seed=42,
)
print(f"✓ Generated {len(scatterers)} synthetic scatterers")

#%% Create main visualization
fig, ax = plot_simulation_setup(
    geometry,
    scatterers,
    wavefront_angle_deg=WAVEFRONT_ANGLE_DEG,
    figsize=(12, 8),
    language=LANGUAGE,
)
print(f"✓ Created simulation setup figure")

#%% Optionally save figure
if SAVE_PNG:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_PATH / "simulation_setup_schematic.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Saved figure to {output_file}")
else:
    print(f"  Tip: Set SAVE_PNG=True to export PNG")

plt.show()
print("✓ Visualization complete")
