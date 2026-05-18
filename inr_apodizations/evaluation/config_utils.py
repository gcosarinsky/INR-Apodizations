"""Configuration helpers shared by numeric phantom evaluation scripts."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_grid_reflector_points(cfg: dict[str, Any]) -> np.ndarray:
    """Build reflector positions ``(N, 2)`` in mm from ``phantom.grid`` config.

    Args:
        cfg: Full evaluation config mapping.

    Returns:
        Reflector coordinates as ``(N, 2)`` array in mm.

    Raises:
        ValueError: If phantom mode or grid fields are invalid.
    """
    phantom_cfg = cfg.get("phantom", {})
    mode = str(phantom_cfg.get("mode", "")).strip().lower()
    if mode != "grid":
        raise ValueError(
            "This reflector-profile flow requires `phantom.mode: grid` in the evaluation config."
        )

    grid_cfg = phantom_cfg.get("grid")
    if not isinstance(grid_cfg, dict):
        raise ValueError("Missing `phantom.grid` block in config.")

    required = (
        "x_count",
        "z_count",
        "x_center_mm",
        "z_start_mm",
        "x_spacing_mm",
        "z_spacing_mm",
    )
    missing = [name for name in required if name not in grid_cfg]
    if missing:
        raise ValueError(f"Missing required fields in `phantom.grid`: {missing}")

    x_count = int(grid_cfg["x_count"])
    z_count = int(grid_cfg["z_count"])
    x_center_mm = float(grid_cfg["x_center_mm"])
    z_start_mm = float(grid_cfg["z_start_mm"])
    x_spacing_mm = float(grid_cfg["x_spacing_mm"])
    z_spacing_mm = float(grid_cfg["z_spacing_mm"])

    if x_count <= 0 or z_count <= 0:
        raise ValueError("`x_count` and `z_count` must be positive integers.")
    if x_spacing_mm <= 0.0 or z_spacing_mm <= 0.0:
        raise ValueError("`x_spacing_mm` and `z_spacing_mm` must be > 0.")

    x_indices = np.arange(x_count, dtype=np.float64)
    x_offsets = (x_indices - (x_count - 1) / 2.0) * x_spacing_mm
    x_positions = x_center_mm + x_offsets

    points: list[list[float]] = []
    for z_idx in range(z_count):
        z_pos = z_start_mm + z_idx * z_spacing_mm
        for x_pos in x_positions:
            points.append([float(x_pos), float(z_pos)])

    return np.asarray(points, dtype=np.float64)


def resolve_reflector_indices(indices_cfg: Any, n_reflectors: int) -> np.ndarray:
    """Resolve user-selected reflector indices from YAML values.

    Accepted values are ``"all"`` (default), or a list of integer indices.

    Args:
        indices_cfg: Raw config value for reflector indices.
        n_reflectors: Number of available reflectors.

    Returns:
        Sorted and unique reflector indices.

    Raises:
        ValueError: If indices are missing, empty, or out-of-range.
    """
    if isinstance(indices_cfg, str) and indices_cfg.strip().lower() == "all":
        return np.arange(n_reflectors, dtype=np.int32)
    if indices_cfg is None:
        return np.arange(n_reflectors, dtype=np.int32)
    if not isinstance(indices_cfg, list) or len(indices_cfg) == 0:
        raise ValueError(
            "`reflector_lateral_profiles.reflector_indices` must be 'all' or "
            "a non-empty integer list."
        )

    resolved = np.asarray(indices_cfg, dtype=np.int32)
    if np.any(resolved < 0) or np.any(resolved >= n_reflectors):
        raise ValueError(
            "`reflector_lateral_profiles.reflector_indices` contains out-of-range values. "
            f"Valid range: [0, {n_reflectors - 1}]"
        )
    return np.unique(resolved)
