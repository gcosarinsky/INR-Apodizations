"""Scatterer selection utilities for profile-centered evaluation plots.

Provides helpers to resolve which scatterer to use as the center of a
reflector lateral/axial profile figure, given a selection mode or an
explicit index.
"""

from __future__ import annotations

import numpy as np


def select_reflector_scatterer_index(
    scatterers_mm: np.ndarray,
    selection: str = "strongest",
    scatterer_idx: int | None = None,
) -> int:
    """Resolve the reflector index used for profile-centered plots.

    Args:
        scatterers_mm: Scatterer coordinates with columns ``[x_mm, z_mm]``
            or ``[x_mm, z_mm, reflectivity]``.
        selection: Selection mode. Supported values are ``"strongest"``
            (highest absolute reflectivity when column 2 is present) and
            ``"first"``.
        scatterer_idx: Optional explicit index. When provided it takes
            priority over ``selection``.

    Returns:
        Integer index of the selected scatterer.

    Raises:
        ValueError: If the array is empty, the index is out of range, or
            the selection mode is unsupported.
    """
    if scatterers_mm.shape[0] == 0:
        raise ValueError("No scatterers available for reflector profile selection.")

    if scatterer_idx is not None:
        if scatterer_idx < 0 or scatterer_idx >= scatterers_mm.shape[0]:
            raise ValueError(
                f"scatterer_idx={scatterer_idx} is out of range "
                f"[0, {scatterers_mm.shape[0] - 1}]."
            )
        return int(scatterer_idx)

    selection_normalized = selection.strip().lower()
    if selection_normalized == "strongest":
        if scatterers_mm.shape[1] >= 3:
            return int(np.argmax(np.abs(scatterers_mm[:, 2])))
        return 0
    if selection_normalized == "first":
        return 0
    raise ValueError("scatterer_selection must be 'strongest' or 'first'.")


def select_reflector_scatterer(
    scatterers_mm: np.ndarray,
    selection: str = "strongest",
    scatterer_idx: int | None = None,
) -> np.ndarray:
    """Select a single scatterer row to center reflector profile plots.

    Convenience wrapper around :func:`select_reflector_scatterer_index`
    that returns the full row instead of the index.

    Args:
        scatterers_mm: Scatterer coordinates with columns
            ``[x_mm, z_mm]`` or ``[x_mm, z_mm, reflectivity]``.
        selection: Selection mode — ``"strongest"`` or ``"first"``.
        scatterer_idx: Optional explicit index with priority over
            ``selection``.

    Returns:
        Selected scatterer row as a 1-D NumPy array.

    Raises:
        ValueError: If the array is empty, the index is out of range, or
            the selection mode is unsupported.
    """
    idx = select_reflector_scatterer_index(
        scatterers_mm,
        selection=selection,
        scatterer_idx=scatterer_idx,
    )
    return scatterers_mm[idx]
