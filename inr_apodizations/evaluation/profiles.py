"""Lateral and axial profile extraction and FWHM measurement.

Profile extraction is decoupled from the disk/circle ROI masks used for SNR;
the only input needed is the rectangular patch size in mm (or in pixels as a
fallback).  This allows using arbitrary window sizes for resolution
measurements independent of the SNR disk radius.
"""

from __future__ import annotations

import numpy as np

from inr_apodizations.coordinate_manager import CoordinateManager


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mm_to_pixels(value_mm: float, spacing_mm: float) -> int:
    """Convert a physical half-size to a pixel count (minimum 1)."""
    return max(1, round(value_mm / spacing_mm))


def _extract_1d_profile(
    image: np.ndarray,
    ix: int,
    iz: int,
    half_x: int,
    half_z: int,
    nz: int,
    nx: int,
    direction: str,
) -> np.ndarray:
    """Extract a max-projection 1D profile along one direction.

    A rectangular patch of size ``(2*half_z+1, 2*half_x+1)`` centred at
    ``(ix, iz)`` is extracted from ``image``.  The profile along ``direction``
    is computed as the column- or row-wise maximum, and the result is
    NaN-padded at image borders.

    Args:
        image: 2D array ``(nz, nx)``.
        ix: Lateral pixel index of the reflector centre.
        iz: Axial pixel index of the reflector centre.
        half_x: Half-width of the patch in x (pixels).
        half_z: Half-width of the patch in z (pixels).
        nz, nx: Image dimensions.
        direction: Either ``'lateral'`` or ``'axial'``.

    Returns:
        1D float64 profile of length ``2*half_x+1`` (lateral) or
        ``2*half_z+1`` (axial), with NaN where the patch exceeds image bounds.
    """
    if direction == "lateral":
        n_out = 2 * half_x + 1
    else:
        n_out = 2 * half_z + 1
    profile = np.full(n_out, np.nan, dtype=np.float64)

    x0 = ix - half_x
    x1 = ix + half_x + 1
    z0 = iz - half_z
    z1 = iz + half_z + 1

    img_x0 = max(0, x0)
    img_x1 = min(nx, x1)
    img_z0 = max(0, z0)
    img_z1 = min(nz, z1)

    if img_x0 >= img_x1 or img_z0 >= img_z1:
        return profile

    patch = image[img_z0:img_z1, img_x0:img_x1].astype(np.float64, copy=False)

    if direction == "lateral":
        insert = img_x0 - x0
        profile[insert: insert + patch.shape[1]] = np.max(patch, axis=0)
    else:
        insert = img_z0 - z0
        profile[insert: insert + patch.shape[0]] = np.max(patch, axis=1)

    return profile


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_reflector_profiles(
    image: np.ndarray,
    scatterers: np.ndarray,
    cm: CoordinateManager,
    half_width_lateral_mm: float | None = None,
    half_width_axial_mm: float | None = None,
    half_width_lateral_px: int | None = None,
    half_width_axial_px: int | None = None,
) -> dict:
    """Extract lateral and axial max-projection profiles for each reflector.

    The patch size for profile extraction is defined independently of any
    disk/SNR radius.  Physical units (mm) take priority; pixel counts are used
    as fallback.  If neither is provided the defaults are 1.5 mm laterally
    and 1.0 mm axially (typical PSF half-widths).

    Args:
        image: 2D image array ``(nz, nx)``, linear amplitude domain.
        scatterers: ``(N, 2)`` array of reflector positions ``[x, z]`` in mm.
        cm: CoordinateManager providing physical coordinate arrays.
        half_width_lateral_mm: Lateral half-width of the extraction window in mm.
        half_width_axial_mm: Axial half-width of the extraction window in mm.
        half_width_lateral_px: Fallback: lateral half-width in pixels (used only
            when ``half_width_lateral_mm`` is None).
        half_width_axial_px: Fallback: axial half-width in pixels (used only
            when ``half_width_axial_mm`` is None).

    Returns:
        Dictionary with keys:
        - ``lateral_profiles``: float64 array ``(N, 2*half_x+1)``.
        - ``axial_profiles``: float64 array ``(N, 2*half_z+1)``.
        - ``lateral_offsets_mm``: 1D array of lateral offsets from centre in mm.
        - ``axial_offsets_mm``: 1D array of axial offsets from centre in mm.
        - ``half_x_px``, ``half_z_px``: actual half-widths used (pixels).

    Raises:
        ValueError: If ``image`` is not 2D, ``scatterers`` shape is invalid,
            or no width specification is provided.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("image must be 2D (nz, nx)")
    scatterers = np.asarray(scatterers, dtype=np.float64)
    if scatterers.ndim != 2 or scatterers.shape[1] != 2:
        raise ValueError("scatterers must be (N, 2) with columns [x, z] in mm")

    nz, nx = image.shape
    coords = cm.get_coordinates_1d(scaled=False)
    x_coords = np.asarray(coords["x"], dtype=np.float64)
    z_coords = np.asarray(coords["z"], dtype=np.float64)

    dx = float(np.abs(x_coords[1] - x_coords[0])) if nx > 1 else 1.0
    dz = float(np.abs(z_coords[1] - z_coords[0])) if nz > 1 else 1.0

    # Resolve half-widths in pixels. mm takes priority over px fallbacks.
    if half_width_lateral_mm is not None:
        half_x = _mm_to_pixels(half_width_lateral_mm, dx)
    elif half_width_lateral_px is not None:
        half_x = max(1, int(half_width_lateral_px))
    else:
        half_x = _mm_to_pixels(1.5, dx)  # default 1.5 mm

    if half_width_axial_mm is not None:
        half_z = _mm_to_pixels(half_width_axial_mm, dz)
    elif half_width_axial_px is not None:
        half_z = max(1, int(half_width_axial_px))
    else:
        half_z = _mm_to_pixels(1.0, dz)  # default 1.0 mm

    n = scatterers.shape[0]
    lateral_profiles = np.full((n, 2 * half_x + 1), np.nan, dtype=np.float64)
    axial_profiles = np.full((n, 2 * half_z + 1), np.nan, dtype=np.float64)

    for i, (x0, z0) in enumerate(scatterers):
        ix = int(np.argmin(np.abs(x_coords - x0)))
        iz = int(np.argmin(np.abs(z_coords - z0)))
        lateral_profiles[i] = _extract_1d_profile(image, ix, iz, half_x, half_z, nz, nx, "lateral")
        axial_profiles[i] = _extract_1d_profile(image, ix, iz, half_x, half_z, nz, nx, "axial")

    lateral_offsets_mm = np.arange(-half_x, half_x + 1, dtype=np.float64) * dx
    axial_offsets_mm = np.arange(-half_z, half_z + 1, dtype=np.float64) * dz

    return {
        "lateral_profiles": lateral_profiles,
        "axial_profiles": axial_profiles,
        "lateral_offsets_mm": lateral_offsets_mm,
        "axial_offsets_mm": axial_offsets_mm,
        "half_x_px": half_x,
        "half_z_px": half_z,
    }


def compute_fwhm(
    profile: np.ndarray,
    offsets_mm: np.ndarray,
    eps: float = 1e-12,
) -> dict:
    """Compute the FWHM of a 1D profile.

    The profile is normalized to its maximum (ignoring NaN values) before
    measurement.

    The width is estimated by linear interpolation between sample pairs that
    straddle the half-maximum threshold. A NaN is returned if fewer than two
    such crossings are found.

    Args:
        profile: 1D amplitude profile (linear domain, non-negative).
        offsets_mm: Physical offset in mm for each sample; must have the same
            length as ``profile``.
        eps: Small value added when normalizing to avoid division by zero.

    Returns:
        Dictionary with keys:
                - ``fwhm_mm``: FWHM in mm, or NaN.
        - ``peak_value``: Maximum value of the input profile (NaN if all NaN).
        - ``peak_offset_mm``: Offset in mm corresponding to the peak.

    Raises:
        ValueError: If ``profile`` and ``offsets_mm`` lengths differ.
    """
    profile = np.asarray(profile, dtype=np.float64)
    offsets_mm = np.asarray(offsets_mm, dtype=np.float64)
    if profile.shape != offsets_mm.shape:
        raise ValueError(
            f"profile length {profile.shape} does not match offsets_mm length {offsets_mm.shape}"
        )

    valid = np.isfinite(profile)
    if not np.any(valid):
        return {
            "fwhm_mm": float("nan"),
            "peak_value": float("nan"),
            "peak_offset_mm": float("nan"),
        }

    peak_val = float(np.nanmax(profile))
    peak_idx = int(np.nanargmax(profile))
    peak_offset = float(offsets_mm[peak_idx])

    def _interpolated_width(threshold: float) -> float:
        """Estimate width (mm) by interpolating threshold crossings."""
        norm = profile / (peak_val + eps)
        above = norm >= threshold

        crossings = []
        for j in range(len(norm) - 1):
            if not valid[j] or not valid[j + 1]:
                continue
            a, b = norm[j], norm[j + 1]
            if (a < threshold) != (b < threshold):
                # linear interpolation
                t = (threshold - a) / (b - a + 1e-30)
                x_cross = offsets_mm[j] + t * (offsets_mm[j + 1] - offsets_mm[j])
                crossings.append(x_cross)

        if len(crossings) < 2:
            return float("nan")
        return float(crossings[-1] - crossings[0])

    fwhm_mm = _interpolated_width(0.5)

    return {
        "fwhm_mm": fwhm_mm,
        "peak_value": peak_val,
        "peak_offset_mm": peak_offset,
    }


def compute_fwhm_batch(
    profiles: np.ndarray,
    offsets_mm: np.ndarray,
    eps: float = 1e-12,
) -> dict:
    """Apply :func:`compute_fwhm` to a batch of profiles.

    Args:
        profiles: 2D array ``(N, n_samples)`` of amplitude profiles.
        offsets_mm: 1D array ``(n_samples,)`` of physical offsets in mm,
            shared by all profiles.
        eps: Normalization epsilon passed to :func:`compute_fwhm`.

    Returns:
        Dictionary with keys:
        - ``fwhm_mm``: float64 array ``(N,)``.
        - ``peak_values``: float64 array ``(N,)``.
        - ``peak_offsets_mm``: float64 array ``(N,)``.

    Raises:
        ValueError: If ``profiles`` is not 2D or lengths are inconsistent.
    """
    profiles = np.asarray(profiles, dtype=np.float64)
    offsets_mm = np.asarray(offsets_mm, dtype=np.float64)
    if profiles.ndim != 2:
        raise ValueError("profiles must be 2D (N, n_samples)")
    if profiles.shape[1] != offsets_mm.shape[0]:
        raise ValueError(
            f"profiles.shape[1]={profiles.shape[1]} does not match "
            f"offsets_mm length {offsets_mm.shape[0]}"
        )

    n = profiles.shape[0]
    fwhm_mm = np.empty(n, dtype=np.float64)
    peak_vals = np.empty(n, dtype=np.float64)
    peak_offsets = np.empty(n, dtype=np.float64)

    for i in range(n):
        result = compute_fwhm(profiles[i], offsets_mm, eps=eps)
        fwhm_mm[i] = result["fwhm_mm"]
        peak_vals[i] = result["peak_value"]
        peak_offsets[i] = result["peak_offset_mm"]

    return {
        "fwhm_mm": fwhm_mm,
        "peak_values": peak_vals,
        "peak_offsets_mm": peak_offsets,
    }
