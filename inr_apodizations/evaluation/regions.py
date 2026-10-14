"""ROI region utilities for scatterer SNR evaluation.

Provides disk/ellipse mask generation and background statistics extraction
for per-reflector SNR computation.  Profile extraction is intentionally kept
in a separate module (profiles.py) to decouple the two concerns.
"""

from __future__ import annotations

import numpy as np

from inr_apodizations.coordinate_manager import CoordinateManager


def build_disk_context(
    image_shape: tuple[int, int],
    cm: CoordinateManager,
    radius_mm: float,
) -> dict:
    """Precompute geometric context for an elliptical disk ROI.

    The disk is defined as the ellipse
    ``(x_offset / rx)^2 + (z_offset / rz)^2 <= 1`` where ``rx`` and ``rz``
    are the radii expressed in pixels along each axis.  The pixel counts are
    derived from ``radius_mm`` and the physical pixel spacings of ``cm``.

    Args:
        image_shape: ``(nz, nx)`` size of images that will use this context.
        cm: CoordinateManager providing physical coordinate arrays.
        radius_mm: Disk radius in mm (same value applied to both axes; the
            pixel counts may differ if spacing is anisotropic).

    Returns:
        Dictionary with keys:
        - ``nz``, ``nx``: image dimensions.
        - ``x_coords``, ``z_coords``: 1D physical coordinate arrays (mm).
        - ``dx``, ``dz``: pixel spacing in mm.
        - ``disk``: 2D boolean template ``(2r+1, 2r+1)`` for the ellipse.
        - ``r``: bounding-box half-size in pixels (``max(rx, rz)``).
        - ``rx``, ``rz``: half-radii in pixels along x and z.

    Raises:
        ValueError: If ``image_shape`` is not 2D or ``radius_mm <= 0``.
    """
    if len(image_shape) != 2:
        raise ValueError("image_shape must be a 2D tuple (nz, nx)")
    if radius_mm <= 0.0:
        raise ValueError("radius_mm must be > 0")

    nz, nx = image_shape
    coords = cm.get_coordinates_1d(scaled=False)
    x_coords = np.asarray(coords["x"], dtype=np.float64)
    z_coords = np.asarray(coords["z"], dtype=np.float64)

    if x_coords.shape[0] != nx or z_coords.shape[0] != nz:
        raise ValueError(
            "image_shape does not match CoordinateManager 1D coordinate dimensions"
        )

    dx = float(np.abs(x_coords[1] - x_coords[0])) if nx > 1 else 1.0
    dz = float(np.abs(z_coords[1] - z_coords[0])) if nz > 1 else 1.0

    rx = max(1, round(radius_mm / dx))
    rz = max(1, round(radius_mm / dz))
    r = max(rx, rz)
    gy, gx = np.ogrid[-r: r + 1, -r: r + 1]
    disk = (gx / rx) ** 2 + (gy / rz) ** 2 <= 1.0

    return {
        "nz": nz,
        "nx": nx,
        "x_coords": x_coords,
        "z_coords": z_coords,
        "dx": dx,
        "dz": dz,
        "disk": disk,
        "r": r,
        "rx": rx,
        "rz": rz,
    }


def build_roi_masks(
    scatterers: np.ndarray,
    disk_ctx: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-scatterer disk masks and their union for a single image.

    Args:
        scatterers: ``(N, 2)`` array of scatterer positions ``[x, z]`` in mm.
        disk_ctx: Disk geometry context returned by :func:`build_disk_context`.

    Returns:
        Tuple ``(individual_masks, union_mask)`` where:
        - ``individual_masks``: bool array ``(N, nz, nx)``, one disk per scatterer.
        - ``union_mask``: bool array ``(nz, nx)``, union of all disks.

    Raises:
        ValueError: If ``scatterers`` is not a ``(N, 2)`` array.
    """
    scatterers = np.asarray(scatterers, dtype=np.float64)
    if scatterers.ndim != 2 or scatterers.shape[1] != 2:
        raise ValueError("scatterers must be (N, 2) with columns [x, z] in mm")

    nz = int(disk_ctx["nz"])
    nx = int(disk_ctx["nx"])
    x_coords = disk_ctx["x_coords"]
    z_coords = disk_ctx["z_coords"]
    disk = disk_ctx["disk"]
    r = int(disk_ctx["r"])

    n = scatterers.shape[0]
    individual_masks = np.zeros((n, nz, nx), dtype=bool)
    union_mask = np.zeros((nz, nx), dtype=bool)

    for i, (x0, z0) in enumerate(scatterers):
        ix = int(np.argmin(np.abs(x_coords - x0)))
        iz = int(np.argmin(np.abs(z_coords - z0)))

        iz0 = iz - r
        iz1 = iz + r + 1
        ix0 = ix - r
        ix1 = ix + r + 1

        disk_z0 = max(0, -iz0)
        disk_z1 = disk.shape[0] - max(0, iz1 - nz)
        disk_x0 = max(0, -ix0)
        disk_x1 = disk.shape[1] - max(0, ix1 - nx)

        img_z0 = max(0, iz0)
        img_z1 = min(nz, iz1)
        img_x0 = max(0, ix0)
        img_x1 = min(nx, ix1)

        if img_z0 < img_z1 and img_x0 < img_x1:
            patch = disk[disk_z0:disk_z1, disk_x0:disk_x1]
            individual_masks[i, img_z0:img_z1, img_x0:img_x1] = patch
            union_mask[img_z0:img_z1, img_x0:img_x1] |= patch

    return individual_masks, union_mask


def compute_background_statistics(
    image: np.ndarray,
    union_mask: np.ndarray,
    return_hist: bool = False,
    hist_bins: int = 50,
) -> dict:
    """Compute background amplitude statistics from pixels outside all ROIs.

    Args:
        image: 2D image array ``(nz, nx)``.
        union_mask: Boolean mask ``(nz, nx)`` marking scatterer regions to exclude.
        return_hist: If True, include a histogram of background pixel values.
        hist_bins: Number of histogram bins (used only when ``return_hist=True``).

    Returns:
        Dictionary with keys:
        - ``background_rms``: float, RMS amplitude of background pixels.
        - ``background_pixels``: 1D array of background pixel values.
        - ``background_hist_counts``, ``background_hist_edges`` (when ``return_hist=True``).

    Raises:
        ValueError: If ``image`` is not 2D or shapes do not match.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be 2D (nz, nx)")
    union_mask = np.asarray(union_mask, dtype=bool)
    if image.shape != union_mask.shape:
        raise ValueError(
            f"image shape {image.shape} does not match union_mask shape {union_mask.shape}"
        )

    bg_pixels = image[~union_mask].astype(np.float64, copy=False)
    bg_rms = float(np.sqrt(np.mean(bg_pixels ** 2))) if bg_pixels.size > 0 else 0.0

    result: dict = {
        "background_rms": bg_rms,
        "background_pixels": bg_pixels,
    }

    if return_hist:
        counts, edges = np.histogram(bg_pixels, bins=hist_bins)
        result["background_hist_counts"] = counts
        result["background_hist_edges"] = edges

    return result


def compute_peak_amplitudes(
    image: np.ndarray,
    individual_masks: np.ndarray,
) -> np.ndarray:
    """Extract peak amplitude within each scatterer disk mask.

    Args:
        image: 2D image array ``(nz, nx)``.
        individual_masks: Bool array ``(N, nz, nx)`` from :func:`build_roi_masks`.

    Returns:
        1D float64 array of shape ``(N,)`` with the peak amplitude per scatterer.

    Raises:
        ValueError: If ``image`` is not 2D or mask ndim is not 3.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("image must be 2D (nz, nx)")
    individual_masks = np.asarray(individual_masks, dtype=bool)
    if individual_masks.ndim != 3 or individual_masks.shape[1:] != image.shape:
        raise ValueError(
            "individual_masks must have shape (N, nz, nx) matching image shape"
        )

    n = individual_masks.shape[0]
    peaks = np.empty(n, dtype=np.float64)
    for i in range(n):
        pixels = image[individual_masks[i]]
        peaks[i] = float(pixels.max()) if pixels.size > 0 else 0.0
    return peaks
