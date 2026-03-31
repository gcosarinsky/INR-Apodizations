"""Minimal synthetic example for batch scatterer evaluation helpers.

This sandbox script builds a tiny synthetic batch of DAS-like images for three
methods, uses one scatterer list per example, and writes the resulting batch
evaluation figures into ``sandbox/inr_das_experiment/review_outputs/``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import helpers


class ToyCoordinateManager:
    """Small coordinate manager stub exposing the method used by helpers."""

    def __init__(self, x_coords_mm: np.ndarray, z_coords_mm: np.ndarray) -> None:
        """Store 1D image coordinates in millimeters.

        Args:
            x_coords_mm: Lateral coordinates with shape ``(nx,)``.
            z_coords_mm: Axial coordinates with shape ``(nz,)``.
        """
        self._x_coords_mm = np.asarray(x_coords_mm, dtype=np.float64)
        self._z_coords_mm = np.asarray(z_coords_mm, dtype=np.float64)

    def get_coordinates_1d(self, scaled: bool = False) -> dict[str, np.ndarray]:
        """Return the unscaled coordinate vectors expected by the helpers.

        Args:
            scaled: Unused here. Present only to match the real interface.

        Returns:
            Mapping with ``x`` and ``z`` coordinate arrays.

        Raises:
            ValueError: If scaled coordinates are requested in this toy stub.
        """
        if scaled:
            raise ValueError("ToyCoordinateManager only supports scaled=False")
        return {"x": self._x_coords_mm, "z": self._z_coords_mm}


def _add_gaussian_blob(
    image: np.ndarray,
    x_grid_mm: np.ndarray,
    z_grid_mm: np.ndarray,
    x_mm: float,
    z_mm: float,
    amplitude: float,
    sigma_x_mm: float = 0.55,
    sigma_z_mm: float = 0.75,
) -> np.ndarray:
    """Add one synthetic scatterer response to an image.

    Args:
        image: Base image with shape ``(nz, nx)``.
        x_grid_mm: Lateral meshgrid in mm.
        z_grid_mm: Axial meshgrid in mm.
        x_mm: Scatterer lateral position in mm.
        z_mm: Scatterer axial position in mm.
        amplitude: Blob amplitude in linear scale.
        sigma_x_mm: Lateral Gaussian sigma in mm.
        sigma_z_mm: Axial Gaussian sigma in mm.

    Returns:
        Updated image with the synthetic blob added.
    """
    exponent = (
        ((x_grid_mm - x_mm) / sigma_x_mm) ** 2 + ((z_grid_mm - z_mm) / sigma_z_mm) ** 2
    )
    return image + amplitude * np.exp(-0.5 * exponent)


x_coords_mm = np.linspace(-4.0, 4.0, 49)
z_coords_mm = np.linspace(15.0, 35.0, 64)
x_grid_mm, z_grid_mm = np.meshgrid(x_coords_mm, z_coords_mm)
cm = ToyCoordinateManager(x_coords_mm=x_coords_mm, z_coords_mm=z_coords_mm)

scatterers_xy_batch = [
    np.asarray([[-1.8, 20.0], [0.2, 25.5], [2.1, 31.0]], dtype=np.float32),
    np.asarray([[-2.4, 18.5], [1.5, 28.0]], dtype=np.float32),
]

method_scales = {
    "uniform": 0.85,
    "hanning": 1.00,
    "inr_after": 1.18,
}
background_levels = {
    "uniform": 0.060,
    "hanning": 0.040,
    "inr_after": 0.028,
}

images_abs_eval: dict[str, np.ndarray] = {}
for method_name, method_scale in method_scales.items():
    images_batch = []
    for example_idx, scatterers_xy in enumerate(scatterers_xy_batch):
        image = np.full(z_grid_mm.shape, background_levels[method_name], dtype=np.float32)
        for scatterer_idx, (x_mm, z_mm) in enumerate(scatterers_xy):
            base_amplitude = 0.55 + 0.18 * scatterer_idx + 0.08 * example_idx
            image = _add_gaussian_blob(
                image,
                x_grid_mm,
                z_grid_mm,
                x_mm=float(x_mm),
                z_mm=float(z_mm),
                amplitude=base_amplitude * method_scale,
            )
        images_batch.append(image)
    images_abs_eval[method_name] = np.stack(images_batch, axis=0)

uniform_metrics = helpers.compute_scatterer_metrics(
    image=images_abs_eval["uniform"],
    scatterers=scatterers_xy_batch,
    cm=cm,
    radius_mm=1.1,
    return_masks=True,
    return_background_hist=True,
    hist_bins=40,
)

print("Batch metric keys:", list(uniform_metrics.keys()))
print("Per-example entries:", len(uniform_metrics["per_example"]))
print("Aggregated keys:", sorted(uniform_metrics["aggregated"].keys()))
print("Total points in batch:", uniform_metrics["aggregated"]["n_scatterers_total"])

output_dir = Path(__file__).resolve().parent / "review_outputs" / "minimal_batch_scatterer_eval"
helpers.plot_scatterer_evaluation(
    images_abs=images_abs_eval,
    scatterers_xy=scatterers_xy_batch,
    cm=cm,
    output_dir=str(output_dir),
    radius_mm=1.1,
    hist_bins=40,
    compare_pairs=[("uniform", "inr_after"), ("hanning", "inr_after")],
    extent=(
        float(x_coords_mm.min()),
        float(x_coords_mm.max()),
        float(z_coords_mm.max()),
        float(z_coords_mm.min()),
    ),
    vmin_db=-40.0,
    vmax_db=0.0,
    cmap="gray",
    example_suffix="synthetic_batch",
)

print(f"Synthetic batch scatterer evaluation saved to: {output_dir}")