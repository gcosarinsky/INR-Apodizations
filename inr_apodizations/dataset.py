from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import numpy as np


def generate_unit_gaussian_mask(scatterers, x_grid, z_grid, sigma_x=0.001, sigma_z=0.001, background_points=None):
    """Generate a unit-amplitude Gaussian mask centered at scatterer positions.

    Args:
        scatterers: Array-like with shape ``(n_scatterers, >=2)`` where the first
            two columns contain ``x`` and ``z`` coordinates.
        x_grid: Meshgrid x coordinates with shape ``(nz, nx)``. Ignored when
            ``background_points`` is provided.
        z_grid: Meshgrid z coordinates with shape ``(nz, nx)``. Ignored when
            ``background_points`` is provided.
        sigma_x: Lateral Gaussian standard deviation.
        sigma_z: Axial Gaussian standard deviation.
        background_points: Optional array with shape ``(n_points, 2)`` containing
            ``(x, z)`` coordinates where the mask should be evaluated.

    Returns:
        A ``float32`` mask with shape ``(nz, nx)`` when evaluated on a grid, or
        shape ``(n_points,)`` when evaluated on ``background_points``.
    """
    scatterers_array = np.asarray(scatterers, dtype=np.float32)
    sigma_x = np.float32(sigma_x)
    sigma_z = np.float32(sigma_z)
    inv_two_sigma_x2 = np.float32(1.0) / (np.float32(2.0) * sigma_x * sigma_x)
    inv_two_sigma_z2 = np.float32(1.0) / (np.float32(2.0) * sigma_z * sigma_z)

    if scatterers_array.size == 0:
        if background_points is not None:
            return np.zeros(len(background_points), dtype=np.float32)
        return np.zeros_like(x_grid, dtype=np.float32)

    scatterers_x = scatterers_array[:, 0][:, None]
    scatterers_z = scatterers_array[:, 1][:, None]

    if background_points is not None:
        points = np.asarray(background_points, dtype=np.float32)
        dx = points[:, 0][None, :] - scatterers_x
        dz = points[:, 1][None, :] - scatterers_z
        exponent = -(dx * dx * inv_two_sigma_x2 + dz * dz * inv_two_sigma_z2)
        return np.exp(exponent, dtype=np.float32).sum(axis=0, dtype=np.float32)

    x_grid_array = np.asarray(x_grid, dtype=np.float32)
    z_grid_array = np.asarray(z_grid, dtype=np.float32)
    dx = x_grid_array[None, :, :] - scatterers_x[:, None, :]
    dz = z_grid_array[None, :, :] - scatterers_z[:, None, :]
    exponent = -(dx * dx * inv_two_sigma_x2 + dz * dz * inv_two_sigma_z2)
    return np.exp(exponent, dtype=np.float32).sum(axis=0, dtype=np.float32)


def generate_das_modulated_target(das_image, scatterers, x_grid, z_grid, sigma_x=0.001, sigma_z=0.001):
    """
    Generate target by modulating a DAS image with a unit-amplitude Gaussian mask.

    das_image: ndarray (nz, nx), real or complex
    scatterers: (n_scatterers, >=2) - columns include x, z as first two values
    x_grid, z_grid: meshgrid arrays of shape (nz, nx)

    TODO: target could be RF instead of abs
    """
    gaussian_mask = generate_unit_gaussian_mask(
        scatterers,
        x_grid,
        z_grid,
        sigma_x=sigma_x,
        sigma_z=sigma_z,
    )
    target = np.abs(das_image).astype(np.float32) * gaussian_mask
    return target.astype(np.float32)


# # --------- From cookiecutter template, not used ---------
# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = RAW_DATA_DIR / "dataset.csv",
#     output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     # ----------------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Processing dataset...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Processing dataset complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()
