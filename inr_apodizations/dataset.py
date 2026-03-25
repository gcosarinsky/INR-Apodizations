from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import numpy as np


def generate_unit_gaussian_mask(scatterers, x_grid, z_grid, sigma_x=0.001, sigma_z=0.001, background_points=None):
    """
    Generate a unit-amplitude Gaussian mask centered at scatterer positions.

    scatterers: (n_scatterers, >=2) - columns include x, z as first two values
    x_grid, z_grid: meshgrid arrays of shape (nz, nx) OR None if using background_points
    background_points: if specified, x_grid and z_grid are ignored and this should be shape (N, 2)

    Returns:
    - If background_points is None: mask array of shape (nz, nx)
    - If background_points is specified: mask array of shape (N,)
    """
    if background_points is not None:
        mask = np.zeros(len(background_points), dtype=np.float32)
        for scat in scatterers:
            x0, z0 = scat[:2]
            dx = background_points[:, 0] - x0
            dz = background_points[:, 1] - z0
            gaussian = np.exp(-(dx**2 / (2 * sigma_x**2) + dz**2 / (2 * sigma_z**2)))
            mask += gaussian
    else:
        mask = np.zeros_like(x_grid, dtype=np.float32)
        for scat in scatterers:
            x0, z0 = scat[:2]
            gaussian = np.exp(
                -((x_grid - x0) ** 2 / (2 * sigma_x ** 2) + (z_grid - z0) ** 2 / (2 * sigma_z ** 2))
            )
            mask += gaussian

    return mask.astype(np.float32)


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
