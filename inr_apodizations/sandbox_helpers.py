"""Provisional facade for sandbox helper utilities.

This module centralizes access to sandbox helper functions so scripts can
import from the package namespace during migration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping
import sys

import numpy as np

from inr_apodizations.config import PROJ_ROOT
from inr_apodizations.evaluation.scatterers import select_reflector_scatterer
from inr_apodizations.utils import to_db


_SANDBOX_EXPERIMENT_DIR = PROJ_ROOT / "sandbox" / "inr_das_experiment"
if str(_SANDBOX_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_SANDBOX_EXPERIMENT_DIR))

# Re-export all sandbox helper utilities from the canonical sandbox module.
from helpers import *  # type: ignore  # noqa: F401,F403,E402


def build_reflector_profile_context(
    *,
    scatterers_mm: np.ndarray,
    profile_cfg: dict,
    images_linear: Mapping[str, np.ndarray],
    images_db: Mapping[str, np.ndarray],
    target_image: np.ndarray | None,
) -> dict:
    """Build a normalized context dict for reflector profile plotting.

    Args:
        scatterers_mm: Scatterers for one example in mm with at least [x, z].
        profile_cfg: User reflector profile config block.
        images_linear: Mapping of method name to linear image.
        images_db: Mapping of method name to dB image.
        target_image: Optional target image.

    Returns:
        Dictionary with selected scatterer, selected images and plotting flags.

    Raises:
        ValueError: If configuration is invalid or no images are selected.
    """
    if scatterers_mm is None or len(scatterers_mm) == 0:
        raise ValueError("Reflector profile plotting requires scatterers for the selected example.")

    line_length_mm = float(profile_cfg.get("line_length_mm", 5.0))
    overlay_profiles = bool(profile_cfg.get("overlay_profiles", True))
    use_db_profiles = bool(profile_cfg.get("use_db", True))
    include_target_profile = bool(profile_cfg.get("include_target", True))
    scatterer_selection = str(profile_cfg.get("scatterer_selection", "strongest"))
    scatterer_idx_raw = profile_cfg.get("scatterer_idx")
    scatterer_idx = None if scatterer_idx_raw is None else int(scatterer_idx_raw)

    selected_scatterer = select_reflector_scatterer(
        scatterers_mm,
        selection=scatterer_selection,
        scatterer_idx=scatterer_idx,
    )

    if use_db_profiles:
        profile_image_source = dict(images_db)
    else:
        profile_image_source = {
            name: np.abs(image) for name, image in images_linear.items()
        }

    if include_target_profile and target_image is not None:
        profile_image_source["target"] = (
            to_db(target_image, ref=float(np.max(np.abs(target_image))))
            if use_db_profiles
            else np.abs(target_image)
        )

    requested_profile_images = profile_cfg.get("images")
    if requested_profile_images is not None and not isinstance(requested_profile_images, list):
        raise ValueError("`reflector_lateral_profile.images` must be a YAML list when provided.")

    if requested_profile_images is None or len(requested_profile_images) == 0:
        profile_image_names = list(profile_image_source.keys())
    else:
        profile_image_names = [str(name).strip().lower() for name in requested_profile_images]

    missing_profile_images = [
        image_name for image_name in profile_image_names if image_name not in profile_image_source
    ]
    if missing_profile_images:
        raise ValueError(
            "Unknown images requested in `reflector_lateral_profile.images`: "
            f"{missing_profile_images}. Available images: {list(profile_image_source.keys())}"
        )

    selected_profile_images = {
        image_name: profile_image_source[image_name] for image_name in profile_image_names
    }
    if len(selected_profile_images) == 0:
        raise ValueError("No images available for `reflector_lateral_profile` plotting.")

    return {
        "selected_scatterer": selected_scatterer,
        "line_length_mm": line_length_mm,
        "overlay_profiles": overlay_profiles,
        "use_db_profiles": use_db_profiles,
        "selected_profile_images": selected_profile_images,
    }
