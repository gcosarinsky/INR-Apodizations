"""Unified SNR calculation for scatterer evaluation.

A single, reusable SNR implementation used by metrics computation,
persistence helpers, and plotting functions.
"""

from __future__ import annotations

import numpy as np


def compute_reflector_snr(
    peak_amplitudes: np.ndarray,
    background_rms: np.ndarray | float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute pointwise SNR as peak amplitude divided by background RMS.

    Args:
        peak_amplitudes: 1D array ``(N,)`` of scatterer peak amplitudes.
        background_rms: Either a scalar (one background level for all points) or
            a 1D array ``(N,)`` with a per-point background RMS value.
        eps: Small value added to the denominator to avoid division by zero.

    Returns:
        1D float64 array of shape ``(N,)`` with ``SNR = peak / (bg + eps)``.

    Raises:
        ValueError: If ``peak_amplitudes`` is not 1D or ``background_rms``
            shape is incompatible.
    """
    peaks = np.asarray(peak_amplitudes, dtype=np.float64)
    if peaks.ndim != 1:
        raise ValueError("peak_amplitudes must be a 1D array")

    bg = np.asarray(background_rms, dtype=np.float64)
    if bg.ndim == 0:
        bg = np.full(peaks.shape, float(bg), dtype=np.float64)
    elif bg.ndim == 1:
        if bg.shape != peaks.shape:
            raise ValueError(
                f"background_rms shape {bg.shape} does not match "
                f"peak_amplitudes shape {peaks.shape}"
            )
    else:
        raise ValueError("background_rms must be a scalar or 1D array")

    return peaks / np.maximum(bg, eps)
