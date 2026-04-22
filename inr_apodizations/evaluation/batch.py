"""Batch input normalization and validation for evaluation utilities.

Provides a single, public implementation of the scatterer-batch normalization
pattern that was previously duplicated across sandbox modules, plus a shared
helper for validating 2D/3D image batches.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def normalize_scatterer_batch(
    scatterers: np.ndarray | Sequence[np.ndarray],
    batch_size: int,
    dtype: type = np.float64,
) -> tuple[list[np.ndarray], bool]:
    """Convert heterogeneous scatterer input to a per-example list.

    Accepted input formats:
    - ``(N, 2)`` array  – replicated ``batch_size`` times (is_replicated=True).
    - ``(B, N, 2)`` array – one ``(N, 2)`` slice per example.
    - Sequence or object-array of ``(Ni, 2)`` arrays – variable counts per example.

    Args:
        scatterers: Scatterer coordinates in mm, columns ``[x, z]``.
        batch_size: Expected number of examples in the batch.
        dtype: NumPy dtype for all output arrays.

    Returns:
        Tuple ``(normalized_list, is_replicated)`` where ``normalized_list``
        contains one ``(Ni, 2)`` array per example and ``is_replicated`` is True
        when a single ``(N, 2)`` array was broadcast to all examples.

    Raises:
        ValueError: If input shape is incompatible or batch size does not match.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    if isinstance(scatterers, np.ndarray):
        if scatterers.ndim == 2:
            arr = np.asarray(scatterers, dtype=dtype)
            if arr.shape[1] != 2:
                raise ValueError("scatterers (N, 2): column count must be 2 (x, z in mm)")
            return [arr.copy() for _ in range(batch_size)], True

        if scatterers.ndim == 3:
            if scatterers.shape[0] != batch_size:
                raise ValueError(
                    f"scatterers (B, N, 2): first dimension {scatterers.shape[0]} "
                    f"does not match batch_size={batch_size}"
                )
            if scatterers.shape[2] != 2:
                raise ValueError("scatterers (B, N, 2): last dimension must be 2 (x, z in mm)")
            return [np.asarray(item, dtype=dtype) for item in scatterers], False

        if scatterers.ndim == 1 and scatterers.dtype == object:
            items: list = list(scatterers)
        else:
            raise ValueError(
                "scatterers must be (N, 2), (B, N, 2), or a sequence/object-array of (Ni, 2) arrays"
            )
    else:
        items = list(scatterers)

    if len(items) != batch_size:
        raise ValueError(
            f"scatterers sequence length {len(items)} does not match batch_size={batch_size}"
        )

    normalized: list[np.ndarray] = []
    for i, item in enumerate(items):
        arr = np.asarray(item, dtype=dtype)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"scatterers[{i}] must have shape (Ni, 2) with columns [x, z] in mm; "
                f"got shape {arr.shape}"
            )
        normalized.append(arr)

    return normalized, False


def validate_image_batch(
    images: dict[str, np.ndarray],
) -> tuple[bool, int]:
    """Validate that all images in a dict share a consistent dimensionality.

    All images must be either 2D ``(nz, nx)`` or 3D ``(B, nz, nx)``; mixing
    is not allowed.  When 3D, all must share the same batch size.

    Args:
        images: Mapping from method name to image array.

    Returns:
        Tuple ``(uses_batch, batch_size)`` where ``uses_batch`` is True when
        images are 3D and ``batch_size`` is 1 for 2D inputs.

    Raises:
        ValueError: If any image has unsupported dimensions or batch sizes differ.
    """
    ndims = {name: np.asarray(img).ndim for name, img in images.items()}
    invalid = [name for name, nd in ndims.items() if nd not in (2, 3)]
    if invalid:
        raise ValueError(
            "All images must be 2D (nz, nx) or 3D (B, nz, nx). "
            f"Invalid methods: {invalid}"
        )

    uses_batch = any(nd == 3 for nd in ndims.values())
    if uses_batch and any(nd != 3 for nd in ndims.values()):
        raise ValueError(
            "Cannot mix 2D and 3D images in the same call. "
            "Convert all to 3D (B, nz, nx) or all to 2D."
        )

    if not uses_batch:
        return False, 1

    batch_sizes = {name: int(np.asarray(img).shape[0]) for name, img in images.items()}
    unique = set(batch_sizes.values())
    if len(unique) != 1:
        raise ValueError(
            f"All 3D images must share the same batch size; found: {batch_sizes}"
        )
    return True, unique.pop()
