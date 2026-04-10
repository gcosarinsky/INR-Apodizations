"""Utility functions for tuner candidate generation.

This module centralizes generation of explicit candidate architectures
for the Keras Tuner workflow.
"""
from __future__ import annotations

import itertools
import random
from typing import List


def _build_discrete_values(min_value: int, max_value: int, step: int, name: str) -> List[int]:
    if step <= 0:
        raise ValueError(f"{name}_step must be > 0. Received: {step}.")
    if min_value > max_value:
        raise ValueError(f"{name}_min must be <= {name}_max. Received: {min_value} > {max_value}.")
    values = list(range(int(min_value), int(max_value) + 1, int(step)))
    if not values:
        raise ValueError(f"{name} interval produced an empty set of values.")
    return values


def generate_candidate_architectures(tuning_cfg: dict, fallback_seed: int) -> List[List[int]]:
    """Generate candidate INR architectures as explicit neuron-count lists.

    Parameters
    ----------
    tuning_cfg : dict
        Tuning section from the experiment config.
    fallback_seed : int
        Seed used when `candidate_seed` is not provided.

    Returns
    -------
    List[List[int]]
        Candidate architectures, each represented as `[units_layer_0, ..., units_layer_n]`.
    """
    layer_values = _build_discrete_values(
        min_value=int(tuning_cfg["n_layers_min"]),
        max_value=int(tuning_cfg["n_layers_max"]),
        step=int(tuning_cfg.get("n_layers_step", 1)),
        name="n_layers",
    )
    unit_values = _build_discrete_values(
        min_value=int(tuning_cfg["units_min"]),
        max_value=int(tuning_cfg["units_max"]),
        step=int(tuning_cfg["units_step"]),
        name="units",
    )

    non_increasing = bool(tuning_cfg.get("non_increasing", True))
    all_candidates: List[List[int]] = []
    seen: set[tuple[int, ...]] = set()

    for n_layers in layer_values:
        for architecture in itertools.product(unit_values, repeat=n_layers):
            if non_increasing and any(architecture[i] < architecture[i + 1] for i in range(n_layers - 1)):
                continue
            if architecture in seen:
                continue
            seen.add(architecture)
            all_candidates.append(list(architecture))

    if not all_candidates:
        raise ValueError("No candidate architectures were generated from the provided tuning ranges.")

    max_candidates = tuning_cfg.get("max_candidates")
    if max_candidates is not None:
        max_candidates = int(max_candidates)
        if max_candidates <= 0:
            raise ValueError(f"max_candidates must be > 0 when provided. Received: {max_candidates}.")
        if len(all_candidates) > max_candidates:
            # Prioritize candidates with fewer total neurons.
            # Sort by (total_neurons, architecture_tuple) to make selection deterministic
            # and stable for architectures with the same total.
            sorted_candidates = sorted(
                all_candidates, key=lambda arch: (sum(arch), tuple(arch))
            )
            all_candidates = sorted_candidates[:max_candidates]

    return all_candidates
