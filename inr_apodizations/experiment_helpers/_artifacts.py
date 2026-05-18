"""Run artifact persistence: JSON, YAML, Keras model, history, NumPy arrays."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Mapping

import numpy as np
import tensorflow as tf
import yaml


def to_json_serializable(obj):
    """Recursively convert numpy/TF values into JSON-serializable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, tf.Tensor):
        try:
            val = obj.numpy()
        except Exception:
            return str(obj)
        return to_json_serializable(val)
    if isinstance(obj, np.ndarray):
        return to_json_serializable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)


def save_json_artifact(path: str, payload: dict) -> None:
    """Persist a JSON payload with stable indentation and UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(to_json_serializable(payload), file, indent=2)


def history_length(history: Mapping[str, list]) -> int:
    """Return the number of recorded epochs from a Keras-like history mapping."""
    if not history:
        return 0
    lengths = [len(values) for values in history.values() if isinstance(values, list)]
    if not lengths:
        return 0
    return max(lengths)


def merge_training_histories(
    previous_history: Mapping[str, list] | None,
    stage_history: Mapping[str, list],
) -> dict[str, list]:
    """Merge two history mappings while preserving epoch alignment.

    Missing metrics in either side are padded with ``None`` so all metric
    arrays have a consistent total length.
    """
    prev = dict(previous_history or {})
    stage = dict(stage_history or {})

    prev_len = history_length(prev)
    stage_len = history_length(stage)
    all_keys = sorted(set(prev.keys()) | set(stage.keys()))
    merged: dict[str, list] = {}
    for key in all_keys:
        prev_values = list(prev.get(key, []))
        stage_values = list(stage.get(key, []))
        if len(prev_values) < prev_len:
            prev_values.extend([None] * (prev_len - len(prev_values)))
        if len(stage_values) < stage_len:
            stage_values.extend([None] * (stage_len - len(stage_values)))
        merged[key] = prev_values + stage_values
    return merged


def load_previous_history(run_dir: str) -> tuple[dict[str, list], list[dict]]:
    """Load accumulated history and stage metadata from a previous run directory.

    Priority:
    1) ``history_full.json``
    2) ``history.json``
    Stage metadata is read from ``history_stages.json`` when available.
    """
    history_full_path = os.path.join(run_dir, "history_full.json")
    history_path = os.path.join(run_dir, "history.json")
    stages_path = os.path.join(run_dir, "history_stages.json")

    if os.path.isfile(history_full_path):
        with open(history_full_path, encoding="utf-8") as file:
            previous_history = json.load(file)
    elif os.path.isfile(history_path):
        with open(history_path, encoding="utf-8") as file:
            previous_history = json.load(file)
    else:
        raise FileNotFoundError(
            "Could not find history_full.json or history.json in resume source run: "
            f"{run_dir}"
        )

    if os.path.isfile(stages_path):
        with open(stages_path, encoding="utf-8") as file:
            stages = json.load(file)
        if not isinstance(stages, list):
            stages = []
    else:
        stages = []
    return previous_history, stages


def save_artifacts(output_dir: str, model: tf.keras.Model, history: dict, config: dict) -> None:
    """Save model and minimal artifacts into ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.keras")
    model.save(model_path)

    serializable_history = to_json_serializable(history)
    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as file:
        json.dump(serializable_history, file, indent=2)

    with open(os.path.join(output_dir, "train_config_info.yml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def save_debug_arrays(output_dir: str, arrays: dict[str, np.ndarray]) -> None:
    """Persist selected NumPy arrays for quick inspection."""
    os.makedirs(output_dir, exist_ok=True)
    for name, array in arrays.items():
        np.save(os.path.join(output_dir, f"{name}.npy"), array)


def create_run_directories(processed_root: str, output_root: str) -> tuple[str, str, str]:
    """Create timestamped run directories for processed and experiment outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_dir = os.path.join(processed_root, timestamp)
    output_dir = os.path.join(output_root, timestamp)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return timestamp, processed_dir, output_dir
