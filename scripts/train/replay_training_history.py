"""Replay training history plots from a saved training run.

Loads ``history_full.json`` (preferred) or ``history.json`` from an existing
run directory and regenerates all training-curve figures using
``plot_training_curves``. Optionally reads ``train_config_info.yml``
(for the regularization lambda) and ``validation_mae_summary.json``
(for hanning reference lines).

Usage:
    Edit the ``RUN_DIR`` variable below to point to the desired training run
    folder, then run this script directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from inr_apodizations import config
from inr_apodizations.experiment_helpers import plot_training_curves

# ============================================================================
# Configuration — edit RUN_DIR to point to the training run to replay
# ============================================================================

RUN_DIR = config.PROJ_ROOT / "scripts" / "outputs" / "train" / "20260429_153130"


# ============================================================================
# Load history artifact (prefer accumulated history_full.json)
# ============================================================================

run_dir = Path(RUN_DIR)
if not run_dir.is_dir():
    raise FileNotFoundError(f"RUN_DIR does not exist: {run_dir}")

history_json_path = run_dir / "history_full.json"
if not history_json_path.is_file():
    history_json_path = run_dir / "history.json"
if not history_json_path.is_file():
    raise FileNotFoundError(f"history_full.json/history.json not found in: {run_dir}")

with open(history_json_path, encoding="utf-8") as f:
    history: dict = json.load(f)

print(f"Loaded history artifact from: {history_json_path}")
print(f"  Keys: {list(history.keys())}")
print(f"  Epochs recorded: {len(next(iter(history.values()), []))}")


# ============================================================================
# Load optional train_config_info.yml — extract regularization lambda
# ============================================================================

weight_reg_lambda: float | None = None
config_yml_path = run_dir / "train_config_info.yml"
if config_yml_path.is_file():
    with open(config_yml_path, encoding="utf-8") as f:
        run_cfg: dict = yaml.safe_load(f)
    reg_cfg = run_cfg.get("resolved_weight_regularization", {})
    if reg_cfg.get("enabled", False):
        raw_lambda = reg_cfg.get("lambda")
        if raw_lambda is not None:
            try:
                weight_reg_lambda = float(raw_lambda)
            except Exception:
                pass
    print(f"Loaded train_config_info.yml  →  weight_reg_lambda={weight_reg_lambda}")
else:
    print(f"train_config_info.yml not found in {run_dir}, skipping lambda extraction.")


# ============================================================================
# Load optional validation_mae_summary.json — extract hanning reference values
# ============================================================================

reference_mae: dict | None = None
reference_relative_y_pred: dict | None = None
reference_relative_y_true: dict | None = None

summary_path = run_dir / "validation_mae_summary.json"
if summary_path.is_file():
    with open(summary_path, encoding="utf-8") as f:
        summary: dict = json.load(f)

    hanning_pw_mae = (
        summary.get("validation_pixel_weighted_mae", {}).get("hanning")
    )
    if hanning_pw_mae is not None:
        reference_mae = {"hanning": float(hanning_pw_mae)}

    hanning_rel_y_pred = (
        summary.get("validation_relative_mae", {})
        .get("relative_mae_y_pred", {})
        .get("hanning")
    )
    if hanning_rel_y_pred is not None:
        reference_relative_y_pred = {"hanning": float(hanning_rel_y_pred)}

    hanning_rel_y_true = (
        summary.get("validation_relative_mae", {})
        .get("relative_mae_y_true", {})
        .get("hanning")
    )
    if hanning_rel_y_true is not None:
        reference_relative_y_true = {"hanning": float(hanning_rel_y_true)}

    print(
        f"Loaded validation_mae_summary.json  →  "
        f"ref_mae={reference_mae}, "
        f"ref_rel_y_pred={reference_relative_y_pred}, "
        f"ref_rel_y_true={reference_relative_y_true}"
    )
else:
    print(f"validation_mae_summary.json not found in {run_dir}, skipping reference lines.")


# ============================================================================
# Regenerate figures
# ============================================================================

history_dir = run_dir / "history"
history_dir.mkdir(parents=True, exist_ok=True)
output_path = str(history_dir / "training_history.png")

plot_training_curves(
    history=history,
    output_path=output_path,
    reference_mae=reference_mae,
    reference_relative_y_pred=reference_relative_y_pred,
    reference_relative_y_true=reference_relative_y_true,
    weight_reg_lambda=weight_reg_lambda,
)

print(f"Figures saved to: {history_dir}")
