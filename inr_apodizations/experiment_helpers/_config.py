"""Experiment and training configuration parsing, including resume and regularization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from ._artifacts import history_length, load_previous_history


def load_experiment_config(config_path: str) -> dict:
    """Load the experiment YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_target_regeneration_override(config: dict) -> tuple[float | None, float | None, float | None]:
    """Extract optional target regeneration overrides from experiment configuration.

    Args:
        config: Experiment configuration mapping.

    Returns:
        Tuple ``(sigma_x, sigma_z, alpha)`` or ``(None, None, None)`` when disabled.

    Raises:
        ValueError: If the override section is enabled but incomplete.
    """
    override_cfg = dict(config.get("target_regeneration", {}))
    if not bool(override_cfg.get("enabled", False)):
        return None, None, None

    sigma_x = override_cfg.get("sigma_x")
    sigma_z = override_cfg.get("sigma_z")
    if sigma_x is None or sigma_z is None:
        raise ValueError(
            "target_regeneration.sigma_x and target_regeneration.sigma_z must be set "
            "when regeneration is enabled"
        )

    alpha = override_cfg.get("alpha")
    if alpha is not None:
        alpha_val = float(alpha)
        if not (0.0 <= alpha_val < 1.0):
            raise ValueError("target_regeneration.alpha must be in [0.0, 1.0)")
        alpha = alpha_val

    return float(sigma_x), float(sigma_z), (float(alpha) if alpha is not None else None)


def _resolve_optional_path(raw_value: str | None) -> Path | None:
    """Resolve optional config path as absolute Path, preserving None/empty.

    Args:
        raw_value: Optional config value (may be None, empty string, or a path).

    Returns:
        Absolute Path object, or None if input is None/empty.
    """
    if raw_value is None:
        return None
    token = str(raw_value).strip()
    if not token:
        return None
    path = Path(token)
    from inr_apodizations import config
    return path if path.is_absolute() else (config.PROJ_ROOT / path)


def parse_resume_config(config: dict) -> dict:
    """Parse and validate resume configuration from experiment config.

    Args:
        config: Experiment configuration dict.

    Returns:
        Dictionary with parsed resume settings:
        - enabled: bool
        - source_run_dir: Path | None (absolute, resolved)
        - source_model_path: Path | None (absolute, resolved)
        - restore_optimizer: bool
        - epochs_mode: str ("additional" or "target_total")
        - additional_epochs: int
        - resolved_model_path: Path | None (final model path to load)

    Raises:
        ValueError: If configuration is invalid or inconsistent.
        FileNotFoundError: If model file does not exist when resume is enabled.
    """
    resume_cfg = dict(config.get("resume", {}))
    resume_enabled = bool(resume_cfg.get("enabled", False))

    if not resume_enabled:
        return {
            "enabled": False,
            "source_run_dir": None,
            "source_model_path": None,
            "restore_optimizer": True,
            "epochs_mode": "additional",
            "additional_epochs": 0,
            "resolved_model_path": None,
        }

    resume_source_run_dir = _resolve_optional_path(resume_cfg.get("source_run_dir"))
    resume_source_model_path = _resolve_optional_path(resume_cfg.get("source_model_path"))
    resume_restore_optimizer = bool(resume_cfg.get("restore_optimizer", True))
    resume_epochs_mode = str(resume_cfg.get("epochs_mode", "additional")).strip().lower()
    resume_additional_epochs = int(resume_cfg.get("additional_epochs", 0))

    if resume_epochs_mode not in {"additional", "target_total"}:
        raise ValueError("resume.epochs_mode must be 'additional' or 'target_total'")

    if resume_additional_epochs < 0:
        raise ValueError("resume.additional_epochs must be >= 0")

    if resume_source_run_dir is None and resume_source_model_path is None:
        raise ValueError(
            "resume.enabled=true requires at least one source: "
            "resume.source_run_dir or resume.source_model_path"
        )

    resume_model_path = resume_source_model_path
    if resume_source_run_dir is not None and resume_model_path is None:
        resume_model_path = resume_source_run_dir / "model.keras"

    if resume_model_path is None:
        raise ValueError("Could not resolve model path for resume mode")

    if not resume_model_path.is_file():
        raise FileNotFoundError(f"Resume model file not found: {resume_model_path}")

    return {
        "enabled": True,
        "source_run_dir": resume_source_run_dir,
        "source_model_path": resume_source_model_path,
        "restore_optimizer": resume_restore_optimizer,
        "epochs_mode": resume_epochs_mode,
        "additional_epochs": resume_additional_epochs,
        "resolved_model_path": resume_model_path,
    }


def setup_resume_state(
    resume_cfg: dict, target_epochs_from_config: int, config: dict
) -> tuple[bool, int, int, Path | None, dict, list]:
    """Setup resume training state: load model, history, and compute epoch schedule.

    This is the primary entry point for resume functionality. It handles:
    - Loading previous training history
    - Computing initial and target epochs based on epochs_mode
    - Validation of epoch progression

    Args:
        resume_cfg: Parsed resume config dict (from parse_resume_config).
        target_epochs_from_config: Target epochs value from training config.
        config: Full experiment config dict (used for logging).

    Returns:
        Tuple ``(resume_enabled, initial_epoch, target_epochs, resume_model_path, previous_history, history_stages)``:
        - resume_enabled: bool
        - initial_epoch: int (0 if not resuming)
        - target_epochs: int (final epoch target for this training stage)
        - resume_model_path: Path | None (path to model to load, or None if not resuming)
        - previous_history: dict (empty if not resuming)
        - history_stages: list[dict] (empty if not resuming)

    Raises:
        ValueError: If epoch progression is invalid.
        FileNotFoundError: If history files are not found in resume source.
    """
    resume_enabled = bool(resume_cfg.get("enabled", False))

    if not resume_enabled:
        return (False, 0, int(target_epochs_from_config), None, {}, [])

    resume_source_run_dir = resume_cfg.get("source_run_dir")
    resume_epochs_mode = str(resume_cfg.get("epochs_mode", "additional"))
    resume_additional_epochs = int(resume_cfg.get("additional_epochs", 0))
    resume_model_path = resume_cfg.get("resolved_model_path")

    initial_epoch = 0
    target_epochs = int(target_epochs_from_config)
    previous_history = {}
    history_stages = []

    if resume_source_run_dir is not None:
        previous_history, history_stages = load_previous_history(str(resume_source_run_dir))
        initial_epoch = history_length(previous_history)

        if resume_epochs_mode == "additional":
            resolved_additional_epochs = resume_additional_epochs
            if resolved_additional_epochs == 0:
                resolved_additional_epochs = int(target_epochs_from_config)
            target_epochs = initial_epoch + resolved_additional_epochs
        else:
            target_epochs = int(target_epochs_from_config)

        if target_epochs <= initial_epoch:
            raise ValueError(
                "Resume target epochs must be greater than previously recorded epochs. "
                f"initial_epoch={initial_epoch}, target_epochs={target_epochs}"
            )

        print(
            "Resume schedule:",
            {
                "initial_epoch": initial_epoch,
                "target_epochs": target_epochs,
                "epochs_mode": resume_epochs_mode,
                "additional_epochs": resume_additional_epochs,
            },
        )

    return (resume_enabled, initial_epoch, target_epochs, resume_model_path, previous_history, history_stages)


def parse_weight_regularization_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Parse and validate weight-regularization settings from experiment config.

    Args:
        cfg: Full experiment configuration mapping.

    Returns:
        Flat dictionary with resolved regularization parameters and auto-init settings.

    Raises:
        ValueError: If configuration values are invalid.
    """
    weight_reg_cfg = dict(cfg.get("training", {}).get("weight_regularization", {}))
    weight_reg_enabled = bool(weight_reg_cfg.get("enabled", False))
    weight_reg_type = str(weight_reg_cfg.get("type", "hinge_low_norm")).strip().lower()
    if weight_reg_enabled and weight_reg_type not in ("hinge_low_norm", "hinge"):
        raise ValueError(
            "training.weight_regularization.type must be 'hinge_low_norm' or 'hinge'"
        )

    weight_reg_lambda = float(weight_reg_cfg.get("lambda", 1e-3))
    weight_reg_tau = float(weight_reg_cfg.get("tau", 0.30))
    weight_reg_epsilon = float(weight_reg_cfg.get("epsilon", 1e-8))
    weight_reg_normalize = bool(weight_reg_cfg.get("normalize_norm", True))
    weight_reg_auto_cfg = dict(weight_reg_cfg.get("auto_init", {}))
    weight_reg_auto_enabled = bool(weight_reg_auto_cfg.get("enabled", False))
    weight_reg_auto_ratio = float(weight_reg_auto_cfg.get("ratio", 0.5))
    weight_reg_auto_eps = float(weight_reg_auto_cfg.get("epsilon", 1e-12))
    weight_reg_auto_norm_fraction = float(weight_reg_auto_cfg.get("norm_fraction", 0.5))

    if weight_reg_lambda < 0.0:
        raise ValueError("training.weight_regularization.lambda must be >= 0")
    if weight_reg_tau < 0.0:
        raise ValueError("training.weight_regularization.tau must be >= 0")
    if weight_reg_epsilon <= 0.0:
        raise ValueError("training.weight_regularization.epsilon must be > 0")
    if weight_reg_auto_ratio < 0.0:
        raise ValueError("training.weight_regularization.auto_init.ratio must be >= 0")
    if weight_reg_auto_eps <= 0.0:
        raise ValueError("training.weight_regularization.auto_init.epsilon must be > 0")
    if weight_reg_auto_norm_fraction <= 0.0:
        raise ValueError("training.weight_regularization.auto_init.norm_fraction must be > 0")

    resolved_weight_reg_type = "hinge_low_norm" if weight_reg_type == "hinge" else weight_reg_type
    return {
        "enabled": weight_reg_enabled,
        "type": resolved_weight_reg_type,
        "lambda": weight_reg_lambda,
        "tau": weight_reg_tau,
        "epsilon": weight_reg_epsilon,
        "normalize_norm": weight_reg_normalize,
        "auto_init": {
            "enabled": weight_reg_auto_enabled,
            "ratio": weight_reg_auto_ratio,
            "epsilon": weight_reg_auto_eps,
            "norm_fraction": weight_reg_auto_norm_fraction,
        },
    }


def parse_lateral_regularization_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Parse and validate lateral-regularization settings from experiment config.

    Reads ``training.lateral_regularization`` and returns a resolved flat dict.
    The ``auto_init`` subsection allows automatic scaling of ``lambda`` based on
    the initial model prediction, analogous to weight-regularization auto-init.

    Args:
        cfg: Full experiment configuration mapping.

    Returns:
        Flat dictionary with keys:
        - ``enabled``: bool
        - ``lambda``: float (may be overridden at runtime by auto-init)
        - ``q_power``: float
        - ``q_epsilon``: float
        - ``abs_xrel_feature_index``: int
        - ``z_feature_index``: int
        - ``normalize_by_uniform``: bool
        - ``channel_reduction``: str (``"mean"`` or ``"sum"``)
        - ``auto_init``: dict with ``enabled``, ``ratio``, ``epsilon``

    Raises:
        ValueError: If configuration values are invalid.
    """
    lat_cfg = dict(cfg.get("training", {}).get("lateral_regularization", {}))
    enabled = bool(lat_cfg.get("enabled", False))
    lam = float(lat_cfg.get("lambda", 0.0))
    q_power = float(lat_cfg.get("q_power", 1.0))
    q_epsilon = float(lat_cfg.get("q_epsilon", 1e-8))
    abs_xrel_feature_index = int(lat_cfg.get("abs_xrel_feature_index", 0))
    z_feature_index = int(lat_cfg.get("z_feature_index", 1))
    normalize_by_uniform = bool(lat_cfg.get("normalize_by_uniform", True))
    channel_reduction = str(lat_cfg.get("channel_reduction", "mean")).strip().lower()

    auto_cfg = dict(lat_cfg.get("auto_init", {}))
    auto_enabled = bool(auto_cfg.get("enabled", False))
    auto_ratio = float(auto_cfg.get("ratio", 0.1))
    auto_eps = float(auto_cfg.get("epsilon", 1e-12))

    if lam < 0.0:
        raise ValueError("training.lateral_regularization.lambda must be >= 0")
    if q_power <= 0.0:
        raise ValueError("training.lateral_regularization.q_power must be > 0")
    if q_epsilon <= 0.0:
        raise ValueError("training.lateral_regularization.q_epsilon must be > 0")
    if channel_reduction not in {"mean", "sum"}:
        raise ValueError(
            "training.lateral_regularization.channel_reduction must be 'mean' or 'sum'"
        )
    if auto_ratio < 0.0:
        raise ValueError("training.lateral_regularization.auto_init.ratio must be >= 0")
    if auto_eps <= 0.0:
        raise ValueError("training.lateral_regularization.auto_init.epsilon must be > 0")

    return {
        "enabled": enabled,
        "lambda": lam,
        "q_power": q_power,
        "q_epsilon": q_epsilon,
        "abs_xrel_feature_index": abs_xrel_feature_index,
        "z_feature_index": z_feature_index,
        "normalize_by_uniform": normalize_by_uniform,
        "channel_reduction": channel_reduction,
        "auto_init": {
            "enabled": auto_enabled,
            "ratio": auto_ratio,
            "epsilon": auto_eps,
        },
    }


def parse_mixer_head_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Parse and validate optional mixer-head settings from experiment config.

    Reads ``model.mixer_head`` and returns a normalized config dict that can be
    passed directly to ``DasInrApodMixer``.

    Args:
        cfg: Full experiment configuration mapping.

    Returns:
        Dictionary with keys:
        - ``enabled``: bool
        - ``hidden_units``: list[int]
        - ``activation``: str
        - ``output_activation``: str | None

    Raises:
        ValueError: If configuration values are invalid.
    """
    model_cfg = cfg.get("model", {})
    mixer_head_cfg_raw = model_cfg.get("mixer_head", {}) if isinstance(model_cfg, Mapping) else {}

    if mixer_head_cfg_raw is None:
        mixer_head_cfg_raw = {}
    if not isinstance(mixer_head_cfg_raw, Mapping):
        raise ValueError("model.mixer_head must be a mapping when provided")

    mixer_head_cfg = dict(mixer_head_cfg_raw)
    enabled = bool(mixer_head_cfg.get("enabled", False))

    hidden_units_raw = mixer_head_cfg.get("hidden_units", [])
    if hidden_units_raw is None:
        hidden_units = []
    elif isinstance(hidden_units_raw, (int, float)):
        hidden_units = [int(hidden_units_raw)]
    elif isinstance(hidden_units_raw, (list, tuple)):
        hidden_units = [int(unit) for unit in hidden_units_raw]
    else:
        raise ValueError(
            "model.mixer_head.hidden_units must be int/float/list/tuple or null"
        )

    if any(unit <= 0 for unit in hidden_units):
        raise ValueError("model.mixer_head.hidden_units entries must be > 0")

    activation = str(mixer_head_cfg.get("activation", "relu")).strip()
    if len(activation) == 0:
        raise ValueError("model.mixer_head.activation must be a non-empty string")

    output_activation_raw = mixer_head_cfg.get("output_activation", "relu")
    if output_activation_raw is None:
        output_activation = None
    else:
        output_activation = str(output_activation_raw).strip()
        if len(output_activation) == 0:
            output_activation = None

    return {
        "enabled": enabled,
        "hidden_units": hidden_units,
        "activation": activation,
        "output_activation": output_activation,
    }
