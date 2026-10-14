"""I/O and path utilities shared by numeric phantom evaluators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from inr_apodizations.config import PROJ_ROOT


def load_config_yaml(cfg_path: Path) -> dict[str, Any]:
    """Load a YAML file and return a mapping.

    Args:
        cfg_path: Absolute or relative path to a YAML file.

    Returns:
        Parsed YAML mapping. Returns an empty dict when the file content is not a mapping.
    """
    with open(cfg_path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a project-relative path against the repository root.

    Args:
        path_value: Absolute or project-relative path.

    Returns:
        Absolute path.
    """
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJ_ROOT / path).resolve()


def remap_legacy_outputs_path(path: Path) -> Path:
    """Map legacy sandbox output paths to the scripts outputs layout.

    This keeps old ``simulation_info.yml`` files usable after moving outputs.

    Args:
        path: Path potentially pointing to the legacy outputs layout.

    Returns:
        Remapped path when a legacy marker is found, otherwise the input path.
    """
    legacy_marker = "sandbox/inr_das_experiment/outputs"
    new_marker = "scripts/outputs"

    path_norm = str(path).replace("\\", "/")
    if legacy_marker not in path_norm:
        return path

    remapped_norm = path_norm.replace(legacy_marker, new_marker, 1)
    return Path(remapped_norm)


def resolve_latest_delayed_samples_path(io_cfg: dict[str, Any]) -> tuple[Path, Path]:
    """Resolve delayed-samples artifact from latest simulation metadata.

    Args:
        io_cfg: Evaluator ``io`` config block.

    Returns:
        Tuple ``(delayed_samples_file, latest_run_folder)``.

    Raises:
        FileNotFoundError: If expected folders/files are missing.
        ValueError: If metadata format/content is invalid.
    """
    simulation_root_cfg = io_cfg.get(
        "simulation_output_root", "scripts/outputs/evaluation/numeric_phantom/delayed_samples"
    )
    simulation_root = resolve_project_path(simulation_root_cfg)
    if not simulation_root.exists() or not simulation_root.is_dir():
        raise FileNotFoundError(
            "Missing `io.simulation_output_root` or it is not a folder: "
            f"{simulation_root}"
        )

    run_folders = [path for path in simulation_root.iterdir() if path.is_dir()]
    if len(run_folders) == 0:
        raise FileNotFoundError(
            "No simulation runs found in `io.simulation_output_root`: "
            f"{simulation_root}"
        )

    latest_run = max(run_folders, key=lambda path: path.stat().st_mtime)
    simulation_info_path = latest_run / "simulation_info.yml"
    if not simulation_info_path.exists():
        raise FileNotFoundError(
            "simulation_info.yml not found in latest simulation run: "
            f"{simulation_info_path}"
        )

    simulation_info = load_config_yaml(simulation_info_path)
    delayed_samples_cfg = simulation_info.get("delayed_samples_path")
    if not isinstance(delayed_samples_cfg, str) or len(delayed_samples_cfg.strip()) == 0:
        raise ValueError(
            "Missing valid `delayed_samples_path` in simulation_info.yml: "
            f"{simulation_info_path}"
        )

    delayed_samples_path = Path(delayed_samples_cfg)
    if not delayed_samples_path.is_absolute():
        delayed_samples_path = resolve_project_path(delayed_samples_path)

    if not delayed_samples_path.exists() or delayed_samples_path.is_dir():
        remapped_path = remap_legacy_outputs_path(delayed_samples_path)
        if remapped_path != delayed_samples_path and remapped_path.exists() and remapped_path.is_file():
            delayed_samples_path = remapped_path

    if not delayed_samples_path.exists() or delayed_samples_path.is_dir():
        raise FileNotFoundError(
            "`delayed_samples_path` does not point to a valid file: "
            f"{delayed_samples_path}"
        )

    return delayed_samples_path, latest_run


def resolve_mixer_artifacts(model_path_cfg: str | Path) -> tuple[Path, Path, Path, Path]:
    """Resolve mixer artifacts from a mixer run directory or model file path.

    Args:
        model_path_cfg: Path to mixer run directory or ``model.keras`` file.

    Returns:
        Tuple ``(model_file, combiner_weights_file, train_info_file, run_dir)``.

    Raises:
        FileNotFoundError: If required artifacts are missing.
    """
    model_path = resolve_project_path(model_path_cfg)
    if not model_path.exists():
        raise FileNotFoundError(f"Mixer model path does not exist: {model_path}")

    if model_path.is_file():
        if model_path.name != "model.keras":
            raise FileNotFoundError(
                "`io.model_path` must point to a mixer run directory or to `model.keras`. "
                f"Got file path: {model_path}"
            )
        model_file = model_path
        run_dir = model_path.parent
    else:
        run_dir = model_path
        model_file = run_dir / "model.keras"

    if not model_file.exists() or not model_file.is_file():
        raise FileNotFoundError(f"Missing mixer model file: {model_file}")

    combiner_weights_file = run_dir / "mixer_combiner_weights.npz"
    if not combiner_weights_file.exists() or not combiner_weights_file.is_file():
        raise FileNotFoundError(
            "Missing mixer combiner weights file. Expected: "
            f"{combiner_weights_file}"
        )

    train_info = run_dir / "train_config_info.yml"
    if not train_info.exists() or not train_info.is_file():
        raise FileNotFoundError(f"Missing train_config_info.yml: {train_info}")

    return model_file, combiner_weights_file, train_info, run_dir


def extract_mixer_train_metadata(
    train_info_path: Path,
) -> tuple[bool, int, str, list[str] | None]:
    """Extract mixer model metadata from ``train_config_info.yml``.

    Args:
        train_info_path: Path to mixer ``train_config_info.yml``.

    Returns:
        Tuple ``(scaled_features, n_apodizations, physical_feature_set, physical_feature_components)``.

    Raises:
        ValueError: If required metadata is missing or invalid.
    """
    train_info = load_config_yaml(train_info_path)
    experiment_cfg = train_info.get("experiment")
    if not isinstance(experiment_cfg, dict):
        raise ValueError(f"Invalid format in {train_info_path}: missing `experiment` block")

    model_cfg = experiment_cfg.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Invalid format in {train_info_path}: missing `experiment.model` block")

    if "scaled_features" not in model_cfg:
        raise ValueError(f"Missing `experiment.model.scaled_features` in {train_info_path}.")
    scaled_features = bool(model_cfg["scaled_features"])

    physical_feature_set = str(model_cfg.get("physical_feature_set", "distance_depth_edge"))
    raw_components = model_cfg.get("physical_feature_components")
    physical_feature_components = (
        [str(token) for token in raw_components] if isinstance(raw_components, list) else None
    )

    resolved_mixer = train_info.get("resolved_mixer")
    n_apodizations = None
    if isinstance(resolved_mixer, dict) and "n_apodizations" in resolved_mixer:
        n_apodizations = int(resolved_mixer["n_apodizations"])
    elif "n_apodizations" in model_cfg:
        n_apodizations = int(model_cfg["n_apodizations"])

    if n_apodizations is None or n_apodizations <= 0:
        raise ValueError(
            f"Missing valid `n_apodizations` in {train_info_path} "
            "(checked `resolved_mixer` and `experiment.model`)."
        )

    return scaled_features, n_apodizations, physical_feature_set, physical_feature_components
