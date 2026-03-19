import pymust
import numpy as np
import yaml

def cfg_to_must_param(cfg):
    """
    Convert a config dictionary into a pymust.utils.Param object.

    Expected config structure:
        cfg["probe"]:
            - pitch (mm)
            - element_width (mm)
            - element_height (mm)
            - n_elements
            - fc (MHz)
            - bandwidth (%)
        cfg["plane_wave_acquisition"]:
            - fs (MHz)
        cfg["c1"]:
            - speed of sound (mm/us)

    Returns:
        pymust.utils.Param: Parameter object with SI units required by pymust.
    """
    probe = cfg["probe"]
    acq = cfg["plane_wave_acquisition"]

    param = pymust.utils.Param()
    param.fs = acq["fs"] * 1e6                    # MHz -> Hz
    param.fc = probe["fc"] * 1e6                  # MHz -> Hz
    param.pitch = probe["pitch"] * 1e-3           # mm -> m
    param.Nelements = probe["n_elements"]
    param.c = cfg["c1"] * 1e3                     # mm/us -> m/s
    param.bandwidth = probe["bandwidth"]          # Percent bandwidth
    param.width = probe["element_width"] * 1e-3   # mm -> m
    param.height = probe["element_height"] * 1e-3 # mm -> m
    param.radius = np.inf

    # Optional linear array element positions:
    # x0 = param.pitch * (param.Nelements - 1) / 2
    # param.elements = np.arange(param.Nelements) * param.pitch - x0

    return param

def save_config_yaml(config_path, cfg, extra_params):
    """
    Save configuration and extra parameters to a YAML file.
    All lists and tuples are saved in flat (flow) style.

    Args:
        config_path (Path): Output YAML file path.
        cfg (dict): Main configuration dictionary.
        extra_params (dict): Additional parameters to save.
    """
    def flat_seq_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    yaml.add_representer(list, flat_seq_representer, Dumper=yaml.SafeDumper)
    yaml.add_representer(tuple, flat_seq_representer, Dumper=yaml.SafeDumper)

    config_to_save = dict(cfg)
    config_to_save.update(extra_params)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_to_save, f, allow_unicode=True, Dumper=yaml.SafeDumper)
    