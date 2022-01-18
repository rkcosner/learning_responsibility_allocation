import json
from tbsim.configs.registry import get_registered_experiment_config


def translate_l5kit_cfg(cfg):
    """
    Translate a tbsim config to a l5kit config

    Args:
        cfg (ExperimentConfig): an ExperimentConfig instance

    Returns:
        cfg for l5kit
    """
    rcfg = dict()

    rcfg["raster_params"] = cfg.env.rasterizer.to_dict()
    rcfg["raster_params"]["dataset_meta_key"] = cfg.train.dataset_meta_key
    rcfg["model_params"] = cfg.algo
    if "data_generation_params" in cfg.env.keys():
        rcfg["data_generation_params"] = cfg.env["data_generation_params"]
    return rcfg


def get_experiment_config_from_file(file_path):
    ext_cfg = json.load(open(file_path, "r"))
    cfg = get_registered_experiment_config(ext_cfg["registered_name"])
    cfg.update(**ext_cfg)
    return cfg