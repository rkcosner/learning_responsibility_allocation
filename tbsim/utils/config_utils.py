import json
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.configs.base import ExperimentConfig
from tbsim.configs.config import Dict


def translate_l5kit_cfg(cfg: ExperimentConfig):
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


def get_experiment_config_from_file(file_path, locked=False):
    ext_cfg = json.load(open(file_path, "r"))
    cfg = get_registered_experiment_config(ext_cfg["registered_name"])
    cfg.update(**ext_cfg)
    cfg.lock(locked)
    return cfg


def translate_avdata_cfg(cfg: ExperimentConfig):
    rcfg = Dict()
    assert cfg.algo.step_time == 0.5  # TODO: support interpolation
    rcfg.step_time = cfg.algo.step_time
    rcfg.avdata_source = cfg.train.avdata_source
    rcfg.dataset_path = cfg.train.dataset_path
    rcfg.history_num_frames = cfg.algo.history_num_frames
    rcfg.future_num_frames = cfg.algo.future_num_frames
    rcfg.max_agents_distance = cfg.env.data_generation_params.max_agents_distance
    rcfg.num_other_agents = cfg.env.data_generation_params.other_agents_num
    rcfg.max_agents_distance_simulation = cfg.env.simulation.distance_th_close
    rcfg.pixel_size = cfg.env.rasterizer.pixel_size
    rcfg.raster_size = int(cfg.env.rasterizer.raster_size)
    rcfg.build_cache = cfg.train.on_ngc
    rcfg.lock()
    return rcfg
