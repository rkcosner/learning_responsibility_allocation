import argparse
import sys
import os
import json
from torch.utils.data import DataLoader
from collections import OrderedDict

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from tbsim.utils.log_utils import PrintLogger
import tbsim.utils.train_utils as TrainUtils
from tbsim.algos.l5kit_algos import L5TrafficModel
from tbsim.configs import ExperimentConfig, L5KitEnvConfig, L5KitTrainConfig, L5RasterizedPlanningConfig
from tbsim.envs.env_l5kit import EnvL5KitSimulation
from tbsim.utils.config_utils import translate_l5kit_cfg


def test_l5_env():
    cfg = ExperimentConfig(
        train_config=L5KitTrainConfig(),
        env_config=L5KitEnvConfig(),
        algo_config=L5RasterizedPlanningConfig()
    )
    l5_config = translate_l5kit_cfg(cfg)

    os.environ["L5KIT_DATA_FOLDER"] = os.path.abspath(cfg.train.dataset_path)
    dm = LocalDataManager(None)
    rasterizer = build_rasterizer(l5_config, dm)
    valid_zarr = ChunkedDataset(dm.require(cfg.train.dataset_valid_key)).open()
    env_dataset = EgoDataset(l5_config, valid_zarr, rasterizer)
    env = EnvL5KitSimulation(cfg.env, dataset=env_dataset, seed=cfg.seed, num_scenes=cfg.train.rollout.num_episodes)
    env.reset()
    ac = {"ego": env.get_random_ego_actions()}
    env.step(ac)
    print(env.get_metrics())

if __name__ == "__main__":
    test_l5_env()