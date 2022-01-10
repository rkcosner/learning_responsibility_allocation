import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tbsim.utils.config_utils import translate_l5kit_cfg
from tqdm import tqdm
from tempfile import gettempdir
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDatasetVectorized, EgoDatasetMixed
from l5kit.planning.vectorized.closed_loop_model import VectorizedUnrollModel
from l5kit.planning.vectorized.open_loop_model import VectorizedModel
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.rasterization import build_rasterizer
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.datasets.factory import datamodule_factory
from tbsim.algos.factory import algo_factory
import pdb
import json
import os

os.environ["L5KIT_DATA_FOLDER"] = "/home/chenyx/repos/l5kit/prediction-dataset"
from tbsim.algos.l5kit_algos import L5TrafficModel, L5TransformerTrafficModel
from tbsim.configs import (
    ExperimentConfig,
    L5KitEnvConfig,
    L5KitTrainConfig,
    L5RasterizedPlanningConfig,
    L5KitMixedEnvConfig,
    L5KitVectorizedEnvConfig,
    L5TransformerPredConfig,
)
from tbsim.models.transformer_model import TransformerModel

config_file = "/home/chenyx/repos/behavior-generation/tbsim/experiments/templates/l5_mixed_transformer_plan.json"

ext_cfg = json.load(open(config_file, "r"))
cfg = get_registered_experiment_config(ext_cfg["registered_name"])
cfg.update(**ext_cfg)
# cfg["env"]["rasterizer"]["map_type"] = "scene_semantic"
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = os.path.abspath(cfg.train.dataset_path)
cfg.algo.tgt_mask_N = 100
l5_config = translate_l5kit_cfg(cfg)
algo_config = cfg.algo
dm = LocalDataManager(None)

train_zarr = ChunkedDataset(dm.require(cfg.train.dataset_valid_key)).open()
vectorizer = build_vectorizer(l5_config, dm)
rasterizer = build_rasterizer(l5_config, dm)
train_dataset = EgoDatasetMixed(l5_config, train_zarr, vectorizer, rasterizer)
# model = L5TransformerTrafficModel.load_from_checkpoint(
#     "2503969/iter9999_ep0_valLoss0.53.ckpt",
#     algo_config=cfg.algo,
# )
model = L5TransformerTrafficModel(algo_config=cfg.algo)
train_cfg = cfg["train_data_loader"]
train_dataloader = DataLoader(
    train_dataset,
    shuffle=train_cfg["shuffle"],
    batch_size=1,
    num_workers=1,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
tr_it = iter(train_dataloader)
data = next(tr_it)
for key, obj in data.items():
    data[key] = obj.to(device)
out_dict = model.nets["policy"].forward(data)
loss = model.nets["policy"].compute_losses(out_dict, data)
# loss = model.training_step(data, algo_config.tgt_mask_N)
