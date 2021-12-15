import os

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tempfile import gettempdir

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDatasetVectorized, EgoDatasetMixed
from l5kit.planning.vectorized.closed_loop_model import VectorizedUnrollModel
from l5kit.planning.vectorized.open_loop_model import VectorizedModel
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.rasterization import build_rasterizer
import pdb
import os


os.environ["L5KIT_DATA_FOLDER"] = "/home/chenyx/repos/l5kit/prediction-dataset"

dm = LocalDataManager(None)
# get config
cfg = load_config_data("./config.yaml")

train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()

vectorizer = build_vectorizer(cfg, dm)
rasterizer = build_rasterizer(cfg, dm)
train_dataset = EgoDatasetMixed(cfg, train_zarr, vectorizer, rasterizer)

from tbsim.algos.l5kit_algos import L5TrafficModel, L5TransformerTrafficModel
from tbsim.configs import (
    ExperimentConfig,
    L5KitEnvConfig,
    L5KitTrainConfig,
    L5RasterizedPlanningConfig,
    L5KitVectorizedEnvConfig,
    L5TransformerPredConfig,
)
from tbsim.models.Transformer_model import Transformer_model

algo_config = L5TransformerPredConfig()
model = Transformer_model(algo_config)
model.cuda()

train_cfg = cfg["train_data_loader"]
train_dataloader = DataLoader(
    train_dataset,
    shuffle=train_cfg["shuffle"],
    batch_size=train_cfg["batch_size"],
    num_workers=train_cfg["num_workers"],
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

tr_it = iter(train_dataloader)
data = next(tr_it)
out_dict = model.forward(data)
loss = model.compute_losses(out_dict, data)
