"""Functions and classes for dataset I/O"""
from typing import Optional
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from l5kit.rasterization import build_rasterizer
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset, AgentDataset

from tbsim.configs.base import TrainConfig
from tbsim.external.l5_ego_dataset import (
    EgoDatasetVectorized,
    EgoDatasetMixed,
)

class L5RasterizedDatasetModule(pl.LightningDataModule):
    def __init__(
            self,
            l5_config,
            train_config: TrainConfig,
            mode: str,
    ):
        super().__init__()
        self.train_dataset = None
        self.valid_dataset = None
        self.rasterizer = None
        self._train_config = train_config
        self._l5_config = l5_config

        assert mode in ["ego", "agents"]
        self._mode = mode

    def setup(self, stage: Optional[str] = None):
        os.environ["L5KIT_DATA_FOLDER"] = os.path.abspath(self._train_config.dataset_path)
        dm = LocalDataManager(None)
        self.rasterizer = build_rasterizer(self._l5_config, dm)

        train_zarr = ChunkedDataset(dm.require(self._train_config.dataset_train_key)).open()
        valid_zarr = ChunkedDataset(dm.require(self._train_config.dataset_valid_key)).open()
        self.ego_trainset = EgoDataset(self._l5_config, train_zarr, self.rasterizer)
        self.ego_validset = EgoDataset(self._l5_config, valid_zarr, self.rasterizer)
        self.agents_trainset = AgentDataset(self._l5_config, train_zarr, self.rasterizer)
        self.agents_validset = AgentDataset(self._l5_config, valid_zarr, self.rasterizer)

        if self._mode == "ego":
            self.train_dataset = self.ego_trainset
            self.valid_dataset = self.ego_validset
        else:
            self.train_dataset = self.agents_trainset
            self.valid_dataset = self.agents_validset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self._train_config.training.batch_size,
            num_workers=self._train_config.training.num_data_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            shuffle=True,
            batch_size=self._train_config.validation.batch_size,
            num_workers=self._train_config.validation.num_data_workers,
            drop_last=True,
        )
    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass


class L5MixedDatasetModule(L5RasterizedDatasetModule):
    def __init__(
            self,
            l5_config,
            train_config: TrainConfig,
            mode: str
    ):
        super(L5MixedDatasetModule, self).__init__(
            l5_config=l5_config, train_config=train_config, mode=mode)
        self.vectorizer = None

    def setup(self, stage: Optional[str] = None):
        os.environ["L5KIT_DATA_FOLDER"] = os.path.abspath(self._train_config.dataset_path)
        dm = LocalDataManager(None)
        self.rasterizer = build_rasterizer(self._l5_config, dm)
        self.vectorizer = build_vectorizer(self._l5_config, dm)

        train_zarr = ChunkedDataset(dm.require(self._train_config.dataset_train_key)).open()
        valid_zarr = ChunkedDataset(dm.require(self._train_config.dataset_valid_key)).open()

        self.ego_trainset = EgoDatasetMixed(self._l5_config, train_zarr, self.vectorizer, self.rasterizer)
        self.ego_validset = EgoDatasetMixed(self._l5_config, valid_zarr, self.vectorizer, self.rasterizer)

        if self._mode == "ego":
            self.train_dataset = self.ego_trainset
            self.valid_dataset = self.ego_validset
        else:
            raise NotImplementedError("Agent mixed dataset is not supported yet!")