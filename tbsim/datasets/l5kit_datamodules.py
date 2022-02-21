"""Functions and classes for dataset I/O"""
import abc
from collections import OrderedDict
import numpy as np

from typing import Optional
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from l5kit.rasterization import build_rasterizer
from l5kit.rasterization.rasterizer import Rasterizer
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset, AgentDataset

from tbsim.configs.base import TrainConfig
from tbsim.external.l5_ego_dataset import (
    EgoDatasetMixed, EgoReplayBufferMixed
)


class LazyRasterizer(Rasterizer):
    """
    Only creates the actual rasterizer when a member function is called.

    A Rasterizer class is non-pickleable, which means that pickle complains about it when we try to do
    multi-process training (e.g., multiGPU training). This class is a way to circumvent the issue by only
    creating the rasterizer object when it's being used in the spawned processes.
    """
    def __init__(self, l5_config, data_manager):
        super(LazyRasterizer, self).__init__()
        self._l5_config = l5_config
        self._dm = data_manager
        self._rasterizer = None

    @property
    def rasterizer(self):
        if self._rasterizer is None:
            self._rasterizer = build_rasterizer(self._l5_config, self._dm)
        return self._rasterizer

    def rasterize(self, *args, **kwargs):
        return self.rasterizer.rasterize(*args, **kwargs)

    def to_rgb(self,*args, **kwargs):
        return self.rasterizer.to_rgb(*args, **kwargs)

    def num_channels(self) -> int:
        return self.rasterizer.num_channels()



class L5BaseDatasetModule(abc.ABC):
    pass


class L5RasterizedDataModule(pl.LightningDataModule, L5BaseDatasetModule):
    def __init__(
            self,
            l5_config: dict,
            train_config: TrainConfig,
            mode: str,
    ):
        super().__init__()
        self.train_dataset = None
        self.valid_dataset = None
        self.experience_dataset = None  # replay buffer
        self.rasterizer = None
        self._train_config = train_config
        self._l5_config = l5_config

        assert mode in ["ego", "agents"]
        self._mode = mode

    @property
    def modality_shapes(self):
        dm = LocalDataManager(None)
        rasterizer = build_rasterizer(self._l5_config, dm)
        return OrderedDict(image=(rasterizer.num_channels(), 224, 224))

    def setup(self, stage: Optional[str] = None):
        os.environ["L5KIT_DATA_FOLDER"] = os.path.abspath(self._train_config.dataset_path)
        dm = LocalDataManager(None)
        self.rasterizer = LazyRasterizer(self._l5_config, dm)

        train_zarr = ChunkedDataset(dm.require(self._train_config.dataset_train_key)).open()
        valid_zarr = ChunkedDataset(dm.require(self._train_config.dataset_valid_key)).open()
        self.ego_trainset = EgoDataset(self._l5_config, train_zarr, self.rasterizer)
        self.ego_validset = EgoDataset(self._l5_config, valid_zarr, self.rasterizer)
        # TODO: Fix NGC issue (see gitlab issue page)
        # agents_mask = np.zeros(len(train_zarr.agents), dtype=np.bool)
        # agents_mask[np.arange(0, len(agents_mask), 100)] = True
        # self.agents_trainset = AgentDataset(self._l5_config, train_zarr, self.rasterizer, agents_mask=agents_mask)
        # agents_mask = np.zeros(len(valid_zarr.agents), dtype=np.bool)
        # agents_mask[np.arange(0, len(agents_mask), 100)] = True
        # self.agents_validset = AgentDataset(self._l5_config, valid_zarr, self.rasterizer, agents_mask=agents_mask)

        if self._mode == "ego":
            self.train_dataset = self.ego_trainset
            self.valid_dataset = self.ego_validset
        else:
            raise NotImplementedError
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


class L5MixedDataModule(L5RasterizedDataModule):
    def __init__(
            self,
            l5_config,
            train_config: TrainConfig,
            mode: str
    ):
        super(L5MixedDataModule, self).__init__(
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
            self.experience_dataset = EgoReplayBufferMixed(
                self._l5_config,
                vectorizer=self.vectorizer,
                rasterizer=self.rasterizer,
                capacity=self._train_config.training.buffer_size
            )
            self.train_dataset = self.ego_trainset
            self.valid_dataset = self.ego_validset
        else:
            raise NotImplementedError("Agent mixed dataset is not supported yet!")