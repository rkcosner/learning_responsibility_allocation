import os
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tbsim.configs.base import TrainConfig

from avdata import AgentBatch, AgentType, UnifiedDataset
from avdata.data_structures.batch import agent_collate_fn
from avdata.caching.df_cache import DataFrameCache
from avdata.data_structures.scene_metadata import SceneMetadata
from avdata.simulation import SimulationScene
from avdata.visualization.vis import plot_agent_batch


class UnifiedDataModule(pl.LightningDataModule):
    def __init__(self, data_config, train_config: TrainConfig):
        super(UnifiedDataModule, self).__init__()
        self._data_config = data_config
        self._train_config = train_config
        self.train_dataset = None
        self.valid_dataset = None

    @property
    def modality_shapes(self):
        # TODO: better way to figure out channel size?
        return dict(image=(7, self._data_config.pixel_size, self._data_config.pixel_size))

    def setup(self, stage = None):
        data_cfg = self._data_config
        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance
        kwargs = dict(
            desired_data=[data_cfg.avdata_source],
            rebuild_cache=data_cfg.build_cache,
            rebuild_maps=data_cfg.build_cache,
            future_sec=(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            data_dirs={
                data_cfg.avdata_source: data_cfg.dataset_path,
            },
            only_types=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_map=True,
            map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
            },
            verbose=True,
            num_workers=os.cpu_count(),
        )
        # print(kwargs)
        self.train_dataset = UnifiedDataset(**kwargs)
        self.valid_dataset = self.train_dataset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self._train_config.training.batch_size,
            num_workers=self._train_config.training.num_data_workers,
            drop_last=True,
            collate_fn=self.train_dataset.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            shuffle=True,
            batch_size=self._train_config.validation.batch_size,
            num_workers=self._train_config.validation.num_data_workers,
            drop_last=True,
            collate_fn=self.valid_dataset.collate_fn
        )

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass
