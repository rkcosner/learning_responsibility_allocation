import os
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tbsim.configs.base import TrainConfig

from avdata import AgentBatch, AgentType, UnifiedDataset


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
        return dict(
            image=(7 + self._data_config.history_num_frames + 1,  # semantic map + num_history + current
                   self._data_config.raster_size,
                   self._data_config.raster_size)
        )

    def setup(self, stage = None):
        data_cfg = self._data_config
        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance

        kwargs = dict(
            centric = data_cfg.centric,
            desired_data=[data_cfg.avdata_source_train],
            desired_dt=data_cfg.step_time,
            future_sec=(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            data_dirs={
                data_cfg.avdata_source_root: data_cfg.dataset_path,
            },
            only_types=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_map=True,
            incl_neighbor_map = self._data_config.incl_neighbor_map,
            map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center
            },
            verbose=False,
            max_agent_num = 1+data_cfg.other_agents_num,
            vectorize_lane = data_cfg.vectorize_lane,
            num_workers=os.cpu_count(),
        )
        print(kwargs)
        self.train_dataset = UnifiedDataset(**kwargs)

        kwargs["desired_data"] = [data_cfg.avdata_source_valid]
        kwargs["rebuild_cache"] = self._train_config.on_ngc
        self.valid_dataset = UnifiedDataset(**kwargs)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self._train_config.training.batch_size,
            num_workers=self._train_config.training.num_data_workers,
            drop_last=True,
            collate_fn=self.train_dataset.get_collate_fn(return_dict=True),
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            shuffle=True,
            batch_size=self._train_config.validation.batch_size,
            num_workers=self._train_config.validation.num_data_workers,
            drop_last=True,
            collate_fn=self.valid_dataset.get_collate_fn(return_dict=True),
            persistent_workers=True
        )

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass
