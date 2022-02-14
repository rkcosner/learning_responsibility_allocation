import abc
import numpy as np
from typing import List, Dict

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.l5_utils as L5Utils
from tbsim.utils.geometry_utils import transform_points_tensor
import tbsim.utils.metrics as Metrics


class EnvMetrics(abc.ABC):
    def __init__(self):
        self._per_step = []
        self._per_step_mask = []

    def reset(self):
        self._per_step = []
        self._per_step_mask = []

    @abc.abstractmethod
    def add_step(self, state_info: Dict, scene_to_agents_index: List):
        pass

    @abc.abstractmethod
    def get_episode_metrics(self):
        pass

    def __len__(self):
        return len(self._per_step)

    @staticmethod
    def step_aggregate_per_scene(met, scene_to_agents_index, agg_func=np.mean):
        """
        Aggregate per-step metrics for each scene.

        1. if there are more than one agent per scene, aggregate their metrics for each scene using @agg_func.
        2. if there are zero agent per scene, the returned mask should have 0 for that scene

        Args:
            met (np.ndarray): metrics for all agents and scene [all_agents_scene]
            scene_to_agents_index (list): agent index in each scene for N scenes
            agg_func: function to aggregate metrics value across all agents in a scene

        Returns:
            met_per_scene (np.ndarray): [N]
            met_per_scene_mask (np.ndarray): [N]
        """
        num_agents_per_scene = np.array([len(s) for s in scene_to_agents_index])
        # split metrics for each scene
        agents_off_road_per_scene = np.split(
            met,
            np.cumsum(num_agents_per_scene)[:-1]
        )
        met_per_scene = np.zeros(len(num_agents_per_scene))
        met_mask_per_scene = num_agents_per_scene > 0
        for i, s_met in enumerate(agents_off_road_per_scene):
            if len(s_met) > 0:
                met_per_scene[i] = agg_func(s_met)
        return met_per_scene, met_mask_per_scene


def compute_weighted_average(met, met_mask):
    """
    Compute average metrics across timesteps given an availability mask
    Args:
        met (np.ndarray): measurements, [num_scene, num_steps]
        met_mask (np.ndarray): measurement masks [num_scene, num_steps]

    Returns:
        avg_met (np.ndarray): [num_scene]
    """
    return (met * met_mask).sum(axis=1) / (met_mask.sum(axis=1) + 1e-8)


class OffRoadRate(EnvMetrics):
    @staticmethod
    def compute_per_step(state_info: dict):
        obs = TensorUtils.to_tensor(state_info)
        drivable_region = L5Utils.get_drivable_region_map(obs["image"])
        centroid_raster = transform_points_tensor(obs["centroid"][:, None], obs["raster_from_world"])[:, 0]
        off_road = Metrics.batch_detect_off_road(centroid_raster, drivable_region)  # [num_agents]
        off_road = TensorUtils.to_numpy(off_road)
        return off_road

    def add_step(self, state_info: dict, scene_to_agents_index: List):
        met = self.compute_per_step(state_info)
        met, met_mask = self.step_aggregate_per_scene(met, scene_to_agents_index, agg_func=np.mean)
        self._per_step.append(met)
        self._per_step_mask.append(met_mask)

    def get_episode_metrics(self):
        met = np.stack(self._per_step, axis=0).transpose((1, 0))  # [num_scene, num_steps]
        met_mask = np.stack(self._per_step_mask, axis=0).transpose((1, 0))  # [num_scene, num_steps]
        return compute_weighted_average(met, met_mask)