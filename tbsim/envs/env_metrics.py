import abc
import numpy as np
from typing import List, Dict

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.l5_utils as L5Utils
from tbsim.utils.geometry_utils import transform_points_tensor, detect_collision, CollisionType
import tbsim.utils.metrics as Metrics


class EnvMetrics(abc.ABC):
    def __init__(self):
        self._per_step = None
        self._per_step_mask = None
        self.reset()

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def add_step(self, state_info: Dict, scene_to_agents_index: List):
        pass

    @abc.abstractmethod
    def get_episode_metrics(self) -> Dict[str, np.ndarray]:
        pass

    def __len__(self):
        return len(self._per_step)


def step_aggregate_per_scene(agent_met, agent_scene_index, all_scene_index, agg_func=np.mean):
    """
    Aggregate per-step metrics for each scene.

    1. if there are more than one agent per scene, aggregate their metrics for each scene using @agg_func.
    2. if there are zero agent per scene, the returned mask should have 0 for that scene

    Args:
        agent_met (np.ndarray): metrics for all agents and scene [num_agents, ...]
        agent_scene_index (np.ndarray): scene index for each agent [num_agents]
        all_scene_index (list, np.ndarray): a list of scene indices [num_scene]
        agg_func: function to aggregate metrics value across all agents in a scene

    Returns:
        met_per_scene (np.ndarray): [num_scene]
        met_per_scene_mask (np.ndarray): [num_scene]
    """
    met_per_scene = split_agents_by_scene(agent_met, agent_scene_index, all_scene_index)
    met_agg_per_scene = []
    for met in met_per_scene:
        if len(met) > 0:
            met_agg_per_scene.append(agg_func(met))
        else:
            met_agg_per_scene.append(np.zeros_like(agent_met[0]))
    met_mask_per_scene = [len(met) > 0 for met in met_per_scene]
    return np.stack(met_agg_per_scene, axis=0), np.array(met_mask_per_scene)


def split_agents_by_scene(agent, agent_scene_index, all_scene_index):
    assert agent.shape[0] == agent_scene_index.shape[0]
    agent_split = []
    for si in all_scene_index:
        agent_split.append(agent[agent_scene_index == si])
    return agent_split


def agent_index_by_scene(agent_scene_index, all_scene_index):
    agent_split = []
    for si in all_scene_index:
        agent_split.append(np.where(agent_scene_index == si)[0])
    return agent_split


def masked_average_per_episode(met, met_mask):
    """
    Compute average metrics across timesteps given an availability mask
    Args:
        met (np.ndarray): measurements, [num_scene, num_steps]
        met_mask (np.ndarray): measurement masks [num_scene, num_steps]

    Returns:
        avg_met (np.ndarray): [num_scene]
    """
    return (met * met_mask).sum(axis=1) / (met_mask.sum(axis=1) + 1e-8)


def masked_max_per_episode(met, met_mask):
    """

    Args:
        met (np.ndarray): measurements, [num_scene, num_steps]
        met_mask (np.ndarray): measurement masks [num_scene, num_steps]

    Returns:
        avg_max (np.ndarray): [num_scene]
    """
    return (met * met_mask).max(axis=1)


class OffRoadRate(EnvMetrics):
    def reset(self):
        self._per_step = []
        self._per_step_mask = []

    @staticmethod
    def compute_per_step(state_info: dict, all_scene_index: np.ndarray):
        obs = TensorUtils.to_tensor(state_info)
        drivable_region = L5Utils.get_drivable_region_map(obs["image"])
        centroid_raster = transform_points_tensor(obs["centroid"][:, None], obs["raster_from_world"])[:, 0]
        off_road = Metrics.batch_detect_off_road(centroid_raster, drivable_region)  # [num_agents]
        off_road = TensorUtils.to_numpy(off_road)
        return off_road

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        met = self.compute_per_step(state_info, all_scene_index)
        met, met_mask = step_aggregate_per_scene(
            met,
            state_info["scene_index"],
            all_scene_index,
            agg_func=lambda x: float(np.mean(x, axis=0))
        )
        self._per_step.append(met)
        self._per_step_mask.append(met_mask)

    def get_episode_metrics(self):
        met = np.stack(self._per_step, axis=0).transpose((1, 0))  # [num_scene, num_steps]
        met_mask = np.stack(self._per_step_mask, axis=0).transpose((1, 0))  # [num_scene, num_steps]
        return masked_average_per_episode(met, met_mask)


class CollisionRate(EnvMetrics):
    """Compute collision rate across all agents in a batch of data."""
    def __init__(self):
        super(CollisionRate, self).__init__()
        self._all_scene_index = None
        self._agent_scene_index = None

    def reset(self):
        self._per_step = {CollisionType.FRONT: [], CollisionType.REAR: [], CollisionType.SIDE:[], "coll_any": []}
        self._all_scene_index = None
        self._agent_scene_index = None

    def __len__(self):
        return len(self._per_step["coll_any"])

    @staticmethod
    def compute_per_step(state_info: dict, all_scene_index: np.ndarray):
        """Compute per-agent and per-scene collision rate and type"""
        agent_scene_index = state_info["scene_index"]
        pos_per_scene = split_agents_by_scene(state_info["centroid"], agent_scene_index, all_scene_index)
        yaw_per_scene = split_agents_by_scene(state_info["yaw"], agent_scene_index, all_scene_index)
        extent_per_scene = split_agents_by_scene(state_info["extent"][..., :2], agent_scene_index, all_scene_index)
        agent_index_per_scene = agent_index_by_scene(agent_scene_index, all_scene_index)

        num_scenes = len(all_scene_index)
        num_agents = len(agent_scene_index)

        coll_rates = dict()
        for k in CollisionType:
            coll_rates[k] = np.zeros(num_agents)
        coll_rates["coll_any"] = np.zeros(num_agents)

        # for each scene, compute collision rate
        for i in range(num_scenes):
            num_agents_in_scene = pos_per_scene[i].shape[0]
            for j in range(num_agents_in_scene):
                other_agent_mask = np.arange(num_agents_in_scene) != j
                coll = detect_collision(
                    ego_pos=pos_per_scene[i][j],
                    ego_yaw=yaw_per_scene[i][j],
                    ego_extent=extent_per_scene[i][j],
                    other_pos=pos_per_scene[i][other_agent_mask],
                    other_yaw=yaw_per_scene[i][other_agent_mask],
                    other_extent=extent_per_scene[i][other_agent_mask]
                )
                if coll is not None:
                    coll_rates[coll[0]][agent_index_per_scene[i][j]] = 1.
                    coll_rates["coll_any"][agent_index_per_scene[i][j]] = 1.

        coll_counts = dict()
        for k in coll_rates:
            coll_counts[k], _ = step_aggregate_per_scene(
                coll_rates[k],
                agent_scene_index,
                all_scene_index,
                agg_func=np.sum
            )

        return coll_rates, coll_counts

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        self._all_scene_index = all_scene_index
        self._agent_scene_index = state_info["scene_index"]
        met_all, _ = self.compute_per_step(state_info, all_scene_index)
        for k in self._per_step:
            self._per_step[k].append(met_all[k])

    def get_episode_metrics(self):
        met_all = dict()
        for coll_type, coll_all_agents in self._per_step.items():
            coll_all_agents = np.stack(coll_all_agents)  # [num_steps, num_agents]
            coll_all_agents_ep = np.max(coll_all_agents, axis=0)  # whether an agent has ever collided into another
            met, met_mask = step_aggregate_per_scene(
                agent_met=coll_all_agents_ep,
                agent_scene_index=self._agent_scene_index,
                all_scene_index=self._all_scene_index
            )
            met_all[str(coll_type)] = met * met_mask
        return met_all