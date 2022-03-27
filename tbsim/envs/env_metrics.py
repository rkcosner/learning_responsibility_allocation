import abc
import numpy as np
from typing import List, Dict

import torch
from l5kit.geometry import transform_points, angular_distance

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
    assert met.shape == met_mask.shape
    return (met * met_mask).sum(axis=1) / (met_mask.sum(axis=1) + 1e-8)


def masked_max_per_episode(met, met_mask):
    """

    Args:
        met (np.ndarray): measurements, [num_scene, num_steps]
        met_mask (np.ndarray): measurement masks [num_scene, num_steps]

    Returns:
        avg_max (np.ndarray): [num_scene]
    """
    assert met.shape == met_mask.shape
    return (met * met_mask).max(axis=1)


class OffRoadRate(EnvMetrics):
    """Compute the fraction of the time that the agent is in undrivable regions"""
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
        self._agent_track_id = None

    def reset(self):
        self._per_step = {CollisionType.FRONT: [], CollisionType.REAR: [], CollisionType.SIDE:[], "coll_any": []}
        self._all_scene_index = None
        self._agent_scene_index = None
        self._agent_track_id = None

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

        # compute per-scene collision counts (for visualization purposes)
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
        if self._all_scene_index is None:  # start of an episode
            self._all_scene_index = all_scene_index
            self._agent_scene_index = state_info["scene_index"]
            self._agent_track_id = state_info["track_id"]

        met_all, _ = self.compute_per_step(state_info, all_scene_index)

        # reassign metrics according to the track id of the initial state (in case some agents go missing)
        for k, met in met_all.items():
            met_a = np.zeros(len(self._agent_track_id))  # assume no collision for missing agents
            for i, (sid, tid) in enumerate(zip(state_info["scene_index"], state_info["track_id"])):
                agent_index = np.bitwise_and(self._agent_track_id == tid, self._agent_scene_index == sid)
                assert np.sum(agent_index) == 1  # make sure there is no new agent
                met_a[agent_index] = met[i]
            met_all[k] = met_a

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


class LearnedMetric(EnvMetrics):
    def __init__(self, metric_algo):
        super(LearnedMetric, self).__init__()
        self.metric_algo = metric_algo
        self.traj_len = metric_algo.algo_config.future_num_frames
        self.state_buffer = []
        self._all_scene_index = None

    def reset(self):
        self.state_buffer = []
        self._per_step = []
        self._per_step_mask = []
        self._all_scene_index = None

    def __len__(self):
        return len(self.state_buffer)

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        self.state_buffer.append(state_info)
        self._all_scene_index = all_scene_index

    def get_episode_metrics(self, state_buffer=None, traj_len=None, all_scene_index=None):
        if traj_len is None:
            traj_len = self.metric_algo.algo_config.future_num_frames

        if state_buffer is None:
            state_buffer = self.state_buffer

        if all_scene_index is None:
            all_scene_index = self._all_scene_index

        num_steps = len(state_buffer) - traj_len

        if num_steps <= 0:
            print("WARNING: LearnedScore metric needs trajectories longer than {} to compute".format(traj_len))
            return None

        ep_metrics = dict()

        for state_i in range(num_steps):
            # assemble score function input
            state = dict(state_buffer[state_i])  # avoid changing the original state_dict
            agent_from_world = state["agent_from_world"]
            yaw_world = state["yaw"]

            # transform traversed trajectories into the ego frame of a given state
            traj_inds = range(state_i + 1, state_i + traj_len + 1)
            traj_pos = [state_buffer[traj_i]["centroid"] for traj_i in traj_inds]
            traj_yaw = [state_buffer[traj_i]["yaw"] for traj_i in traj_inds]
            traj_pos = np.stack(traj_pos, axis=1)  # [B, T, 2]
            traj_yaw = np.stack(traj_yaw, axis=1)  # [B, T]
            assert traj_pos.shape[0] == traj_yaw.shape[0]

            agent_traj_pos = transform_points(points=traj_pos, transf_matrix=agent_from_world)
            agent_traj_yaw = angular_distance(traj_yaw, yaw_world[:, None])

            state["target_positions"] = agent_traj_pos
            state["target_yaws"] = agent_traj_yaw[:, :, None]

            state_torch = TensorUtils.to_torch(state, self.metric_algo.device)
            with torch.no_grad():
                metrics = self.metric_algo.get_metrics(state_torch)
            metrics= TensorUtils.to_numpy(metrics)

            for k in metrics:
                if k not in ep_metrics:
                    ep_metrics[k] = []
                met, met_mask = step_aggregate_per_scene(metrics[k], state["scene_index"], all_scene_index)
                assert np.all(met_mask > 0)  # since we will always use it for all agents
                ep_metrics[k].append(met)

        ep_metrics_agg = dict()
        for k in ep_metrics:
            met = np.stack(ep_metrics[k], axis=1)  # [num_scene, T, ...]
            ep_metrics_agg[k] = np.mean(met, axis=1)
            for met_horizon in [10, 50, 100, 150]:
                if num_steps >= met_horizon:
                    ep_metrics_agg[k + "_@{}".format(met_horizon)] = np.mean(met[:, :met_horizon], axis=1)
        return ep_metrics_agg