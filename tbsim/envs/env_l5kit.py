import numpy as np
from typing import List, Dict
from torch.utils.data.dataloader import default_collate

from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.simulation.unroll import ClosedLoopSimulator

import tbsim.utils.tensor_utils as TensorUtils


class SimulationException(Exception):
    pass


class BatchedEnv(object):
    pass


class EnvL5KitSimulation(BatchedEnv):
    def __init__(self, env_config, num_scenes, dataset, seed=0):
        """
        A gym-like interface for simulating traffic behaviors (both ego and other agents) with L5Kit's SimulationDataset

        Args:
            env_config (L5KitEnvConfig): a Config object specifying the behavior of the L5Kit CloseLoopSimulator
            num_scenes (int): number of scenes to run in parallel
            dataset (EgoDataset): an EgoDataset instance that contains scene data for simulation
        """
        self._sim_cfg = SimulationConfig(
            disable_new_agents=True,
            distance_th_far=env_config.simulation.distance_th_far,
            distance_th_close=env_config.simulation.distance_th_close,
            num_simulation_steps=env_config.simulation.num_simulation_steps,
            start_frame_index=env_config.simulation.start_frame_index,
            show_info=True
        )

        self._npr = np.random.RandomState(seed=seed)
        self.dataset = dataset
        self._num_scenes = num_scenes

        self._active_scenes = None
        self._active_scenes_dataset = None

        self._frame_index = 0
        self._cached_observation = None
        self._done = False

    def reset(self, scene_indices: List=None):
        """
        Reset the previous simulation episode.

        Args:
            scene_indices (List): Optional, a list of scene indices to initialize the simulation episode
        """
        num_scenes = len(self.dataset.dataset.scenes)
        if scene_indices is None:
            scene_indices = self._npr.randint(low=0, high=num_scenes, size=(self.num_instances,))
        else:
            assert len(scene_indices) == self._num_scenes
            assert np.max(scene_indices) < num_scenes and np.min(scene_indices) >= 0
        self._active_scenes = scene_indices
        self._active_scenes_dataset = SimulationDataset.from_dataset_indices(self.dataset, scene_indices, self._sim_cfg)
        self._frame_index = 0
        self._cached_observation = None
        self._done = False

    @property
    def num_instances(self):
        return self._num_scenes

    def get_metrics(self):
        """
        Get metrics of the current episode (may compute before is_done==True)

        Returns: a dictionary of metrics, each containing an array of measurement same length as the number of scenes
        """
        return {"ADE": np.zeros(self._num_scenes), "FDE": np.zeros(self._num_scenes)}

    def get_observation(self):
        if self._done:
            return None

        if self._cached_observation is not None:
            return self._cached_observation

        agent_obs = self._active_scenes_dataset.rasterise_agents_frame_batch(self._frame_index)
        ego_obs = self._active_scenes_dataset.rasterise_frame_batch(self._frame_index)
        agent_obs = default_collate(list(agent_obs.values()))
        ego_obs = default_collate(ego_obs)
        self._cached_observation = {"agents": agent_obs, "ego": ego_obs}
        return self._cached_observation

    def is_done(self):
        return self._done

    def get_reward(self):
        raise NotImplementedError

    @property
    def horizon(self):
        return len(self._active_scenes_dataset)

    def step(self, actions):
        """
        Step the simulation with control inputs

        Args:
            actions (Dict): a dictionary containing either or both "ego_control" or "agents_control"
                - ego_control (Dict): a dictionary containing future ego position and orientation for all scenes
                - agents_control (Dict): a dictionary containing future agent positions and orientations for all scenes
        """
        if self._done:
            raise SimulationException("Simulation episode has ended")

        actions = TensorUtils.to_numpy(actions)

        ego_control = actions.get("ego_control", None)
        agents_control = actions.get("agents_control", None)

        should_update = self._frame_index + 1 < self.horizon

        if ego_control is not None and should_update:
            # update the next frame's ego position and orientation using control input
            ClosedLoopSimulator.update_ego(
                dataset=self._active_scenes_dataset,
                frame_idx=self._frame_index + 1,
                input_dict=TensorUtils.to_numpy(self.get_observation()["ego"]),
                output_dict=TensorUtils.to_numpy(ego_control)
            )

        if agents_control is not None and should_update:
            # update the next frame's agent positions and orientations using control input
            ClosedLoopSimulator.update_agents(
                dataset=self._active_scenes_dataset,
                frame_idx=self._frame_index + 1,
                input_dict=TensorUtils.to_numpy(self.get_observation()["agents"]),
                output_dict=TensorUtils.to_numpy(agents_control)
            )

        self._frame_index += 1
        self._cached_observation = None

        if self._frame_index == self.horizon:
            self._done = True


