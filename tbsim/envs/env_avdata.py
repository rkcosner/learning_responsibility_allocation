import numpy as np
from copy import deepcopy
from typing import List, Dict
from torch.utils.data.dataloader import default_collate
from collections import OrderedDict
from l5kit.geometry import transform_points, transform_point

from avdata import AgentBatch, AgentType, UnifiedDataset
from avdata.simulation import SimulationScene
from avdata.visualization.vis import plot_agent_batch

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.vis_utils import render_state_avdata
from tbsim.envs.base import BaseEnv, BatchedEnv, SimulationException
from tbsim.policies.common import RolloutAction, Action
import tbsim.envs.env_metrics as EnvMetrics
from tbsim.utils.timer import Timers
from tbsim.utils.avdata_utils import parse_avdata_batch


class EnvUnifiedSimulation(BaseEnv, BatchedEnv):
    def __init__(
            self,
            env_config,
            num_scenes,
            dataset: UnifiedDataset,
            seed=0,
            prediction_only=False,
            metrics=None,
            renderer=None
    ):
        """
        A gym-like interface for simulating traffic behaviors (both ego and other agents) with UnifiedDataset

        Args:
            env_config (NuscEnvConfig): a Config object specifying the behavior of the simulator
            num_scenes (int): number of scenes to run in parallel
            dataset (UnifiedDataset): a UnifiedDataset instance that contains scene data for simulation
            prediction_only (bool): if set to True, ignore the input action command and only record the predictions
        """
        print(env_config)
        self._npr = np.random.RandomState(seed=seed)
        self.dataset = dataset
        self._env_config = env_config

        self._num_total_scenes = len(dataset.scene_index)
        self._num_scenes = num_scenes
        self._all_scene_info = dataset.scene_index

        # indices of the scenes (in dataset) that are being used for simulation
        self._current_scenes: List[SimulationScene] = None # corresponding dataset of the scenes

        self._frame_index = 0
        self._done = False
        self._prediction_only = prediction_only

        self._cached_observation = None
        self.episode_buffer = []

        self.timers = Timers()

        self._metrics = dict() if metrics is None else metrics

    @property
    def current_scene_names(self):
        return deepcopy([scene.scene_info.name for scene in self._current_scenes])

    @property
    def current_num_agents(self):
        return sum(len(scene.agents) for scene in self._current_scenes)

    @property
    def current_agent_names(self):
        names = []
        for scene in self._current_scenes:
            names.extend([a.name for a in scene.agents])
        return names

    @property
    def num_instances(self):
        return self._num_scenes

    @property
    def total_num_scenes(self):
        return self._num_total_scenes

    def is_done(self):
        return self._done

    def get_reward(self):
        # TODO
        return np.zeros(self._num_scenes)

    @property
    def horizon(self):
        return self._env_config.simulation.num_simulation_steps

    def reset(self, scene_indices: List = None):
        """
        Reset the previous simulation episode. Randomly sample a batch of new scenes unless specified in @scene_indices

        Args:
            scene_indices (List): Optional, a list of scene indices to initialize the simulation episode
        """
        if scene_indices is None:
            # randomly sample a batch of scenes for close-loop rollouts
            all_indices = np.arange(self._num_total_scenes)
            scene_indices = self._npr.choice(
                all_indices, size=(self.num_instances,), replace=False
            )

        scene_info = [self._all_scene_info[i] for i in scene_indices]

        for si in scene_info:
            assert si in self._all_scene_info

        self._num_scenes = len(scene_info)

        assert (
                np.max(scene_indices) < self._num_total_scenes
                and np.min(scene_indices) >= 0
        )

        self._current_scenes = []
        for i, si in enumerate(scene_info):
            sim_scene: SimulationScene = SimulationScene(
                env_name=self._env_config.name,
                scene_name=si.name,
                scene_info=si,
                dataset=self.dataset,
                init_timestep=self._env_config.simulation.start_frame_index,
                freeze_agents=True,
            )
            sim_scene.reset()
            self._current_scenes.append(sim_scene)

        self._frame_index = 0
        self._cached_observation = None
        self._done = False

        for v in self._metrics.values():
            v.reset()

        self.episode_buffer = []
        for _ in range(self.num_instances):
            self.episode_buffer.append(dict(ego_obs=dict(), ego_action=dict(), agents_obs=dict(), agents_action=dict()))

    def render(self, actions_to_take):
        scene_ims = []
        ego_inds = [i for i, name in enumerate(self.current_agent_names) if name == "ego"]
        for i in ego_inds:
            im = render_state_avdata(
                batch=self.get_observation()["agents"],
                batch_idx=i,
                action=actions_to_take
            )
            scene_ims.append(im)
        return np.stack(scene_ims)

    def get_random_action(self):
        ac = self._npr.randn(self.current_num_agents, 1, 3)
        agents = Action(
            positions=ac[:, :, :2],
            yaws=ac[:, :, 2:3]
        )

        return RolloutAction(agents=agents)

    def get_info(self):
        for scene_buffer in self.episode_buffer:
            for mk in scene_buffer:
                for k in scene_buffer[mk]:
                    scene_buffer[mk][k] = np.stack(scene_buffer[mk][k])

        return {
            "scene_index": self.current_scene_names,
        }

    def get_metrics(self):
        """
        Get metrics of the current episode (may compute before is_done==True)

        Returns: a dictionary of metrics, each containing an array of measurement same length as the number of scenes
        """
        metrics = dict()
        # aggregate per-step metrics
        self._add_per_step_metrics(self.get_observation()["agents"])
        for met_name, met in self._metrics.items():
            assert len(met) == self._frame_index + 1, len(met)
            met_vals = met.get_episode_metrics()
            if isinstance(met_vals, dict):
                for k, v in met_vals.items():
                    metrics[met_name + "_" + k] = v
            else:
                metrics[met_name] = met_vals
        return metrics

    def get_observation(self):
        if self._cached_observation is not None:
            return self._cached_observation[0]

        self.timers.tic("get_obs")

        raw_obs = []
        for scene in self._current_scenes:
            raw_obs.extend(scene.get_obs(collate=False))
        agent_obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
        agent_obs = parse_avdata_batch(agent_obs)
        agent_obs = TensorUtils.to_numpy(agent_obs, ignore_if_unspecified=True)

        # cache observations
        self._cached_observation = (dict(agents=agent_obs), raw_obs)
        self.timers.toc("get_obs")

        return self._cached_observation[0]

    def _add_per_step_metrics(self, obs):
        for k, v in self._metrics.items():
            if len(v) >= self._frame_index + 1:
                return
            v.add_step(obs, self.current_scene_names)

    def _step(self, step_actions: RolloutAction):
        if self.is_done():
            raise SimulationException("Cannot step in a finished episode")
        obs = self.get_observation()["agents"]
        # record metrics
        self._add_per_step_metrics(obs)

        action = step_actions.agents.to_dict()
        assert action["positions"].shape[0] == obs["centroid"].shape[0]
        idx = 0
        # print(action["positions"])
        for scene in self._current_scenes:
            scene_action = dict()
            for agent in scene.agents:
                curr_yaw = obs["curr_agent_state"][idx, -1]
                curr_pos = obs["curr_agent_state"][idx, :2]
                world_from_agent = np.array(
                    [
                        [np.cos(curr_yaw), np.sin(curr_yaw)],
                        [-np.sin(curr_yaw), np.cos(curr_yaw)],
                    ]
                )
                next_state = np.zeros(3, dtype=obs["agent_fut"].dtype)
                if not np.any(np.isnan(action["positions"][idx, 0])):
                    next_state[:2] = action["positions"][idx, 0] @ world_from_agent + curr_pos
                    next_state[2] = curr_yaw + action["yaws"][idx, 0, 0]
                    # next_state[:2] = obs["agent_fut"][idx, 0, :2] @ world_from_agent + curr_pos
                    # yaw_ac = np.arctan2(obs["agent_fut"][idx, 0, -2], obs["agent_fut"][idx, 0, -1])
                    # next_state[2] = curr_yaw + yaw_ac
                scene_action[agent.name] = next_state
                idx += 1
            scene.step(scene_action)

        self._cached_observation = None

        if self._frame_index + 1 == self.horizon:
            self._done = True
        else:
            self._frame_index += 1

    def step(self, actions: RolloutAction, num_steps_to_take: int = 1, render=False):
        """
        Step the simulation with control inputs

        Args:
            actions (RolloutAction): action for controlling ego and/or agents
            num_steps_to_take (int): how many env steps to take. Must be less or equal to length of the input actions
            render (bool): whether to render state and actions and return renderings
        """
        actions = actions.to_numpy()
        assert num_steps_to_take == 1
        renderings = []
        if render:
            renderings.append(self.render(actions))
        self._step(step_actions=actions)
        return renderings