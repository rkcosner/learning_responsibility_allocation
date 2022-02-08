import numpy as np
from copy import deepcopy
from typing import List, Dict
from torch.utils.data.dataloader import default_collate

from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.simulation.unroll import ClosedLoopSimulator, SimulationOutput
from l5kit.cle.metrics import DisplacementErrorL2Metric
from l5kit.geometry import transform_points, rotation33_as_yaw

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.vis_utils import render_state_l5kit
from tbsim.envs.base import BaseEnv, BatchedEnv, SimulationException
from tbsim.utils.geometry_utils import batch_nd_transform_points


class EnvL5KitSimulation(BaseEnv, BatchedEnv):
    def __init__(self, env_config, num_scenes, dataset, seed=0, prediction_only=False):
        """
        A gym-like interface for simulating traffic behaviors (both ego and other agents) with L5Kit's SimulationDataset

        Args:
            env_config (L5KitEnvConfig): a Config object specifying the behavior of the L5Kit CloseLoopSimulator
            num_scenes (int): number of scenes to run in parallel
            dataset (EgoDataset): an EgoDataset instance that contains scene data for simulation
            prediction_only (bool): if set to True, ignore the input action command and only record the predictions
        """
        self._sim_cfg = SimulationConfig(
            disable_new_agents=True,
            distance_th_far=env_config.simulation.distance_th_far,
            distance_th_close=env_config.simulation.distance_th_close,
            num_simulation_steps=env_config.simulation.num_simulation_steps,
            start_frame_index=env_config.simulation.start_frame_index,
            show_info=True,
        )

        self._npr = np.random.RandomState(seed=seed)
        self.dataset = dataset
        self.rasterizer = dataset.rasterizer
        self._num_total_scenes = len(dataset.dataset.scenes)
        self._num_scenes = num_scenes

        self._current_scene_indices = None  # indices of the scenes (in dataset) that are being used for simulation
        self._current_scene_dataset = None  # corresponding dataset of the scenes

        self._frame_index = 0
        self._cached_observation = None
        self._done = False
        self._prediction_only = prediction_only

        self._ego_states = dict()
        self._agents_states = dict()

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
        assert len(scene_indices) == self.num_instances
        assert np.max(scene_indices) < self._num_total_scenes and np.min(scene_indices) >= 0

        self._current_scene_indices = scene_indices
        self._current_scene_dataset = SimulationDataset.from_dataset_indices(
            self.dataset, scene_indices, self._sim_cfg
        )
        self._frame_index = 0
        self._cached_observation = None
        self._done = False
        for k in self._current_scene_indices:
            self._ego_states[k] = []
            self._agents_states[k] = []

    def get_random_ego_actions(self):
        ac = np.random.randn(self._num_scenes, 1, 3)
        return {"positions": ac[:, :, :2], "yaws": ac[:, :, 2:3]}

    @property
    def num_instances(self):
        return self._num_scenes

    def get_info(self):
        return {
            "l5_sim_states": self._get_l5_sim_states(),
            "l5_scene_indices": deepcopy(self._current_scene_indices),
        }

    def get_metrics(self):
        """
        Get metrics of the current episode (may compute before is_done==True)

        Returns: a dictionary of metrics, each containing an array of measurement same length as the number of scenes
        """
        # TODO: phase out the dependencies on l5kit Metrics
        met = DisplacementErrorL2Metric()
        sim_states = self._get_l5_sim_states()
        ego_ade = np.zeros(self._num_scenes)
        ego_fde = np.zeros(self._num_scenes)
        for si, scene_states in enumerate(sim_states):
            err = TensorUtils.to_numpy(met.compute(scene_states))
            ego_ade[si] = np.mean(err)
            ego_fde[si] = err[self._frame_index]

        # TODO: compute agent metrics

        return {
            "ego_ADE": ego_ade,
            "ego_FDE": ego_fde,
        }

    def get_state(self):
        """Get the current raw state of the scenes (ego and agents)"""
        obs = self.get_observation()
        agents_s = obs["agents"]
        agent_dict = dict()
        for idx_agent in range(len(agents_s["track_id"])):
            agent_in = {k: v[idx_agent] for k, v in agents_s.items() if k != "image"}
        # TODO: finish implementation

    def get_observation(self):
        if self._done:
            return None

        if self._cached_observation is not None:
            return self._cached_observation

        agent_obs = self._current_scene_dataset.rasterise_agents_frame_batch(self._frame_index)
        ego_obs = self._current_scene_dataset.rasterise_frame_batch(self._frame_index)
        if len(agent_obs) > 0:
            agent_obs = default_collate(list(agent_obs.values()))

        ego_obs = default_collate(ego_obs)
        self._cached_observation = TensorUtils.to_numpy({"agents": agent_obs, "ego": ego_obs})
        return self._cached_observation

    def is_done(self):
        return self._done

    def get_reward(self):
        # TODO
        return np.zeros(self._num_scenes)

    def render(self, actions_to_take, **kwargs):
        ims = []
        for i in range(self._num_scenes):
            im = render_state_l5kit(
                self,
                state_obs=self.get_observation(),
                action=actions_to_take,
                scene_index=i,
                step_index=self._frame_index,
                **kwargs
            )
            ims.append(im)
        return ims

    def _get_l5_sim_states(self) -> List[SimulationOutput]:
        simulated_outputs: List[SimulationOutput] = []
        for scene_idx in self._current_scene_indices:
            simulated_outputs.append(
                SimulationOutput(
                    scene_idx,
                    self._current_scene_dataset,
                    self._ego_states,
                    self._agents_states,
                )
            )
        return simulated_outputs

    @property
    def horizon(self):
        return len(self._current_scene_dataset)

    def _step(self, ego_control=None, agents_control=None, ego_samples=None, agents_samples=None):
        obs = self.get_observation()
        should_update = self._frame_index + 1 < self.horizon and not self._prediction_only
        if ego_control is not None:
            if should_update:
                # update the next frame's ego position and orientation using control input
                ClosedLoopSimulator.update_ego(
                    dataset=self._current_scene_dataset,
                    frame_idx=self._frame_index + 1,
                    input_dict=obs["ego"],
                    output_dict=ego_control,
                )

            # record state
            ego_in_out = ClosedLoopSimulator.get_ego_in_out(
                obs["ego"], ego_control, keys_to_exclude=set(("image",))
            )
            for i, scene_idx in enumerate(self._current_scene_indices):
                # If applicable, insert prediction samples for visualization purposes
                if ego_samples is not None:
                    ego_in_out[scene_idx].outputs["samples"] = dict()
                    for k in ego_samples:
                        ego_in_out[scene_idx].outputs["samples"][k] = ego_samples[k][i]
                self._ego_states[scene_idx].append(ego_in_out[scene_idx])

        if agents_control is not None:
            if should_update:
                # update the next frame's agent positions and orientations using control input
                ClosedLoopSimulator.update_agents(
                    dataset=self._current_scene_dataset,
                    frame_idx=self._frame_index + 1,
                    input_dict=obs["agents"],
                    output_dict=agents_control,
                )
            agents_in_out = ClosedLoopSimulator.get_agents_in_out(
                obs["agents"], agents_control, keys_to_exclude=set(("image",))
            )
            for i, scene_idx in enumerate(self._current_scene_indices):
                self._agents_states[scene_idx].append(agents_in_out.get(scene_idx, []))

        # TODO: accumulate sim trajectories
        self._cached_observation = None

        if self._frame_index + 1 == self.horizon:
            self._done = True
        else:
            self._frame_index += 1

    def step(self, actions, num_steps_to_take: int = 1):
        """
        Step the simulation with control inputs

        Args:
            actions (Dict): a dictionary containing either or both "ego_control" or "agents_control"
                - ego_control (Dict): a dictionary containing future ego position and orientation for all scenes
                - agents_control (Dict): a dictionary containing future agent positions and orientations for all scenes
            num_steps_to_take (int): how many env steps to take. Must be less or equal to length of the input actions
        """

        if self._done:
            raise SimulationException("Simulation episode has ended")

        # use dataset actions if we are doing prediction-only rollouts
        if self._prediction_only:
            for _ in range(num_steps_to_take):
                self._step()
            return

        # otherwise, use @actions to update simulation
        actions = TensorUtils.to_numpy(actions)

        obs = self.get_observation()
        actions_world = dict()
        # Convert action to world frame
        actions_world["ego"] = dict(
            positions=transform_points(actions["ego"]["positions"], obs["ego"]["world_from_agent"]),
            yaws=obs["ego"]["yaw"][..., None, None] + actions["ego"]["yaws"]
        )
        for step_i in range(num_steps_to_take):
            step_actions_world = TensorUtils.map_ndarray(actions_world, lambda x: x[..., step_i:, :])
            obs = self.get_observation()

            # transform step action back to the agent frame
            step_actions = dict()
            for k in step_actions_world:
                step_actions[k] = dict(
                    positions=transform_points(step_actions_world[k]["positions"], obs["ego"]["agent_from_world"]),
                    yaws=step_actions_world[k]["yaws"] - obs["ego"]["yaw"][..., None, None]
                )

            ego_control = step_actions.get("ego", None)
            ego_samples = step_actions.get("ego_samples", None)
            self._step(ego_control=ego_control, ego_samples=ego_samples)