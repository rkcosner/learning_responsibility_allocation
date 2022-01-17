import numpy as np
from copy import deepcopy
from typing import List, Dict
from torch.utils.data.dataloader import default_collate

from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.simulation.unroll import ClosedLoopSimulator, SimulationOutput
from l5kit.cle.metrics import DisplacementErrorL2Metric

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.envs.base import BaseEnv, BatchedEnv, SimulationException
from l5kit.geometry import compute_agent_pose
import pdb


class EnvL5KitSimulation(BaseEnv, BatchedEnv):
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
            show_info=True,
        )
        if "generate_agent_obs" not in env_config.keys():
            self.generate_agent_obs = True
        else:
            self.generate_agent_obs = env_config.generate_agent_obs
        self._npr = np.random.RandomState(seed=seed)
        self.dataset = dataset
        self._num_total_scenes = len(dataset.dataset.scenes)
        self._num_scenes = num_scenes

        self._current_scene_indices = None  # indices of the scenes (in dataset) that are being used for simulation
        self._current_scene_dataset = None  # corresponding dataset of the scenes

        self._frame_index = 0
        self._cached_observation = None
        self._done = False

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
        assert (
            np.max(scene_indices) < self._num_total_scenes
            and np.min(scene_indices) >= 0
        )

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
        if self.generate_agent_obs:
            agent_obs = self._current_scene_dataset.rasterise_agents_frame_batch(
                self._frame_index
            )
            if len(agent_obs) > 0:
                agent_obs = default_collate(list(agent_obs.values()))
        else:
            agent_obs = None
        ego_obs = self._current_scene_dataset.rasterise_frame_batch(self._frame_index)

        ego_obs = default_collate(ego_obs)
        self._cached_observation = TensorUtils.to_numpy(
            {"agents": agent_obs, "ego": ego_obs}
        )
        return self._cached_observation

    def is_done(self):
        return self._done

    def get_reward(self):
        # TODO
        return np.zeros(self._num_scenes)

    def render(self):
        raise NotImplementedError

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

    def step(self, actions):
        """
        Step the simulation with control inputs

        Args:
            actions (Dict): a dictionary containing either or both "ego_control" or "agents_control"
                - ego_control (Dict): a dictionary containing future ego position and orientation for all scenes
                - agents_control (Dict): a dictionary containing future agent positions and orientations for all scenes
        """
        for k in actions:
            assert k in ["ego", "agents"]

        if self._done:
            raise SimulationException("Simulation episode has ended")

        actions = TensorUtils.to_numpy(actions)
        obs = self.get_observation()

        ego_control = actions.get("ego", None)
        agents_control = actions.get("agents", None)

        should_update = self._frame_index + 1 < self.horizon
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

            for scene_idx in self._current_scene_indices:
                self._ego_states[scene_idx].append(ego_in_out[scene_idx])

            # if "all_other_agents_track_id" in ego_control:
            #     if should_update:
            #         # batch update all surrounding agents if the model is scene-centric
            #         for scene_idx in self._current_scene_indices:

            #             in_dict = ego_in_out[scene_idx].inputs
            #             out_dict = ego_in_out[scene_idx].outputs
            #             agents_track_id = out_dict["all_other_agents_track_id"]
            #             idx = np.where(agents_track_id != 0)[0]
            #             agent_centroid_m = (
            #                 in_dict["centroid"]
            #                 + out_dict["all_other_positions"][idx, 0]
            #             )
            #             agent_yaw_rad = (
            #                 in_dict["yaw"] + out_dict["all_other_yaws"][idx, 0, 0]
            #             )
            #             world_from_agent = np.zeros([idx.shape[0], 3, 3])
            #             for j in range(agent_yaw_rad.shape[0]):
            #                 world_from_agent[j] = compute_agent_pose(
            #                     agent_centroid_m[j], agent_yaw_rad[j]
            #                 )
            #             extents = in_dict["all_other_agents_history_extents"][idx, 0]
            #             extents = np.hstack(
            #                 (extents, 1.8 * np.ones((extents.shape[0], 1)))
            #             )
            #             agent_obs = {
            #                 "world_from_agent": world_from_agent,
            #                 "yaw": np.tile(in_dict["yaw"], idx.shape[0]),
            #                 "track_id": agents_track_id[idx],
            #                 "extent": extents,
            #                 "scene_index": np.tile(scene_idx, idx.shape[0]),
            #             }
            #             agent_control = {
            #                 "positions": out_dict["all_other_positions"][idx],
            #                 "yaws": out_dict["all_other_yaws"][idx],
            #             }

            #             ClosedLoopSimulator.update_agents(
            #                 dataset=self._current_scene_dataset,
            #                 frame_idx=self._frame_index + 1,
            #                 input_dict=agent_obs,
            #                 output_dict=agent_control,
            #             )

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
            for scene_idx in self._current_scene_indices:
                self._agents_states[scene_idx].append(agents_in_out.get(scene_idx, []))

        # TODO: accumulate sim trajectories
        self._cached_observation = None

        if self._frame_index + 1 == self.horizon:
            self._done = True
        else:
            self._frame_index += 1
