import numpy as np
from copy import deepcopy

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.policies.common import RolloutAction


class RolloutLogger(object):
    """Log trajectories and other essential info during rollout for visualization and evaluation"""
    def __init__(self, obs_keys=None):
        if obs_keys is None:
            obs_keys = dict()
        self._obs_keys = obs_keys
        self._scene_buffer = dict()
        self._serialized_scene_buffer = None
        self._scene_indices = None
        self._agent_id_per_scene = dict()

    def _combine_obs(self, obs):
        combined = dict()
        if "ego" in obs and obs["ego"] is not None:
            combined.update(obs["ego"])
        if "agents" in obs and obs["agents"] is not None:
            for k in obs["agents"].keys():
                if k in combined:
                    combined[k] = np.concatenate((combined[k], obs["agents"][k]), axis=0)
                else:
                    combined[k] = obs["agents"][k]
        return combined

    def _combine_action(self, action: RolloutAction):
        combined = dict(action=dict())
        if action.has_ego:
            combined["action"] = action.ego.to_dict()
            if action.ego_info is not None and "action_samples" in action.ego_info:
                combined["action_samples"] = action.ego_info["action_samples"]
        if action.has_agents:
            agents_action = action.agents.to_dict()
            for k in agents_action:
                if k in combined["action"]:
                    combined["action"][k] = np.concatenate((combined["action"][k], agents_action[k]), axis=0)
                else:
                    combined["action"][k] = agents_action[k]
            if action.agents_info is not None and "action_samples" in action.agents_info:
                if "action_samples" in combined:
                    samples = action.agents_info["action_samples"]
                    for k in samples:
                        if k in combined["action_samples"]:
                            combined["action_samples"][k] = np.concatenate((combined["action_samples"][k], samples[k]), axis=0)
                        else:
                            combined["action_samples"][k] = samples[k]

        return combined

    def _maybe_initialize(self, obs):
        if self._scene_indices is None:
            self._scene_indices = np.unique(obs["scene_index"])
            for si in self._scene_indices:
                self._agent_id_per_scene[si] = obs["track_id"][obs["scene_index"] == si]
            for si in self._scene_indices:
                self._scene_buffer[si] = dict()

    def _append_buffer(self, obs, action):
        """
        scene_index:
            dict(
                action_positions=[[num_agent, ...], [num_agent, ...], ],
                action_yaws=[[num_agent, ...], [num_agent, ...], ],
                centroid=[[num_agent, ...], [num_agent, ...], ],
                ...
            )
        """
        self._serialized_scene_buffer = None  # need to re-serialize

        # TODO: move this to __init__ as arg
        state = {k: obs[k] for k in self._obs_keys}
        state["action_positions"] = action["action"]["positions"][:, [0]]
        state["action_yaws"] = action["action"]["yaws"][:, [0]]
        if "action_samples" in action:
            # only collect up to 10 samples to save space
            state["action_sample_positions"] = action["action_samples"]["positions"][:, :10]
            state["action_sample_yaws"] = action["action_samples"]["yaws"][:, :10]

        for si in self._scene_indices:

            scene_mask = np.where(si == obs["scene_index"])[0]
            scene_state = TensorUtils.map_ndarray(state, lambda x: x[scene_mask])

            reassignment_index = np.zeros(len(scene_mask), dtype=np.int64)
            scene_track_id = scene_state["track_id"]
            for i, ti in enumerate(scene_track_id):
                reassignment_index[i] = np.where(ti == self._agent_id_per_scene[si])[0]
            for k in scene_state:
                if k not in self._scene_buffer[si]:
                    # don't need to do anything special for the initial step
                    assert scene_state[k].shape[0] == self._agent_id_per_scene[si].shape[0]
                    self._scene_buffer[si][k] = []
                    self._scene_buffer[si][k].append(scene_state[k])
                else:
                    # otherwise, use the previous state as a template and only assign values for agents
                    # in the current state (i.e., use the previous value for missing agents)
                    prev_state = np.copy(self._scene_buffer[si][k][-1])
                    prev_state[reassignment_index] = scene_state[k]
                    self._scene_buffer[si][k].append(prev_state)

    def get_serialized_scene_buffer(self):
        """
        scene_index:
            dict(
                action_positions=[num_agent, T, ...],
                action_yaws=[num_agent, T, ...],
                centroid=[num_agent, T, ...],
                ...
            )
        """

        if self._serialized_scene_buffer is not None:
            return self._serialized_scene_buffer

        buffer_len = None
        serialized = dict()
        for si in self._scene_buffer:
            serialized[si] = dict()
            for k in self._scene_buffer[si]:
                bf = [e[:, None] for e in self._scene_buffer[si][k]]
                serialized[si][k] = np.concatenate(bf, axis=1)
                if buffer_len is None:
                    buffer_len = serialized[si][k].shape[1]
                assert serialized[si][k].shape[1] == buffer_len

        self._serialized_scene_buffer = serialized
        return deepcopy(self._serialized_scene_buffer)

    def get_trajectory(self):
        """Get per-scene rollout trajectory in the world coordinate system"""
        buffer = self.get_serialized_scene_buffer()
        traj = dict()
        for si in buffer:
            traj[si] = dict(
                positions=buffer[si]["centroid"],
                yaws=buffer[si]["yaw"][..., None]
            )
        return traj

    def get_track_id(self):
        return deepcopy(self._agent_id_per_scene)

    def get_stats(self):
        # TODO
        raise NotImplementedError()

    def log_step(self, obs, action: RolloutAction):
        combined_obs = self._combine_obs(obs)
        combined_action = self._combine_action(action)
        assert combined_obs["scene_index"].shape[0] == combined_action["action"]["positions"].shape[0]
        self._maybe_initialize(combined_obs)
        self._append_buffer(combined_obs, combined_action)
