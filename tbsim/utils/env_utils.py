from typing import OrderedDict
import numpy as np
import pytorch_lightning as pl
import torch

from tbsim.envs.base import BatchedEnv, BaseEnv
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.timer import Timers
from tbsim.policies.common import RolloutAction
from tbsim.policies.wrappers import RolloutWrapper
from l5kit.simulation.unroll import ClosedLoopSimulator
import tbsim.utils.geometry_utils as GeoUtils


def set_initial_states(env, obs, adjustment_plan ,device):
    obs = TensorUtils.to_torch(obs, device)
    bs, T = obs["ego"]["target_positions"].shape[:2]
    
    ego_global = GeoUtils.batch_nd_transform_points(
        obs["ego"]["history_positions"][:, 0],
        obs["ego"]["world_from_agent"]
    )
    # for obj in ["ego","agents"]:
    #     for key in ["scene_index","track_id"]:
    #         obs[obj][key] = obs[obj][key].int()
    ego_yaw = obs["ego"]["history_yaws"][:, 0].flatten()+obs["ego"]["yaw"]
    offset_x = 8.0
    offset_y = 4.0
    agent_indices = []
    positions = []
    yaws = []
    for i in range(bs):
        scene_idx = obs["ego"]["scene_index"][i].item()
        if scene_idx in adjustment_plan:
            for agent_id in adjustment_plan[scene_idx]:
                agent_idx = torch.where(
                    (obs["agents"]["scene_index"] == scene_idx) &
                    (obs["agents"]["track_id"] == agent_id)
                )[0]
                if agent_idx.nelement()==1:
                    agent_idx = agent_idx.item()
                else:
                    continue
                agent_indices.append(agent_idx)
                
                if adjustment_plan[scene_idx][agent_id] == 0:
                    # in front of the ego vehicle
                    agent_pos = ego_global[i]+torch.tensor(
                        [offset_x*torch.cos(ego_yaw[i]), offset_x*torch.sin(ego_yaw[i])]).to(device)
                elif adjustment_plan[scene_idx][agent_id] == 1:
                    # behind of the ego vehicle
                    agent_pos = ego_global[i]+torch.tensor(
                        [-offset_x*torch.cos(ego_yaw[i]), -offset_x*torch.sin(ego_yaw[i])]).to(device)
                elif adjustment_plan[scene_idx][agent_id] == 2:
                    # left of the ego vehicle
                    agent_pos = ego_global[i]+torch.tensor(
                        [-offset_y*torch.sin(ego_yaw[i]), offset_y*torch.cos(ego_yaw[i])]).to(device)
                elif adjustment_plan[scene_idx][agent_id] == 3:
                    # right of the ego vehicle
                    agent_pos = ego_global[i]+torch.tensor(
                        [offset_y*torch.sin(ego_yaw[i]), -offset_y*torch.cos(ego_yaw[i])]).to(device)
                elif adjustment_plan[scene_idx][agent_id] == 4:
                    # two vehicle length ahead of the ego vehicle
                    agent_pos = ego_global[i]+torch.tensor(
                        [2*offset_x*torch.cos(ego_yaw[i]), 2*offset_x*torch.sin(ego_yaw[i])]).to(device)
                agent_pos = GeoUtils.batch_nd_transform_points(
                    agent_pos, obs["agents"]["agent_from_world"][agent_idx])
                agent_pos = agent_pos.tile(T, 1)
                agent_yaw = (ego_yaw[i]-obs["agents"]
                             ["yaw"][agent_idx]).tile(T, 1)
                positions.append(agent_pos)
                yaws.append(agent_yaw)
    agent_obs = dict()
    for k,v in obs["agents"].items():
        agent_obs[k]=v[agent_indices]

    
    agent_action = OrderedDict(positions=torch.stack(positions,0),yaws=torch.stack(yaws,0))
    agent_obs = TensorUtils.to_numpy(agent_obs)
    agent_action = TensorUtils.to_numpy(agent_action)
    ClosedLoopSimulator.update_agents(
        dataset=env._current_scene_dataset,
        frame_idx=env._frame_index + 1,
        input_dict=agent_obs,
        output_dict=agent_action,
    )


def rollout_episodes(
    env,
    policy,
    num_episodes,
    skip_first_n=1,
    n_step_action=1,
    render=False,
    scene_indices=None,
    device=None,
    obs_to_torch=True,
    adjustment_plan=None,
    horizon = None,
):
    """
    Rollout an environment for a number of episodes
    Args:
        env (BaseEnv): a base simulation environment (gym-like)
        policy (RolloutWrapper): a policy that controls agents in the environment
        num_episodes (int): number of episodes to rollout for
        skip_first_n (int): number of steps to skip at the begining
        n_step_action (int): number of steps to take between querying models
        render (bool): if True, return a sequence of rendered frames
        scene_indices (tuple, list): (Optional) scenes indices to rollout with
        adjustment_plan (dict): (Optional) initialization condition
        horizon (int): (Optional) override horizon of the simulation

    Returns:
        stats (dict): A dictionary of rollout stats for each episode (metrics, rewards, etc.)
        info (dict): A dictionary of environment info for each episode
        renderings (list): A list of rendered frames in the form of np.ndarray, one for each episode
    """
    stats = {}
    info = {}
    renderings = []
    is_batched_env = isinstance(env, BatchedEnv)
    timers = Timers()

    for ei in range(num_episodes):
        env.reset(scene_indices=scene_indices)

        done = env.is_done()
        counter = 0
        frames = []
        while not done:
            timers.tic("step")
            with timers.timed("obs"):
                obs = env.get_observation()
                # import pdb
                # pdb.set_trace()
            with timers.timed("to_torch"):
                if obs_to_torch:
                    device = policy.device if device is None else device
                    obs_torch = TensorUtils.to_torch(obs, device=device, ignore_if_unspecified=True)
                else:
                    obs_torch = obs

            if counter < skip_first_n:
                # skip the first N steps to warm up environment state (velocity, etc.)
                # DIFF
                env.step(RolloutAction(), num_steps_to_take=1, render=False)
                if adjustment_plan is not None:
                    set_initial_states(env, obs, adjustment_plan ,device)
                # env.step(env.get_gt_action(obs), num_steps_to_take=1, render=False)
                counter += 1
            else:
                with timers.timed("network"):
                    action = policy.get_action(obs_torch, step_index=counter)
                with timers.timed("env_step"):
                    ims = env.step(
                        action, num_steps_to_take=n_step_action, render=render
                    )  # List of [num_scene, h, w, 3]
                if render:
                    frames.extend(ims)
                counter += n_step_action
            timers.toc("step")
            print(timers)

            done = env.is_done()
            
            if horizon is not None and counter>=horizon:
                break
        metrics = env.get_metrics()

        for k, v in metrics.items():
            if k not in stats:
                stats[k] = []
            if is_batched_env:
                stats[k] = np.concatenate([stats[k], v], axis=0)
            else:
                stats[k].append(v)

        env_info = env.get_info()
        for k, v in env_info.items():
            if k not in info:
                info[k] = []
            if is_batched_env:
                info[k].extend(v)
            else:
                info[k].append(v)

        if render:
            frames = np.stack(frames)
            if is_batched_env:
                # [step, scene] -> [scene, step]
                frames = frames.transpose((1, 0, 2, 3, 4))
            renderings.append(frames)

    return stats, info, renderings


class RolloutCallback(pl.Callback):
    def __init__(
            self,
            env,
            num_episodes=1,
            n_step_action=1,
            every_n_steps=100,
            warm_start_n_steps=1,
            verbose=False,
    ):
        self._env = env
        self._num_episodes = num_episodes
        self._every_n_steps = every_n_steps
        self._warm_start_n_steps = warm_start_n_steps
        self._verbose = verbose
        self._n_step_action = n_step_action

    def print_if_verbose(self, msg):
        if self._verbose:
            print(msg)

    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        should_run = (
            trainer.global_step >= self._warm_start_n_steps
            and trainer.global_step % self._every_n_steps == 0
        )
        if should_run:
            try:
                stats, _, _ = rollout_episodes(
                    env=self._env,
                    policy=RolloutWrapper(ego_policy=pl_module),
                    num_episodes=self._num_episodes,
                    n_step_action=self._n_step_action,
                )
                self.print_if_verbose(
                    "\nStep %i rollout (%i episodes): "
                    % (trainer.global_step, self._num_episodes)
                )
                for k, v in stats.items():
                    # Set on_step=True and on_epoch=False to force the logger to log stats at the step
                    # See https://github.com/PyTorchLightning/pytorch-lightning/issues/9772 for explanation
                    pl_module.log(
                        "rollout/metrics_" + k, np.mean(v), on_step=True, on_epoch=False
                    )
                    self.print_if_verbose(("rollout/metrics_" + k, np.mean(v)))
                self.print_if_verbose("\n")
            except Exception as e:
                print("Rollout failed because:")
                print(e)
