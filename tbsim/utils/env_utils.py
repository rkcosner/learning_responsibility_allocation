import numpy as np
import pytorch_lightning as pl
import torch
from typing import Tuple, Dict

from tbsim.envs.base import BatchedEnv, BaseEnv
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.dynamics as dynamics
from tbsim.utils.l5_utils import get_current_states
from tbsim.algos.algo_utils import optimize_trajectories
from tbsim.utils.geometry_utils import transform_points_tensor
from l5kit.geometry import transform_points


def rollout_episodes(env, policy, num_episodes, n_step_action=1, render=False, scene_indices=None):
    """
    Rollout an environment for a number of episodes
    Args:
        env (BaseEnv): a base simulation environment (gym-like)
        policy (RolloutWrapper): a policy that controls agents in the environment
        num_episodes (int): number of episodes to rollout for
        n_step_action (int): number of steps to take between querying models
        render (bool): if True, return a sequence of rendered frames
        scene_indices (tuple, list): (Optional) scenes indices to rollout with

    Returns:
        stats (dict): A dictionary of rollout stats for each episode (metrics, rewards, etc.)
        info (dict): A dictionary of environment info for each episode
        renderings (list): A list of rendered frames in the form of np.ndarray, one for each episode
    """
    stats = {}
    info = {}
    renderings = []
    is_batched_env = isinstance(env, BatchedEnv)

    for ei in range(num_episodes):
        env.reset(scene_indices=scene_indices)

        done = env.is_done()
        frames = []
        while not done:
            obs = env.get_observation()
            obs = TensorUtils.to_torch(obs, device=policy.device)

            action = policy.get_action(obs)
            ims = env.step(action, num_steps_to_take=n_step_action, render=render)  # List of [num_scene, h, w, 3]
            if render:
                frames.extend(ims)
            done = env.is_done()

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
                frames = frames.transpose((1, 0, 2, 3, 4))  # [step, scene] -> [scene, step]
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
            stats, _, _ = rollout_episodes(
                env=self._env,
                policy=RolloutWrapper(ego_policy=pl_module),
                num_episodes=self._num_episodes,
                n_step_action=self._n_step_action
            )
            self.print_if_verbose("\nStep %i rollout (%i episodes): " % (trainer.global_step, self._num_episodes))
            for k, v in stats.items():
                # Set on_step=True and on_epoch=False to force the logger to log stats at the step
                # See https://github.com/PyTorchLightning/pytorch-lightning/issues/9772 for explanation
                pl_module.log("rollout/metrics_" + k, np.mean(v), on_step=True, on_epoch=False)
                self.print_if_verbose(("rollout/metrics_" + k, np.mean(v)))
            self.print_if_verbose("\n")


class Action(object):
    def __init__(self, positions, yaws):
        assert positions.ndim == 3  # [B, T, 2]
        assert positions.shape[:-1] == yaws.shape[:-1]
        assert positions.shape[-1] == 2
        assert yaws.shape[-1] == 1
        self._positions = positions
        self._yaws = yaws

    @property
    def positions(self):
        return TensorUtils.clone(self._positions)

    @property
    def yaws(self):
        return TensorUtils.clone(self._yaws)

    def to_dict(self):
        return dict(
            positions=self.positions,
            yaws=self.yaws
        )

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def transform(self, trans_mats, rot_rads):
        if isinstance(self.positions, np.ndarray):
            pos = transform_points(self.positions, trans_mats)
        else:
            pos = transform_points_tensor(self.positions, trans_mats)

        yaw = self.yaws + rot_rads
        return self.__class__(pos, yaw)

    def to_numpy(self):
        return self.__class__(**TensorUtils.to_numpy(self.to_dict()))


class Plan(object):
    def __init__(self, positions, yaws, availabilities, controls=None):
        assert positions.ndim == 3  # [B, T, 2]
        b, t, s = positions.shape
        assert s == 2
        assert yaws.shape == (b, t, 1)
        assert availabilities.shape ==  (b, t)
        self._positions = positions
        self._yaws = yaws
        self._availabilities = availabilities
        self._controls = controls

    @property
    def positions(self):
        return TensorUtils.clone(self._positions)

    @property
    def yaws(self):
        return TensorUtils.clone(self._yaws)

    @property
    def availabilities(self):
        return TensorUtils.clone(self._availabilities)

    @property
    def controls(self):
        return TensorUtils.clone(self._controls)

    def transform(self, trans_mats, rot_rads):
        if isinstance(self.positions, np.ndarray):
            pos = transform_points(self.positions, trans_mats)
        else:
            pos = transform_points_tensor(self.positions, trans_mats)

        yaw = self.yaws + rot_rads
        return self.__class__(pos, yaw, self.availabilities, self.controls)

    def to_dict(self):
        p = dict(
            positions=self.positions,
            yaws=self.yaws,
            availabilities=self.availabilities,
        )
        if self._controls is not None:
            p["controls"] = self.controls
        return p

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def to_numpy(self):
        return self.__class__(**TensorUtils.to_numpy(self.to_dict()))


class RolloutAction(object):
    def __init__(self, ego=None, ego_info=None, agents=None, agents_info=None):
        assert ego is None or isinstance(ego, Action)
        assert agents is None or isinstance(agents, Action)
        assert ego_info is None or isinstance(ego_info, dict)
        assert agents_info is None or isinstance(agents_info, dict)

        self.ego = ego
        self.ego_info = ego_info
        self.agents = agents
        self.agents_info = agents_info

    @property
    def has_ego(self):
        return self.ego is not None

    @property
    def has_agents(self):
        return self.agents is not None

    def to_dict(self):
        d = dict()
        if self.has_ego:
            d["ego"] = self.ego.to_dict()
            d["ego_info"] = self.ego_info
        if self.has_agents:
            d["agents"] = self.agents.to_dict()
            d["agents_info"] = self.agents_info
        return d

    def to_numpy(self):
        return self.__class__(
            ego=self.ego.to_numpy() if self.has_ego else None,
            ego_info=TensorUtils.to_numpy(self.ego_info) if self.has_ego else None,
            agents=self.agents.to_numpy() if self.has_agents else None,
            agents_info=TensorUtils.to_numpy(self.agents_info) if self.has_agents else None,
        )


class OptimController(object):
    """An optimization-based controller"""
    def __init__(
            self,
            dynamics_type,
            dynamics_kwargs,
            step_time: float,
            optimizer_kwargs = None,
    ):
        self.step_time = step_time
        self.optimizer_kwargs = dict() if optimizer_kwargs is None else optimizer_kwargs
        if dynamics_type in ["Unicycle", dynamics.DynType.UNICYCLE]:
            self.dyn = dynamics.Unicycle(
                "dynamics",
                max_steer=dynamics_kwargs["max_steer"],
                max_yawvel=dynamics_kwargs["max_yawvel"],
                acce_bound=dynamics_kwargs["acce_bound"]
            )
        elif dynamics_type in ["Bicycle", dynamics.DynType.BICYCLE]:
            self.dyn = dynamics.Bicycle(
                acc_bound=dynamics_kwargs["acce_bound"],
                ddh_bound=dynamics_kwargs["ddh_bound"],
                max_hdot=dynamics_kwargs["max_yawvel"],
                max_speed=dynamics_kwargs["max_speed"]
            )
        else:
            raise NotImplementedError("dynamics type {} is not implemented", dynamics_type)

    def get_action(self, obs, plan: Plan, init_u=None, **kwargs) -> Tuple[Action, Dict]:
        target_pos = plan.positions
        target_yaw = plan.yaws
        target_avails = plan.availabilities
        device = target_pos.device
        num_action_steps = target_pos.shape[-2]
        init_x = get_current_states(obs, dyn_type=self.dyn.type())
        if init_u is None:
            init_u = torch.randn(*init_x.shape[:-1], num_action_steps, self.dyn.udim).to(device)
        if target_avails is None:
            target_avails = torch.ones(target_pos.shape[:-1]).to(device)
        targets = torch.cat((target_pos, target_yaw), dim=-1)
        assert init_u.shape[-2] == num_action_steps
        predictions, raw_traj, final_u, losses = optimize_trajectories(
            init_u=init_u,
            init_x=init_x,
            target_trajs=targets,
            target_avails=target_avails,
            dynamics_model=self.dyn,
            step_time=self.step_time,
            data_batch=obs,
            **self.optimizer_kwargs
        )
        action = Action(**predictions)
        return action, {}


class GTPlanner(object):
    """A (fake) planner tha sets ground truth trajectory as (sub)goal"""
    def __init__(self, device):
        self.device = device

    @staticmethod
    def get_plan(obs, **kwargs) -> Tuple[Plan, Dict]:
        plan = Plan(
            positions=obs["target_positions"],
            yaws=obs["target_yaws"],
            availabilities=obs["target_availabilities"],
        )
        return plan, {}


class HierarchicalWrapper(object):
    """A wrapper policy that feeds subgoal from a planner to a controller"""
    def __init__(self, planner, controller):
        self.device = planner.device
        self.planner = planner
        self.controller = controller

    def get_action(self, obs) -> Tuple[Action, Dict]:
        plan, plan_info = self.planner.get_plan(obs)
        actions, action_info = self.controller.get_action(
            obs,
            plan=plan,
            init_u=plan.controls
        )
        action_info["plan"] = plan.to_dict()
        action_info["plan_info"] = plan_info
        return actions, action_info


class PolicyWrapper(object):
    """A convenient wrapper for specifying run-time keyword arguments"""
    def __init__(self, model, get_action_kwargs=None, get_plan_kwargs=None):
        self.model = model
        self.device = model.device
        self.action_kwargs = get_action_kwargs
        self.plan_kwargs = get_plan_kwargs

    def get_action(self, obs) -> Tuple[Action, Dict]:
        return self.model.get_action(obs, **self.action_kwargs)

    def get_plan(self, obs) -> Tuple[Plan, Dict]:
        return self.model.get_plan(obs, **self.plan_kwargs)

    @classmethod
    def wrap_controller(cls, model, **kwargs):
        return cls(model=model, get_action_kwargs=kwargs)

    @classmethod
    def wrap_planner(cls, model, **kwargs):
        return cls(model=model, get_plan_kwargs=kwargs)


class RolloutWrapper(object):
    """A wrapper policy that can (optionally) control both ego and other agents in a scene"""
    def __init__(self, ego_policy=None, agents_policy=None):
        self.device = ego_policy.device if agents_policy is None else agents_policy.device
        self.ego_policy = ego_policy
        self.agents_policy = agents_policy

    def get_action(self, obs) -> RolloutAction:
        ego_action = None
        ego_action_info = None
        agents_action = None
        agents_action_info = None
        if self.ego_policy is not None:
            ego_action, ego_action_info = self.ego_policy.get_action(obs["ego"])
        if self.agents_policy is not None:
            agents_action, agents_action_info = self.agents_policy.get_action(obs["agents"])
        return RolloutAction(ego_action, ego_action_info, agents_action, agents_action_info)
