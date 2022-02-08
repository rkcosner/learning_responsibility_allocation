import numpy as np
import pytorch_lightning as pl
import torch

from tbsim.envs.base import BatchedEnv, BaseEnv
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.dynamics as dynamics
from tbsim.models.l5kit_models import optimize_trajectories, get_current_states


def rollout_episodes(env, policy, num_episodes, n_step_action=1):
    """
    Rollout an environment for a number of episodes
    Args:
        env (BaseEnv): a base simulation environment (gym-like)
        policy: a policy that
        num_episodes (int): number of episodes to rollout for
        n_step_action (int): number of steps to take between querying models

    Returns:
        stats (dict): A dictionary of rollout stats for each episode (metrics, rewards, etc.)
        info (dict): A dictionary of environment info for each episode
    """
    stats = {}
    info = {}
    is_batched_env = isinstance(env, BatchedEnv)

    for ei in range(num_episodes):
        env.reset()

        done = env.is_done()
        while not done:
            obs = env.get_observation()
            obs = TensorUtils.to_torch(obs, device=policy.device)

            action = policy.get_action(obs, sample=False)
            env.step(action, num_steps_to_take=n_step_action)
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

    return stats, info


class RolloutCallback(pl.Callback):
    def __init__(self, env, num_episodes=1, n_step_action=1, every_n_steps=100, warm_start_n_steps=1, verbose=False):
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
            stats, _ = rollout_episodes(
                env=self._env,
                policy=pl_module,
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



class RolloutWrapper(object):
    def __init__(self, model: pl.LightningModule, num_prediction_steps, action_frequency=1):
        assert action_frequency <= num_prediction_steps
        self.model = model
        self.device = model.device
        self.action_freq = action_frequency
        self._action_length =  num_prediction_steps
        self._cached_action = None
        self._iter_i = 0

    def reset(self):
        self._iter_i = 0
        self._cached_action = None

    def get_action(self, obs, **kwargs):
        if self._iter_i % self.action_freq == 0:
            self._iter_i = 0
            self._cached_action = self.model.get_action(obs, **kwargs)
            assert "agents" not in self._cached_action  # TODO: support agents actions

        def get_step_action(action_tensor):
            assert action_tensor.shape[-2] == self._action_length
            return action_tensor[..., self._iter_i:, :]

        step_action = TensorUtils.map_tensor(self._cached_action, get_step_action)
        self._iter_i += 1
        return step_action

    @classmethod
    def wrap(cls, model, action_frequency):
        return cls(
            model=model,
            num_prediction_steps=model.algo_config.future_num_frames,
            action_frequency=action_frequency
        )


class OptimController(object):
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

    def get_action(self, obs, plan, init_u=None, **kwargs):
        obs = obs["ego"]
        target_pos = plan["predictions"]["positions"]
        target_yaw = plan["predictions"]["yaws"]
        target_avails = plan["availabilities"]
        device = target_pos.device
        num_action_steps = target_pos.shape[-2]
        init_x = get_current_states(obs, dyn_type=self.dyn.type())
        if init_u is None:
            init_u = torch.randn(*init_x.shape[:-1], num_action_steps, self.dyn.udim).to(device)
        if target_avails is None:
            target_avails = torch.ones(target_pos.shape[:-1]).to(device)
        targets = torch.cat((target_pos, target_yaw), dim=-1)
        assert init_u.shape[-2] == num_action_steps
        traj, raw_traj, final_u, losses = optimize_trajectories(
            init_u=init_u,
            init_x=init_x,
            target_trajs=targets,
            target_avails=target_avails,
            dynamics_model=self.dyn,
            step_time=self.step_time,
            data_batch=obs,
            **self.optimizer_kwargs
        )
        return dict(ego=traj)


class GTPlanner(object):
    def __init__(self, device):
        self.device = device

    @staticmethod
    def get_plan(obs, **kwargs):
        obs = obs["ego"]
        plan = dict(
            predictions=dict(
                positions=obs["target_positions"],
                yaws=obs["target_yaws"],
            ),
            availabilities=obs["target_availabilities"],
        )
        return dict(ego=plan)


class HierarchicalWrapper(object):
    def __init__(self, planner, controller=None):
        self.device = planner.device
        self.planner = planner
        self.controller = controller

    @staticmethod
    def verify_plan(plan):
        assert "predictions" in plan
        assert "availabilities" in plan
        pred = plan["predictions"]
        assert pred["positions"].ndim == 3  # [B, T, 2]
        b, t, s = pred["positions"].shape
        assert s == 2
        assert pred["yaws"].shape == (b, t, 1)
        assert plan["availabilities"].shape ==  (b, t)

    def get_action(self, obs, num_steps_to_goal=None, **kwargs):
        with torch.no_grad():
            all_plans = self.planner.get_plan(obs, sample=kwargs.get("sample", False))
        plan = all_plans["ego"]
        self.verify_plan(plan)
        plan_len = plan["predictions"]["positions"].shape[-2]
        if num_steps_to_goal is not None:
            assert num_steps_to_goal > 0
            if num_steps_to_goal > plan_len:
                n_steps_to_pad = num_steps_to_goal - plan_len
                plan = TensorUtils.pad_sequence(plan, padding=(n_steps_to_pad, 0), batched=True)
            else:
                plan = TensorUtils.map_tensor(plan, func=lambda x: x[:, :num_steps_to_goal])
        init_u = plan.get("controls", None)
        actions = self.controller.get_action(
            obs,
            plan=plan,
            init_u=init_u
        )
        output = actions
        output["ego_plan"] = plan
        if "ego_samples" in all_plans:
            output["ego_samples"] = all_plans["ego_samples"]
        return output
