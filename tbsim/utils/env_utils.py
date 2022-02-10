import numpy as np
import pytorch_lightning as pl
from tbsim.envs.base import BatchedEnv, BaseEnv
import tbsim.utils.tensor_utils as TensorUtils
import pdb
import torch


def rollout_episodes(env, policy, num_episodes, skip_first_n=0):
    """
    Rollout an environment for a number of episodes
    Args:
        env (BaseEnv): a base simulation environment (gym-like)
        policy: a policy that
        num_episodes (int): number of episodes to rollout for

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
        counter = 0
        while not done:

            obs = env.get_observation()
            obs = TensorUtils.to_torch(obs, device=policy.device)

            if counter < skip_first_n:
                fake_action = {
                    "ego": {
                        "positions": obs["ego"]["target_positions"],
                        "yaws": obs["ego"]["target_yaws"],
                    }
                }

                env.step(fake_action)
            else:
                action = policy.get_action(obs)
                env.step(action)
            done = env.is_done()
            counter += 1

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


def replay_episodes(env, policy, num_episodes):
    """
    replay an environment for a number of episodes
    Args:
        env (BaseEnv): a base simulation environment (gym-like)
        policy: a policy that
        num_episodes (int): number of episodes to rollout for

    Returns:
        stats (dict): A dictionary of rollout stats for each episode (metrics, rewards, etc.)
        info (dict): A dictionary of environment info for each episode
    """
    stats = {}
    info = {}
    is_batched_env = isinstance(env, BatchedEnv)
    pred_seq = [list() for i in range(num_episodes * env.num_instances)]
    for ei in range(num_episodes):
        env.reset()
        counter = 0
        done = env.is_done()
        while not done:

            obs = env.get_observation()
            obs = TensorUtils.to_torch(obs, device=policy.device)
            action = policy.get_action(obs)
            yaw = action["ego"]["yaws"]
            yaw += obs["ego"]["yaw"].view(-1, 1, 1)
            pos = action["ego"]["positions"]
            s = torch.sin(yaw).unsqueeze(-1)
            c = torch.cos(yaw).unsqueeze(-1)
            centroid = obs["ego"]["centroid"]
            rotM = torch.cat(
                (torch.cat((c, -s), dim=-1), torch.cat((s, c), dim=-1)), dim=-2
            )
            world_xy = ((pos.unsqueeze(-2)) @ (rotM.transpose(-1, -2))).squeeze(-2)
            world_xy += centroid.view(-1, 1, 2).type(torch.float)

            counter += 1
            for k in range(env.num_instances):
                pred_seq[ei * env.num_instances + k].append(
                    TensorUtils.to_numpy(world_xy[k])
                )
            fake_action = {
                "ego": {
                    "positions": obs["ego"]["target_positions"],
                    "yaws": obs["ego"]["target_yaws"],
                }
            }
            env.step(fake_action)
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
        info["predictions"] = pred_seq
        for k, v in env_info.items():
            if k not in info:
                info[k] = []
            if is_batched_env:
                info[k].extend(v)
            else:
                info[k].append(v)

    return stats, info


class RolloutCallback(pl.Callback):
    def __init__(
        self,
        env,
        num_episodes=1,
        every_n_steps=100,
        warm_start_n_steps=1,
        verbose=False,
    ):
        self._env = env
        self._num_episodes = num_episodes
        self._every_n_steps = every_n_steps
        self._warm_start_n_steps = warm_start_n_steps
        self._verbose = verbose

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
                self._env, pl_module, num_episodes=self._num_episodes, skip_first_n=1
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
