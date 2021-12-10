import numpy as np
import pytorch_lightning as pl
from tbsim.envs.base import BatchedEnv, BaseEnv
import tbsim.utils.tensor_utils as TensorUtils
import pdb


def rollout_episodes(env, policy, num_episodes):
    """
    Rollout an environment for a number of episodes
    Args:
        env (BaseEnv): a base simulation environment (gym-like)
        policy: a policy that
        num_episodes:

    Returns:

    """
    stats = {}
    is_batched_env = isinstance(env, BatchedEnv)

    for ei in range(num_episodes):
        env.reset()

        done = env.is_done()
        while not done:
            obs = env.get_observation()
            obs = TensorUtils.to_torch(obs, device=policy.device)

            action = policy.get_action(obs)
            env.step(action)
            done = env.is_done()

        metrics = env.get_metrics()
        for k, v in metrics.items():
            if k not in stats:
                stats[k] = []
            if is_batched_env:
                stats[k] = np.concatenate([stats[k], v], axis=0)
            else:
                stats[k].append(v)

    return stats


class RolloutCallback(pl.Callback):
    def __init__(self, env, num_episodes=1, every_n_steps=100, warm_start_n_steps=1):
        self._env = env
        self._num_episodes = num_episodes
        self._every_n_steps = every_n_steps
        self._warm_start_n_steps = warm_start_n_steps

    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        should_run = (
            trainer.global_step >= self._warm_start_n_steps
            and trainer.global_step % self._every_n_steps == 0
        )
        if should_run:
            stats = rollout_episodes(
                self._env, pl_module, num_episodes=self._num_episodes
            )
            print("Step %i rollout: " % trainer.global_step)
            for k, v in stats.items():
                pl_module.log("rollout/metrics_" + k, np.mean(v))
                print("rollout/metrics_" + k, np.mean(v))
            print("\n")
