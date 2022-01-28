import numpy as np
import pytorch_lightning as pl
from tbsim.envs.base import BatchedEnv, BaseEnv
import tbsim.utils.tensor_utils as TensorUtils


def rollout_episodes(env, policy, num_episodes):
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
        while not done:
            obs = env.get_observation()
            obs = TensorUtils.to_torch(obs, device=policy.device)

            action = policy.get_action(obs, sample=False)
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
    def __init__(self, env, num_episodes=1, every_n_steps=100, warm_start_n_steps=1, verbose=False):
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
            stats, _ = rollout_episodes(self._env, pl_module, num_episodes=self._num_episodes)
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
        self.device= model.device
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
