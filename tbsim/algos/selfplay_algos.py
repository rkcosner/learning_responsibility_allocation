import numpy as np

from pytorch_lightning import LightningModule

from torch.utils.data import DataLoader
import torch.nn as nn
from copy import deepcopy

from tbsim.external.l5_ego_dataset import ExperienceIterableWrapper
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.configs.base import ExperimentConfig
from tbsim.envs.env_l5kit import EnvL5KitSimulation
from tbsim.utils.env_utils import rollout_episodes, PolicyWrapper, RolloutWrapper, HierarchicalPolicy
from tbsim.algos.l5kit_algos import SpatialPlanner, L5TrafficModelGC


class SelfPlay(LightningModule):
    def __init__(self, cfg: ExperimentConfig, data_module):
        super().__init__()

        self.buffer_ds = data_module.experience_dataset
        self.env_cfg = deepcopy(cfg.env)
        with self.env_cfg.unlocked():
            self.env_cfg.generate_agent_obs = True
            self.env_cfg.simulation.num_simulation_steps = cfg.train.train_rollout.horizon

        self.train_env = EnvL5KitSimulation(
            env_config=self.env_cfg,
            dataset=data_module.train_dataset,
            num_scenes=cfg.train.train_rollout.num_scenes,
            seed=cfg.seed
        )

        self.eval_env = EnvL5KitSimulation(
            env_config=self.env_cfg,
            dataset=data_module.valid_dataset,
            num_scenes=cfg.train.test_rollout.num_scenes,
            seed=cfg.seed
        )

        self.cfg = cfg
        self.algo_cfg = cfg.algo

        self.algos = dict()
        self.train_algo: LightningModule = None
        self.policy: RolloutWrapper = None
        self.build_policy(data_module.modality_shapes)

    def build_policy(self, modality_shapes):
        raise NotImplementedError

    def forward(self, obs_dict):
        return self.policy.get_action(obs_dict)

    def on_train_batch_start(self, batch, batch_idx: int, unused = 0) -> None:
        if self.global_step % self.cfg.train.train_rollout.every_n_steps == 0 and self.global_step > 0:
            self.populate_buffer(self.cfg.train.train_rollout.num_episodes)

    def training_step(self, data_batch, batch_idx):
        self.train_algo.train()

        outputs = self.train_algo.training_step(data_batch, batch_idx)
        loss = outputs["loss"]
        for lk, l in outputs["all_losses"].items():
            self.log("train/losses_" + lk, l)

        for mk, m in outputs["all_metrics"].items():
            self.log("train/metrics_" + mk, m)
        return loss

    def configure_optimizers(self):
        return self.train_algo.configure_optimizers()

    def populate_buffer(self, num_episodes, log=True):
        self.policy.eval()

        print("Populating dataset with {}*{} episodes".format(num_episodes, self.cfg.train.train_rollout.num_scenes))
        stats, info, _ = rollout_episodes(
            env=self.train_env,
            policy=self.policy,
            num_episodes=num_episodes,
            skip_first_n=1,
            n_step_action=self.cfg.train.train_rollout.n_step_action,
            render=False,
            scene_indices=None,
            device=self.device
        )

        self.buffer_ds.append_experience(info["experience"])
        if log:
            for k, v in stats.items():
                # Set on_step=True and on_epoch=False to force the logger to log stats at the step
                # See https://github.com/PyTorchLightning/pytorch-lightning/issues/9772 for explanation
                self.log(
                    "train_rollout/metrics_" + k, np.mean(v), on_step=True, on_epoch=False
                )

    def test_step(self):
        self.policy.eval()

        stats, _, _ = rollout_episodes(
            env=self.train_env,
            policy=self.policy,
            num_episodes=self.cfg.train.test_rollout.num_episodes,
            skip_first_n=1,
            n_step_action=self.cfg.train.test_rollout.n_step_action,
            render=False,
            scene_indices=None,
            device=self.device
        )

        for k, v in stats.items():
            # Set on_step=True and on_epoch=False to force the logger to log stats at the step
            # See https://github.com/PyTorchLightning/pytorch-lightning/issues/9772 for explanation
            self.log(
                "rollout/metrics_" + k, np.mean(v), on_step=True, on_epoch=False
            )

    def _dataloader(self):
        for v in self.algos.values():
            v.to(self.device)
        self.populate_buffer(self.cfg.train.train_rollout.num_episodes_warm_start, log=False)
        dataset = ExperienceIterableWrapper(self.buffer_ds)
        return DataLoader(
            dataset=dataset,
            batch_size=self.cfg.train.training.batch_size,
            num_workers=self.cfg.train.training.num_data_workers
        )

    def train_dataloader(self):
        """Get train loader."""
        return self._dataloader()


class SelfPlayHierarchical(SelfPlay):
    def build_policy(self, modality_shapes):
        planner_cfg = get_experiment_config_from_file(self.algo_cfg.planner_config_path)
        # build policies from pretrained checkpoints
        planner = SpatialPlanner.load_from_checkpoint(
            self.algo_cfg.planner_ckpt_path,
            algo_config=planner_cfg.algo,
            modality_shapes=modality_shapes,
        )
        print("Loading planner from {}".format(self.algo_cfg.planner_ckpt_path))
        self.algos["planner"] = planner

        controller_cfg = get_experiment_config_from_file(self.algo_cfg.controller_config_path)
        controller = L5TrafficModelGC.load_from_checkpoint(
            self.algo_cfg.controller_ckpt_path,
            algo_config=controller_cfg.algo,
            modality_shapes=modality_shapes
        )
        print("Loading controller from {}".format(self.algo_cfg.controller_config_path))
        self.algos["controller"] = controller

        planner = PolicyWrapper.wrap_planner(
            planner,
            mask_drivable=self.algo_cfg.policy.mask_drivable,
            sample=self.algo_cfg.policy.sample
        )

        policy = HierarchicalPolicy(planner, controller)
        self.policy = RolloutWrapper(ego_policy=policy, agents_policy=policy)
        self.train_algo = controller  # only finetune controller through self-play