from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from tbsim.models.multiagent_models import (
    AgentAwareRasterizedModel
)
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.policies.common import Action, Plan, Trajectory
from tbsim.utils.loss_utils import discriminator_loss


class MATrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(MATrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        assert modality_shapes["image"][0] == 15

        self.model = AgentAwareRasterizedModel(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            global_feature_dim=algo_config.global_feature_dim,
            agent_feature_dim=algo_config.agent_feature_dim,
            roi_size=algo_config.context_size,
            future_num_frames=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            step_time=algo_config.step_time,
            decoder_kwargs=algo_config.decoder,
            goal_conditional=algo_config.goal_conditional,
            goal_feature_dim=algo_config.goal_feature_dim,
            use_rotated_roi=algo_config.use_rotated_roi,
            use_transformer=algo_config.use_transformer,
            roi_layer_key=algo_config.roi_layer_key,
            use_gan=algo_config.use_GAN
        )

    @property
    def checkpoint_monitor_keys(self):
        return {
            "valLoss": "val/losses_prediction_loss"
        }

    def forward(self, obs_dict, plan=None):
        return self.model(obs_dict, plan)

    def training_step(self, batch, batch_idx):
        pout = self.model.forward(batch)
        losses = self.model.compute_losses(pout, batch)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        metrics = self.model.compute_metrics(pout, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        pout = self.model.forward(batch)
        losses = TensorUtils.detach(self.model.compute_losses(pout, batch))
        metrics = self.model.compute_metrics(pout, batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

    def get_plan(self, obs_dict, **kwargs):
        """If using the model as a planner (setting subgoals)"""
        preds = self(obs_dict)
        ego_preds = self.model.get_ego_predictions(preds)
        avails = torch.ones(ego_preds["predictions"]["positions"].shape[:-1]).to(
            ego_preds["predictions"]["positions"].device)

        plan = Plan(
            positions=ego_preds["predictions"]["positions"],
            yaws=ego_preds["predictions"]["yaws"],
            availabilities=avails
        )

        return plan, {}

    def get_action(self, obs_dict, **kwargs):
        """If using the model as an actor (generating actions)"""
        # extract agent features from obs
        feats = self.model.extract_features(obs_dict)
        if "plan_samples" in kwargs:
            # if evaluating multiple plan samples per obs
            plan_samples = kwargs["plan_samples"]
            b, n = plan_samples.positions.shape[:2]
            # reuse features by tiling the feature tensors to the same size as plan samples
            feats_tiled = TensorUtils.repeat_by_expand_at(feats, repeats=n, dim=0)
            # flatten the sample dimension to the batch dimension
            plan_tiled = TensorUtils.join_dimensions(plan_samples.to_dict(), begin_axis=0, end_axis=2)
            plan_tiled = Plan.from_dict(plan_tiled)

            obs_tiled = TensorUtils.repeat_by_expand_at(obs_dict, repeats=n, dim=0)
            preds = self.model.forward_prediction(feats_tiled, obs_tiled, plan=plan_tiled)
        else:
            plan = kwargs.get("plan", None)
            preds = self.model.forward_prediction(feats, obs_dict, plan)

        ego_preds = self.model.get_ego_predictions(preds)
        action = Action(
            positions=ego_preds["predictions"]["positions"],
            yaws=ego_preds["predictions"]["yaws"]
        )
        return action, {}

    def get_prediction(self, obs_dict, **kwargs):
        """If using the model as a trajectory predictor (generating trajs for non-ego agents)"""
        plan = kwargs.get("plan", None)
        preds = self(obs_dict, plan)
        agent_preds = self.model.get_agents_predictions(preds)
        agent_trajs = Trajectory(
            positions=agent_preds["predictions"]["positions"],
            yaws=agent_preds["predictions"]["yaws"]
        )
        return agent_trajs, {}


class MAGANTrafficModel(MATrafficModel):

    @property
    def checkpoint_monitor_keys(self):
        return {
            "valLoss": "val/losses_prediction_loss"
        }

    def forward(self, obs_dict, plan=None):
        return self.model(obs_dict, plan)

    def discriminator_loss(self, pred_batch):
        d_loss = discriminator_loss(
            pred_batch["likelihood_pred"], pred_batch["likelihood_GT"])
        self.log("train/discriminator_loss", d_loss)
        return d_loss

    def training_step(self, batch, batch_idx, optimizer_idx):

        pout = self.model.forward(batch)
        if optimizer_idx == 0:
            losses = self.model.compute_losses(pout, batch)
            total_loss = 0.0
            for lk, l in losses.items():
                loss = l * self.algo_config.loss_weights[lk]
                self.log("train/losses_" + lk, loss)
                total_loss += loss

            metrics = self.model.compute_metrics(pout, batch)
            for mk, m in metrics.items():
                self.log("train/metrics_" + mk, m)

            return total_loss
        elif optimizer_idx == 1:
            d_loss = self.discriminator_loss(pout)

            for mk in ["likelihood_pred", "likelihood_GT"]:
                self.log("train/metrics_" + mk, pout[mk].mean())
            return d_loss

    def validation_step(self, batch, batch_idx):
        pout = self.model.forward(batch)
        losses = TensorUtils.detach(self.model.compute_losses(pout, batch))
        metrics = self.model.compute_metrics(pout, batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        gen_params = list()
        discr_params = list()
        for com_name, com in self.model.named_children():
            if com_name not in ["GAN", "traj_encoder"]:
                gen_params += list(com.parameters())
            else:
                discr_params += list(com.parameters())
        gen_optim_params = self.algo_config.optim_params.policy
        discr_optim_params = self.algo_config.optim_params.GAN
        gen_optim = optim.Adam(
            params=gen_params,
            lr=gen_optim_params["learning_rate"]["initial"],
            weight_decay=gen_optim_params["regularization"]["L2"],
        )
        discr_optim = optim.Adam(
            params=discr_params,
            lr=discr_optim_params["learning_rate"]["initial"],
            weight_decay=discr_optim_params["regularization"]["L2"],
        )

        return [gen_optim, discr_optim], []

    def get_plan(self, obs_dict, **kwargs):
        preds = self(obs_dict)
        ego_preds = self.model.get_ego_predictions(preds)
        avails = torch.ones(ego_preds["predictions"]["positions"].shape[:-1]).to(
            ego_preds["predictions"]["positions"].device)

        plan = Plan(
            positions=ego_preds["predictions"]["positions"],
            yaws=ego_preds["predictions"]["yaws"],
            availabilities=avails
        )

        return plan, {}

    def get_action(self, obs_dict, **kwargs):
        if "plan" in kwargs:
            plan = kwargs["plan"]
        else:
            plan = None
        preds = self(obs_dict, plan)
        ego_preds = self.model.get_ego_predictions(preds)
        action = Action(
            positions=ego_preds["predictions"]["positions"],
            yaws=ego_preds["predictions"]["yaws"]
        )
        return action, {}

    def get_prediction(self, obs_dict, **kwargs):
        if "plan" in kwargs:
            plan = kwargs["plan"]
        else:
            plan = None
        preds = self(obs_dict, plan)
        agent_preds = self.model.get_agents_predictions(preds)
        agent_trajs = Trajectory(
            positions=agent_preds["predictions"]["positions"],
            yaws=agent_preds["predictions"]["yaws"]
        )
        return agent_trajs, {}
