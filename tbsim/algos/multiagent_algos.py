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
from tbsim.utils.env_utils import Action, Plan


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
            roi_layer_key=algo_config.roi_layer_key
        )

    @property
    def checkpoint_monitor_keys(self):
        return {
            "valLoss": "val/losses_prediction_loss"
        }

    def forward(self, obs_dict):
        return self.model(obs_dict)

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
        preds = self(obs_dict)
        ego_preds = self.model.get_ego_predictions(preds)
        avails = torch.ones(ego_preds["predictions"]["positions"].shape[:-1]).to(ego_preds["predictions"]["positions"].device)

        plan = Plan(
            positions=ego_preds["predictions"]["positions"],
            yaws=ego_preds["predictions"]["yaws"],
            availabilities=avails
        )

        return plan, {}

    def get_action(self, obs_dict, **kwargs):
        preds = self(obs_dict)
        ego_preds = self.model.get_ego_predictions(preds)
        action = Action(
            positions=ego_preds["predictions"]["positions"],
            yaws=ego_preds["predictions"]["yaws"]
        )
        return action, {}