from typing import Dict, List
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

import tbsim.models.base_models as base_models
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.metrics as Metrics

import tbsim.models.vaes as vaes
import tbsim.utils.l5_utils as L5Utils
from tbsim.utils.geometry_utils import get_upright_box, transform_points_tensor
from tbsim.utils.loss_utils import (
    trajectory_loss,
    goal_reaching_loss,
    collision_loss
)


class MultiAgentRasterizedModel(nn.Module):
    """Raster-based model for planning.
    """

    def __init__(
            self,
            model_arch: str,
            input_image_shape,
            agent_feature_dim: int,
            future_num_frames: int,
            context_size: tuple,
            dynamics_type: str,
            dynamics_kwargs: dict,
            step_time: float,
            decoder_kwargs=None,
            weights_scaling = (1.0, 1.0, 1.0),
    ) -> None:

        super().__init__()
        self.map_encoder = base_models.RasterizeAgentEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,  # [C, H, W]
            global_feature_dim=None,
            agent_feature_dim=agent_feature_dim,
            output_activation=nn.ReLU
        )

        self.traj_decoder = base_models.MLPTrajectoryDecoder(
            feature_dim=agent_feature_dim,
            state_dim=3,
            num_steps=future_num_frames,
            dynamics_type=dynamics_type,
            dynamics_kwargs=dynamics_kwargs,
            step_time=step_time,
            network_kwargs=decoder_kwargs
        )
        assert len(context_size) == 2
        self.context_size = nn.Parameter(torch.Tensor(context_size), requires_grad=False)
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)

    @staticmethod
    def get_ego_predictions(pred_batch):
        return TensorUtils.map_tensor(pred_batch, lambda x: x[:, 0])

    @staticmethod
    def get_agents_predictions(pred_batch):
        return TensorUtils.map_tensor(pred_batch, lambda x: x[:, 1:])

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch["image"]

        # create ROI boxes
        curr_pos_all = torch.cat((
                data_batch["history_positions"].unsqueeze(1),
                data_batch["all_other_agents_history_positions"],
        ), dim=1)[:, :, 0]

        b, a = curr_pos_all.shape[:2]
        curr_pos_raster = transform_points_tensor(curr_pos_all, data_batch["raster_from_agent"].float())
        extents = torch.ones_like(curr_pos_raster) * self.context_size  # [B, A, 2]
        rois_raster = get_upright_box(curr_pos_raster, extent=extents).reshape(b * a, 2, 2)
        rois_raster = torch.flatten(rois_raster, start_dim=1)  # [B * A, 4]

        roi_indices = torch.arange(0, b).unsqueeze(1).expand(-1, a).reshape(-1, 1).to(rois_raster.device)  # [B * A, 1]
        indexed_rois_raster = torch.cat((roi_indices, rois_raster), dim=1)  # [B * A, 5]

        # get per-agent predictions
        agent_feats, _, _ = self.map_encoder(image_batch, rois=indexed_rois_raster)
        agent_feats = agent_feats.reshape(b, a, -1)
        traj = self.traj_decoder.forward(inputs=agent_feats)["trajectories"]

        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws},
            "rois_raster": rois_raster.reshape(b, a, 2, 2)
        }
        return out_dict

    def compute_metrics(self, pred_batch, data_batch):
        metrics = dict()
        ego_preds = self.get_ego_predictions(pred_batch)
        pos_preds = TensorUtils.to_numpy(ego_preds["predictions"]["positions"])

        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        ade = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, pos_preds, avail
        )
        fde = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, pos_preds, avail
        )

        metrics["ego_ADE"] = np.mean(ade)
        metrics["ego_FDE"] = np.mean(fde)

        agents_preds = self.get_agents_predictions(pred_batch)
        pos_preds = TensorUtils.to_numpy(agents_preds["predictions"]["positions"])
        num_frames = pos_preds.shape[2]
        pos_preds = pos_preds.reshape(-1, num_frames, 2)
        all_targets = L5Utils.batch_to_target_all_agents(data_batch)
        gt = TensorUtils.to_numpy(all_targets["target_positions"][:, 1:]).reshape(-1, num_frames, 2)
        avail = TensorUtils.to_numpy(all_targets["target_availabilities"][:, 1:]).reshape(-1, num_frames)

        ade = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, pos_preds, avail
        )
        fde = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, pos_preds, avail
        )

        metrics["agents_ADE"] = np.mean(ade)
        metrics["agents_FDE"] = np.mean(fde)

        return metrics

    def compute_losses(self, pred_batch, data_batch):
        all_targets = L5Utils.batch_to_target_all_agents(data_batch)
        target_traj = torch.cat((all_targets["target_positions"], all_targets["target_yaws"]), dim=-1)
        pred_loss = trajectory_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=all_targets["target_availabilities"],
            weights_scaling=self.weights_scaling
        )
        goal_loss = goal_reaching_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=all_targets["target_availabilities"],
            weights_scaling=self.weights_scaling
        )

        # compute collision loss
        pred_edges = L5Utils.get_edges_from_batch(
            data_batch=data_batch,
            all_predictions=pred_batch["predictions"]
        )

        coll_loss = collision_loss(pred_edges=pred_edges)
        losses = OrderedDict(prediction_loss=pred_loss, goal_loss=goal_loss, collision_loss=coll_loss)
        return losses