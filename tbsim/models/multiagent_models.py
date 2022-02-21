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
    """A model that predict each agent's future trajectories by featurizing their local image feature."""
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
        preds = self.traj_decoder.forward(inputs=agent_feats)

        traj = preds["trajectories"]

        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws},
            "rois_raster": rois_raster.reshape(b, a, 2, 2)
        }
        if self.traj_decoder.dyn is not None:
            out_dict["controls"] = preds["controls"]
        return out_dict

    def compute_metrics(self, pred_batch, data_batch):
        metrics = dict()

        # ego ADE & FDE
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


        # agent ADE & FDE
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


        # pairwise collisions
        targets_all = L5Utils.batch_to_target_all_agents(data_batch)
        raw_type = torch.cat(
            (data_batch["type"].unsqueeze(1), data_batch["all_other_agents_types"]),
            dim=1,
        ).type(torch.int64)

        pred_edges = L5Utils.generate_edges(
            raw_type,
            targets_all["extents"],
            pos_pred=targets_all["target_positions"],
            yaw_pred=targets_all["target_yaws"],
        )

        coll_rates = TensorUtils.to_numpy(
            Metrics.batch_pairwise_collision_rate(pred_edges)
        )
        for c in coll_rates:
            metrics["coll_" + c] = float(coll_rates[c])

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
        if self.traj_decoder.dyn is not None:
            losses["yaw_reg_loss"] = torch.mean(pred_batch["controls"][..., 1] ** 2)
        return losses


class AgentAwareRasterizedModel(MultiAgentRasterizedModel):
    """Ego-centric model that is aware of other agents' future trajectories through auxiliary prediction task"""
    def __init__(
            self,
            model_arch: str,
            input_image_shape,
            ego_feature_dim: int,
            agent_feature_dim: int,
            future_num_frames: int,
            context_size: tuple,
            dynamics_type: str,
            dynamics_kwargs: dict,
            step_time: float,
            decoder_kwargs: dict = None,
            disable_dynamics_for_other_agents: bool = False,
            goal_conditional: bool = False,
            goal_feature_dim : int = 32,
            weights_scaling : tuple = (1.0, 1.0, 1.0),
    ) -> None:

        nn.Module.__init__(self)

        self.map_encoder = base_models.RasterizeAgentEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,  # [C, H, W]
            global_feature_dim=ego_feature_dim,
            agent_feature_dim=agent_feature_dim,
            output_activation=nn.ReLU
        )

        self.goal_conditional = goal_conditional
        goal_dim = 0
        if self.goal_conditional:
            self.goal_encoder = base_models.MLP(
                input_dim=3,
                output_dim=goal_feature_dim,
                output_activation=nn.ReLU
            )
            goal_dim = goal_feature_dim

        self.ego_decoder = base_models.MLPTrajectoryDecoder(
            feature_dim=ego_feature_dim + goal_dim,
            state_dim=3,
            num_steps=future_num_frames,
            dynamics_type=dynamics_type,
            dynamics_kwargs=dynamics_kwargs,
            step_time=step_time,
            network_kwargs=decoder_kwargs
        )

        other_dyn_type = None if disable_dynamics_for_other_agents else dynamics_type
        self.agent_decoder = base_models.MLPTrajectoryDecoder(
            feature_dim=agent_feature_dim,
            state_dim=3,
            num_steps=future_num_frames,
            dynamics_type=other_dyn_type,
            dynamics_kwargs=dynamics_kwargs,
            step_time=step_time,
            network_kwargs=decoder_kwargs
        )

        assert len(context_size) == 2
        self.context_size = nn.Parameter(torch.Tensor(context_size), requires_grad=False)
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch["image"]

        # create ROI boxes only for other agents
        curr_pos_agents = data_batch["all_other_agents_history_positions"][:, :, 0]

        b, a = curr_pos_agents.shape[:2]
        curr_pos_raster = transform_points_tensor(curr_pos_agents, data_batch["raster_from_agent"].float())
        extents = torch.ones_like(curr_pos_raster) * self.context_size  # [B, A, 2]
        rois_raster = get_upright_box(curr_pos_raster, extent=extents).reshape(b * a, 2, 2)
        rois_raster = torch.flatten(rois_raster, start_dim=1)  # [B * A, 4]

        roi_indices = torch.arange(0, b).unsqueeze(1).expand(-1, a).reshape(-1, 1).to(rois_raster.device)  # [B * A, 1]
        indexed_rois_raster = torch.cat((roi_indices, rois_raster), dim=1)  # [B * A, 5]

        agent_feats, _, ego_feats = self.map_encoder(image_batch, rois=indexed_rois_raster)
        # use per-agent features for predicting non-ego trajectories
        agent_feats = agent_feats.reshape(b, a, -1)
        agent_pred = self.agent_decoder.forward(inputs=agent_feats)

        # use global feature for predicting ego trajectories
        if self.goal_conditional:
            # optionally condition the ego features on a goal location
            target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
            goal_inds = L5Utils.get_last_available_index(data_batch["target_availabilities"])
            goal_state = torch.gather(
                target_traj,  # [B, T, 3]
                dim=1,
                index=goal_inds[:, None, None].expand(-1, 1, target_traj.shape[-1])
            ).squeeze(1)  # -> [B, 3]
            goal_feat = self.goal_encoder(goal_state) # -> [B, D]
            ego_feats = torch.cat((ego_feats, goal_feat), dim=-1)

        ego_pred = self.ego_decoder.forward(inputs=ego_feats)

        # process predictions
        all_pred = dict()
        for k in agent_pred:
            all_pred[k] = torch.cat((ego_pred[k].unsqueeze(1), agent_pred[k]), dim=1)

        traj = all_pred["trajectories"]
        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws},
            "rois_raster": rois_raster.reshape(b, a, 2, 2)
        }
        if self.ego_decoder.dyn is not None and self.ego_decoder.dyn is not None:
            out_dict["controls"] = all_pred["controls"]
        elif self.ego_decoder.dyn is not None and self.ego_decoder.dyn is None:
            out_dict["controls"] = ego_pred["controls"]

        return out_dict

    def compute_losses(self, pred_batch, data_batch):
        all_targets = L5Utils.batch_to_target_all_agents(data_batch)
        target_traj = torch.cat((all_targets["target_positions"], all_targets["target_yaws"]), dim=-1)
        pred_loss = trajectory_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=all_targets["target_availabilities"],
            weights_scaling=self.weights_scaling
        )

        # compute goal loss only for the ego
        goal_loss = goal_reaching_loss(
            predictions=pred_batch["trajectories"][:, 0],
            targets=target_traj[:, 0],
            availabilities=all_targets["target_availabilities"][:, 0],
            weights_scaling=self.weights_scaling
        )

        # compute collision loss
        # since we don't assume control over other agents, the loss should only backprop through the ego prediction
        preds = TensorUtils.clone(pred_batch["predictions"])
        preds["positions"][:, 1:] = preds["positions"][:, 1:].detach()
        preds["yaws"][:, 1:] = preds["yaws"][:, 1:].detach()
        pred_edges = L5Utils.get_edges_from_batch(
            data_batch=data_batch,
            all_predictions=preds
        )

        coll_loss = collision_loss(pred_edges=pred_edges)
        losses = OrderedDict(prediction_loss=pred_loss, goal_loss=goal_loss, collision_loss=coll_loss)

        if "controls" in pred_batch:
            losses["yaw_reg_loss"] = torch.mean(pred_batch["controls"][..., 1] ** 2)
        return losses