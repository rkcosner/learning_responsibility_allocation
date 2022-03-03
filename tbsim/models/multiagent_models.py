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
from tbsim.models.roi_align import ROI_align, generate_ROIs, Indexing_ROI_result
from tbsim.models.Transformer import SimpleTransformer


class AgentAwareRasterizedModel(nn.Module):
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
            goal_conditional: bool = False,
            goal_feature_dim : int = 32,
            weights_scaling : tuple = (1.0, 1.0, 1.0),
    ) -> None:

        nn.Module.__init__(self)

        self.map_encoder = base_models.RasterizeROIEncoder(
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

        # self.ego_decoder = base_models.MLPTrajectoryDecoder(
        #     feature_dim=ego_feature_dim + goal_dim,
        #     state_dim=3,
        #     num_steps=future_num_frames,
        #     dynamics_type=dynamics_type,
        #     dynamics_kwargs=dynamics_kwargs,
        #     step_time=step_time,
        #     network_kwargs=decoder_kwargs
        # )

        # other_dyn_type = None if disable_dynamics_for_other_agents else dynamics_type
        self.decoder = base_models.MLPTrajectoryDecoder(
            feature_dim=agent_feature_dim + goal_dim,
            state_dim=3,
            num_steps=future_num_frames,
            dynamics_type=dynamics_type,
            dynamics_kwargs=dynamics_kwargs,
            step_time=step_time,
            network_kwargs=decoder_kwargs
        )

        self.transformer = SimpleTransformer(src_dim=agent_feature_dim)
        assert len(context_size) == 2
        self.context_size = nn.Parameter(torch.Tensor(context_size), requires_grad=False)
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)

    @staticmethod
    def get_ego_predictions(pred_batch):
        return TensorUtils.map_tensor(pred_batch, lambda x: x[:, 0])

    @staticmethod
    def get_agents_predictions(pred_batch):
        return TensorUtils.map_tensor(pred_batch, lambda x: x[:, 1:])

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

    def _get_roi_boxes(self, pos, yaw, trans_mat):
        b, a = pos.shape[:2]
        curr_pos_raster = transform_points_tensor(pos, trans_mat.float())
        extents = torch.ones_like(curr_pos_raster) * self.context_size  # [B, A, 2]
        rois_raster = get_upright_box(curr_pos_raster, extent=extents).reshape(b * a, 2, 2)
        rois_raster = torch.flatten(rois_raster, start_dim=1)  # [B * A, 4]

        roi_indices = torch.arange(0, b).unsqueeze(1).expand(-1, a).reshape(-1, 1).to(rois_raster.device)  # [B * A, 1]
        indexed_rois_raster = torch.cat((roi_indices, rois_raster), dim=1)  # [B * A, 5]
        return indexed_rois_raster

    def _get_roi_boxes_rotated(self, pos, yaw, centroid, avails, trans_mat, patch_size):
        rois, indices = generate_ROIs(pos, yaw, centroid, trans_mat, avails, patch_size, mode="last")
        return rois, indices

    def _get_goal_states(self, data_batch):
        all_targets = L5Utils.batch_to_target_all_agents(data_batch)
        target_traj = torch.cat((all_targets["target_positions"], all_targets["target_yaws"]), dim=2)
        goal_inds = L5Utils.get_last_available_index(all_targets["target_availabilities"])  # [B, A]
        goal_state = torch.gather(
            target_traj,  # [B, A, T, 3]
            dim=2,
            index=goal_inds[:, :, None, None].expand(-1, -1, 1, target_traj.shape[-1])  # [B, A, 1, 3]
        ).squeeze(2)  # -> [B, A, 3]
        return goal_state

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch["image"]

        # create ROI boxes only for other agents
        curr_pos_all = torch.cat((
            data_batch["history_positions"].unsqueeze(1),
            data_batch["all_other_agents_history_positions"],
        ), dim=1)[:, :, 0]  # histories are reversed
        b, a = curr_pos_all.shape[:2]

        # curr_yaw_all = torch.cat((
        #     data_batch["history_yaws"].unsqueeze(1),
        #     data_batch["all_other_agents_history_yaws"],
        # ), dim=1)[:, :, 0]  # histories are reversed

        states_all = L5Utils.batch_to_raw_all_agents(data_batch, self.decoder.step_time)

        # rois = self._get_roi_boxes(curr_pos_all, curr_yaw_all, trans_mat=data_batch["raster_from_agent"])
        rois, indices = self._get_roi_boxes_rotated(
            pos=states_all["history_positions"],
            yaw=states_all["history_yaws"],
            centroid=data_batch["centroid"],
            avails=states_all["history_availabilities"],
            trans_mat=data_batch["raster_from_world"],
            patch_size=torch.Tensor([30, 30, 30, 30]).to(image_batch.device)
        )

        all_feats, _, _ = self.map_encoder(image_batch, rois=rois)  # approximately B * A
        split_sizes = [len(l) for l in indices]
        all_feats_list = torch.split(all_feats, split_sizes)
        all_feats = Indexing_ROI_result(all_feats_list, indices, emb_size=(b, a, all_feats.shape[-1]))
        all_feats = all_feats.reshape(b, a, -1)

        all_feats = self.transformer(
            all_feats,
            states_all["history_availabilities"][:, :, -1],
            states_all["history_positions"][:, :, -1]
        )

        if self.goal_conditional:
            # optionally condition the prediction on goal locations
            goal_state = self._get_goal_states(data_batch)
            goal_feat = TensorUtils.time_distributed(goal_state, self.goal_encoder) # -> [B, A, D]
            all_feats = torch.cat((all_feats, goal_feat), dim=-1)

        curr_states = L5Utils.get_current_states_all_agents(
            data_batch,
            self.decoder.step_time,
            dyn_type=self.decoder.dyn.type()
        )

        all_pred = self.decoder.forward(inputs=all_feats, current_states=curr_states)

        traj = all_pred["trajectories"]
        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws},
            # "rois_raster": rois[:, 1:].reshape(b, a, 2, 2)
        }
        if self.decoder.dyn is None:
            out_dict["controls"] = all_pred["controls"]

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

        goal_loss = goal_reaching_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=all_targets["target_availabilities"],
            weights_scaling=self.weights_scaling
        )

        # compute collision loss
        preds = TensorUtils.clone(pred_batch["predictions"])
        # preds["positions"][:, 1:] = preds["positions"][:, 1:].detach()
        # preds["yaws"][:, 1:] = preds["yaws"][:, 1:].detach()
        pred_edges = L5Utils.get_edges_from_batch(
            data_batch=data_batch,
            all_predictions=preds
        )

        coll_loss = collision_loss(pred_edges=pred_edges)
        losses = OrderedDict(prediction_loss=pred_loss, goal_loss=goal_loss, collision_loss=coll_loss)
        if "controls" in pred_batch:
            # regularize the magnitude of yaw
            losses["yaw_reg_loss"] = torch.mean(pred_batch["controls"][..., 1] ** 2)

        return losses