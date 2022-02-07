from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from tbsim.models.l5kit_models import (
    RasterizedPlanningModel,
    RasterizedVAEModel,
    RasterizedGCModel,
    optimize_trajectories,
    get_current_states
)
from tbsim.models.base_models import MLPTrajectoryDecoder, RasterizedMapUNet, RasterizedMapKeyPointNet
from tbsim.models.transformer_model import TransformerModel
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.metrics as Metrics
import tbsim.utils.l5_utils as L5Utils
from tbsim.utils.geometry_utils import transform_points_tensor
import tbsim.dynamics as dynamics


class L5TrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        """
        Creates networks and places them into @self.nets.
        """
        super(L5TrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        assert modality_shapes["image"][0] == 15

        traj_decoder = MLPTrajectoryDecoder(
            feature_dim=algo_config.map_feature_dim,
            state_dim=3,
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            step_time=algo_config.step_time,
            network_kwargs=algo_config.decoder
        )

        self.nets["policy"] = RasterizedPlanningModel(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            trajectory_decoder=traj_decoder,
            map_feature_dim=algo_config.map_feature_dim,
            weights_scaling=[1.0, 1.0, 1.0],
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )

    @property
    def checkpoint_monitor_keys(self):
        return {
            "valLoss": "val/losses_prediction_loss"
        }

    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]

    def _compute_metrics(self, pred_batch, data_batch):
        metrics = {}
        predictions = pred_batch["predictions"]
        preds = TensorUtils.to_numpy(predictions["positions"])
        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        ade = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, preds, avail
        )
        fde = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, preds, avail
        )

        metrics["ego_ADE"] = np.mean(ade)
        metrics["ego_FDE"] = np.mean(fde)

        targets_all = L5Utils.batch_to_target_all_agents(data_batch)
        raw_type = torch.cat(
            (data_batch["type"].unsqueeze(1), data_batch["all_other_agents_types"]),
            dim=1,
        ).type(torch.int64)

        pred_edges = L5Utils.generate_edges(
            raw_type, targets_all["extents"],
            pos_pred=targets_all["target_positions"],
            yaw_pred=targets_all["target_yaws"]
        )

        coll_rates = TensorUtils.to_numpy(Metrics.batch_pairwise_collision_rate(pred_edges))
        for c in coll_rates:
            metrics["coll_" + c] = float(coll_rates[c])

        return metrics

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        pout = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(pout, batch)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        metrics = self._compute_metrics(pout, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        metrics = self._compute_metrics(pout, batch)
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

    def get_action(self, obs_dict, **kwargs):
        return {"ego": self(obs_dict["ego"])}


class L5TrafficModelGC(L5TrafficModel):
    def __init__(self, algo_config, modality_shapes):
        """
        Creates networks and places them into @self.nets.
        """
        pl.LightningModule.__init__(self)
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        assert modality_shapes["image"][0] == 15

        traj_decoder = MLPTrajectoryDecoder(
            feature_dim=algo_config.map_feature_dim + algo_config.goal_feature_dim,
            state_dim=3,
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            step_time=algo_config.step_time,
            network_kwargs=algo_config.decoder
        )

        self.nets["policy"] = RasterizedGCModel(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            trajectory_decoder=traj_decoder,
            map_feature_dim=algo_config.map_feature_dim,
            weights_scaling=[1.0, 1.0, 1.0],
            goal_feature_dim=algo_config.goal_feature_dim,
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )


class SpatialPlanner(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(SpatialPlanner, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        assert modality_shapes["image"][0] == 15

        self.nets["policy"] = RasterizedMapUNet(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            output_channel=4,  # (pixel, x_residua, y_residua, yaw)
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )

    @property
    def checkpoint_monitor_keys(self):
        return {
            "posErr": "val/metrics_goal_pos_err",
            "valLoss": "val/losses_pixel_cls_loss"
        }

    def decode_spatial_map(self, pred_map):
        # decode map as predictions
        b, c, h, w = pred_map.shape
        pixel_logit, pixel_loc_flat = torch.max(torch.flatten(pred_map[:, 0], start_dim=1), dim=1)
        pixel_loc_x = torch.remainder(pixel_loc_flat, w)
        pixel_loc_y = torch.floor(pixel_loc_flat.float() / float(w)).long()
        pixel_loc = torch.stack((pixel_loc_x, pixel_loc_y), dim=1)  # [B, 2]
        local_pred = torch.gather(
            input=torch.flatten(pred_map, 2),  # [B, C, H * W]
            dim=2,
            index=TensorUtils.unsqueeze_expand_at(pixel_loc_flat, size=c, dim=1)[:, :, None]  # [B, C, 1]
        ).squeeze(-1)
        residual_pred = local_pred[:, 1:3]
        yaw_pred = local_pred[:, 3:4]
        return pixel_loc, residual_pred, yaw_pred, pixel_logit

    def get_last_available_index(self, avails):
        """
        Args:
            avails (torch.Tensor): target availabilities [B, (A), T]

        Returns:
            last_indices (torch.Tensor): index of the last available frame
        """
        num_frames = avails.shape[-1]
        inds = torch.arange(0, num_frames).to(avails.device)  # [T]
        inds = (avails > 0).float() * inds  # [B, (A), T] arange indices with unavailable indices set to 0
        last_inds = inds.max(dim=-1)[1]  # [B, (A)] calculate the index of the last availale frame
        return last_inds

    def get_goal_supervision(self, data_batch):
        b, _, h, w = data_batch["image"].shape  # [B, C, H, W]

        # use last available step as goal location
        goal_index = self.get_last_available_index(data_batch["target_availabilities"])[:, None, None]

        # gather by goal index
        goal_pos_agent = torch.gather(
            data_batch["target_positions"],  # [B, T, 2]
            dim=1,
            index=goal_index.expand(-1, 1, data_batch["target_positions"].shape[-1])
        )  # [B, 1, 2]

        goal_yaw_agent = torch.gather(
            data_batch["target_yaws"],  # [B, T, 1]
            dim=1,
            index=goal_index.expand(-1, 1, data_batch["target_yaws"].shape[-1])
        )  # [B, 1, 1]

        # create spatial supervisions
        goal_pos_raster = transform_points_tensor(
            goal_pos_agent,
            data_batch["raster_from_agent"].float()
        ).squeeze(1)  # [B, 2]
        # make sure all pixels are within the raster image
        goal_pos_raster[:, 0] = goal_pos_raster[:, 0].clip(0, w - 1e-5)
        goal_pos_raster[:, 1] = goal_pos_raster[:, 1].clip(0, h - 1e-5)

        goal_pos_pixel = torch.floor(goal_pos_raster).float()  # round down pixels
        goal_pos_residual = goal_pos_raster - goal_pos_pixel  # compute rounding residuals (range 0-1)
        # compute flattened pixel location
        goal_pos_pixel_flat = goal_pos_pixel[:, 1] * w + goal_pos_pixel[:, 0]
        raster_sup_flat = TensorUtils.to_one_hot(goal_pos_pixel_flat.long(), num_class=h * w)
        raster_sup = raster_sup_flat.reshape(b, h, w)
        return {
            "goal_position_residual": goal_pos_residual,  # [B, 2]
            "goal_spatial_map": raster_sup,  # [B, H, W]
            "goal_position_pixel": goal_pos_pixel,  # [B, 2]
            "goal_position_pixel_flat": goal_pos_pixel_flat,  # [B]
            "goal_position": goal_pos_agent.squeeze(1),  # [B, 2]
            "goal_yaw": goal_yaw_agent.squeeze(1),  # [B, 1]
            "goal_index": goal_index.reshape(b)  # [B]
        }

    def forward(self, obs_dict):
        pred_map = self.nets["policy"](obs_dict["image"])
        pred_map[:, 1:3] = torch.sigmoid(pred_map[:, 1:3])  # x, y residuals are within [0, 1]
        # decode map as predictions
        pixel_pred, res_pred, yaw_pred, pred_logit = self.decode_spatial_map(pred_map)
        # transform prediction to agent coordinate
        pixel_pred = transform_points_tensor(
            pixel_pred.unsqueeze(1).float(),
            obs_dict["agent_from_raster"].float()
        ).squeeze(1)
        pos_pred = transform_points_tensor(
            (pixel_pred + res_pred).unsqueeze(1),
            obs_dict["agent_from_raster"].float()
        ).squeeze(1)

        # normalize pixel location map
        location_map = pred_map[:, 0]
        location_map = torch.softmax(location_map.flatten(1), dim=1).reshape(location_map.shape)

        return dict(
            predictions=dict(
                positions=pos_pred,
                yaws=yaw_pred
            ),
            pixel_predictions=dict(
                positions=pixel_pred,
                yaws=yaw_pred
            ),
            confidence=torch.sigmoid(pred_logit),
            spatial_prediction=pred_map,
            normalized_location_map=location_map
        )

    def _compute_metrics(self, pred_batch, data_batch):
        metrics = dict()
        goal_sup = data_batch["goal"]
        pos_norm_err = torch.norm(pred_batch["predictions"]["positions"] - goal_sup["goal_position"], dim=-1)
        metrics["goal_pos_err"] = torch.mean(pos_norm_err)
        metrics["goal_yaw_err"] = torch.mean(torch.abs(pred_batch["predictions"]["yaws"] - goal_sup["goal_yaw"]))

        pixel_pred = torch.argmax(torch.flatten(pred_batch["spatial_prediction"][:, 0], start_dim=1), dim=1) # [B]
        metrics["goal_selection_err"] = torch.mean((goal_sup["goal_position_pixel_flat"].long() != pixel_pred).float())
        metrics["goal_cls_err"] = torch.mean((pred_batch["confidence"] < 0.5).float())
        metrics = TensorUtils.to_numpy(metrics)
        for k, v in metrics.items():
            metrics[k] = float(v)
        return metrics

    def _compute_losses(self, pred_batch, data_batch):
        losses = dict()
        pred_map = pred_batch["spatial_prediction"]
        b, c, h, w = pred_map.shape

        goal_sup = data_batch["goal"]
        # compute pixel classification loss
        location_prediction = pred_map[:, 0]
        losses["pixel_bce_loss"] = torch.binary_cross_entropy_with_logits(
            input=location_prediction,  # [B, H, W]
            target=goal_sup["goal_spatial_map"],  # [B, H, W]
        ).mean()

        losses["pixel_ce_loss"] = torch.nn.CrossEntropyLoss()(
            input=location_prediction.flatten(start_dim=1),  # [B, H * W]
            target=goal_sup["goal_position_pixel_flat"].long(),  # [B]
        )

        # compute residual and yaw loss
        gather_inds = TensorUtils.unsqueeze_expand_at(
            goal_sup["goal_position_pixel_flat"].long(), size=c, dim=1
        )[..., None]  # -> [B, C, 1]

        local_pred = torch.gather(
            input=torch.flatten(pred_map, 2),  # [B, C, H * W]
            dim=2,
            index=gather_inds  # [B, C, 1]
        ).squeeze(-1)  # -> [B, C]
        residual_pred = local_pred[:, 1:3]
        yaw_pred = local_pred[:, 3:4]
        losses["pixel_res_loss"] = torch.nn.MSELoss()(residual_pred, goal_sup["goal_position_residual"])
        losses["pixel_yaw_loss"] = torch.nn.MSELoss()(yaw_pred, goal_sup["goal_yaw"])

        return losses

    def training_step(self, batch, batch_idx):
        pout = self(batch)
        batch["goal"] = self.get_goal_supervision(batch)
        losses = self._compute_losses(pout, batch)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        metrics = self._compute_metrics(pout, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        pout = self(batch)
        batch["goal"] = self.get_goal_supervision(batch)
        losses = TensorUtils.detach(self._compute_losses(pout, batch))
        metrics = self._compute_metrics(pout, batch)
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

    def get_action(self, obs_dict, **kwargs):
        return {"ego": self(obs_dict["ego"])}



class L5VAETrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(L5VAETrafficModel, self).__init__()
        assert modality_shapes["image"][0] == 15

        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = RasterizedVAEModel(
            algo_config=algo_config,
            modality_shapes=modality_shapes,
            weights_scaling=[1.0, 1.0, 1.0]
        )

    @property
    def checkpoint_monitor_keys(self):
        return {
            "valLoss": "val/losses_prediction_loss"
        }

    def forward(self, obs_dict):
        return self.nets["policy"].predict(obs_dict)["predictions"]

    def _compute_metrics(self, pred_batch, sample_batch, data_batch):
        metrics = {}

        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        # compute ADE & FDE based on posterior params
        recon_preds = TensorUtils.to_numpy(pred_batch["predictions"]["positions"])
        metrics["ego_ADE"] = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, recon_preds, avail
        ).mean()
        metrics["ego_FDE"] = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, recon_preds, avail
        ).mean()

        # compute ADE & FDE based on trajectory samples
        sample_preds = TensorUtils.to_numpy(sample_batch["predictions"]["positions"])
        conf = np.ones(sample_preds.shape[0:2]) / float(sample_preds.shape[1])
        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "max").mean()
        metrics["ego_avg_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "max").mean()

        return metrics

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        pout = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(pout, batch)
        # take samples to measure trajectory diversity
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=self.algo_config.vae.num_eval_samples)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        metrics = self._compute_metrics(pout, samples, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=self.algo_config.vae.num_eval_samples)
        metrics = self._compute_metrics(pout, samples, batch)
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

    def get_action(self, obs_dict, sample=True, num_viz_samples=10, **kwargs):
        if sample:
            preds = self.nets["policy"].sample(obs_dict["ego"], n=1)["predictions"]  # [B, 1, T, 3]
            preds = TensorUtils.squeeze(preds, dim=1)
        else:
            preds = self.nets["policy"].predict(obs_dict["ego"])["predictions"]

        # get trajectory samples for visualization purposes
        ego_samples = self.nets["policy"].sample(obs_dict["ego"], n=num_viz_samples)["predictions"]
        return {"ego": preds, "ego_samples": ego_samples}


class L5TransformerTrafficModel(pl.LightningModule):
    def __init__(self, algo_config):
        """
        Creates networks and places them into @self.nets.
        """
        super(L5TransformerTrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = TransformerModel(algo_config)

    @property
    def checkpoint_monitor_keys(self):
        return {
            "valLoss": "val/losses_prediction_loss"
        }

    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        tgt_mask_p = 1 - min(1.0, float(batch_idx / self.algo_config.tgt_mask_N))
        pout = self.nets["policy"](batch, tgt_mask_p)
        losses = self.nets["policy"].compute_losses(pout, batch)
        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)

        total_loss = 0.0
        for v in losses.values():
            total_loss += v

        metrics = self._compute_metrics(pout["predictions"], batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m, prog_bar=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        metrics = self._compute_metrics(pout["predictions"], batch)
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

    def get_action(self, obs_dict, **kwargs):
        return {"ego": self(obs_dict["ego"])}
