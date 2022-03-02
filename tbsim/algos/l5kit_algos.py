from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import tbsim.utils.torch_utils as TorchUtils

from tbsim.models.rasterized_models import (
    RasterizedPlanningModel,
    RasterizedVAEModel,
    RasterizedGCModel,
)
from tbsim.models.base_models import (
    MLPTrajectoryDecoder,
    RasterizedMapUNet,
    RasterizedMapKeyPointNet,
)
from tbsim.models.transformer_model import TransformerModel
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.metrics as Metrics
import tbsim.utils.l5_utils as L5Utils
from tbsim.utils.env_utils import Plan, Action
import tbsim.algos.algo_utils as AlgoUtils
from tbsim.utils.geometry_utils import transform_points_tensor


class L5TrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, do_log=True):
        """
        Creates networks and places them into @self.nets.
        """
        super(L5TrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self._do_log = do_log
        assert modality_shapes["image"][0] == 15

        traj_decoder = MLPTrajectoryDecoder(
            feature_dim=algo_config.map_feature_dim,
            state_dim=3,
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            step_time=algo_config.step_time,
            network_kwargs=algo_config.decoder,
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
        return {"valLoss": "val/losses_prediction_loss"}

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
            losses[lk] = l * self.algo_config.loss_weights[lk]
            total_loss += losses[lk]

        metrics = self._compute_metrics(pout, batch)

        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return {
            "loss": total_loss,
            "all_losses": losses,
            "all_metrics": metrics
        }

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

    def get_plan(self, obs_dict, **kwargs):
        preds = self(obs_dict)
        plan = Plan(
            positions=preds["positions"],
            yaws=preds["yaws"],
            availabilities=torch.ones(preds["positions"].shape[:-1]).to(
                preds["positions"].device
            ),  # [B, T]
        )
        return plan, {}

    def get_action(self, obs_dict, **kwargs):
        preds = self(obs_dict)
        action = Action(
            positions=preds["positions"],
            yaws=preds["yaws"]
        )
        return action, {}


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
            network_kwargs=algo_config.decoder,
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

    def get_action(self, obs_dict, **kwargs):
        obs_dict = dict(obs_dict)
        if "plan" in kwargs:
            plan = kwargs["plan"]
            assert isinstance(plan, Plan)
            obs_dict["target_positions"] = plan.positions
            obs_dict["target_yaws"] = plan.yaws
            obs_dict["target_availabilities"] = plan.availabilities
        preds = self(obs_dict)
        action = Action(
            positions=preds["positions"],
            yaws=preds["yaws"]
        )
        return action, {}


class SpatialPlanner(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(SpatialPlanner, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        assert modality_shapes["image"][0] == 15

        self.nets["policy"] = RasterizedMapUNet(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            output_channel=4,  # (pixel, x_residual, y_residual, yaw)
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )

    @property
    def checkpoint_monitor_keys(self):
        keys = {"posErr": "val/metrics_goal_pos_err"}
        if self.algo_config.loss_weights.pixel_bce_loss > 0:
            keys["valBCELoss"] = "val/losses_pixel_bce_loss"
        if self.algo_config.loss_weights.pixel_ce_loss > 0:
            keys["valCELoss"] = "val/losses_pixel_ce_loss"
        return keys

    def forward(self, obs_dict, mask_drivable=False, num_samples=None):
        pred_map = self.nets["policy"](obs_dict["image"])
        assert pred_map.shape[1] == 4  # [location_logits, residual_x, residual_y, yaw]

        pred_map[:, 1:3] = torch.sigmoid(pred_map[:, 1:3])
        location_map = pred_map[:, 0]

        # get normalized probability map
        location_prob_map = torch.softmax(location_map.flatten(1), dim=1).reshape(location_map.shape)

        if mask_drivable:
            # At test time: optionally mask out undrivable regions
            drivable_map = L5Utils.get_drivable_region_map(obs_dict["image"])
            for i, m in enumerate(drivable_map):
                if m.sum() == 0:  # if nowhere is drivable, set it to all True's to avoid decoding problems
                    drivable_map[i] = True

            location_prob_map = location_prob_map * drivable_map.float()

        # decode map as predictions
        pixel_pred, res_pred, yaw_pred, pred_logit = AlgoUtils.decode_spatial_prediction(
            prob_map=location_prob_map,
            residual_yaw_map=pred_map[:, 1:],
            num_samples=num_samples
        )

        # transform prediction to agent coordinate
        pos_pred = transform_points_tensor(
            (pixel_pred + res_pred),
            obs_dict["agent_from_raster"].float()
        )

        return dict(
            predictions=dict(
                positions=pos_pred,
                yaws=yaw_pred
            ),
            confidence=torch.sigmoid(pred_logit),
            spatial_prediction=pred_map,
            location_map=location_map,
            location_prob_map=location_prob_map
        )

    def _compute_metrics(self, pred_batch, data_batch):
        metrics = dict()
        goal_sup = data_batch["goal"]
        goal_pred = TensorUtils.squeeze(pred_batch["predictions"], dim=1)

        pos_norm_err = torch.norm(
            goal_pred["positions"] - goal_sup["goal_position"], dim=-1
        )
        metrics["goal_pos_err"] = torch.mean(pos_norm_err)

        metrics["goal_yaw_err"] = torch.mean(
            torch.abs(goal_pred["yaws"] - goal_sup["goal_yaw"])
        )

        pixel_pred = torch.argmax(
            torch.flatten(pred_batch["location_map"], start_dim=1), dim=1
        )  # [B]
        metrics["goal_selection_err"] = torch.mean(
            (goal_sup["goal_position_pixel_flat"].long() != pixel_pred).float()
        )
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
        pout = self.forward(batch)
        batch["goal"] = AlgoUtils.get_spatial_goal_supervision(batch)
        losses = self._compute_losses(pout, batch)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        with torch.no_grad():
            metrics = self._compute_metrics(pout, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        pout = self(batch)
        batch["goal"] = AlgoUtils.get_spatial_goal_supervision(batch)
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

    def get_plan(self, obs_dict, mask_drivable=False, sample=False, **kwargs):
        num_samples = 1 if sample else None
        preds = self.forward(obs_dict, mask_drivable=mask_drivable, num_samples=num_samples)
        plan_preds = TensorUtils.squeeze(preds["predictions"], dim=1)
        plan_dict = dict(
            predictions=TensorUtils.to_sequence(plan_preds),
            availabilities=torch.ones(plan_preds["positions"].shape[0], 1).to(self.device),
        )
        # pad plans to the same size as the future trajectories
        n_steps_to_pad = self.algo_config.future_num_frames - 1
        plan_dict = TensorUtils.pad_sequence(plan_dict, padding=(n_steps_to_pad, 0), batched=True, pad_values=0.)
        plan = Plan(
            positions=plan_dict["predictions"]["positions"],
            yaws=plan_dict["predictions"]["yaws"],
            availabilities=plan_dict["availabilities"]
        )

        return plan, dict(location_map=preds["location_map"])


class L5VAETrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(L5VAETrafficModel, self).__init__()
        assert modality_shapes["image"][0] == 15

        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = RasterizedVAEModel(
            algo_config=algo_config,
            modality_shapes=modality_shapes,
            weights_scaling=[1.0, 1.0, 1.0],
        )

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}

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

    def get_action(self, obs_dict, sample=True, **kwargs):
        if sample:
            preds = self.nets["policy"].sample(obs_dict, n=1)["predictions"]  # [B, 1, T, 3]
            preds = TensorUtils.squeeze(preds, dim=1)
        else:
            preds = self.nets["policy"].predict(obs_dict)["predictions"]

        # get trajectory samples for visualization purposes
        action = Action(
            positions=preds["positions"],
            yaws=preds["yaws"]
        )
        return action, dict()


class L5TransformerTrafficModel(pl.LightningModule):
    def __init__(self, algo_config):
        """
        Creates networks and places them into @self.nets.
        """
        super(L5TransformerTrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = TransformerModel(algo_config)
        device = TorchUtils.get_torch_device(algo_config.try_to_use_cuda)
        self.nets["policy"].to(device)
        self.rasterizer = None

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}

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

        pout = self.nets["policy"](batch, batch_idx)
        losses = self.nets["policy"].compute_losses(pout, batch)
        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)

        total_loss = 0.0
        for v in losses.values():
            total_loss += v

        metrics = self._compute_metrics(pout["predictions"], batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m, prog_bar=False)
        tqdm_dict = {"g_loss": total_loss}
        output = OrderedDict(
            {"loss": total_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        return output

    def validation_step(self, batch, batch_idx):
        pout = self.nets["policy"](batch, batch_idx)
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
        preds = self(obs_dict)
        action = Action(
            positions=preds["positions"],
            yaws=preds["yaws"]
        )
        return action, {}

    def _compute_metrics(self, predictions, batch):
        metrics = {}
        preds = TensorUtils.to_numpy(predictions["positions"])
        gt = TensorUtils.to_numpy(batch["target_positions"])
        avail = TensorUtils.to_numpy(batch["target_availabilities"])

        ade = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, preds, avail
        )
        fde = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, preds, avail
        )

        metrics["ego_ADE"] = np.mean(ade)
        metrics["ego_FDE"] = np.mean(fde)
        return metrics


class L5TransformerGANTrafficModel(pl.LightningModule):
    def __init__(self, algo_config):
        """
        Creates networks and places them into @self.nets.
        """
        super(L5TransformerGANTrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = TransformerModel(algo_config)
        device = TorchUtils.get_torch_device(algo_config.try_to_use_cuda)
        self.nets["policy"].to(device)
        self.rasterizer = None

    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]

    def _compute_metrics(self, pout, batch):
        predictions = pout["predictions"]
        metrics = {}
        preds = TensorUtils.to_numpy(predictions["positions"])
        gt = TensorUtils.to_numpy(batch["target_positions"])
        avail = TensorUtils.to_numpy(batch["target_availabilities"])

        ade = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, preds, avail
        )
        fde = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, preds, avail
        )

        metrics["ego_ADE"] = np.mean(ade)
        metrics["ego_FDE"] = np.mean(fde)
        metrics["positive_likelihood"] = TensorUtils.to_numpy(
            pout["scene_predictions"]["likelihood"]
        ).mean()
        metrics["negative_likelihood"] = TensorUtils.to_numpy(
            pout["scene_predictions"]["likelihood_new"]
        ).mean()
        return metrics

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

            optimizer_idx (int): index of the optimizer, 0 for discriminator and 1 for generator

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        pout = self.nets["policy"](batch, batch_idx)

        # adversarial loss is binary cross-entropy
        if optimizer_idx == 0:
            real_label = torch.ones_like(pout["scene_predictions"]["likelihood"])
            fake_label = torch.zeros_like(pout["scene_predictions"]["likelihood_new"])
            d_loss_real = self.adversarial_loss(
                pout["scene_predictions"]["likelihood"], real_label
            )
            d_loss_fake = self.adversarial_loss(
                pout["scene_predictions"]["likelihood_new"], fake_label
            )
            d_loss = d_loss_real + d_loss_fake
            # if batch_idx % 200 == 0:
            #     print("positive:", pout["likelihood"][0:5])
            #     print("negative:", pout["likelihood_new"][0:5])
            return d_loss
        if optimizer_idx == 1:

            losses = self.nets["policy"].compute_losses(pout, batch)

            for lk, l in losses.items():
                self.log("train/losses_" + lk, l)

            g_loss = 0.0
            for v in losses.values():
                g_loss += v

            g_loss += (
                torch.mean(1.0 - pout["scene_predictions"]["likelihood_new"])
                * self.algo_config.GAN_weight
            )
            metrics = self._compute_metrics(pout, batch)
            for mk, m in metrics.items():
                self.log("train/metrics_" + mk, m, prog_bar=False)
            return g_loss

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
        optim_params = self.algo_config.optim_params
        optim_generator = TorchUtils.optimizer_from_optim_params(
            net_optim_params=optim_params["policy"],
            net=self.nets["policy"].Transformermodel,
        )

        optim_params_discriminator = self.algo_config.optim_params_discriminator
        optim_discriminator = TorchUtils.optimizer_from_optim_params(
            net_optim_params=optim_params_discriminator,
            net=self.nets["policy"].Discriminator,
        )
        return [optim_discriminator, optim_generator], []

    def get_action(self, obs_dict, **kwargs):
        preds = self(obs_dict)
        action = Action(
            positions=preds["positions"],
            yaws=preds["yaws"]
        )
        return action, {}
