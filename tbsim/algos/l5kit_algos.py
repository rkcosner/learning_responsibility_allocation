from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl

from tbsim.models.l5kit_models import RasterizedPlanningModel
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

import tbsim.utils.metrics as Metrics
from tbsim.algos.base import PolicyAlgo


class L5TrafficModel(PolicyAlgo):

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = RasterizedPlanningModel(
            model_arch=self.algo_config.model_architecture,
            num_input_channels=self.modality_shapes["image"][0], # [C, H, W]
            num_targets=3 * self.algo_config.future_num_frames,  # X, Y, Yaw * number of future states,
            weights_scaling=[1., 1., 1.],
            criterion=nn.MSELoss(reduction="none")
        )

    def process_batch_for_training(self, batch):
        return TensorUtils.to_device(TensorUtils.to_float(batch), self.device)

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["prediction_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def train_on_batch(self, batch, step, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            step (int): training step number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(L5TrafficModel, self).train_on_batch(batch, step, validate=validate)
            pout = self.nets["policy"](batch)
            losses = self.nets["policy"].compute_losses(pout, batch)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

            info["losses"] = TensorUtils.detach(losses)

            # TODO: metric compute needs to go somewhere else
            preds = TensorUtils.to_numpy(pout["predictions"]["positions"])
            gt = TensorUtils.to_numpy(batch["target_positions"])
            conf = np.ones((preds.shape[0], 1))
            avail = TensorUtils.to_numpy(batch["target_availabilities"])

            preds = preds[:, None]

            ade = Metrics.batch_average_displacement_error(gt, preds, conf, avail, mode="oracle")
            fde = Metrics.batch_final_displacement_error(gt, preds, conf, avail, mode="oracle")

            info["ADE"] = ade
            info["FDE"] = fde

        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(PolicyAlgo, self).log_info(info)
        log["Loss"] = info["losses"]["prediction_loss"].item()
        log["Metrics_ADE"] = info["ADE"]
        log["Metrics_FDE"] = info["FDE"]
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict):
        pass


class L5TrafficModelPL(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        """
        Creates networks and places them into @self.nets.
        """
        super(L5TrafficModelPL, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = RasterizedPlanningModel(
            model_arch=self.algo_config.model_architecture,
            num_input_channels=modality_shapes["image"][0], # [C, H, W]
            num_targets=3 * self.algo_config.future_num_frames,  # X, Y, Yaw * number of future states,
            weights_scaling=[1., 1., 1.],
            criterion=nn.MSELoss(reduction="none")
        )

    def process_batch_for_training(self, batch):
        return TensorUtils.to_device(TensorUtils.to_float(batch), self.device)

    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]

    def _compute_metrics(self, predictions, batch):
        metrics = {}
        preds = TensorUtils.to_numpy(predictions["positions"])
        gt = TensorUtils.to_numpy(batch["target_positions"])
        conf = np.ones((preds.shape[0], 1))
        avail = TensorUtils.to_numpy(batch["target_availabilities"])

        preds = preds[:, None]

        ade = Metrics.batch_average_displacement_error(gt, preds, conf, avail, mode="oracle")
        fde = Metrics.batch_final_displacement_error(gt, preds, conf, avail, mode="oracle")

        metrics["ADE"] = np.mean(ade)
        metrics["FDE"] = np.mean(fde)
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
        for lk, l in losses.items():
            self.log("train/losses_" + lk, l, on_step=True, logger=True)

        total_loss = 0.
        for v in losses.values():
            total_loss += v

        metrics = self._compute_metrics(pout["predictions"], batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m, on_step=True, prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        metrics =self._compute_metrics(pout["predictions"], batch)
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
        optim = TorchUtils.optimizer_from_optim_params(
            net_optim_params=optim_params["policy"], net=self.nets["policy"]
        )
        return optim

    def get_action(self, obs_dict):
        pass
