from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from tbsim.models.l5kit_models import RasterizedPlanningModel
from tbsim.models.transformer_model import TransformerModel
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.torch_utils as TorchUtils
import tbsim.utils.metrics as Metrics
import pdb


class L5TrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        """
        Creates networks and places them into @self.nets.
        """
        super(L5TrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = RasterizedPlanningModel(
            model_arch=self.algo_config.model_architecture,
            num_input_channels=modality_shapes["image"][0],  # [C, H, W]
            num_targets=3
            * self.algo_config.future_num_frames,  # X, Y, Yaw * number of future states,
            weights_scaling=[1.0, 1.0, 1.0],
            criterion=nn.MSELoss(reduction="none"),
        )

    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]

    def _compute_metrics(self, predictions, batch):
        metrics = {}
        # preds = # TODO
        # gt = # TODO
        # avail = TensorUtils.to_numpy(batch["target_availabilities"])
        #
        # ade = Metrics.single_mode_metrics(
        #     Metrics.batch_average_displacement_error, gt, preds, avail
        # )
        # fde = Metrics.single_mode_metrics(
        #     Metrics.batch_final_displacement_error, gt, preds, avail
        # )
        #
        # metrics["ADE"] = np.mean(ade)
        # metrics["FDE"] = np.mean(fde)
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
            self.log("train/losses_" + lk, l)

        total_loss = 0.0
        for v in losses.values():
            total_loss += v

        # metrics = self._compute_metrics(pout["predictions"], batch)
        # for mk, m in metrics.items():
        #     self.log("train/metrics_" + mk, m, prog_bar=True)

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
        optim_params = self.algo_config.optim_params
        optim = TorchUtils.optimizer_from_optim_params(
            net_optim_params=optim_params["policy"], net=self.nets["policy"]
        )
        return optim

    def get_action(self, obs_dict):
        return {"ego": self(obs_dict["ego"])}


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

    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]

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
        optim_params = self.algo_config.optim_params
        optim = TorchUtils.optimizer_from_optim_params(
            net_optim_params=optim_params["policy"], net=self.nets["policy"]
        )
        return optim

    def get_action(self, obs_dict):
        return {"ego": self(obs_dict["ego"])}

    # def sim_step(self, env, obs_dict):
    #     action = {"ego": self(obs_dict["ego"])}


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
            batch_size = pout["likelihood"].shape[0]
            real_label = torch.ones((batch_size, 1), device=pout["likelihood"].device)
            fake_label = torch.zeros((batch_size, 1), device=pout["likelihood"].device)
            d_loss_real = self.adversarial_loss(pout["likelihood"], real_label)
            d_loss_fake = self.adversarial_loss(pout["likelihood_new"], fake_label)
            d_loss = d_loss_real + d_loss_fake
            if batch_idx % 200 == 0:
                print("positive:", pout["likelihood"][0:5])
                print("negative:", pout["likelihood_new"][0:5])
            return d_loss
        if optimizer_idx == 1:

            losses = self.nets["policy"].compute_losses(pout, batch)

            for lk, l in losses.items():
                self.log("train/losses_" + lk, l)

            g_loss = 0.0
            for v in losses.values():
                g_loss += v

            g_loss += (
                torch.mean(1.0 - pout["likelihood_new"]) * self.algo_config.GAN_weight
            )

            metrics = self._compute_metrics(pout["predictions"], batch)
            for mk, m in metrics.items():
                self.log("train/metrics_" + mk, m, prog_bar=False)
            return g_loss

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
        optim_params = self.algo_config.optim_params
        optim = TorchUtils.optimizer_from_optim_params(
            net_optim_params=optim_params["policy"],
            net=self.nets["policy"].Transformermodel,
        )

        optim_params_discriminator = self.algo_config.optim_params_discriminator
        optim_params_discriminator = TorchUtils.optimizer_from_optim_params(
            net_optim_params=optim_params_discriminator["policy"],
            net=self.nets["policy"].summary_dec,
        )
        return [optim_params_discriminator, optim], []

    def get_action(self, obs_dict):
        return {"ego": self(obs_dict["ego"])}

    # def sim_step(self, env, obs_dict):
    #     action = {"ego": self(obs_dict["ego"])}
