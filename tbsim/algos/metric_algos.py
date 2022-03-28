from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from tbsim.models.learned_metrics import PermuteEBM
import tbsim.utils.tensor_utils as TensorUtils



class EBMMetric(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, do_log=True):
        """
        Creates networks and places them into @self.nets.
        """
        super(EBMMetric, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self._do_log = do_log
        assert modality_shapes["image"][0] == 15

        self.nets["ebm"] = PermuteEBM(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            map_feature_dim=algo_config.map_feature_dim,
            traj_feature_dim=algo_config.traj_feature_dim,
            embedding_dim=algo_config.embedding_dim,
            embed_layer_dims=algo_config.embed_layer_dims,
        )

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_infoNCE_loss"}

    def forward(self, obs_dict):
        return self.nets["ebm"](obs_dict)

    def _compute_metrics(self, pred_batch, data_batch):
        scores = pred_batch["scores"]
        pred_inds = torch.argmax(scores, dim=1)
        gt_inds = torch.arange(scores.shape[0]).to(scores.device)
        cls_acc = torch.mean((pred_inds == gt_inds).float()).item()

        return dict(cls_acc=cls_acc)

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
        pout = self.nets["ebm"](batch)
        losses = self.nets["ebm"].compute_losses(pout, batch)
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
        pout = self.nets["ebm"](batch)
        losses = TensorUtils.detach(self.nets["ebm"].compute_losses(pout, batch))
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

    def get_metrics(self, obs_dict):
        preds = self.forward(obs_dict)
        return dict(
            scores=preds["scores"]
        )
