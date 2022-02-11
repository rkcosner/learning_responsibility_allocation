import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from tbsim.models.l5kit_models import RasterizedPlanningModel, MLPTrajectoryDecoder, RasterizedVAEModel
from tbsim.algos.l5kit_algos import L5VAETrafficModel
from tbsim.models.transformer_model import TransformerModel
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.metrics as Metrics
from tbsim.algos.factory import algo_factory
from tbsim.utils.config_utils import get_experiment_config_from_file


class L5RasterizedJointModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(L5RasterizedJointModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()

        # load the (pretrained) ego traffic model
        ego_config = get_experiment_config_from_file(algo_config.ego_model_config_path)
        ego_algo = L5VAETrafficModel.load_from_checkpoint(
            checkpoint_path=algo_config.ego_ckpt_path,
            algo_config=ego_config.algo,
            modality_shapes=modality_shapes
        )
        self.nets["ego_policy"] = ego_algo.nets["policy"]

        traj_decoder = MLPTrajectoryDecoder(
            feature_dim=algo_config.map_feature_dim,
            state_dim=3,
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            step_time=algo_config.step_time
        )

        self.nets["agents_policy"] = RasterizedPlanningModel(
            model_arch=algo_config.model_architecture,
            num_input_channels=modality_shapes["image"][0],  # [C, H, W]
            trajectory_decoder=traj_decoder,
            map_feature_dim=algo_config.map_feature_dim,
            weights_scaling=[1.0, 1.0, 1.0]
        )

    def forward(self, obs_dict):
        return {
            "ego": self.nets["ego_policy"].predict(obs_dict["ego"])["predictions"],
            "agents": self.nets["agents_policy"].forward(obs_dict["agents"])["predictions"]
        }

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
        return metrics

    def training_step(self, batch, batch_idx):
        ego_batch = batch["ego"]
        agents_batch = batch["agents"]
        with torch.no_grad():
            # Take samples from ego policy behaviors
            samples = self.nets["ego_policy"].sample(ego_batch, n=1)

        # Override future position of the ego in the agents batch
        modified_agents_batch = dict(agents_batch)

        pout = self.nets["agents_policy"](agents_batch)
        losses = self.nets["policy"].compute_losses(pout, modified_agents_batch)
        # take samples to measure trajectory diversity
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        metrics = self._compute_metrics(pout, modified_agents_batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

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
