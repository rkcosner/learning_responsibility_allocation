import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.metrics as Metrics
import tbsim.models.l5kit_models as l5m
import tbsim.models.vaes as vaes


class L5TrafficVAE(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(L5TrafficVAE, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        trajectory_shape = (self.algo_config.future_num_frames, 3)

        prior = vaes.FixedGaussianPrior(latent_dim=algo_config.vae.latent_dim)

        map_encoder = l5m.RasterizedMapEncoder(
            model_arch=algo_config.model_architecture,
            num_input_channels=modality_shapes["image"][0],
            feature_dim=algo_config.visual_feature_dim
        )

        c_encoder = l5m.ConditionEncoder(
            map_encoder=map_encoder,
            trajectory_shape=trajectory_shape,
            condition_dim=algo_config.vae.condition_dim,
            mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            rnn_hidden_size=algo_config.vae.encoder.rnn_hidden_size
        )

        q_encoder = l5m.PosteriorEncoder(
            condition_dim=algo_config.vae.condition_dim,
            trajectory_shape=trajectory_shape,
            output_shapes=prior.posterior_param_shapes,
            mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            rnn_hidden_size=algo_config.vae.encoder.rnn_hidden_size
        )

        decoder = l5m.ConditionFlatDecoder(
            condition_dim=algo_config.vae.condition_dim,
            latent_dim=prior.latent_dim,
            output_shapes=OrderedDict(trajectories=trajectory_shape),
            mlp_layer_dims=algo_config.vae.decoder.mlp_layer_dims
        )

        model = vaes.CVAE(
            q_net=q_encoder,
            c_net=c_encoder,
            decoder=decoder,
            prior=prior,
            target_criterion=nn.MSELoss(reduction="none")
        )

        self.nets["cvae"] = model

    def forward(self, batch_inputs: dict):
        trajectories = torch.cat((batch_inputs["target_positions"], batch_inputs["target_yaws"]), dim=-1)
        inputs = OrderedDict(trajectories=trajectories)
        condition_inputs = OrderedDict(image=batch_inputs["image"])
        return self.nets["cvae"](inputs=inputs, condition_inputs=condition_inputs)

    def sample(self, batch_inputs: dict, n: int):
        condition_inputs = OrderedDict(image=batch_inputs["image"])
        return self.nets["cvae"].sample(condition_inputs=condition_inputs, n=n)

    def _compute_metrics(self, outputs, batch):
        metrics = {}
        preds = TensorUtils.to_numpy(outputs["x_recons"]["trajectories"][..., :2])
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

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        pout = self.forward(batch_inputs=batch)
        avails = batch["target_availabilities"].unsqueeze(2)   # [B, T, 1]
        trajectories = torch.cat((batch["target_positions"], batch["target_yaws"]), dim=-1)
        target_weights = OrderedDict(trajectories=torch.ones_like(trajectories) * avails)
        targets = OrderedDict(trajectories=trajectories)

        losses = self.nets["cvae"].compute_losses(
            outputs=pout,
            targets=targets,
            target_weights=target_weights,
            kl_weight=self.algo_config.vae.kl_weight
        )
        total_loss = 0.0
        for lk, l in losses.items():
            self.log("train/losses_" + lk, l, prog_bar=True)
            total_loss += l

        metrics = self._compute_metrics(pout, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        pout = self.forward(batch_inputs=batch)
        avails = batch["target_availabilities"].unsqueeze(2)   # [B, T, 1]
        trajectories = torch.cat((batch["target_positions"], batch["target_yaws"]), dim=-1)
        target_weights = OrderedDict(trajectories=torch.ones_like(trajectories) * avails)
        targets = OrderedDict(trajectories=trajectories)

        losses = self.nets["cvae"].compute_losses(
            outputs=pout,
            targets=targets,
            target_weights=target_weights,
            kl_weight=self.algo_config.vae.kl_weight
        )
        metrics = self._compute_metrics(pout, batch)
        return {"losses": TensorUtils.detach(losses), "metrics": metrics}

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

    def get_action(self, obs_dict):
        preds = self.sample(obs_dict["ego"], n=1)  # [B, 1, T, 3]
        actions = dict(
            positions=preds["trajectories"][:, 0, :, :2],
            yaws=preds["trajectories"][:, 0, :, 2:3]
        )
        return {"ego": actions}


class L5TrafficSlimVAE(L5TrafficVAE):
    """CVAE with lean memory footprint (by reusing encoded features)"""
    def __init__(self, algo_config, modality_shapes):
        pl.LightningModule.__init__(self)
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        trajectory_shape = (self.algo_config.future_num_frames, 3)

        prior = vaes.FixedGaussianPrior(latent_dim=algo_config.vae.latent_dim)

        map_encoder = l5m.RasterizedMapEncoder(
            model_arch=algo_config.model_architecture,
            num_input_channels=modality_shapes["image"][0],
            feature_dim=algo_config.visual_feature_dim
        )

        traj_encoder = l5m.RNNTrajectoryEncoder(
            trajectory_dim=trajectory_shape[-1],
            rnn_hidden_size=algo_config.vae.encoder.rnn_hidden_size
        )

        c_net = l5m.ConditionNet(
            condition_input_shapes=OrderedDict(
                map_feature=(map_encoder.output_shape()[-1],)
            ),
            condition_dim=algo_config.vae.condition_dim,
            mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims
        )

        q_net = l5m.PosteriorNet(
            input_shapes=OrderedDict(
                traj_feature=(traj_encoder.output_shape()[-1],)
            ),
            condition_dim=algo_config.vae.condition_dim,
            param_shapes=prior.posterior_param_shapes,
            mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims
        )

        decoder = l5m.ConditionFlatDecoder(
            condition_dim=algo_config.vae.condition_dim,
            latent_dim=prior.latent_dim,
            output_shapes=OrderedDict(trajectories=trajectory_shape),
            mlp_layer_dims=algo_config.vae.decoder.mlp_layer_dims
        )

        model = vaes.CVAE(
            q_net=q_net,
            c_net=c_net,
            decoder=decoder,
            prior=prior,
            target_criterion=nn.MSELoss(reduction="none")
        )

        self.nets["cvae"] = model
        self.nets["map_encoder"] = map_encoder
        self.nets["traj_encoder"] = traj_encoder

    def forward(self, batch_inputs: dict):
        trajectories = torch.cat((batch_inputs["target_positions"], batch_inputs["target_yaws"]), dim=-1)
        traj_feat = self.nets["traj_encoder"](trajectories)
        map_feat = self.nets["map_encoder"](batch_inputs["image"])
        input_feats = dict(traj_feature=traj_feat)
        condition_feats = dict(map_feature=map_feat)
        return self.nets["cvae"](inputs=input_feats, condition_inputs=condition_feats)

    def sample(self, batch_inputs: dict, n: int):
        condition_feat = dict(map_feature=self.nets["map_encoder"](batch_inputs["image"]))
        return self.nets["cvae"].sample(condition_inputs=condition_feat, n=n)