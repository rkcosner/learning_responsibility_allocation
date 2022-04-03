from typing import Dict, List
from collections import OrderedDict

import torch
import torch.nn as nn

import tbsim.models.base_models as base_models
import tbsim.models.vaes as vaes
import tbsim.utils.l5_utils as L5Utils
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.loss_utils import (
    trajectory_loss,
    goal_reaching_loss,
    collision_loss
)


class RasterizedPlanningModel(nn.Module):
    """Raster-based model for planning.
    """

    def __init__(
            self,
            model_arch: str,
            input_image_shape,
            map_feature_dim: int,
            weights_scaling: List[float],
            trajectory_decoder: nn.Module,
            use_spatial_softmax=False,
            spatial_softmax_kwargs=None,
    ) -> None:

        super().__init__()
        self.map_encoder = base_models.RasterizedMapEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,
            feature_dim=map_feature_dim,
            use_spatial_softmax=use_spatial_softmax,
            spatial_softmax_kwargs=spatial_softmax_kwargs,
            output_activation=nn.ReLU
        )
        self.traj_decoder = trajectory_decoder
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch["image"]
        map_feat = self.map_encoder(image_batch)

        if self.traj_decoder.dyn is not None:
            curr_states = L5Utils.get_current_states(data_batch, dyn_type=self.traj_decoder.dyn.type())
        else:
            curr_states = None
        dec_output = self.traj_decoder.forward(inputs=map_feat, current_states=curr_states)
        traj = dec_output["trajectories"]

        pred_positions = traj[:, :, :2]
        pred_yaws = traj[:, :, 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }
        if self.traj_decoder.dyn is not None:
            out_dict["controls"] = dec_output["controls"]
            out_dict["curr_states"] = curr_states
        return out_dict

    def compute_losses(self, pred_batch, data_batch):
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        pred_loss = trajectory_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=data_batch["target_availabilities"],
            weights_scaling=self.weights_scaling
        )
        goal_loss = goal_reaching_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=data_batch["target_availabilities"],
            weights_scaling=self.weights_scaling
        )

        # compute collision loss
        pred_edges = L5Utils.get_edges_from_batch(
            data_batch=data_batch,
            ego_predictions=pred_batch["predictions"]
        )

        coll_loss = collision_loss(pred_edges=pred_edges)
        losses = OrderedDict(
            prediction_loss=pred_loss,
            goal_loss=goal_loss,
            collision_loss=coll_loss
        )
        if self.traj_decoder.dyn is not None:
            losses["yaw_reg_loss"] = torch.mean(pred_batch["controls"][..., 1] ** 2)
        return losses


class RasterizedGCModel(RasterizedPlanningModel):
    def __init__(
            self,
            model_arch: str,
            input_image_shape: int,
            map_feature_dim: int,
            goal_feature_dim: int,
            weights_scaling: List[float],
            trajectory_decoder: nn.Module,
            use_spatial_softmax=False,
            spatial_softmax_kwargs=None,
    ) -> None:
        super(RasterizedGCModel, self).__init__(
            model_arch=model_arch,
            input_image_shape=input_image_shape,
            map_feature_dim=map_feature_dim,
            weights_scaling=weights_scaling,
            trajectory_decoder=trajectory_decoder,
            use_spatial_softmax=use_spatial_softmax,
            spatial_softmax_kwargs=spatial_softmax_kwargs,
        )

        self.goal_encoder = base_models.MLP(
            input_dim=trajectory_decoder.state_dim,
            output_dim=goal_feature_dim,
            output_activation=nn.ReLU
        )

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch["image"]
        map_feat = self.map_encoder(image_batch)
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        goal_inds = L5Utils.get_last_available_index(data_batch["target_availabilities"])
        goal_state = torch.gather(
            target_traj,  # [B, T, 3]
            dim=1,
            index=goal_inds[:, None, None].expand(-1, 1, target_traj.shape[-1])
        ).squeeze(1)  # -> [B, 3]
        goal_feat = self.goal_encoder(goal_state) # -> [B, D]
        input_feat = torch.cat((map_feat, goal_feat), dim=-1)

        if self.traj_decoder.dyn is not None:
            curr_states = L5Utils.get_current_states(data_batch, dyn_type=self.traj_decoder.dyn.type())
        else:
            curr_states = None
        preds = self.traj_decoder.forward(inputs=input_feat, current_states=curr_states)

        traj = preds["trajectories"]
        pred_positions = traj[:, :, :2]
        pred_yaws = traj[:, :, 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }
        if self.traj_decoder.dyn is not None:
            out_dict["controls"] = preds["controls"]
        return out_dict


class RasterizedGANModel(nn.Module):
    """
    GAN-based latent variable model (e.g., social GAN)
    """

    def __init__(self, algo_config, modality_shapes, weights_scaling):
        super().__init__()
        trajectory_shape = (algo_config.future_num_frames, 3)

        self.map_encoder = base_models.RasterizedMapEncoder(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],
            feature_dim=algo_config.map_feature_dim,
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs
        )
        self.traj_encoder = base_models.RNNTrajectoryEncoder(
            trajectory_dim=3,
            rnn_hidden_size=algo_config.traj_encoder.rnn_hidden_size,
            feature_dim=algo_config.traj_encoder.feature_dim,
            mlp_layer_dims=algo_config.traj_encoder.mlp_layer_dims
        )
        self.traj_decoder = base_models.MLPTrajectoryDecoder(
            feature_dim=algo_config.map_feature_dim + algo_config.gan.latent_dim,
            state_dim=trajectory_shape[-1],
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            network_kwargs=algo_config.decoder,
            step_time=algo_config.step_time
        )
        self.gan_disc = base_models.MLP(
            input_dim=algo_config.map_feature_dim + algo_config.traj_encoder.feature_dim,
            output_dim=1,
            layer_dims=algo_config.gan.disc_layer_dims
        )

        self.generator_mods = nn.ModuleList(modules=[self.map_encoder, self.traj_encoder, self.traj_decoder])
        self.discriminator_mods = nn.ModuleList(modules=[self.map_encoder, self.traj_encoder, self.gan_disc])

        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)
        self.algo_config = algo_config

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch["image"]
        map_feat = self.map_encoder(image_batch)

        if self.traj_decoder.dyn is not None:
            curr_states = L5Utils.get_current_states(data_batch, dyn_type=self.traj_decoder.dyn.type())
        else:
            curr_states = None

        gan_noise = torch.randn(image_batch.shape[0], self.algo_config.gan.latent_dim).to(image_batch.device)
        input_feats = torch.cat((map_feat, gan_noise), dim=-1)
        dec_output = self.traj_decoder.forward(inputs=input_feats, current_states=curr_states)
        traj = dec_output["trajectories"]

        pred_positions = traj[:, :, :2]
        pred_yaws = traj[:, :, 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }
        if self.traj_decoder.dyn is not None:
            out_dict["controls"] = dec_output["controls"]

        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        pred_traj_feats = self.traj_encoder(traj)
        pred_score = self.gan_disc(torch.cat((map_feat, pred_traj_feats), dim=-1))
        out_dict["gen_score"] = torch.sigmoid(pred_score).squeeze(-1)

        pred_traj_feats = self.traj_encoder(traj.detach())
        real_traj_feats = self.traj_encoder(target_traj)
        disc_pred_score = self.gan_disc(torch.cat((map_feat, pred_traj_feats), dim=-1))
        disc_real_score = self.gan_disc(torch.cat((map_feat, real_traj_feats), dim=-1))

        out_dict["disc_pred_score"] = torch.sigmoid(disc_pred_score).squeeze(-1)
        out_dict["disc_real_score"] = torch.sigmoid(disc_real_score).squeeze(-1)

        return out_dict

    def sample(self, data_batch, n):
        image_batch = data_batch["image"]
        map_feat = self.map_encoder(image_batch)
        map_feat = TensorUtils.repeat_by_expand_at(map_feat, repeats=n, dim=0)

        if self.traj_decoder.dyn is not None:
            curr_states = L5Utils.get_current_states(data_batch, dyn_type=self.traj_decoder.dyn.type())
            curr_states = TensorUtils.repeat_by_expand_at(curr_states, repeats=n, dim=0)
        else:
            curr_states = None

        gan_noise = torch.randn(map_feat.shape[0], self.algo_config.gan.latent_dim).to(image_batch.device)
        input_feats = torch.cat((map_feat, gan_noise), dim=-1)
        dec_output = self.traj_decoder.forward(inputs=input_feats, current_states=curr_states)
        traj = dec_output["trajectories"]

        pred_positions = traj[:, :, :2]
        pred_yaws = traj[:, :, 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }

        return TensorUtils.reshape_dimensions(out_dict, begin_axis=0, end_axis=1, target_dims=(image_batch.shape[0], n))

    def compute_losses(self, pred_batch, data_batch):
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        pred_loss = trajectory_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=data_batch["target_availabilities"],
            weights_scaling=self.weights_scaling
        )

        # compute gan loss
        if self.algo_config.gan.loss_type == "gan":
            adv_loss_func = nn.BCELoss()
        elif self.algo_config.gan.loss_type == "lsgan":
            adv_loss_func = nn.MSELoss()
        else:
            raise Exception("GAN loss {} is not supported".format(self.algo_config.gan.loss_type))

        device = target_traj.device
        valid = torch.ones(target_traj.shape[0]).to(device)
        fake = torch.zeros(target_traj.shape[0]).to(device)
        gen_loss = adv_loss_func(pred_batch["gen_score"], valid)
        real_loss = adv_loss_func(pred_batch["disc_real_score"], valid)
        fake_loss = adv_loss_func(pred_batch["disc_pred_score"], fake)
        disc_loss = (real_loss + fake_loss) / 2

        losses = OrderedDict(
            prediction_loss=pred_loss,
            gan_gen_loss=gen_loss,
            gan_disc_loss=disc_loss
        )
        if self.traj_decoder.dyn is not None:
            losses["yaw_reg_loss"] = torch.mean(pred_batch["controls"][..., 1] ** 2)
        return losses


class RasterizedVAEModel(nn.Module):
    def __init__(self, algo_config, modality_shapes, weights_scaling):
        super(RasterizedVAEModel, self).__init__()
        trajectory_shape = (algo_config.future_num_frames, 3)
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)
        prior = vaes.FixedGaussianPrior(latent_dim=algo_config.vae.latent_dim)

        goal_dim = 0 if not algo_config.goal_conditional else algo_config.goal_feature_dim

        map_encoder = base_models.RasterizedMapEncoder(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],
            feature_dim=algo_config.map_feature_dim,
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )

        traj_decoder = base_models.MLPTrajectoryDecoder(
            feature_dim=algo_config.vae.condition_dim + goal_dim + algo_config.vae.latent_dim,
            state_dim=trajectory_shape[-1],
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            network_kwargs=algo_config.decoder,
            step_time=algo_config.step_time
        )

        if algo_config.goal_conditional:
            goal_encoder = base_models.MLP(
                input_dim=traj_decoder.state_dim,
                output_dim=algo_config.goal_feature_dim,
                output_activation=nn.ReLU
            )
        else:
            goal_encoder = None

        c_encoder = base_models.ConditionEncoder(
            map_encoder=map_encoder,
            trajectory_shape=trajectory_shape,
            condition_dim=algo_config.vae.condition_dim,
            mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            goal_encoder=goal_encoder,
            rnn_hidden_size=algo_config.vae.encoder.rnn_hidden_size
        )

        q_encoder = base_models.PosteriorEncoder(
            condition_dim=algo_config.vae.condition_dim + goal_dim,
            trajectory_shape=trajectory_shape,
            output_shapes=prior.posterior_param_shapes,
            mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            rnn_hidden_size=algo_config.vae.encoder.rnn_hidden_size
        )

        decoder = base_models.ConditionDecoder(traj_decoder)

        self.vae = vaes.CVAE(
            q_net=q_encoder,
            c_net=c_encoder,
            decoder=decoder,
            prior=prior
        )

        self.dyn = traj_decoder.dyn

    def _traj_to_preds(self, traj):
        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]
        return {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }

    def _get_goal_states(self, data_batch) -> torch.Tensor:
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=-1)
        goal_inds = L5Utils.get_last_available_index(data_batch["target_availabilities"])  # [B]
        goal_state = torch.gather(
            target_traj,  # [B, T, 3]
            dim=1,
            index=goal_inds[:, None, None].expand(-1, 1, target_traj.shape[-1])  # [B, 1, 3]
        ).squeeze(1)  # -> [B, 3]
        return goal_state

    def forward(self, batch_inputs: dict):
        trajectories = torch.cat((batch_inputs["target_positions"], batch_inputs["target_yaws"]), dim=-1)
        inputs = OrderedDict(trajectories=trajectories)
        condition_inputs = OrderedDict(image=batch_inputs["image"], goal=self._get_goal_states(batch_inputs))

        decoder_kwargs = dict()
        if self.dyn is not None:
            decoder_kwargs["current_states"] = L5Utils.get_current_states(batch_inputs, self.dyn.type())

        outs = self.vae.forward(inputs=inputs, condition_inputs=condition_inputs, decoder_kwargs=decoder_kwargs)
        outs.update(self._traj_to_preds(outs["x_recons"]["trajectories"]))
        if self.dyn is not None:
            outs["controls"] = outs["x_recons"]["controls"]
        return outs

    def sample(self, batch_inputs: dict, n: int):
        condition_inputs = OrderedDict(image=batch_inputs["image"], goal=self._get_goal_states(batch_inputs))

        decoder_kwargs = dict()
        if self.dyn is not None:
            curr_states = L5Utils.get_current_states(batch_inputs, self.dyn.type())
            decoder_kwargs["current_states"] = TensorUtils.repeat_by_expand_at(curr_states, repeats=n, dim=0)

        outs = self.vae.sample(condition_inputs=condition_inputs, n=n, decoder_kwargs=decoder_kwargs)
        return self._traj_to_preds(outs["trajectories"])

    def predict(self, batch_inputs: dict):
        condition_inputs = OrderedDict(image=batch_inputs["image"], goal=self._get_goal_states(batch_inputs))

        decoder_kwargs = dict()
        if self.dyn is not None:
            decoder_kwargs["current_states"] = L5Utils.get_current_states(batch_inputs, self.dyn.type())

        outs = self.vae.predict(condition_inputs=condition_inputs, decoder_kwargs=decoder_kwargs)
        return self._traj_to_preds(outs["trajectories"])

    def compute_losses(self, pred_batch, data_batch):
        kl_loss = self.vae.compute_kl_loss(pred_batch)
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        pred_loss = trajectory_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=data_batch["target_availabilities"],
            weights_scaling=self.weights_scaling
        )
        losses = OrderedDict(prediction_loss=pred_loss, kl_loss=kl_loss)
        if self.dyn is not None:
            losses["yaw_reg_loss"] = torch.mean(pred_batch["controls"][..., 1] ** 2)
        return losses

