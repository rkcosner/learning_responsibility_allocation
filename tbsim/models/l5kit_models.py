from typing import Dict, List
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

import tbsim.models.base_models as base_models
import tbsim.models.vaes as vaes
import tbsim.dynamics as dynamics
import tbsim.utils.l5_utils as L5Utils
from tbsim.utils.loss_utils import (
    trajectory_loss,
    goal_reaching_loss,
    collision_loss
)


def optimize_trajectories(
        init_u,
        init_x,
        target_trajs,
        target_avails,
        dynamics_model,
        step_time: float,
        data_batch = None,
        goal_loss_weight=1.0,
        traj_loss_weight=0.0,
        coll_loss_weight=0.0,
        num_optim_iterations: int = 50
):
    curr_u = init_u.detach().clone()
    curr_u.requires_grad = True
    action_optim = optim.LBFGS([curr_u], max_iter=20, lr=1.0, line_search_fn='strong_wolfe')

    for oidx in range(num_optim_iterations):
        def closure():
            action_optim.zero_grad()

            # get trajectory with current params
            _, pos, yaw = dynamics.forward_dynamics(
                dyn_model=dynamics_model,
                initial_states=init_x,
                actions=curr_u,
                step_time=step_time
            )
            curr_trajs = torch.cat((pos, yaw), dim=-1)
            # compute trajectory optimization losses
            losses = dict()
            losses["goal_loss"] = goal_reaching_loss(
                predictions=curr_trajs,
                targets=target_trajs,
                availabilities=target_avails
            ) * goal_loss_weight
            losses["traj_loss"] = trajectory_loss(
                predictions=curr_trajs,
                targets=target_trajs,
                availabilities=target_avails
            ) * traj_loss_weight
            if coll_loss_weight > 0:
                assert data_batch is not None
                coll_edges = L5Utils.get_edges_from_batch(
                    data_batch,
                    ego_predictions=dict(positions=pos, yaws=yaw)
                )
                for c in coll_edges:
                    coll_edges[c] = coll_edges[c][:, :target_trajs.shape[-2]]
                vv_edges = dict(VV=coll_edges["VV"])
                if vv_edges["VV"].shape[0] > 0:
                    losses["coll_loss"] = collision_loss(vv_edges) * coll_loss_weight

            total_loss = torch.hstack(list(losses.values())).sum()

            # backprop
            total_loss.backward()
            return total_loss
        action_optim.step(closure)

    final_raw_trajs, final_pos, final_yaw = dynamics.forward_dynamics(
        dyn_model=dynamics_model,
        initial_states=init_x,
        actions=curr_u,
        step_time=step_time
    )
    final_trajs = torch.cat((final_pos, final_yaw), dim=-1)
    losses = dict()
    losses["goal_loss"] = goal_reaching_loss(
        predictions=final_trajs,
        targets=target_trajs,
        availabilities=target_avails
    )
    losses["traj_loss"] = trajectory_loss(
        predictions=final_trajs,
        targets=target_trajs,
        availabilities=target_avails
    )

    return dict(positions=final_pos, yaws=final_yaw), final_raw_trajs, curr_u, losses


def get_current_states(batch: dict, dyn_type: dynamics.DynType) -> torch.Tensor:
    bs = batch["curr_speed"].shape[0]
    if dyn_type == dynamics.DynType.BICYCLE:
        current_states = torch.zeros(bs, 6).to(batch["curr_speed"].device)  # [x, y, yaw, vel, dh, veh_len]
        current_states[:, 3] = batch["curr_speed"].abs()
        current_states[:, [4]] = (batch["history_yaws"][:, 0] - batch["history_yaws"][:, 1]).abs()
        current_states[:, 5] = batch["extent"][:, 0]  # [l, w, h]
    else:
        current_states = torch.zeros(bs, 4).to(batch["curr_speed"].device)  # [x, y, vel, yaw]
        current_states[:, 2] = batch["curr_speed"]
    return current_states


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
            curr_states = get_current_states(data_batch, dyn_type=self.traj_decoder.dyn.type())
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
        losses = OrderedDict(prediction_loss=pred_loss, goal_loss=goal_loss, collision_loss=coll_loss)
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
        goal_state = target_traj[:, -1]
        goal_feat = self.goal_encoder(goal_state)
        input_feat = torch.cat((map_feat, goal_feat), dim=-1)

        if self.traj_decoder.dyn is not None:
            curr_states = get_current_states(data_batch, dyn_type=self.traj_decoder.dyn.type())
        else:
            curr_states = None
        traj = self.traj_decoder.forward(inputs=input_feat, current_states=curr_states)["trajectories"]

        pred_positions = traj[:, :, :2]
        pred_yaws = traj[:, :, 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }
        return out_dict


class RasterizedVAEModel(nn.Module):
    def __init__(self, algo_config, modality_shapes, weights_scaling):
        super(RasterizedVAEModel, self).__init__()
        trajectory_shape = (algo_config.future_num_frames, 3)
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)
        prior = vaes.FixedGaussianPrior(latent_dim=algo_config.vae.latent_dim)

        map_encoder = base_models.RasterizedMapEncoder(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],
            feature_dim=algo_config.map_feature_dim,
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )

        c_encoder = base_models.ConditionEncoder(
            map_encoder=map_encoder,
            trajectory_shape=trajectory_shape,
            condition_dim=algo_config.vae.condition_dim,
            mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            rnn_hidden_size=algo_config.vae.encoder.rnn_hidden_size
        )

        q_encoder = base_models.PosteriorEncoder(
            condition_dim=algo_config.vae.condition_dim,
            trajectory_shape=trajectory_shape,
            output_shapes=prior.posterior_param_shapes,
            mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            rnn_hidden_size=algo_config.vae.encoder.rnn_hidden_size
        )

        traj_decoder = base_models.MLPTrajectoryDecoder(
            feature_dim=algo_config.vae.condition_dim + algo_config.vae.latent_dim,
            state_dim=trajectory_shape[-1],
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            network_kwargs=algo_config.decoder
        )

        decoder = base_models.ConditionDecoder(traj_decoder)

        self.vae = vaes.CVAE(
            q_net=q_encoder,
            c_net=c_encoder,
            decoder=decoder,
            prior=prior
        )

    def _traj_to_preds(self, traj):
        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]
        return {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }

    def forward(self, batch_inputs: dict):
        trajectories = torch.cat((batch_inputs["target_positions"], batch_inputs["target_yaws"]), dim=-1)
        inputs = OrderedDict(trajectories=trajectories)
        condition_inputs = OrderedDict(image=batch_inputs["image"])
        outs = self.vae(inputs=inputs, condition_inputs=condition_inputs)
        outs.update(self._traj_to_preds(outs["x_recons"]["trajectories"]))
        return outs

    def sample(self, batch_inputs: dict, n: int):
        condition_inputs = OrderedDict(image=batch_inputs["image"])
        outs = self.vae.sample(condition_inputs=condition_inputs, n=n)
        return self._traj_to_preds(outs["trajectories"])

    def predict(self, batch_inputs: dict):
        condition_inputs = OrderedDict(image=batch_inputs["image"])
        outs = self.vae.predict(condition_inputs=condition_inputs)
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
        return losses

