import warnings
from typing import Dict, List, Union
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

from tbsim.models.base_models import SplitMLP, MLP, MIMOMLP
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.dynamics as dynamics
import tbsim.utils.l5_utils as L5Utils


def forward_dynamics(
        dyn_model: dynamics.Dynamics,
        initial_states: torch.Tensor,
        actions: torch.Tensor,
        step_time: float
):
    """
    Integrate the state forward with initial state x0, action u
    Args:
        dyn_model (dynamics.Dynamics): dynamics model
        initial_states (Torch.tensor): state tensor of size [..., 4]
        actions (Torch.tensor): action tensor of size [..., 2]
        step_time (float): delta time between steps
    Returns:
        state tensor of size [B,Num_agent,T,4]
    """
    num_steps = actions.shape[-2]
    x = [initial_states.squeeze(-2)] + [None] * num_steps
    for t in range(num_steps):
        x[t + 1] = (
                dyn_model.step(
                    x[t], actions[..., t, :], step_time, bound=True
                )
        )

    x = torch.stack(x[1:], dim=-2)
    pos = dyn_model.state2pos(x)
    yaw = dyn_model.state2yaw(x)
    return x, pos, yaw


def trajectory_loss(predictions, targets, availabilities, weights_scaling=None, crit=nn.MSELoss(reduction="none")):
    """

    Args:
        predictions (torch.Tensor): predicted trajectory [B, (A), T, D]
        targets (torch.Tensor): target trajectory [B, (A), T, D]
        availabilities (torch.Tensor): [B, (A), T]
        weights_scaling (torch.Tensor): [D]
        crit (nn.Module): loss function

    Returns:
        loss (torch.Tensor)
    """
    assert availabilities.shape == predictions.shape[:-1]
    assert predictions.shape == targets.shape
    if weights_scaling is None:
        weights_scaling = torch.zeros(targets.shape[-1]).to(targets.device)
    assert weights_scaling.shape[-1] == targets.shape[-1]
    target_weights = (availabilities.unsqueeze(-1) * weights_scaling)
    loss = torch.mean(crit(predictions, targets) * target_weights)
    return loss


def goal_reaching_loss(predictions, targets, availabilities, weights_scaling=None, crit=nn.MSELoss(reduction="none")):
    """

    Args:
        predictions (torch.Tensor): predicted trajectory [B, (A), T, D]
        targets (torch.Tensor): target trajectory [B, (A), T, D]
        availabilities (torch.Tensor): [B, (A), T]
        weights_scaling (torch.Tensor): [D]
        crit (nn.Module): loss function

    Returns:
        loss (torch.Tensor)
    """
    # compute loss mask by finding the last available target
    num_frames = availabilities.shape[-1]
    inds = torch.arange(0, num_frames).to(targets.device)  # [T]
    inds = (availabilities > 0).float() * inds  # [B, (A), T] arange indices with unavailable indices set to 0
    last_inds = inds.max(dim=-1)[1]  # [B, (A)] calculate the index of the last availale frame
    goal_mask = TensorUtils.to_one_hot(last_inds, num_class=num_frames)  # [B, (A), T] with the last frame set to 1
    # filter out samples that do not have available frames
    available_samples_mask = availabilities.sum(-1) > 0  # [B, (A)]
    goal_mask = goal_mask * available_samples_mask.unsqueeze(-1).float()  # [B, (A), T]
    goal_loss = trajectory_loss(
        predictions,
        targets,
        availabilities=goal_mask,
        weights_scaling=weights_scaling,
        crit=crit
    )
    return goal_loss


class RasterizedPlanningModel(nn.Module):
    """Raster-based model for planning.
    """

    def __init__(
            self,
            model_arch: str,
            num_input_channels: int,
            map_feature_dim: int,
            weights_scaling: List[float],
            trajectory_decoder: nn.Module,
    ) -> None:

        super().__init__()
        self.map_encoder = RasterizedMapEncoder(
            model_arch=model_arch,
            num_input_channels=num_input_channels,
            feature_dim=map_feature_dim,
            output_activation=nn.ReLU
        )
        self.traj_decoder = trajectory_decoder
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch["image"]
        map_feat = self.map_encoder(image_batch)

        # raw_ego = L5Utils.batch_to_raw_ego(data_batch, step_time=self.traj_decoder.step_time)
        # all_states, curr_states = L5Utils.raw_to_states(*raw_ego)
        # org_curr_states = curr_states[..., 0, :]

        curr_states = torch.zeros(image_batch.shape[0], 4).to(image_batch.device)  # [x, y, vel, yaw]
        curr_states[:, 2] = data_batch["curr_speed"]
        traj = self.traj_decoder.forward(inputs=map_feat, current_state=curr_states)

        pred_positions = traj[:, :, :2]
        pred_yaws = traj[:, :, 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }
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
        losses = OrderedDict(prediction_loss=pred_loss, goal_loss=goal_loss)
        return losses


class RasterizedMapEncoder(nn.Module):
    """A basic image-based rasterized map encoder"""
    def __init__(
            self,
            model_arch: str,
            num_input_channels: int = 3,  # C
            feature_dim: int = 128,
            output_activation = nn.ReLU
    ) -> None:
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = num_input_channels
        self._feature_dim = feature_dim
        self._output_activation = output_activation

        if model_arch == "resnet18":
            self.map_model = resnet18()
            self.map_model.fc = nn.Linear(in_features=512, out_features=feature_dim)
        elif model_arch == "resnet50":
            self.map_model = resnet50()
            self.map_model.fc = nn.Linear(in_features=2048, out_features=feature_dim)
        else:
            raise NotImplementedError(f"Model arch {model_arch} unknown")

        if self.num_input_channels != 3:
            self.map_model.conv1 = nn.Conv2d(
                self.num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

    def output_shape(self, input_shape=None):
        return [self._feature_dim]

    def forward(self, map_inputs):
        feat = self.map_model(map_inputs)
        if self._output_activation is not None:
            feat = self._output_activation()(feat)
        return feat


class RNNTrajectoryEncoder(nn.Module):
    def __init__(self, trajectory_dim, rnn_hidden_size, feature_dim=None, mlp_layer_dims: tuple = ()):
        super(RNNTrajectoryEncoder, self).__init__()
        self.lstm = nn.LSTM(trajectory_dim, hidden_size=rnn_hidden_size, batch_first=True)
        if feature_dim is not None:
            self.mlp = MLP(
                input_dim=rnn_hidden_size,
                output_dim=feature_dim,
                layer_dims=mlp_layer_dims,
                output_activation=nn.ReLU
            )
            self._feature_dim = feature_dim
        else:
            self.mlp = None
            self._feature_dim = rnn_hidden_size

    def output_shape(self, input_shape=None):
        num_frame = 1 if input_shape is None else input_shape[0]
        return [num_frame, self._feature_dim]

    def forward(self, input_trajectory):
        traj_feat = self.lstm(input_trajectory)[0][:, -1, :]
        if self.mlp is not None:
            traj_feat = TensorUtils.time_distributed(traj_feat, op=self.mlp)
        return traj_feat


class PosteriorEncoder(nn.Module):
    """Posterior Encoder (x, x_c -> q) for CVAE"""
    def __init__(
            self,
            condition_dim: int,
            trajectory_shape: tuple,  # [T, D]
            output_shapes: OrderedDict,
            mlp_layer_dims: tuple = (128, 128),
            rnn_hidden_size: int = 100
    ) -> None:
        super(PosteriorEncoder, self).__init__()
        self.trajectory_shape = trajectory_shape

        # TODO: history encoder
        self.traj_encoder = RNNTrajectoryEncoder(
            trajectory_dim=trajectory_shape[-1],
            rnn_hidden_size=rnn_hidden_size
        )
        self.mlp = SplitMLP(
            input_dim=(rnn_hidden_size + condition_dim),
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims,
            output_activation=nn.ReLU
        )

    def forward(self, inputs, condition_features) -> Dict[str, torch.Tensor]:
        traj_feat = self.traj_encoder(inputs["trajectories"])
        feat = torch.cat((traj_feat, condition_features), dim=-1)
        return self.mlp(feat)


class ConditionEncoder(nn.Module):
    """Condition Encoder (x -> c) for CVAE"""
    def __init__(
            self,
            map_encoder: nn.Module,
            trajectory_shape: tuple,  # [T, D]
            condition_dim: int,
            mlp_layer_dims: tuple = (128, 128),
            rnn_hidden_size: int = 100
    ) -> None:
        super(ConditionEncoder, self).__init__()
        self.map_encoder = map_encoder
        self.trajectory_shape = trajectory_shape

        # TODO: history encoder
        # self.hist_lstm = nn.LSTM(trajectory_shape[-1], hidden_size=rnn_hidden_size, batch_first=True)
        visual_feature_size = self.map_encoder.output_shape()[0]
        self.mlp = MLP(
            input_dim=visual_feature_size,
            output_dim=condition_dim,
            layer_dims=mlp_layer_dims,
            output_activation=nn.ReLU
        )

    def forward(self, condition_inputs):
        map_feat = self.map_encoder(condition_inputs["image"])
        return self.mlp(map_feat)


class PosteriorNet(nn.Module):
    def __init__(
            self,
            input_shapes: OrderedDict,
            condition_dim: int,
            param_shapes: OrderedDict,
            mlp_layer_dims: tuple=()
    ):
        super(PosteriorNet, self).__init__()
        all_shapes = deepcopy(input_shapes)
        all_shapes["condition_features"] = (condition_dim,)
        self.mlp = MIMOMLP(
            input_shapes=all_shapes,
            output_shapes=param_shapes,
            layer_dims=mlp_layer_dims,
            output_activation=None
        )

    def forward(self, inputs: dict, condition_features: torch.Tensor):
        all_inputs = dict(inputs)
        all_inputs["condition_features"] = condition_features
        return self.mlp(all_inputs)


class ConditionNet(nn.Module):
    def __init__(
            self,
            condition_input_shapes: OrderedDict,
            condition_dim: int,
            mlp_layer_dims: tuple=()
    ):
        super(ConditionNet, self).__init__()
        self.mlp = MIMOMLP(
            input_shapes=condition_input_shapes,
            output_shapes=OrderedDict(feat=(condition_dim,)),
            layer_dims=mlp_layer_dims,
            output_activation=nn.ReLU
        )

    def forward(self, inputs: dict):
        return self.mlp(inputs)["feat"]


class ConditionDecoder(nn.Module):
    """Decoding (z, c) -> x' using a flat MLP"""
    def __init__(self, decoder_model: nn.Module):
        super(ConditionDecoder, self).__init__()
        self.decoder_model = decoder_model

    def forward(self, latents, condition_features):
        return self.decoder_model(torch.cat((latents, condition_features), dim=-1))


class TrajectoryDecoder(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            state_dim: int = 3,
            num_steps: int = None,
            dynamics_type: Union[str, dynamics.DynType] = None,
            dynamics_kwargs: dict = None,
            step_time: float = None,
            network_kwargs: dict = None
    ):
        """
        A class that predict future trajectories based on input features
        Args:
            feature_dim (int): dimension of the input feature
            state_dim (int): dimension of the output trajectory at each step
            num_steps (int): (optional) number of future state to predict
            dynamics_type (str, dynamics.DynType): (optional) if specified, the network predicts action
                for the dynamics model instead of future states. The actions are then used to predict
                the future trajectories.
            step_time (float): time between steps. required for using dynamics models
            network_kwargs (dict): keyword args for the decoder networks
        """
        super(TrajectoryDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.num_steps = num_steps
        self.step_time = step_time
        self._create_dynamics(dynamics_type, dynamics_kwargs)
        self._create_networks(network_kwargs)

    def _create_dynamics(self, dynamics_type, dynamics_kwargs):
        if dynamics_type in ["Unicycle", dynamics.DynType.UNICYCLE]:
            self.dyn = dynamics.Unicycle(
                "dynamics",
                max_steer=dynamics_kwargs["max_steer"],
                max_yawvel=dynamics_kwargs["max_yawvel"],
                acce_bound=dynamics_kwargs["acce_bound"]
            )
        else:
            self.dyn = None

    def _create_networks(self, network_kwargs):
        raise NotImplementedError

    def _forward_networks(self, inputs, num_steps=None):
        raise NotImplementedError

    def _forward_dynamics(self, current_state, actions):
        assert self.dyn is not None
        assert current_state.shape[-1] == self.dyn.xdim
        assert actions.shape[-1] == self.dyn.udim
        assert isinstance(self.step_time, float) and self.step_time > 0
        _, pos, yaw = forward_dynamics(
            self.dyn,
            initial_states=current_state,
            actions=actions,
            step_time=self.step_time
        )
        traj = torch.cat((pos, yaw), dim=-1)
        return traj

    def forward(self, inputs, current_state=None, num_steps=None):
        preds = self._forward_networks(inputs, num_steps)
        if self.dyn is not None:
            preds = self._forward_dynamics(current_state=current_state, actions=preds)
        return preds


class MLPTrajectoryDecoder(TrajectoryDecoder):
    def _create_networks(self, net_kwargs):
        if net_kwargs is None:
            net_kwargs = dict()
        assert isinstance(self.num_steps, int)
        pred_dim = self.state_dim if self.dyn is None else self.dyn.udim
        self.mlp = MLP(
            input_dim= self.feature_dim,
            output_dim=pred_dim * self.num_steps,
            output_activation=None,
            **net_kwargs
        )

    def _forward_networks(self, inputs, num_steps=None):
        pred_dim = self.state_dim if self.dyn is None else self.dyn.udim
        return self.mlp(inputs).reshape(-1, self.num_steps, pred_dim)

