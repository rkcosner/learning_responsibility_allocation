import warnings
from typing import Dict, List
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

from tbsim.models.base_models import SplitMLP, MLP, MIMOMLP
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.dynamics as dynamics
import tbsim.utils.l5_utils as L5Utils


class RasterizedPlanningModelOld(nn.Module):
    """Raster-based model for planning.
    """

    def __init__(
            self,
            model_arch: str,
            num_input_channels: int,
            num_targets: int,
            weights_scaling: List[float],
            criterion: nn.Module,
            pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = num_input_channels
        self.num_targets = num_targets
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        if model_arch == "resnet18":
            self.model = resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=512, out_features=num_targets)
        elif model_arch == "resnet50":
            self.model = resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
        else:
            raise NotImplementedError(f"Model arch {model_arch} unknown")

        if self.num_input_channels != 3:
            self.model.conv1 = nn.Conv2d(
                self.num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        outputs = self.model(image_batch)
        batch_size = len(data_batch["image"])

        predicted = outputs.view(batch_size, -1, 3)
        # [batch_size, num_steps, 2->(XY)]
        pred_positions = predicted[:, :, :2]
        # [batch_size, num_steps, 1->(yaw)]
        pred_yaws = predicted[:, :, 2:3]
        out_dict = {
            "raw_outputs": outputs,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }
        return out_dict

    def compute_losses(self, pred_batch, data_batch):
        if self.criterion is None:
            raise NotImplementedError("Loss function is undefined.")

        batch_size = data_batch["image"].shape[0]
        # [batch_size, num_steps * 2]
        targets = (torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)).view(
            batch_size, -1
        )
        # [batch_size, num_steps]
        target_weights = (data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling).view(
            batch_size, -1
        )
        loss = torch.mean(self.criterion(pred_batch["raw_outputs"], targets) * target_weights)
        losses = OrderedDict(prediction_loss=loss)
        return losses


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


class RasterizedPlanningModel(nn.Module):
    """Raster-based model for planning.
    """

    def __init__(
            self,
            model_arch: str,
            num_input_channels: int,
            num_future_frames: int,
            weights_scaling: List[float],
            criterion: nn.Module,
            dynamics_model: dynamics.Dynamics = None,
            step_time = 0.1
    ) -> None:

        super().__init__()
        self.map_encoder = RasterizedMapEncoder(
            model_arch=model_arch,
            num_input_channels=num_input_channels,
            feature_dim=128,
            output_activation=nn.ReLU
        )
        self.dynamics_model = dynamics_model
        self.pred_step_dim = 3 if self.dynamics_model is None else 2  # [x, y, yaw] or [acc, dh]
        self.output_fc = nn.Linear(128, num_future_frames * self.pred_step_dim)
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)
        self.criterion = criterion
        self.step_time = step_time

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch["image"]
        map_feat = self.map_encoder(image_batch)
        preds = self.output_fc(map_feat)
        preds = preds.reshape(preds.shape[0], -1, self.pred_step_dim)
        if self.dynamics_model is not None:
            raw_ego = L5Utils.batch_to_raw_ego(data_batch, step_time=self.step_time)
            all_states, curr_states = L5Utils.raw_to_states(*raw_ego)
            _, pos, yaw = forward_dynamics(
                self.dynamics_model,
                initial_states=curr_states[:, 0, ...],
                actions=preds,
                step_time=self.step_time
            )
            traj = torch.cat((pos, yaw), dim=-1)
        else:
            traj = preds

        pred_positions = traj[:, :, :2]
        pred_yaws = traj[:, :, 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }
        return out_dict

    def compute_losses(self, pred_batch, data_batch):
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        target_weights = (data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling)
        loss = torch.mean(self.criterion(pred_batch["trajectories"], target_traj) * target_weights)
        losses = OrderedDict(prediction_loss=loss)
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


class ConditionFlatDecoder(nn.Module):
    """Decoding (z, c) -> x' using a flat MLP"""
    def __init__(
            self,
            condition_dim: int,
            latent_dim: int,
            output_shapes: OrderedDict,
            mlp_layer_dims : tuple = (128, 128),
    ):
        super(ConditionFlatDecoder, self).__init__()
        self.mlp = SplitMLP(
            input_dim=(condition_dim + latent_dim),
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims
        )

    def forward(self, latents, condition_features):
        return self.mlp(torch.cat((latents, condition_features), dim=-1))