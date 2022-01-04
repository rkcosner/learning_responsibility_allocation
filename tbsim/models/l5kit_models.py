import warnings
from typing import Dict, List
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

from tbsim.models.base_models import SplitMLP, MLP


class RasterizedPlanningModel(nn.Module):
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


class RasterizedMapEncoder(nn.Module):
    """A basic image-based rasterized map encoder"""
    def __init__(
            self,
            model_arch: str,
            num_input_channels: int = 3,  # C
            visual_feature_dim: int = 128,
    ) -> None:
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = num_input_channels
        self._visual_feature_dim = visual_feature_dim

        if model_arch == "resnet18":
            self.map_model = resnet18()
            self.map_model.fc = nn.Linear(in_features=512, out_features=visual_feature_dim)
        elif model_arch == "resnet50":
            self.map_model = resnet50()
            self.map_model.fc = nn.Linear(in_features=2048, out_features=visual_feature_dim)
        else:
            raise NotImplementedError(f"Model arch {model_arch} unknown")

        if self.num_input_channels != 3:
            self.map_model.conv1 = nn.Conv2d(
                self.num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

    def output_shape(self, input_shape=None):
        return [self._visual_feature_dim]

    def forward(self, map_inputs):
        return self.map_model(map_inputs)


class PosteriorEncoder(nn.Module):
    """Posterior Encoder (x, c -> q) for CVAE"""
    def __init__(
            self,
            map_encoder: nn.Module,
            trajectory_shape: tuple,  # [T, D]
            output_shapes: dict,
            mlp_layer_dims: tuple = (128, 128),
            rnn_hidden_size: int = 100
    ) -> None:
        super(PosteriorEncoder, self).__init__()
        self.map_encoder = map_encoder
        self.trajectory_shape = trajectory_shape

        # TODO: history encoder
        self.traj_lstm = nn.LSTM(trajectory_shape[-1], hidden_size=rnn_hidden_size, batch_first=True)
        visual_feature_size = self.map_encoder.output_shape()[0]
        self.mlp = SplitMLP(
            input_dim=(visual_feature_size + rnn_hidden_size),
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims,
            output_activation=nn.ReLU
        )

    def forward(self, inputs, condition_inputs) -> Dict[str, torch.Tensor]:
        map_feat = self.map_encoder(condition_inputs["image"])
        traj_feat = self.traj_lstm(inputs["trajectories"])[0][:, -1, :]
        feat = torch.cat((map_feat, traj_feat), dim=-1)
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
    
    
class ConditionFlatDecoder(nn.Module):
    """Decoding (z, c) -> x' using a flat MLP"""
    def __init__(
            self,
            condition_dim: int,
            latent_dim: int,
            output_shapes: dict,
            mlp_layer_dims : tuple = (128, 128),
    ):
        super(ConditionFlatDecoder, self).__init__()
        self.mlp = SplitMLP(
            input_dim=(condition_dim + latent_dim),
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims
        )

    def forward(self, latents, conditions):
        return self.mlp(torch.cat((latents, conditions), dim=-1))