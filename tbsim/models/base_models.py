import numpy as np
import math
import textwrap
from collections import OrderedDict
from typing import Dict, Union
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50

from tbsim.utils.tensor_utils import reshape_dimensions, flatten
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.dynamics as dynamics


class MLP(nn.Module):
    """
    Base class for simple Multi-Layer Perceptrons.
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            layer_dims: tuple=(),
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            dropouts=None,
            normalization=False,
            output_activation=None,
    ):
        """
        Args:
            input_dim (int): dimension of inputs
            output_dim (int): dimension of outputs
            layer_dims ([int]): sequence of integers for the hidden layers sizes
            layer_func: mapping per layer - defaults to Linear
            layer_func_kwargs (dict): kwargs for @layer_func
            activation: non-linearity per layer - defaults to ReLU
            dropouts ([float]): if not None, adds dropout layers with the corresponding probabilities
                after every layer. Must be same size as @layer_dims.
            normalization (bool): if True, apply layer normalization after each layer
            output_activation: if provided, applies the provided non-linearity to the output layer
        """
        super(MLP, self).__init__()
        layers = []
        dim = input_dim
        if layer_func_kwargs is None:
            layer_func_kwargs = dict()
        if dropouts is not None:
            assert(len(dropouts) == len(layer_dims))
        for i, l in enumerate(layer_dims):
            layers.append(layer_func(dim, l, **layer_func_kwargs))
            if normalization:
                layers.append(nn.LayerNorm(l))
            layers.append(activation())
            if dropouts is not None and dropouts[i] > 0.:
                layers.append(nn.Dropout(dropouts[i]))
            dim = l
        layers.append(layer_func(dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        self._layer_func = layer_func
        self.nets = layers
        self._model = nn.Sequential(*layers)

        self._layer_dims = layer_dims
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._dropouts = dropouts
        self._act = activation
        self._output_act = output_activation

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return [self._output_dim]

    def forward(self, inputs):
        """
        Forward pass.
        """
        return self._model(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = str(self.__class__.__name__)
        act = None if self._act is None else self._act.__name__
        output_act = None if self._output_act is None else self._output_act.__name__

        indent = ' ' * 4
        msg = "input_dim={}\noutput_shape={}\nlayer_dims={}\nlayer_func={}\ndropout={}\nact={}\noutput_act={}".format(
            self._input_dim, self.output_shape(), self._layer_dims,
            self._layer_func.__name__, self._dropouts, act, output_act
        )
        msg = textwrap.indent(msg, indent)
        msg = header + '(\n' + msg + '\n)'
        return msg


class SplitMLP(MLP):
    """
    A multi-output MLP network: The model split and reshapes the output layer to the desired output shapes
    """
    def __init__(
            self,
            input_dim: int,
            output_shapes: OrderedDict,
            layer_dims: tuple=(),
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            dropouts=None,
            normalization=False,
            output_activation=None,
    ):
        """
        Args:
            input_dim (int): dimension of inputs
            output_shapes (dict): named dictionary of output shapes
            layer_dims ([int]): sequence of integers for the hidden layers sizes
            layer_func: mapping per layer - defaults to Linear
            layer_func_kwargs (dict): kwargs for @layer_func
            activation: non-linearity per layer - defaults to ReLU
            dropouts ([float]): if not None, adds dropout layers with the corresponding probabilities
                after every layer. Must be same size as @layer_dims.
            normalization (bool): if True, apply layer normalization after each layer
            output_activation: if provided, applies the provided non-linearity to the output layer
        """

        assert isinstance(output_shapes, OrderedDict)
        output_dim = 0
        for v in output_shapes.values():
            output_dim += np.prod(v)
        self._output_shapes = output_shapes

        super(SplitMLP, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            layer_dims=layer_dims,
            layer_func=layer_func,
            layer_func_kwargs=layer_func_kwargs,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation
        )

    def output_shape(self, input_shape=None):
        return self._output_shapes

    def forward(self, inputs):
        outs = super(SplitMLP, self).forward(inputs)
        out_dict = dict()
        ind = 0
        for k, v in self._output_shapes.items():
            v_dim = int(np.prod(v))
            out_dict[k] = reshape_dimensions(outs[:, ind: ind + v_dim], begin_axis=1, end_axis=2, target_dims=v)
            ind += v_dim
        return out_dict


class MIMOMLP(SplitMLP):
    """
    A multi-input, multi-output MLP: The model flattens and concatenate the input before feeding into an MLP
    """
    def __init__(
            self,
            input_shapes: OrderedDict,
            output_shapes: OrderedDict,
            layer_dims: tuple=(),
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            dropouts=None,
            normalization=False,
            output_activation=None,
    ):
        """
        Args:
            input_shapes (OrderedDict): named dictionary of input shapes
            output_shapes (OrderedDict): named dictionary of output shapes
            layer_dims ([int]): sequence of integers for the hidden layers sizes
            layer_func: mapping per layer - defaults to Linear
            layer_func_kwargs (dict): kwargs for @layer_func
            activation: non-linearity per layer - defaults to ReLU
            dropouts ([float]): if not None, adds dropout layers with the corresponding probabilities
                after every layer. Must be same size as @layer_dims.
            normalization (bool): if True, apply layer normalization after each layer
            output_activation: if provided, applies the provided non-linearity to the output layer
        """
        assert isinstance(input_shapes, OrderedDict)
        input_dim = 0
        for v in input_shapes.values():
            input_dim += np.prod(v)

        self._input_shapes = input_shapes

        super(MIMOMLP, self).__init__(
            input_dim=input_dim,
            output_shapes=output_shapes,
            layer_dims=layer_dims,
            layer_func=layer_func,
            layer_func_kwargs=layer_func_kwargs,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation
        )

    def forward(self, inputs):
        flat_inputs = []
        for k in self._input_shapes.keys():
            flat_inputs.append(flatten(inputs[k]))
        return super(MIMOMLP, self).forward(torch.cat(flat_inputs, dim=1))


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),                                     # (Mohit): argh... forgot to remove this batchnorm
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),                                     # (Mohit): argh... forgot to remove this batchnorm
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        """
        U-Net forward by concatenating input feature (x1) with mirroring encoding feature maps channel-wise (x2)
        Args:
            x1 (torch.Tensor): [B, C1, H1, W1]
            x2 (torch.Tensor): [B, C2, H2, W2]

        Returns:
            output (torch.Tensor): [B, out_channels, H2, W2]
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class IdentityBlock(nn.Module):
    def __init__(self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True):
        super(IdentityBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=1,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        if self.final_relu:
            out = F.relu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True):
        super(ConvBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=1,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, filters3,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if self.final_relu:
            out = F.relu(out)
        return out


class UNetDecoder(nn.Module):
    def __init__(self, input_channel, up_factor, bilinear, batchnorm):
        super(UNetDecoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

        self.up1 = Up(2048, 512, bilinear=True)

        self.up2 = Up(1024, 512 // up_factor, bilinear)

        self.up3 = Up(512, 256 // up_factor, bilinear)

        self.layer1 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        )

    def forward(self, feat_to_decode, encoder_feats, target_hw):
        x = self.conv1(feat_to_decode)
        x = self.up1(x, encoder_feats[-2])
        x = self.up2(x, encoder_feats[-3])
        x = self.up3(x, encoder_feats[-4])

        for layer in [self.layer1, self.layer2, self.layer3, self.conv2]:
            x = layer(x)

        x = F.interpolate(x, size=(target_hw[0], target_hw[1]), mode='bilinear')
        return x


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.
    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(
            self,
            input_shape,
            num_kp=None,
            temperature=1.,
            learnable_temperature=False,
            output_variance=False,
            noise_std=0.0,
    ):
        """
        Args:
            input_shape (list, tuple): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not use spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self._in_w),
            np.linspace(-1., 1., self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial
        probability distribution is created using a softmax, where the support is the
        pixel locations. This distribution is used to compute the expected value of
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.
        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints


class RasterizedMapEncoder(nn.Module):
    """A basic image-based rasterized map encoder"""
    def __init__(
            self,
            model_arch: str,
            input_image_shape: tuple = (3, 224, 224),
            feature_dim: int = 128,
            use_spatial_softmax=False,
            spatial_softmax_kwargs=None,
            output_activation = nn.ReLU
    ) -> None:
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = input_image_shape[0]
        self._feature_dim = feature_dim
        self._output_activation = output_activation

        # configure conv backbone
        if model_arch == "resnet18":
            self.map_model = resnet18()
            out_h = int(math.ceil(input_image_shape[1] / 32.))
            out_w = int(math.ceil(input_image_shape[2] / 32.))
            conv_out_shape = (2048, out_h, out_w)
        elif model_arch == "resnet50":
            self.map_model = resnet50()
            out_h = int(math.ceil(input_image_shape[1] / 32.))
            out_w = int(math.ceil(input_image_shape[2] / 32.))
            conv_out_shape = (2048, out_h, out_w)
        else:
            raise NotImplementedError(f"Model arch {model_arch} unknown")

        # configure spatial reduction pooling layer
        if use_spatial_softmax:
            pooling = SpatialSoftmax(input_shape=conv_out_shape, **spatial_softmax_kwargs)
            pool_out_dim = int(np.prod(pooling.output_shape(conv_out_shape)))
        else:
            pooling = nn.AdaptiveAvgPool2d((1, 1))
            pool_out_dim = conv_out_shape[0]

        self.map_model.conv1 = nn.Conv2d(
            self.num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.map_model.avgpool = pooling
        self.map_model.fc = nn.Linear(in_features=pool_out_dim, out_features=feature_dim)

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
        self._create_networks(network_kwargs, dynamics_kwargs.get("predict_current_states"))

    def _create_dynamics(self, dynamics_type, dynamics_kwargs):
        if dynamics_type in ["Unicycle", dynamics.DynType.UNICYCLE]:
            self.dyn = dynamics.Unicycle(
                "dynamics",
                max_steer=dynamics_kwargs["max_steer"],
                max_yawvel=dynamics_kwargs["max_yawvel"],
                acce_bound=dynamics_kwargs["acce_bound"]
            )
        elif dynamics_type in ["Bicycle", dynamics.DynType.BICYCLE]:
            self.dyn = dynamics.Bicycle(
                acc_bound=dynamics_kwargs["acce_bound"],
                ddh_bound=dynamics_kwargs["ddh_bound"],
                max_hdot=dynamics_kwargs["max_yawvel"],
                max_speed=dynamics_kwargs["max_speed"]
            )
        else:
            self.dyn = None

    def _create_networks(self, network_kwargs, predict_current_states=False):
        raise NotImplementedError

    def _forward_networks(self, inputs, num_steps=None):
        raise NotImplementedError

    def _forward_dynamics(self, current_states, actions):
        assert self.dyn is not None
        assert current_states.shape[-1] == self.dyn.xdim
        assert actions.shape[-1] == self.dyn.udim
        assert isinstance(self.step_time, float) and self.step_time > 0
        _, pos, yaw = dynamics.forward_dynamics(
            self.dyn,
            initial_states=current_states,
            actions=actions,
            step_time=self.step_time
        )
        traj = torch.cat((pos, yaw), dim=-1)
        return traj

    def forward(self, inputs, current_states=None, num_steps=None):
        preds = self._forward_networks(inputs, num_steps)
        if "current_states" in preds:
            current_states = preds["current_states"]
        if self.dyn is not None:
            preds["controls"] = preds["trajectories"]
            preds["trajectories"] = self._forward_dynamics(
                current_states=current_states,
                actions=preds["trajectories"]
            )
        return preds


class MLPTrajectoryDecoder(TrajectoryDecoder):
    def _create_networks(self, net_kwargs, predict_current_states=False):
        if net_kwargs is None:
            net_kwargs = dict()
        assert isinstance(self.num_steps, int)
        if self.dyn is None:
            pred_shapes = OrderedDict(trajectories=(self.num_steps, self.state_dim))
        else:
            pred_shapes = OrderedDict(trajectories=(self.num_steps, self.dyn.udim))
            if predict_current_states:
                pred_shapes["current_states"] = (self.dyn.xdim,)

        self.mlp = SplitMLP(
            input_dim=self.feature_dim,
            output_shapes=pred_shapes,
            output_activation=None,
            **net_kwargs
        )

    def _forward_networks(self, inputs, num_steps=None):
        preds = self.mlp(inputs)
        if "current_states" in preds:
            preds["current_states"][:, [0, 1, 3]] = 0. # ego frame prediction - [x, y, yaw] are 0's
        return preds