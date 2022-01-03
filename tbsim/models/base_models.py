import numpy as np
import textwrap
from collections import OrderedDict
import torch.nn as nn

from tbsim.utils.tensor_utils import reshape_dimensions


class MLP(nn.Module):
    """
    Base class for simple Multi-Layer Perceptrons.
    """
    def __init__(
            self,
            input_dim,
            output_dim,
            layer_dims=(),
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
    A multi-headed MLP network
    """
    def __init__(
            self,
            input_dim,
            output_shapes=None,
            layer_dims=(),
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