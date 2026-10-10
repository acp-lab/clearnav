import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
from typing import List, Optional, Type, Union, Dict
from torchvision import models
from abc import abstractmethod


class FeatureExtractor(nn.Module):
    activation_fn_alias = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "selu": nn.SELU,
        "softplus": nn.Softplus,
        "identity": nn.Identity,
    }

    def __init__(
        self,
        observation_space: spaces.Dict,
        net_arch: Dict,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_dim: int = 1,
    ):
        super().__init__()
        self._features_dim = features_dim
        # self._is_recurrent = False
        if isinstance(activation_fn, str):
            activation_fn = self.activation_fn_alias[activation_fn]

        self._build(observation_space, net_arch, activation_fn)
        # self._build_recurrent(net_arch)

    @abstractmethod
    def _build(self, observation_space, net_arch, activation_fn):
        pass

    # def _build_recurrent(self, net_arch):
    #     if net_arch.get("recurrent", None) is not None:
    #         _hidden_features_dim = set_recurrent_feature_extractor(
    #             self, self._features_dim, net_arch.get("recurrent")
    #         )
    #         self._features_dim = _hidden_features_dim
    #         self._is_recurrent = True

    @abstractmethod
    def extract(self, observations) -> th.Tensor:
        pass

    # def extract_with_recurrent(self, observations):
    #     features = self.extract(observations)
    #     if hasattr(self, "recurrent_extractor"):
    #         features, h = self.recurrent_extractor(
    #             features.unsqueeze(0), observations['latent'].unsqueeze(0)
    #         )
    #         return features[0], h[0]
    #     else:
    #         return features

    def forward(self, observations):
        return self.extract(observations)
        # return self.extract_with_recurrent(observations)

    # @property
    # def is_recurrent(self):
    #     return self._is_recurrent

    @property
    def features_dim(self):
        return self._features_dim


class FlattenExtractor(FeatureExtractor):
    """
    Feature extractor that flattens the input.
    Used as a placeholder when feature extraction is not needed.
    """

    def __init__(self, observation_space: spaces.Space) -> None:
        super().__init__(observation_space, {})

    def _build(self, observation_space, net_arch, activation_fn):
        self.flatten = nn.Flatten()
        self._features_dim = spaces.utils.flatdim(observation_space)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)


def create_mlp(
    input_dim: int,
    layer: List[int],
    output_dim: Optional[int] = None,
    activation_fn: Type[nn.Module] = nn.ReLU,
    batch_norm: Union[bool, List] = False,
    squash_output: bool = False,
    with_bias: bool = True,
    layer_norm: Union[bool, List] = False,
    device: th.device = th.device("cpu"),
) -> nn.Module:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param layer: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param batch_norm: Whether to use batch normalization or not
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :param layer_norm: If set to False, Whether to use layer normalization or not
    :param device: Device on which the neural network should be run.

    :return:
    """
    batch_norm = (
        [batch_norm] * len(layer) if isinstance(batch_norm, bool) else batch_norm
    )
    layer_norm = (
        [layer_norm] * len(layer) if isinstance(layer_norm, bool) else layer_norm
    )
    for each_batch_norm, each_layer_norm in zip(batch_norm, layer_norm):
        assert not (each_batch_norm and each_layer_norm), (
            "batch normalization and layer normalization should not be both implemented."
        )

    # if input batch_norm list length is shorter than layer, then complete the list with False
    if len(batch_norm) < len(layer):
        batch_norm += [False] * (len(layer) - len(batch_norm))
    if len(layer_norm) < len(layer):
        layer_norm += [False] * (len(layer) - len(layer_norm))

    if len(layer) > 0:
        modules = [nn.Linear(input_dim, layer[0], bias=with_bias)]
        if batch_norm[0]:
            modules.append(nn.BatchNorm1d(layer[0]))
        elif layer_norm[0]:
            modules.append(nn.LayerNorm(layer[0]))
        modules.append(activation_fn())
    else:
        modules = []

    for idx in range(len(layer) - 1):
        modules.append(nn.Linear(layer[idx], layer[idx + 1], bias=with_bias))
        if batch_norm[idx + 1]:
            modules.append(nn.BatchNorm1d(layer[idx + 1]))
        elif layer_norm[idx + 1]:
            modules.append(nn.LayerNorm(layer[idx + 1]))
        modules.append(activation_fn())

    if output_dim is not None:
        last_layer_dim = layer[-1] if len(layer) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
        # if batch_norm:
        #     modules.append(nn.BatchNorm1d(output_dim))
        # elif layer_norm:
        #     modules.append(nn.LayerNorm(output_dim))

    if squash_output:
        if len(modules) >= 0 and not isinstance(modules[-1], nn.Linear):
            modules[-1] = nn.Tanh()
        else:
            modules.append(nn.Tanh())

    if len(modules) == 0:
        modules.append(nn.Flatten())

    net = nn.Sequential(*modules).to(device)

    return net


def create_cnn(
    input_channels: int,
    kernel_size: List[int],
    channel: List[int],
    stride: List[int],
    padding: List[int],
    output_channel: Optional[int] = None,
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    batch_norm: bool = False,
    with_bias: bool = True,
    device: th.device = th.device("cpu"),
) -> nn.Module:
    assert len(kernel_size) == len(stride) == len(padding) == len(channel), (
        "The length of kernel_size, stride, padding and net_arch should be the same."
    )

    if len(channel) > 0:
        modules = [
            nn.Conv2d(
                input_channels,
                channel[0],
                kernel_size=kernel_size[0],
                stride=stride[0],
                padding=padding[0],
                bias=with_bias,
            )
        ]
        if batch_norm:
            modules.append(nn.BatchNorm2d(channel[0]))
        modules.append(activation_fn())
    else:
        modules = []

    for idx in range(len(channel) - 1):
        modules.append(
            nn.Conv2d(
                channel[idx],
                channel[idx + 1],
                kernel_size=kernel_size[idx + 1],
                stride=stride[idx + 1],
                padding=padding[idx + 1],
                bias=with_bias,
            )
        )

        if batch_norm:
            modules.append(nn.BatchNorm2d(channel[idx + 1]))
        modules.append(activation_fn())

    if output_channel is not None:
        last_layer_channel = channel[-1] if len(channel) > 0 else input_channels
        modules.append(
            nn.Conv2d(
                last_layer_channel,
                output_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        if batch_norm:
            modules.append(nn.BatchNorm2d(output_channel))
        modules.append(activation_fn())

    modules.append(nn.Flatten())

    if squash_output:
        modules.append(nn.Tanh())

    net = nn.Sequential(*modules).to(device)
    return net


# def set_recurrent_feature_extractor(cls, input_size, rnn_setting):
#     recurrent_alias = {
#         "GRU": th.nn.GRU,
#     }
#     rnn_class = rnn_setting.get("class")
#     kwargs = rnn_setting.get("kwargs")
#     if isinstance(rnn_class, str):
#         rnn_class = recurrent_alias[rnn_class]
#     cls.__setattr__("recurrent_extractor", rnn_class(input_size=input_size, **kwargs))
#     return kwargs.get("hidden_size")


def set_mlp_feature_extractor(cls, name, observation_space, net_arch, activation_fn):
    layer = net_arch.get("mlp_layer", [])
    features_dim = layer[-1] if len(layer) != 0 else observation_space.shape[0]
    if len(observation_space.shape) == 1:
        input_dim = observation_space.shape[0]
    else:
        input_dim = observation_space.shape[1]

    setattr(
        cls,
        name + "_extractor",
        create_mlp(
            input_dim=input_dim,
            layer=net_arch.get("mlp_layer", []),
            activation_fn=activation_fn,
            batch_norm=net_arch.get("bn", False),
            layer_norm=net_arch.get("ln", False),
        ),
    )
    return features_dim


class TargetExtractor(FeatureExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        net_arch: Dict = {},
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        assert "target" in observation_space.spaces
        super().__init__(
            observation_space=observation_space,
            net_arch=net_arch,
            activation_fn=activation_fn,
        )

    def _build(self, observation_space, net_arch, activation_fn):
        feature_dim = set_mlp_feature_extractor(
            self,
            "target",
            observation_space["target"],
            net_arch["target"],
            activation_fn,
        )
        self._features_dim = feature_dim

    def extract(self, observations) -> th.Tensor:
        return self.target_extractor(observations["target"])


class StateExtractor(FeatureExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        net_arch: Optional[Dict] = {},
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        assert "state" in observation_space.spaces
        super().__init__(observation_space, net_arch, activation_fn)

    def _build(self, observation_space, net_arch, activation_fn):
        feature_dim = set_mlp_feature_extractor(
            self, "state", observation_space["state"], net_arch["state"], activation_fn
        )
        self._features_dim = feature_dim

    def extract(self, observations) -> th.Tensor:
        return self.state_extractor(observations["state"])


class ImageExtractor(FeatureExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(
        self,
        observation_space: spaces.Dict,
        net_arch: Dict = {},
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        assert any(
            "semantic" in key or "color" in key or "depth" in key
            for key in observation_space.keys()
        )
        super().__init__(
            observation_space=observation_space,
            net_arch=net_arch,
            activation_fn=activation_fn,
        )

    def _build(self, observation_space, net_arch, activation_fn):
        """
        builds extractor network for each semantic/color/depth observations,
        stores it as an attribute, and populates self._image_extractor_names
        with name of attribute
        """
        _image_features_dims = 0
        self._image_extractor_names = []
        # for key in observation_space.keys():
        for key in net_arch.keys():
            if "semantic" in key or "color" in key or "depth" in key:
                _image_features_dims += self.set_cnn_feature_extractor(
                    key, observation_space[key], net_arch.get(key, {}), activation_fn
                )
        self._features_dim = _image_features_dims

    def extract(self, observations) -> th.Tensor:
        features = []
        for name in self._image_extractor_names:
            image = observations[name.split("_")[0]]
            if "depth" in name:
                image = self.preprocess_depth(image)
            x = getattr(self, name)(image)
            features.append(x)
        combined_features = th.cat(features, dim=1)

        return combined_features

    def preprocess_depth(self, depth: th.Tensor):
        depth = depth.float()
        inv_depth = 1.0 / (depth + 1e-6)

        if hasattr(self, "input_max_pool_H_W"):
            # apply max pooling
            # note this preserves closer objects, since we use inverted depth
            H, W = self.input_max_pool_H_W
            inv_depth = F.adaptive_max_pool2d(inv_depth, (H, W))
        return inv_depth

    def set_cnn_feature_extractor(
        self, name, observation_space, net_arch, activation_fn
    ):
        """
        creates an attribute containing the specified network under the name,
        name + "_extractor"
        """
        in_channels = observation_space.shape[0]
        if "input_max_pool_H_W" in net_arch:
            H, W = net_arch["input_max_pool_H_W"]
            self.input_max_pool_H_W = net_arch["input_max_pool_H_W"]
            observation_shape = (in_channels, H, W)
        else:
            observation_shape = observation_space.shape
        assert in_channels >= 1
        backbone = net_arch.get("backbone", None)
        assert backbone is None or backbone in self.backbone_alias, (
            f"Backbone {backbone} not supported."
        )

        if backbone is None:
            image_extractor = create_cnn(
                input_channels=in_channels,
                kernel_size=net_arch["kernel_size"],
                channel=net_arch["channels"],
                output_channel=net_arch.get("output_channel", None),
                activation_fn=activation_fn,
                padding=net_arch["padding"],
                stride=net_arch["stride"],
                batch_norm=net_arch.get("cnn_bn", False),
                squash_output=False,
            )
            _image_features_dims = self._get_conv_output(
                image_extractor, observation_shape
            )
            if len(net_arch.get("mlp_layer", [])) > 0:
                image_extractor.add_module(
                    "mlp",
                    create_mlp(
                        input_dim=_image_features_dims,
                        layer=net_arch.get("mlp_layer"),
                        activation_fn=activation_fn,
                        batch_norm=net_arch.get("bn", False),
                        layer_norm=net_arch.get("ln", False),
                    ),
                )
        elif "resnet" in backbone:
            image_extractor = self.backbone_alias[backbone](pretrained=True)
            # replace the first layer to match the input channels
            new_layer = nn.Conv2d(
                in_channels,
                image_extractor.conv1.out_channels,
                kernel_size=image_extractor.conv1.kernel_size,
                stride=image_extractor.conv1.stride,
                padding=image_extractor.conv1.padding,
                bias=image_extractor.conv1.bias is not None,
            )
            # copy weights for available channels if using pretrained
            with th.no_grad():
                if in_channels <= 3:
                    new_layer.weight[:, :in_channels] = image_extractor.conv1.weight[
                        :, :in_channels
                    ]
                else:
                    new_layer.weight[:, :3] = image_extractor.conv1.weight
            image_extractor.conv1 = new_layer
            if len(net_arch.get("mlp_layer", [])) > 0:
                image_extractor.fc = create_mlp(
                    input_dim=image_extractor.fc.in_features,
                    layer=net_arch.get("mlp_layer"),
                    activation_fn=activation_fn,
                    batch_norm=net_arch.get("bn", False),
                    layer_norm=net_arch.get("ln", False),
                )
        elif "efficientnet" in backbone or "mobilenet" in backbone:
            image_extractor = self.backbone_alias[backbone](pretrained=True)
            new_layer = nn.Conv2d(
                in_channels,
                image_extractor.features[0][0].out_channels,
                kernel_size=image_extractor.features[0][0].kernel_size,
                stride=image_extractor.features[0][0].stride,
                padding=image_extractor.features[0][0].padding,
                bias=image_extractor.features[0][0].bias is not None,
            )
            # copy weights for available channels if using pretrained
            with th.no_grad():
                if in_channels <= 3:
                    new_layer.weight[:, :in_channels] = image_extractor.features[0][
                        0
                    ].weight[:, :in_channels]
                else:
                    new_layer.weight[:, :3] = image_extractor.conv1.weight
            image_extractor.features[0][0] = new_layer
            if len(net_arch.get("mlp_layer", [])) > 0:
                image_extractor.classifier[-1] = create_mlp(
                    input_dim=image_extractor.classifier[-1].in_features,
                    layer=net_arch.get("mlp_layer"),
                    activation_fn=activation_fn,
                    batch_norm=net_arch.get("bn", False),
                    layer_norm=net_arch.get("ln", False),
                )
        else:
            raise ValueError(f"Backbone {backbone} not supported.")
        setattr(self, name + "_extractor", image_extractor)
        self._image_extractor_names.append(name + "_extractor")
        return self._get_conv_output(image_extractor, observation_shape)

    def _get_conv_output(self, net, shape):
        net.eval()
        image = th.rand(1, *shape)
        output = net(image)
        net.train()
        return output.numel()


class StateTargetExtractor(FeatureExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        net_arch: Dict = {},
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        obs_keys = list(observation_space.spaces)
        assert ("state" in obs_keys) and ("target" in obs_keys)
        super().__init__(observation_space, net_arch, activation_fn)

    def _build(self, observation_space, net_arch, activation_fn):
        state_features_dim = set_mlp_feature_extractor(
            self,
            "state",
            observation_space["state"],
            net_arch.get("state", {}),
            activation_fn,
        )
        target_features_dim = set_mlp_feature_extractor(
            self,
            "target",
            observation_space["target"],
            net_arch.get("target", {}),
            activation_fn,
        )
        self._features_dim = state_features_dim + target_features_dim

    def extract(self, observations):
        return th.cat(
            [
                self.state_extractor(observations["state"]),
                self.target_extractor(observations["target"]),
            ],
            dim=1,
        )


class StateImageExtractor(ImageExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        net_arch: Dict = {},
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        assert "state" in observation_space.spaces
        super().__init__(observation_space, net_arch, activation_fn)

    def _build(self, observation_space, net_arch, activation_fn):
        super()._build(observation_space, net_arch, activation_fn)
        _state_features_dim = set_mlp_feature_extractor(
            self,
            "state",
            observation_space["state"],
            net_arch.get("state", {}),
            activation_fn,
        )

        self.concatenate = net_arch.get("concatenate", True)
        if self.concatenate:
            # concatenate features
            self._features_dim = _state_features_dim + self._features_dim
        else:
            # add features elementwise
            assert _state_features_dim == self._features_dim
            self._features_dim = self._features_dim

    def extract(self, observations) -> th.Tensor:
        state_features = self.state_extractor(observations["state"])
        image_features = super().extract(observations)
        if self.concatenate:
            combined_feature = th.cat([state_features, image_features], dim=1)
        else:
            combined_feature = state_features + image_features
        return combined_feature


class StateTargetImageExtractor(ImageExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        net_arch: Dict = {},
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        obs_keys = list(observation_space.spaces)
        assert ("state" in obs_keys) and ("target" in obs_keys)
        super().__init__(observation_space, net_arch, activation_fn)

    def _build(self, observation_space, net_arch, activation_fn):
        super()._build(observation_space, net_arch, activation_fn)
        _state_features_dim = set_mlp_feature_extractor(
            self,
            "state",
            observation_space["state"],
            net_arch.get("state", {}),
            activation_fn,
        )
        _target_features_dim = set_mlp_feature_extractor(
            self,
            "target",
            observation_space["target"],
            net_arch.get("target", {}),
            activation_fn,
        )

        self.concatenate = net_arch.get("concatenate", True)
        if self.concatenate:
            # concatenate features
            self._features_dim = (
                _state_features_dim + _target_features_dim + self._features_dim
            )
        else:
            # add features elementwise
            assert _state_features_dim == _target_features_dim == self._features_dim
            self._features_dim = self._features_dim

    def extract(self, observation):
        state_features = self.state_extractor(observation["state"])
        target_features = self.target_extractor(observation["target"])
        image_features = super().extract(observation)
        if self.concatenate:
            combined_feature = th.cat(
                [state_features, target_features, image_features], dim=1
            )
        else:
            combined_feature = state_features + target_features + image_features
        return combined_feature
