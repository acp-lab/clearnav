from typing import Union, List, Dict
import torch as th
from torch import nn
import torch.nn.functional as F


class AccelerationBoundedYaw(nn.Module):
    def __init__(self, min_yaw=-th.pi, max_yaw=th.pi):
        super().__init__()
        self.min_yaw = min_yaw
        self.max_yaw = max_yaw

    def forward(self, z):
        acc = z[:, 0:3]
        yaw = z[:, 3]

        bounded_yaw = self.min_yaw + (self.max_yaw - self.min_yaw) * th.sigmoid(yaw)
        acc_bounded_yaw = th.cat([acc, bounded_yaw.unsqueeze(1)], dim=1)
        return acc_bounded_yaw


class MlpPolicy(nn.Module):
    activation_fn_alias = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "selu": nn.SELU,
        "softplus": nn.Softplus,
        "identity": nn.Identity,
    }

    output_activation_fn_alias = {
        "identity": nn.Identity,
        "acceleration_bounded_yaw": AccelerationBoundedYaw,
    }

    def __init__(
        self,
        in_dim: int,
        net_arch: Dict[str, List[int]],
        activation_fn: Union[str, nn.Module],
        output_activation_fn: Union[str, nn.Module],
        output_activation_kwargs: Dict = None,
        device: th.device = "cpu",
    ):
        super().__init__()

        if isinstance(activation_fn, str):
            activation_fn = self.activation_fn_alias[activation_fn]
        if isinstance(output_activation_fn, str):
            output_activation_fn = self.output_activation_fn_alias[output_activation_fn]

        self.in_dim = in_dim
        self.activation_fn = activation_fn
        self.output_activation_fn = output_activation_fn
        self.output_activation_kwargs = output_activation_kwargs or {}
        self.net_arch = net_arch
        self.device = device
        self._is_recurrent = False
        self._latent_dim = 0

        self._build()

    def _build(self):
        policy_net: List[nn.Module] = []
        pi_layers_dims = self.net_arch.get("mlp_layer", [])

        # iterate through the policy layers and build the policy net
        last_layer_dim_pi = self.in_dim
        for i, curr_layer_dim in enumerate(pi_layers_dims):
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            if i < len(pi_layers_dims) - 1:
                policy_net.append(self.activation_fn())
            else:
                assert not (
                    type(self.output_activation_fn) == nn.Identity
                    and len(self.output_activation_kwargs) > 0
                )
                policy_net.append(
                    self.output_activation_fn(**self.output_activation_kwargs)
                )
            last_layer_dim_pi = curr_layer_dim

        self.out_dim = last_layer_dim_pi
        self.policy_net = nn.Sequential(*policy_net).to(self.device)

    def forward(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def save(self, filepath):
        th.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(th.load(filepath))
        self.to(self.device)

    @property
    def is_recurrent(self):
        return self._is_recurrent

    @property
    def latent_dim(self):
        return self._latent_dim
