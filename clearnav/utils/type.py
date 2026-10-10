from dataclasses import dataclass
import torch as th
from typing import Union, Any
from enum import Enum
import numpy as np


class Uniform:
    mean: Union[float, th.Tensor] = 0
    half: Union[float, th.Tensor] = 0

    def __init__(
        self,
        mean,
        half,
    ):
        self.mean = th.as_tensor(mean)
        self.half = th.as_tensor(half)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.half = self.half.to(device)
        return self

    def generate(self, size, generator=None):
        return (th.rand(size, generator=generator) - 0.5) * self.half + self.mean


class Normal:
    mean: Union[float, th.Tensor] = 0
    std: Union[float, th.Tensor] = 0

    def __init__(
        self,
        mean,
        std,
    ):
        self.mean = th.as_tensor(mean)
        self.std = th.as_tensor(std)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def generate(self, size, generator=None):
        return th.randn(size, generator=generator) * self.std + self.mean


class Cylinder:
    mean: Union[float, th.Tensor] = 0
    half: Union[float, th.Tensor] = 0

    def __init__(
        self,
        mean,
        half,
    ):
        self.mean = th.as_tensor(mean)
        self.half = th.as_tensor(half)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.half = self.half.to(device)
        return self

    def generate(self, size, generator=None):
        if type(size) == tuple:
            n = size[0] if type(size) == tuple else size.shape[0]
        theta = 2.0 * th.pi * th.rand(n, generator=generator)
        x = self.half[0] * th.cos(theta)
        y = self.half[1] * th.sin(theta)
        z = self.half[2] * th.rand(n) - 0.5
        samples = th.stack([x, y, z], dim=1) + self.mean
        return samples


# create a dict designed for tensor that can use detach
class TensorDict(dict):
    def __init__(self, data):
        super().__init__(data)

    # return a new detach, do not change instance itself
    def detach(self):
        return TensorDict({key: self[key].detach() for key in self.keys()})

    def clone(self):
        for key in self.keys():
            self[key] = self[key].clone()

        return self

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return super().__getitem__(key)
        elif isinstance(key, int):
            return TensorDict({k: v[key] for k, v in self.items()})
        elif hasattr(key, "__iter__"):
            return TensorDict({k: v[key] for k, v in self.items()})
        else:
            raise TypeError("Invalid key type. Must be either str or int.")

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, str):
            super().__setitem__(key, value)
        elif isinstance(key, (int, th.Tensor, np.ndarray, list)):
            for k in self.keys():
                self[k][key] = value[k]
        else:
            raise TypeError("Invalid key type. Must be either str or int.")

    def append(self, data):
        if isinstance(data, TensorDict):
            for key, value in data.items():
                self[key] = th.cat([self[key], data[key]])

    def cpu(self):
        for key, value in self.items():
            self[key] = self[key].cpu()

    def as_tensor(self, device=th.device("cpu")):
        d = {}
        for key, value in self.items():
            d[key] = th.as_tensor(value, device=device)

        return d

    def to(self, device):
        for key, value in self.items():
            self[key] = value.to(device)
        return self
