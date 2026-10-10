import torch as th
import numpy as np
from typing import TypeVar
from .maths import safe_atan2

SelfRotation3 = TypeVar("SelfRotation3", bound="Rotation3")


class Rotation3:
    check_valid = True

    def __init__(self, R: th.Tensor = None, num=1, device=th.device("cpu")):
        """
        Create batch of (num, 3, 3) rotation matrices
        """
        self.device = device
        if R is None:
            self._R = th.eye(3, device=device).expand(num, 3, 3).clone()
            self.num = num
        elif isinstance(R, th.Tensor):
            assert R.ndimension() == 3, "expects batch (num, 3, 3)"
            self._R = R
            self.num = len(R)
        else:
            raise ValueError("unsupported type")

    def is_valid_rotation(self, atol=1e-5):
        """
        Checks if a batch of matrices are valid rotation matrices within
        absolute tolerance
        """
        I = th.eye(3, device=self.device).expand(self.num, 3, 3)
        ortho_check = th.allclose(
            th.matmul(self._R, self._R.transpose(1, 2)), I, atol=atol
        )
        normal_check = th.allclose(
            th.norm(self._R, dim=2), th.ones(self.num, 3, device=self.device), atol=atol
        )
        det_check = th.allclose(
            th.det(self._R), th.ones(self.num, device=self.device), atol=atol
        )
        return bool(ortho_check & normal_check & det_check)

    @property
    def R(self):
        return self._R

    def __len__(self):
        return self.num

    def __getitem__(self, indices):
        return Rotation3(self._R[indices], device=self.device)

    def __setitem__(self, indices, value):
        if isinstance(value, Rotation3):
            # Assign directly for a single index
            self._R[indices] = value.R
        elif isinstance(value, th.Tensor):
            # Assign directly for a single index
            self._R[indices] = value
        else:
            raise ValueError("Assigned value must be an instance of rotation")

    @classmethod
    def from_quat(self, q: th.tensor, device=th.device("cpu")) -> SelfRotation3:
        """Calculates the 3x3 rotation matrix from a quaternion
                parameterized as (w,x,y,z).

        Output:
            Rot: 3x3 rotation matrix represented as numpy matrix
        """

        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = th.stack(
            [
                th.stack(
                    [
                        w * w + x * x - y * y - z * z,
                        2 * (x * y - w * z),
                        2 * (x * z + w * y),
                    ],
                    dim=1,
                ),
                th.stack(
                    [
                        2 * (x * y + w * z),
                        w * w - x * x + y * y - z * z,
                        2 * (y * z - w * x),
                    ],
                    dim=1,
                ),
                th.stack(
                    [
                        2 * (x * z - w * y),
                        2 * (y * z + w * x),
                        w * w - x * x - y * y + z * z,
                    ],
                    dim=1,
                ),
            ],
            dim=-2,
        )
        return Rotation3(R, device=device)

    @classmethod
    def from_euler_zyx(self, zyx: th.Tensor, device=th.device("cpu")) -> SelfRotation3:
        """Convert euler angle rotation representation to 3x3
                rotation matrix
        Arg:
            zyx: Nx3 tensor containing euler angles
        Output:
            Rot: Nx3x3 rotation matrix
        """

        # Assignment 1, Problem 1.2
        assert zyx.ndimension() == 2 and zyx.shape[1] == 3

        num = len(zyx)
        Rx = Rotation3(num=num, device=device)
        Rx[:, 1, 1] = th.cos(zyx[:, 0])
        Rx[:, 1, 2] = -th.sin(zyx[:, 0])
        Rx[:, 2, 1] = th.sin(zyx[:, 0])
        Rx[:, 2, 2] = th.cos(zyx[:, 0])

        Ry = Rotation3(num=num, device=device)
        Ry[:, 0, 0] = th.cos(zyx[:, 1])
        Ry[:, 0, 2] = th.sin(zyx[:, 1])
        Ry[:, 2, 0] = -th.sin(zyx[:, 1])
        Ry[:, 2, 2] = th.cos(zyx[:, 1])

        Rz = Rotation3(num=num, device=device)
        Rz[:, 0, 0] = th.cos(zyx[:, 2])
        Rz[:, 0, 1] = -th.sin(zyx[:, 2])
        Rz[:, 1, 0] = th.sin(zyx[:, 2])
        Rz[:, 1, 1] = th.cos(zyx[:, 2])
        R = Rz.R @ Ry.R @ Rx.R
        rot = Rotation3(R, device=device)
        return rot

    @property
    def T(self) -> th.Tensor:
        return self._R.transpose(1, 2)

    @property
    def x_axis(self) -> th.Tensor:
        return self._R[:, :, 0]

    @property
    def y_axis(self) -> th.Tensor:
        return self._R[:, :, 1]

    @property
    def z_axis(self) -> th.Tensor:
        return self._R[:, :, 2]

    def to_euler_zyx(self) -> th.Tensor:
        """
        returned (N, 3) in order of axes x, y, z (rpy)
        https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
        """
        roll = safe_atan2(self._R[:, 2, 1], self._R[:, 2, 2])  # x
        pitch = -th.asin(self._R[:, 2, 0].clamp(-1.0, 1.0))  # y
        yaw = safe_atan2(self._R[:, 1, 0], self._R[:, 0, 0])  # z
        zyx = th.stack([roll, pitch, yaw], dim=-1)
        return zyx

    def roll(self) -> th.Tensor:
        """Extracts the phi component from the rotation matrix"""
        phi = safe_atan2(self._R[:, 2, 1], self._R[:, 2, 2])
        return phi

    def pitch(self) -> th.Tensor:
        """Extracts the theta component from the rotation matrix"""
        theta = -th.asin(self._R[:, 2, 0].clamp(-1.0, 1.0))
        return theta

    def yaw(self) -> th.Tensor:
        """Extracts the psi component from the rotation matrix"""
        psi = safe_atan2(self._R[:, 1, 0], self._R[:, 0, 0])
        return psi

    def to_quat(self):
        """Calculates a quaternion from the class variable
                self.R and returns it

        Output:
            q: An instance of the Quaternion class parameterized
                as [w, x, y, z]
        """
        eulers = self.to_euler_zyx()
        phi, theta, psi = eulers[:, 0], eulers[:, 1], eulers[:, 2]

        w = th.cos(phi / 2) * th.cos(theta / 2) * th.cos(psi / 2) + th.sin(
            phi / 2
        ) * th.sin(theta / 2) * th.sin(psi / 2)
        x = th.sin(phi / 2) * th.cos(theta / 2) * th.cos(psi / 2) - th.cos(
            phi / 2
        ) * th.sin(theta / 2) * th.sin(psi / 2)
        y = th.cos(phi / 2) * th.sin(theta / 2) * th.cos(psi / 2) + th.sin(
            phi / 2
        ) * th.cos(theta / 2) * th.sin(psi / 2)
        z = th.cos(phi / 2) * th.cos(theta / 2) * th.sin(psi / 2) - th.sin(
            phi / 2
        ) * th.sin(theta / 2) * th.cos(psi / 2)
        return th.stack([w, x, y, z], dim=1)

    def to(self, device):
        self._R = self._R.to(device)
        return self

    def clone(self):
        return Rotation3(self._R.clone())

    def detach(self):
        return Rotation3(self._R.detach())
