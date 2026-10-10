import math
import torch
import torch as th
from typing import Union, List, Tuple, Optional, Dict
import torch.nn.functional as F
from enum import Enum

from clearnav.utils import Rotation3, is_multiple
# from pylogtools import timerlog

CONTROL_LATENCY_CONV_MAX_CONTRIB = 0.01  # unitless


class ACTION_TYPE(Enum):
    THRUST_BODY_FRAME = 0
    THRUST_WORLD_FRAME = 1
    THRUST_START_FRAME = 2


class PointMassDynamics:
    action_type_alias: Dict = {
        "thrust_body_frame": ACTION_TYPE.THRUST_BODY_FRAME,
        "thrust_world_frame": ACTION_TYPE.THRUST_WORLD_FRAME,
        "thrust_start_frame": ACTION_TYPE.THRUST_START_FRAME,
        # can add future support for other action types
    }

    def __init__(
        self,
        N: int = 1,
        action_type: str = "thrust_body_frame",
        dt: float = 0.02,
        ctrl_dt: float = 0.02,
        ctrl_delay: float = 0.02,
        grad_decay: float = 0.92,
        exp_smoothing_factor: float = 12.0,
        exp_smoothing_window_dt: float = 1.0,
        avg_velocity_window_dt: float = 2.0,
        vel_smoothing_factor: float = 0.5,
        enable_air_drag: bool = False,
        enable_ctrl_smoothing: bool = False,
        air_drag_theta_1: float = 0.1,
        air_drag_theta_2: float = 0.1,
        device: th.device = th.device("cpu"),
    ):
        assert action_type in self.action_type_alias.keys()
        assert is_multiple(avg_velocity_window_dt, ctrl_dt)
        assert is_multiple(exp_smoothing_window_dt, ctrl_dt)
        assert is_multiple(ctrl_dt, dt)

        # constants
        self.g = th.tensor([[0.0, 0.0, -9.81]], device=device).expand(N, 3)
        self.N = N
        self.action_type = self.action_type_alias[action_type]
        self.ctrl_dt = ctrl_dt
        self.dt = dt
        self.integrator_steps = round(ctrl_dt / dt)
        self.ctrl_delay = ctrl_delay
        self.grad_decay = grad_decay
        self.exp_smoothing_factor = exp_smoothing_factor
        self.device = device
        self.avg_velocity_window_len = round(avg_velocity_window_dt / ctrl_dt)
        self.exp_smoothing_window_len = round(exp_smoothing_window_dt / ctrl_dt)
        self.enable_ctrl_smoothing = enable_ctrl_smoothing
        self.enable_air_drag = enable_air_drag
        self.air_drag_theta_1 = air_drag_theta_1
        self.air_drag_theta_2 = air_drag_theta_2
        self.default_action = -self.g

        # can't apply smoothing in body frame, as reference frame is always changing
        assert not (
            enable_ctrl_smoothing and self.action_type == ACTION_TYPE.THRUST_BODY_FRAME
        )
        if enable_ctrl_smoothing:
            # control smoothing weights should follow an exponential decay and sum to 1
            t = (
                th.arange(self.exp_smoothing_window_len, device=self.device)
                * self.ctrl_dt
            )
            weights = self.exp_smoothing_factor * th.exp(
                -self.exp_smoothing_factor * (t - self.ctrl_delay)
            )
            weights = torch.where(
                t >= self.ctrl_delay, weights, torch.zeros_like(weights)
            )
            weights = weights / weights.sum()  # make a percentage
            weights = weights.unsqueeze(0).unsqueeze(-1)  # add dimensions (1, H, 1)
            self.smoothing_weights = weights.expand(
                self.N, self.exp_smoothing_window_len, 3
            )  # (N, H, 3)

        # exp moving average velocity weights
        t = th.arange(self.avg_velocity_window_len, device=self.device) * self.ctrl_dt
        weights = vel_smoothing_factor * th.exp(-vel_smoothing_factor * t)
        weights = weights / weights.sum()  # make a percentage
        weights = weights.unsqueeze(0).unsqueeze(-1)  # add dimensions (1, H, 1)
        self.velocity_smoothing_weights = weights.expand(
            self.N, self.avg_velocity_window_len, 3
        )  # (N, H, 3)

        # iterative parameters exposed as read-only with @property
        self._position = th.zeros((self.N, 3), device=self.device)
        self._start_position = th.zeros((self.N, 3), device=self.device)
        self._rot_wb = Rotation3(num=self.N, device=self.device)
        self._rot_ws = Rotation3(num=self.N, device=self.device)
        self._velocity = th.zeros((self.N, 3), device=self.device)
        self._omega = th.zeros((self.N, 3), device=self.device)
        self._acceleration = th.zeros((self.N, 3), device=self.device)
        self._last_acceleration = th.zeros((self.N, 3), device=self.device)
        self._moving_average_velocity = th.zeros(
            (self.N, self.avg_velocity_window_len, 3), device=self.device
        )
        self._action_history = self.default_action.unsqueeze(1).repeat(
            1, self.exp_smoothing_window_len, 1
        )
        self._t = th.zeros(self.N, device=self.device)

        self._grad_decay_factor = th.exp(
            th.tensor(-self.grad_decay * self.dt, device=self.device)
        )
        self._position_hook_handle = None
        self._velocity_hook_handle = None

    def reset(
        self,
        pos: Union[th.Tensor, None] = None,
        rot: Union[Rotation3, None] = None,
        vel: Union[th.Tensor, None] = None,
        gravity: Union[th.Tensor, None] = None,
        indices: Optional[List] = None,
    ):
        if indices is None:
            self._position = (
                th.zeros((self.N, 3), device=self.device) if pos is None else pos
            )
            self._start_position = (
                th.zeros((self.N, 3), device=self.device) if pos is None else pos
            )
            self._rot_wb = (
                Rotation3(num=self.N, device=self.device) if rot is None else rot
            )
            self._rot_ws = (
                Rotation3(num=self.N, device=self.device) if rot is None else rot
            )
            self._velocity = (
                th.zeros((self.N, 3), device=self.device) if vel is None else vel
            )
            self._t = th.zeros(self.N, device=self.device)
            self._acceleration = th.zeros((self.N, 3), device=self.device)
            self._last_acceleration = th.zeros((self.N, 3), device=self.device)
            self._moving_average_velocity = th.zeros(
                (self.N, self.avg_velocity_window_len, 3), device=self.device
            )
            self._action_history = self.default_action.unsqueeze(1).repeat(
                1, self.exp_smoothing_window_len, 1
            )
            self.g = th.tensor([[0.0, 0.0, -9.81]], device=self.device).expand(
                self.N, 3
            )
        else:
            # NOTE we use th.scatter to prevent in-place ops which are not
            # supported by autograd, as it will break the computation graph
            indices = indices.to(self.device)
            indices3 = indices.unsqueeze(1).expand(-1, 3)
            self._position = self._position.scatter(0, indices3, pos)
            self._start_position = self._start_position.scatter(0, indices3, pos)
            self._velocity = self._velocity.scatter(0, indices3, vel)
            self.g = self.g.scatter(0, indices3, gravity)

            indices_rot = indices.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
            self._rot_wb = Rotation3(
                self._rot_wb.R.scatter(0, indices_rot, rot.R), device=self.device
            )
            self._rot_ws = Rotation3(
                self._rot_ws.R.scatter(0, indices_rot, rot.R), device=self.device
            )

            self._t = self._t.scatter(
                0, indices, th.zeros(len(indices), device=self.device)
            )
            self._acceleration = self._acceleration.scatter(
                0, indices3, th.zeros((len(indices), 3), device=self.device)
            )
            self._last_acceleration = self._last_acceleration.scatter(
                0, indices3, th.zeros((len(indices), 3), device=self.device)
            )
            self._omega = self._omega.scatter(
                0, indices3, th.zeros((len(indices), 3), device=self.device)
            )

            indices_moving_avg = (
                indices.unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, self.avg_velocity_window_len, 3)
            )
            self._moving_average_velocity = self._moving_average_velocity.scatter(
                0,
                indices_moving_avg,
                th.zeros(
                    (len(indices), self.avg_velocity_window_len, 3), device=self.device
                ),
            )

            indices_action_hist = (
                indices.unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, self.exp_smoothing_window_len, 3)
            )
            self._action_history = self._action_history.scatter(
                0,
                indices_action_hist,
                self.default_action.unsqueeze(1).repeat(
                    1, self.exp_smoothing_window_len, 1
                ),
            )

    def apply_control_smoothing(self, action: torch.Tensor):
        assert action.shape[0] == self.N
        # shift history to the right and put the new set of actions at the 0th index
        self._action_history = torch.cat(
            [action.unsqueeze(1), self._action_history[:, :-1, :]], dim=1
        )
        filtered_action = th.sum(self.smoothing_weights * self._action_history, dim=1)
        return filtered_action

    def _calc_air_drag(self, cur_vel: torch.Tensor):
        vel_norm = torch.norm(cur_vel, dim=1, keepdim=True)
        return (
            self.air_drag_theta_1 * vel_norm * cur_vel - self.air_drag_theta_2 * cur_vel
        )

    def step(self, acc_cmd, target_dir):
        # timerlog.timer.tic("step_dynamics")
        if self.action_type == ACTION_TYPE.THRUST_BODY_FRAME:
            self._step_thrust_body_frame(acc_cmd, target_dir)
        elif self.action_type == ACTION_TYPE.THRUST_WORLD_FRAME:
            self._step_thrust_world_frame(acc_cmd, target_dir)
        elif self.action_type == ACTION_TYPE.THRUST_START_FRAME:
            self._step_thrust_start_frame(acc_cmd, target_dir)
        else:
            raise NotImplementedError

        # add gradient decay to position and velocity update
        if self._position.requires_grad:
            self._position_hook_handle = self._position.register_hook(
                lambda grad: grad * self._grad_decay_factor
            )
        if self._velocity.requires_grad:
            self._velocity_hook_handle = self._velocity.register_hook(
                lambda grad: grad * self._grad_decay_factor
            )

        # timerlog.timer.toc("step_dynamics")

    def _rotate_vector_body_to_world(self, vector_bf: th.Tensor):
        """batch multiplies body rotation matrix with body frame vector"""
        vector_wf = th.matmul(
            self._rot_wb.R, vector_bf.unsqueeze(-1)
        )  # (N,3,3) * (N,3,1)
        return vector_wf.squeeze(-1)  # (N,3,1) -> (3,N)

    def _rotate_vector_world_to_body(self, vector_wf: th.Tensor):
        vector_bf = th.matmul(
            self._rot_wb.T, vector_wf.unsqueeze(-1)
        )  # (N,3,3) * (N,3,1)
        return vector_bf.squeeze(-1)  # (N,3,1) -> (3,N)

    def _rotate_vector_start_to_world(self, vector_sf: th.Tensor):
        """batch multiplies start rotation matrix with start frame vector"""
        vector_wf = th.matmul(
            self._rot_ws.R, vector_sf.unsqueeze(-1)
        )  # (N,3,3) * (N,3,1)
        return vector_wf.squeeze(-1)  # (N,3,1) -> (3,N)

    def _step_thrust_body_frame(self, acc_cmd_bf: th.Tensor, target_dir_bf: th.Tensor):
        """
        update step given mass normalized thrust and x-axis target vectors
        acc_cmd_wf: (N, 3) the mass normalized thrust vector
        target_dir_wf: (N, 3) target direction vector
        """
        acc_cmd_wf = self._rotate_vector_body_to_world(acc_cmd_bf)
        target_dir_wf = self._rotate_vector_body_to_world(target_dir_bf)
        self._step_thrust_world_frame(acc_cmd_wf, target_dir_wf)

    def _step_thrust_start_frame(self, acc_cmd_sf: th.Tensor, target_dir_sf: th.Tensor):
        """
        update step given mass normalized thrust and x-axis target vectors
        acc_cmd_wf: (N, 3) the mass normalized thrust vector
        target_dir_wf: (N, 3) target direction vector
        """
        acc_cmd_wf = self._rotate_vector_start_to_world(acc_cmd_sf)
        target_dir_wf = self._rotate_vector_start_to_world(target_dir_sf)
        self._step_thrust_world_frame(acc_cmd_wf, target_dir_wf)

    def _step_thrust_world_frame(self, acc_cmd_wf: th.Tensor, target_dir_wf: th.Tensor):
        """
        update step given mass normalized thrust and x-axis target vectors
        acc_cmd_wf: (N, 3) the mass normalized thrust vector
        target_dir_wf: (N, 3) target direction vector
        """
        assert acc_cmd_wf.shape == target_dir_wf.shape == (self.N, 3)
        acc_cmd_wf = acc_cmd_wf.to(self.device)
        target_dir_wf = target_dir_wf.to(self.device)
        g = self.g.to(self.device)

        if self.enable_ctrl_smoothing:
            acc_filt = self.apply_control_smoothing(acc_cmd_wf)  # (N, 3)
        else:
            acc_filt = acc_cmd_wf
        assert acc_filt.shape == (self.N, 3)

        if self.enable_air_drag:
            drag = self._calc_air_drag(self._velocity)
            net_acc = acc_filt + g + drag
        else:
            net_acc = acc_filt + g

        # integrate
        for _ in range(self.integrator_steps):
            self._position, self._velocity = self._integrate(
                cur_pos=self._position,
                cur_vel=self._velocity,
                cur_acc=self._acceleration,
                next_acc=net_acc,
                dt=self.dt,
            )

        # update states
        self._t = self._t + self.ctrl_dt
        # shift velocities to the right by one, and put newest velocity at 0th index
        shifted = torch.roll(self._moving_average_velocity, shifts=1, dims=1)
        self._moving_average_velocity = torch.cat(
            [self._velocity.unsqueeze(1), shifted[:, 1:, :]], dim=1
        )
        self._last_acceleration = self._acceleration
        self._acceleration = net_acc
        rot3 = self._calc_orientation(acc_filt, target_dir_wf, self.device)
        self._omega = self._calc_angular_velocity(
            self._rot_wb, rot3, self.ctrl_dt, self.device
        )
        self._rot_wb = rot3

    @staticmethod
    def _calc_orientation(
        acc_wf: th.Tensor,
        target_dir_wf: th.Tensor,
        device=th.device("cpu"),
        atol: float = 5e-3,
        eps: float = 1e-5,
    ) -> Rotation3:
        """
        acc_wf: (N,3)
        target_dir_wf: (N,3)
        atol: absolute tolerance to check if two floats are close
        eps: epsilon used to prevent division by zero

        returns N rotation matrices corresponding to the frame:
            z: along acc_wf
            x: closest vector towards target while z is constrained
            y: orthogonal to z and x

        ref: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/renderer/cameras.html#look_at_rotation
        """

        target_dir_wf = F.normalize(target_dir_wf, eps=eps)

        # first, constrain z_body to acc_wf, using default up direction if acc is zero
        z_body = F.normalize(acc_wf, eps=eps)
        acc_is_zero = th.isclose(th.norm(z_body, dim=1), th.tensor(0.0), atol=atol)
        z_body[acc_is_zero] = th.tensor([0.0, 0.0, 1.0], device=device)

        # if target_dir_wf and z_body are aligned, nudge target down and forward slightly
        is_same = th.isclose(
            th.norm(th.cross(z_body, target_dir_wf, dim=1), dim=1),
            th.tensor(0.0),
            atol=atol,
        )
        target_dir_wf[is_same] = F.normalize(
            target_dir_wf[is_same] + th.tensor([1e-4, 0.0, -1e-4], device=device)
        )

        y_body = F.normalize(th.cross(z_body, target_dir_wf, dim=1), eps=eps)
        x_body = F.normalize(th.cross(y_body, z_body, dim=1), eps=eps)

        # stack x y z (N, 3) columns into (N, 3, 3)
        rot3 = Rotation3(th.stack([x_body, y_body, z_body], dim=-1), device=device)
        return rot3

    @staticmethod
    def _calc_angular_velocity(
        R1: Rotation3,
        R2: Rotation3,
        dt: float,
        device=th.device("cpu"),
    ) -> th.Tensor:
        """
        args:
            R1: rotation at t-1
            R2: rotation at t
        returns:
            omega_body: (B,3), angular rate from t-1 to t
        """
        N = R1.R.shape[0]

        # calculate relative rotation of desired in the current body frame
        R_rel = R1.T @ R2.R
        rot_rel = Rotation3(R_rel)
        rpy = rot_rel.to_euler_zyx()  # returns roll, pitch, yaw
        rpy_rates = rpy / dt

        # jacobian to convert euler angle rates to angular rates
        # see Mobile Robots Lecture 3 slide 35
        # NB: the Jacobian must be evaluated at the *relative* rotation's
        # angles, not the absolute body->world orientation (R1). Using R1
        # pairs an absolute-orientation map with relative-rotation rates and
        # distorts omega by the body's current orientation.
        theta = rot_rel.pitch()
        phi = rot_rel.roll()
        ones = th.ones(N, device=device)
        zeros = th.zeros(N, device=device)
        J_euler_rate_to_angular_rate = th.stack(
            [
                th.stack([ones, zeros, -th.sin(theta)], dim=1),
                th.stack([zeros, th.cos(phi), th.cos(theta) * th.sin(phi)], dim=1),
                th.stack([zeros, -th.sin(phi), th.cos(theta) * th.cos(phi)], dim=1),
            ],
            dim=1,
        )
        omega_body = J_euler_rate_to_angular_rate @ rpy_rates.unsqueeze(-1)
        omega_body = omega_body.squeeze(-1)  # (N,3,1) -> (3,N)
        return omega_body

    @staticmethod
    def _integrate(cur_pos, cur_vel, cur_acc, next_acc, dt):
        """
        Velocity Verlet integration used in BNL
        Static method because it does not modify object state (plus easier to test)

        pos/vel/acc: th.Tensor (N,3)

        Returns updated position and velocity
        https://en.wikipedia.org/wiki/Verlet_integration#Algorithmic_representation
        """
        new_pos = cur_pos + cur_vel * dt + 0.5 * cur_acc * dt**2
        new_vel = cur_vel + 0.5 * (cur_acc + next_acc) * dt
        return new_pos, new_vel

    def detach(self):
        if self._position_hook_handle is not None:
            self._position_hook_handle.remove()
            self._position_hook_handle = None
        if self._velocity_hook_handle is not None:
            self._velocity_hook_handle.remove()
            self._velocity_hook_handle = None
        self._position = self._position.clone().detach()
        self._start_position = self._start_position.clone().detach()
        self._rot_wb = self._rot_wb.clone().detach()
        self._rot_ws = self._rot_ws.clone().detach()
        self._velocity = self._velocity.clone().detach()
        self._omega = self._omega.clone().detach()
        self._acceleration = self._acceleration.clone().detach()
        self._last_acceleration = self._last_acceleration.clone().detach()
        self._moving_average_velocity = self._moving_average_velocity.clone().detach()
        self._action_history = self._action_history.clone().detach()

    # expose properties as read-only
    @property
    def t(self):
        return self._t

    @property
    def position(self):
        return self._position

    @property
    def start_position(self):
        return self._start_position

    @property
    def velocity(self):
        return self._velocity

    @property
    def velocity_sb(self):
        return th.matmul(self._rot_ws.T, self._velocity.unsqueeze(-1)).squeeze(-1)

    @property
    def velocity_bf(self):
        return self._rotate_vector_world_to_body(self._velocity)

    @property
    def moving_average_velocity(self):
        return self._moving_average_velocity.mean(dim=1)

    @property
    def exp_moving_average_velocity(self):
        weighted_avg_velocity = th.sum(
            self.velocity_smoothing_weights * self._moving_average_velocity, dim=1
        )
        return weighted_avg_velocity

    @property
    def speed(self):
        return self._velocity.norm(dim=1, keepdim=True)

    @property
    def acceleration(self):
        return self._acceleration

    @property
    def jerk(self):
        return (self._acceleration - self._last_acceleration) / self.ctrl_dt

    @property
    def quaternion(self):
        q = self._rot_wb.to_quat()
        # ensure w is positive
        q = torch.where(q[:, 0:1] < 0, -q, q)
        return q

    @property
    def quaternion_sb(self):
        R_sw = self._rot_ws.T
        R_wb = self._rot_wb.R
        R_sb = R_sw @ R_wb
        q = Rotation3(R=R_sb).to_quat()
        # ensure w is positive
        q = torch.where(q[:, 0:1] < 0, -q, q)
        return q

    @property
    def euler(self):
        return self._rot_wb.to_euler_zyx()

    @property
    def rot_wb(self):
        return self._rot_wb

    @property
    def rot_ws(self):
        return self._rot_ws

    @property
    def rotation(self):
        return self._rot_wb.R

    @property
    def t(self):
        return self._t

    @property
    def omega(self):
        return self._omega
