from typing import List, Dict, Optional, Tuple, Union
import torch as th
import numpy as np
from gymnasium import spaces
from habitat_sim import SensorType
from abc import abstractmethod
from dacite import from_dict

from pylogtools import timerlog, colorlog

from .dynamics import PointMassDynamics
from ..utils import Rotation3
from ..utils.type import Uniform, Normal, Cylinder
from ..common import habitat_to_std
from .scene_manager import SceneManager, Bounds


class BaseEnv:
    def __init__(
        self,
        num_envs: int = 1,
        seed: int = 42,
        visual: bool = False,
        single_env: bool = False,
        max_episode_steps: int = 1000,
        device: Optional[th.device] = th.device("cpu"),
        requires_grad: bool = False,
        robot_radius: float = 0.1,
        dynamics_kwargs=None,
        random_kwargs=None,
        base_action=[0.0, 0.0, 0.0],
        bounds: Optional[Union[Bounds, Dict]] = None,
        scene_kwargs: Optional[Dict] = None,
        sensor_kwargs: Optional[List] = None,
    ):
        # constants
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.robot_radius = robot_radius
        self.seed = seed
        self.visual = visual
        self.device = device
        self.requires_grad = requires_grad

        dynamics_kwargs = dynamics_kwargs or {}
        self.dynamics = PointMassDynamics(
            N=self.num_envs, device=device, **dynamics_kwargs
        )

        # expose as read-only with @property
        self._sensor_obs = {}
        self._is_collision = None
        self._collision_dis = None
        self._collision_point = None
        self._collision_vector = None
        self._last_loss_metrics = {}
        self._loss_metrics_running = {}

        # state
        self._step_count = th.zeros((self.num_envs,), dtype=th.int32, device=device)
        self._reward = th.zeros((self.num_envs,), device=device)
        self._rewards = th.zeros((self.num_envs,), device=device)
        self._action = th.zeros((self.num_envs, 4))
        self._success = th.zeros(self.num_envs, dtype=bool, device=device)
        self._done = th.zeros(self.num_envs, dtype=bool, device=device)
        self._info = [
            {"TimeLimit.truncated": False, "loss_metrics": {}}
            for _ in range(self.num_envs)
        ]

        # state generators
        self.random_kwargs = random_kwargs or {}
        self.position_rng = self._create_rng("position", self.random_kwargs)
        self.velocity_rng = self._create_rng("velocity", self.random_kwargs)
        self.rotation_rng = self._create_rng("rotation", self.random_kwargs)
        self.velocity_noise_rng = self._create_rng("velocity_noise", self.random_kwargs)
        self.rotation_noise_rng = self._create_rng("rotation_noise", self.random_kwargs)
        self.gravity_rng = self._create_rng(
            "gravity",
            self.random_kwargs,
            default_rng=Uniform([0.0, 0.0, -9.81], [0.0, 0.0, 0.0]),
        )

        self.scene_manager = None
        self.single_env = single_env
        if visual:
            num_scene = 1 if single_env else num_envs
            num_agent_per_scene = num_envs if single_env else 1
            scene_kwargs = scene_kwargs or {}
            self.scene_manager = SceneManager(
                num_scene=num_scene,
                num_agent_per_scene=num_agent_per_scene,
                multi_drone=False,
                sensor_settings=sensor_kwargs,
                env_device=self.device,
                **scene_kwargs,
            )
            self._scene_ids = (
                th.arange(self.num_envs, device=self.device)
                // int(self.scene_manager.num_agent_per_scene)
            )
            self._is_out_bounds_buf = th.zeros(
                self.num_envs, dtype=th.bool, device=self.device
            )
            # the cloud backend answers collision queries with a GPU cdist
            # over a per-scene surface-point cloud (matching the reference
            # implementation's default collision_backend: cloud)
            self._cloud_active = self.scene_manager.collision_backend == "cloud"
        else:
            self._scene_ids = None
            self._is_out_bounds_buf = None
            self._cloud_active = False

        # scene bounding boxes (automatically reset when out-of-bounds)
        if not visual:
            bounds = from_dict(Bounds, bounds) if type(bounds) == dict else bounds
            self._bboxes = self._create_bbox(bounds)
            self._flatten_bboxes = [bbox.flatten() for bbox in self._bboxes]
        self._sensor_list = (
            [sensor["uuid"] for sensor in sensor_kwargs]
            if sensor_kwargs is not None
            else []
        )
        self._visual_sensor_list = [s for s in self._sensor_list if "IMU" not in s]
        self._sensor_kwargs = sensor_kwargs

        # NOTE self.state_size depends on state property which can be overridden
        if visual:
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.state_size,),
                        dtype=np.float32,
                    )
                }
            )
            for sensor_setting in self.scene_manager.sensor_settings:
                if "depth" in sensor_setting["sensor_type"]:
                    max_depth = np.inf
                    self.observation_space.spaces[sensor_setting["uuid"]] = spaces.Box(
                        low=0,
                        high=max_depth,
                        shape=[1] + sensor_setting["resolution"],
                        dtype=np.float32,
                    )
                elif "color" in sensor_setting["sensor_type"]:
                    self.observation_space.spaces[sensor_setting["uuid"]] = spaces.Box(
                        low=0,
                        high=255,
                        shape=[3] + sensor_setting["resolution"],
                        dtype=np.uint8,
                    )
        else:
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.state_size,),
                        dtype=np.float32,
                    )
                }
            )

        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.base_action = th.tensor(base_action, device=self.device)

    @abstractmethod
    def get_observation(self) -> dict:
        if not self.requires_grad:
            if self.visual:
                return {
                    "state": self.state.cpu().clone().numpy(),
                    "depth": self.sensor_obs["depth"],
                }
            else:
                return {
                    "state": self.state.cpu().clone().numpy(),
                }
        else:
            if self.visual:
                return {
                    "state": self.state.to(self.device),
                    "depth": th.from_numpy(self.sensor_obs["depth"]).to(self.device),
                }
            else:
                return {
                    "state": self.state.to(self.device),
                }

    @abstractmethod
    def get_reward(self, action=None):
        rewards = th.zeros(self.num_envs)
        return rewards

    @abstractmethod
    def get_success(self) -> th.Tensor:
        success = th.zeros(self.num_envs, dtype=th.bool)
        return success

    @timerlog.timer.timed
    def step(
        self,
        action: th.Tensor,
        target_dir_wf: th.Tensor = None,
        is_test=False,
        device="cpu",
    ):
        """
        is_test: if false, agents that are done are automatically reset
        """
        # by default, use a target direction of +1 in the x axis for orientation
        # child classes should override this behavior if necessary
        action = action.to(self.device)
        target_dir_wf = (
            th.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            if target_dir_wf is None
            else target_dir_wf
        )

        # we're predicting the delta acceleration from hover
        self.dynamics.step(action + self.base_action, target_dir_wf)

        if self.visual:
            self.scene_manager.set_pose(self.position, self.rot_wb.to_quat())
            self.update_observation()
        self.update_collision()

        observations = self.get_observation()
        self._step_count += 1
        self._success = self.get_success()  # .cpu()
        ret_val = self.get_reward(action)
        if type(ret_val) == tuple:
            self._reward = ret_val[0]  # .cpu()
            loss_metrics = ret_val[1]
        else:
            self._reward = ret_val  # .cpu()
            loss_metrics = {}
        self._last_loss_metrics = loss_metrics
        self._rewards += self._reward
        previous_done = self._done.clone()
        # self._done = self._done | self._success | self.is_collision.cpu() | (self._step_count >= self.max_episode_steps)
        self._done = (
            self._done
            | self._success
            | self.is_collision
            | (self._step_count >= self.max_episode_steps)
        )

        # keep a moving average of loss_metrics (tensorized across envs)
        for metric in loss_metrics:
            if metric not in self._loss_metrics_running:
                self._loss_metrics_running[metric] = th.zeros(
                    self.num_envs, device=self.device
                )
            count = self._step_count.float()
            self._loss_metrics_running[metric] = self._loss_metrics_running[
                metric
            ] + (loss_metrics[metric].to(self.device) - self._loss_metrics_running[metric]) / (
                count + 1
            )

        # record _info for agents that finished on this step
        done_indices = (self._done & ~previous_done).nonzero(as_tuple=True)[0]
        if len(done_indices) > 0:
            done_rewards = self._rewards[done_indices].detach().cpu().numpy()
            done_step_counts = self._step_count[done_indices].cpu().numpy()
            done_success = self._success[done_indices].cpu().numpy()
            done_metrics_cpu = {}
            for metric in self._loss_metrics_running:
                done_metrics_cpu[metric] = (
                    self._loss_metrics_running[metric][done_indices]
                    .detach()
                    .cpu()
                    .numpy()
                )
            for j in range(len(done_indices)):
                indice = done_indices[j].item()
                self._info[indice]["is_success"] = bool(done_success[j])
                self._info[indice]["episode_reward"] = float(done_rewards[j])
                self._info[indice]["episode_avg_step_reward"] = float(
                    done_rewards[j] / max(done_step_counts[j], 1)
                )
                self._info[indice]["episode_length"] = int(done_step_counts[j])
                self._info[indice]["episode_duration"] = float(
                    done_step_counts[j] * self.dynamics.ctrl_dt
                )
                for metric in done_metrics_cpu:
                    self._info[indice]["loss_metrics"][metric] = float(
                        done_metrics_cpu[metric][j]
                    )
                # the following are used by sb3 ppo
                if self.requires_grad:
                    self._info[indice]["terminal_observation"] = {
                        key: observations[key][indice].detach().cpu()
                        for key in observations.keys()
                    }
                else:
                    self._info[indice]["terminal_observation"] = {
                        key: observations[key][indice].cpu()
                        for key in observations.keys()
                    }
                if done_step_counts[j] >= self.max_episode_steps:
                    self._info[indice]["TimeLimit.truncated"] = True

        if not is_test:
            # reset dead agents
            _reward = self._reward.clone()
            _done = self._done.clone()
            _info = self._info.copy()
            if self._done.any():
                self.reset_agents(th.where(self._done)[0])
            return self.get_observation(), _reward, _done, _info

        return observations, self._reward, self._done, self._info

    def reset(self):
        if self.visual:
            self.scene_manager.load_scenes()
        self.reset_agents()
        return self.get_observation()

    def reset_agents(
        self,
        indices: Optional[List] = None,
        pos: Optional[th.Tensor] = None,
        vel: Optional[th.Tensor] = None,
        start_rot: Optional[th.Tensor] = None,
        delta_rot: Optional[th.Tensor] = None,
        gravity: Optional[th.Tensor] = None,
    ):
        og_indices = indices
        if indices is None:
            self._reward = th.zeros(self.num_envs, device=self.device)
            self._rewards = th.zeros(self.num_envs, device=self.device)
            self._done = th.zeros(self.num_envs, dtype=bool, device=self.device)
            self._step_count = th.zeros(
                self.num_envs, dtype=th.int32, device=self.device
            )
            self._loss_metrics_running = {}
        else:
            # buffers may have been created on a different device (e.g. cpu in
            # __init__ or from get_reward results); ensure they match indices
            self._reward = self._reward.to(self.device)
            self._rewards = self._rewards.to(self.device)
            self._done = self._done.to(self.device)
            self._step_count = self._step_count.to(self.device)
            self._reward = self._reward.scatter(
                0, indices, th.zeros(len(indices), device=self.device)
            )
            self._rewards = self._rewards.scatter(
                0, indices, th.zeros(len(indices), device=self.device)
            )
            self._done = self._done.scatter(
                0, indices, th.zeros(len(indices), dtype=th.bool, device=self.device)
            )
            self._step_count[indices] = 0
            for metric in self._loss_metrics_running:
                self._loss_metrics_running[metric][indices] = 0.0

        indices = th.arange(self.num_envs) if indices is None else indices
        size = (len(indices), 3)
        for i in indices:
            self._info[i] = {
                "TimeLimit.truncated": False,
                "loss_metrics": {},
            }

        pos = (
            self.safe_generate(self.position_rng, indices, 1.0).to(self.device)
            if pos is None
            else pos
        )
        vel = self.velocity_rng.generate(size).to(self.device) if vel is None else vel
        gravity = (
            self.gravity_rng.generate(size).to(self.device)
            if gravity is None
            else gravity
        )
        start_rot = (
            Rotation3(num=len(indices), device=self.device)
            if start_rot is None
            else start_rot
        )
        if delta_rot is None:
            # delta_rot is a small euler angle pertubation in the body frame
            euler_zyx = self.rotation_rng.generate(size).to(self.device)
            delta_rot = Rotation3.from_euler_zyx(euler_zyx, device=self.device)
        rot = Rotation3(start_rot.R @ delta_rot.R)
        self.dynamics.reset(pos=pos, rot=rot, vel=vel, gravity=gravity, indices=indices)
        if self.visual:
            self.scene_manager.reset_agents(
                std_positions=pos, std_orientations=rot.to_quat(), indices=indices
            )
            self.update_observation(og_indices)

        self.update_collision(indices)

    def update_observation(self, indices: Optional[int] = None):
        if indices is None:
            img_obs = self.scene_manager.get_observation()
        else:
            img_obs = self.scene_manager.get_observation(indices)
            sensor_device = "cuda" if self.scene_manager.gpu2gpu else "cpu"
            indices = indices.to(sensor_device)
        for sensor_uuid in self._visual_sensor_list:
            if "depth" in sensor_uuid:
                sensor_spec = [
                    sensor
                    for sensor in self._sensor_kwargs
                    if sensor["uuid"] == sensor_uuid
                ][-1]
                near_clip = sensor_spec.get("near", 0.05)
                far_clip = sensor_spec.get("far", 20.0)
                img = th.stack(
                    [each_agent_obs[sensor_uuid] for each_agent_obs in img_obs], dim=0
                ).unsqueeze(1)
                # set invalid pixels to max range
                img = th.nan_to_num(img, nan=far_clip)
                img[img < near_clip] = far_clip
                img[img > far_clip] = far_clip
            elif "color" in sensor_uuid:
                img = th.transpose(
                    th.stack(
                        [each_agent_obs[sensor_uuid] for each_agent_obs in img_obs]
                    )[..., :3],
                    (0, 3, 1, 2),
                )
            elif "semantic" in sensor_uuid:
                img = th.stack(
                    [each_agent_obs[sensor_uuid] for each_agent_obs in img_obs]
                )
            else:
                raise KeyError("Can not find uuid of sensors")
            if indices is None or sensor_uuid not in self._sensor_obs:
                self._sensor_obs[sensor_uuid] = img
            else:
                self._sensor_obs[sensor_uuid][indices] = img
        # self._sensor_obs["IMU"] = self._generate_noise_obs("IMU")

    def _create_rng(
        self,
        key: str,
        random_kwargs: Dict,
        default_rng=Uniform([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ):
        aliases = {"uniform": Uniform, "normal": Normal, "cylinder": Cylinder}
        try:
            rng_class = aliases[random_kwargs[key]["class"]]
            mean = random_kwargs[key]["mean"]
            half = random_kwargs[key]["half"]
            return rng_class(mean, half)
        except:
            return default_rng

    @timerlog.timer.timed
    def safe_generate(
        self, rng: Union[Uniform, Normal], scene_ids, safe_radius, max_iter=1000
    ):
        """
        generate one sample of size (3,) for each scene_id in indices, until
        no samples are in collision with objects in the scene.
        returns samples from rng of size (len(indices), 3)
        """
        size = (len(scene_ids), 3)
        samples = rng.generate(size)
        if not self.visual:
            return samples

        # check for collision with visual environments
        for i, scene_id in enumerate(scene_ids):
            for _ in range(max_iter):
                scene_id = 0 if self.single_env else scene_id
                is_collision = bool(
                    self.scene_manager.get_point_is_collision(
                        std_positions=samples[i].unsqueeze(0),
                        scene_id=int(scene_id),
                        uav_radius=safe_radius,
                    )
                )
                if not is_collision:
                    break
                samples[i] = rng.generate((1, 3))
        return samples

    def render(self):
        pass

    def _create_bbox(self, bounds: Optional[Bounds] = None) -> List[th.Tensor]:
        """
        Define bounding boxes for scene extents
        return bboxes (N, 2, 3)
        """
        # create a single bbox either user provided or default
        if bounds:
            bboxes = [th.tensor([bounds.min, bounds.max]).to(self.device)]
        else:
            # use some default bounding box
            bboxes = [
                th.tensor([[-2.0, -10.0, 0.0], [18.0, 10.0, 7.0]]).to(self.device)
            ]

        return bboxes

    def _update_collision_cloud(self, indices: Optional[List[int]] = None) -> None:
        """Cloud-backend collision update: GPU cdist query against the
        per-scene surface-point clouds (mirrors the reference implementation)."""
        position = self.dynamics.position
        if indices is None:
            pos = position
            sids = self._scene_ids
            point, _dis, out = self.scene_manager.get_collision_query(pos, sids)
            self._collision_point = point
            self._is_out_bounds_buf = out
        else:
            idx = indices.to(self.device).long()
            pos = position[idx]
            sids = self._scene_ids[idx]
            point, _dis, out = self.scene_manager.get_collision_query(pos, sids)
            if self._collision_point is None:
                self._collision_point = th.empty(
                    (self.num_envs, 3), device=self.device
                )
            self._collision_point[idx] = point
            self._is_out_bounds_buf[idx] = out
        self._collision_vector = self._collision_point - position
        self._collision_dis = self._collision_vector.norm(dim=1)
        self._is_collision = (
            self._collision_dis < self.robot_radius
        ) | self._is_out_bounds_buf

    @timerlog.timer.timed
    def update_collision(self, indices: Optional[List[int]] = None):
        if self._cloud_active:
            self._update_collision_cloud(indices)
            return
        if not self.visual:
            if indices is None or self._collision_point is None:
                # get the index of the closest bound from the current position
                value, index = th.hstack(
                    [
                        self.position.clone().detach() - self._bboxes[0][0],
                        self._bboxes[0][1] - self.position.clone().detach(),
                    ]
                ).min(dim=1)
                # collision point is current position projected onto closest bound
                self._collision_point = self.position.clone().detach()
                self._collision_point[th.arange(self.dynamics.N), index % 3] = (
                    self._flatten_bboxes[0][index]
                )
            else:
                value, index = th.hstack(
                    [
                        self.position[indices].clone().detach() - self._bboxes[0][0],
                        self._bboxes[0][1] - self.position[indices].clone().detach(),
                    ]
                ).min(dim=1)
                self._collision_point[indices] = self.position[indices].clone().detach()
                self._collision_point[indices, index % 3] = self._flatten_bboxes[0][
                    index
                ]

            self._is_out_bounds = (self.position < self._bboxes[0][0]).any(dim=1) | (
                self.position > self._bboxes[0][1]
            ).any(dim=1)

        else:
            if indices is None or self._collision_point is None:
                self._collision_point = self.scene_manager.get_collision_point().to(
                    self.device
                )
            else:
                self._collision_point[indices] = self.scene_manager.get_collision_point(
                    indices=indices.cpu()
                ).to(self.device)
            self._is_out_bounds = self.scene_manager.is_out_bounds.to(self.device)
        self._collision_vector = self._collision_point - self.position
        self._collision_dis = (self._collision_vector - 0).norm(dim=1)
        self._is_collision = (
            self._collision_dis < self.robot_radius
        ) | self._is_out_bounds

    def detach(self):
        self.dynamics.detach()
        self._rewards = self._rewards.clone().detach()
        self._reward = self._reward.clone().detach()
        self._action = self._action.clone().detach()
        self._step_count = self._step_count.clone().detach()
        self._done = self._done.clone().detach()

    def close(self):
        self.scene_manager.close() if self.visual else None

    def geodesic_cost(self, positions):
        if self.single_env:
            cost = self.scene_manager.interpolate_geodesic(0, positions, gradient=False)
            cost = cost.squeeze(-1)
        else:
            raise NotImplementedError
        return cost

    def geodesic_gradient(self, positions):
        if self.single_env:
            gradient = self.scene_manager.interpolate_geodesic(0, positions)
        else:
            raise NotImplementedError
        return gradient

    @property
    def t(self):
        return self.dynamics.t

    @property
    def position(self):
        return self.dynamics.position

    @property
    def start_position(self):
        return self.dynamics.start_position

    @property
    def euler(self):
        return self.dynamics.euler

    @property
    def quaternion(self):
        return self.dynamics.quaternion

    @property
    def quaternion_sb(self):
        return self.dynamics.quaternion_sb

    @property
    def velocity_sb(self):
        return self.dynamics.velocity_sb

    @property
    def rotation(self):
        return self.dynamics.rotation

    @property
    def rot_wb(self):
        return self.dynamics.rot_wb

    @property
    def rot_ws(self):
        return self.dynamics.rot_ws

    @property
    def velocity(self):
        return self.dynamics.velocity

    @property
    def velocity_bf(self):
        return self.dynamics.velocity_bf

    @property
    def moving_average_velocity(self):
        return self.dynamics.moving_average_velocity

    @property
    def exp_moving_average_velocity(self):
        return self.dynamics.exp_moving_average_velocity

    @property
    def speed(self):
        return self.dynamics.speed

    @property
    def acceleration(self):
        return self.dynamics.acceleration

    @property
    def jerk(self):
        return self.dynamics.jerk

    @property
    def is_collision(self):
        return self._is_collision

    @property
    def is_out_bounds(self):
        return self._is_out_bounds

    @property
    def done(self):
        return self._done

    @property
    def collision_vector(self):
        return self._collision_vector

    @property
    def collision_dis(self):
        return self._collision_dis

    @property
    def collision_point(self):
        return self._collision_point

    @property
    def sensor_obs(self):
        return self._sensor_obs

    @property
    @abstractmethod
    def state(self):
        return th.hstack(
            [
                self.position,
                self.quaternion,
                self.velocity,
            ]
        )

    @property
    def state_size(self):
        return self.state.shape[1]

    @property
    def omega(self):
        return self.dynamics.omega
