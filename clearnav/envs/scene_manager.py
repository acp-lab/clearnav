import os
import time
import json
import warnings
import numpy as np
from pathlib import Path
from numpy.random import default_rng
import torch as th
import magnum as mn
import trimesh
from typing import List, Union, Optional, Dict
import habitat_sim
from habitat_sim.physics import ManagedRigidObject, RigidObjectManager
from habitat_sim.attributes_managers import ObjectAttributesManager
from dataclasses import dataclass, field
from itertools import cycle
from dacite import from_dict
import quaternion
from scipy.spatial.transform import Rotation as R
from pylogtools import timerlog, colorlog
import open3d as o3d
import heapq
import itertools

from ..common import std_to_habitat, habitat_to_std
from ..utils.type import Uniform, Normal
from ..utils.rotation3 import Rotation3
from .dataloader import SimpleDataLoader, ChildrenPathDataset

import skfmm
import scipy.ndimage

DEBUG = False

origin = mn.Vector3(0.0, 0.0, 0.0)
eye_pos_near = mn.Vector3(0.1, 0.5, 1) * 3
eye_pos_back = mn.Vector3(0, 0.8, 2)
eye_pos_follow_near = mn.Vector3(0.1, 0.5, 1)
eye_pos_follow_back = mn.Vector3(0, 0.5, 1)

# NOTE Color4 channels are BGRA (not RGBA)
opacity = 1.0
red = mn.Color4(0.0, 0.0, 1.0, opacity)
green = mn.Color4(0.0, 1.0, 0.0, opacity)
blue = mn.Color4(1.0, 0.0, 0.0, opacity)
white = mn.Color4(1.0, 1.0, 1.0, opacity)
orange = mn.Color4(0.0, 0.5, 1.0, opacity)
cyan = mn.Color4(1.0, 1.0, 0.0, opacity)

# create a 20 length similar Colorset using orange as primary color
ColorSet3 = []
for i in np.linspace(0, 1, 100):
    color = mn.Color4(1.0 - 0.5 * i, 0.5 * i, 0.5 * i, 1.0)  # RGBA color
    ColorSet3.append(color)


def color_consequence(color1=orange, color2=cyan, factor=1):
    factor = np.array(factor).clip(min=0.0, max=1.0)
    return color1 * (1 - factor) + factor * color2


def calc_camera_transform(
    eye_translation=mn.Vector3(1, 1, 1), lookat=mn.Vector3(0, 0, 0)
):
    # choose y-up to match Habitat's y-up convention
    camera_up = mn.Vector3(0.0, 1.0, 0.0)
    return mn.Matrix4.look_at(
        mn.Vector3(eye_translation), mn.Vector3(lookat), camera_up
    )


@dataclass
class Bounds:
    min: Union[int, List]
    max: Union[int, List]


class UniformObstacleGenerator:
    def __init__(
        self,
        obstacle_sets=["./datasets/clearnav_dataset/configs/objects"],
        set_densities=[0.1],
        seed=None,
        random_kwargs=None,
        num_template_rescales_per_scene=0,
        obstacle_bounds: Optional[Union[Bounds, Dict]] = None,
    ):
        assert len(obstacle_sets) == len(set_densities)
        self._obstacle_sets = [
            self._get_all_children_path(obstacle_set) for obstacle_set in obstacle_sets
        ]
        self.set_densities = set_densities
        self.obstacle_bounds = (
            from_dict(Bounds, obstacle_bounds)
            if type(obstacle_bounds) == dict
            else obstacle_bounds
        )

        # RNGs
        self.rotation_rng = self._create_rng("rotation", random_kwargs)
        self.scale_rng = self._create_rng(
            "scale",
            random_kwargs,
            default_rng=Uniform([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
        )
        self.num_template_rescales_per_scene = num_template_rescales_per_scene

    def _create_rng(
        self,
        key: str,
        random_kwargs: Dict,
        default_rng=Uniform([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ):
        aliases = {"uniform": Uniform, "normal": Normal}
        try:
            rng_class = aliases[random_kwargs[key]["class"]]
            mean = random_kwargs[key]["mean"]
            half = random_kwargs[key]["half"]
            return rng_class(mean, half)
        except:
            return default_rng

    def _get_all_children_path(self, path):
        if os.path.isdir(path):
            file_paths = []
            for root, directories, files in os.walk(path):
                for filename in files:
                    file_paths.append(str.split(filename, ".")[0])
            return file_paths
        else:
            basename = os.path.basename(path)
            directory = os.path.dirname(path)
            file_paths = os.listdir(directory)
            matches = [filename for filename in file_paths if basename in filename]
            return matches

    @timerlog.timer.timed
    def add_obstacles_to_scene(self, scene: habitat_sim.scene):
        template_mgr = scene.get_object_template_manager()
        rigid_mgr = scene.get_rigid_object_manager()
        obstacles = []
        # generate each obstacle set with its corresponding density
        for obstacle_set, set_density in zip(self._obstacle_sets, self.set_densities):
            num_samples, positions, orientations = self.generate_samples(set_density)
            template_ids = self._load_templates(template_mgr, obstacle_set)
            random_indices = th.randint(0, high=len(template_ids), size=(num_samples,))
            for i in range(num_samples):
                new_obj = rigid_mgr.add_object_by_template_id(
                    template_ids[random_indices[i]]
                )
                new_obj.translation = mn.Vector3(positions[i])
                new_obj.rotation = mn.Quaternion(
                    mn.Vector3(*orientations[i][0:3]), orientations[i][3]
                )
                new_obj.motion_type = habitat_sim.physics.MotionType.STATIC
                obstacles.append(new_obj)

        # reload mesh kd-tree so new obstacles get checked for collisions
        scene.recompute_mesh_kdtree()
        return obstacles

    @timerlog.timer.timed
    def remove_obstacles_from_scene(self, scene, obj_refs, reload_kdtree=False):
        rigid_mgr = scene.get_rigid_object_manager()
        for obj in obj_refs:
            rigid_mgr.remove_object_by_id(obj.object_id)
        if reload_kdtree:
            scene.recompute_mesh_kdtree()

    def _load_templates(self, template_mgr, obstacle_set):
        template_ids = []
        for path in obstacle_set:
            id = template_mgr.load_configs(path)[0]
            assert id >= 0
            template_ids.append(id)

            obj_handle = template_mgr.get_template_handle_by_id(id)
            # make rescaled copies of the template random scales
            for i in range(self.num_template_rescales_per_scene):
                template_copy = template_mgr.get_template_by_id(id)
                sample_scale = self.scale_rng.generate(size=(1, 3))
                sample_scale = std_to_habitat(sample_scale)[0][0]
                template_copy.scale = mn.Vector3(*sample_scale)
                obj_scaled_handle = obj_handle + f"_copy_{i}"
                copy_id = template_mgr.register_template(
                    template_copy, obj_scaled_handle
                )
                template_ids.append(copy_id)

        return template_ids

    def generate_samples(self, density):
        # determine the number of points to generate based on density and volume of bounds
        low = th.tensor(self.obstacle_bounds.min)
        high = th.tensor(self.obstacle_bounds.max)
        volume = th.prod(th.abs(high - low))
        num_samples = int(volume * density)

        # Generate random positions within bounds
        positions = low + (high - low) * th.rand((num_samples, 3))
        positions = std_to_habitat(positions, None)[0]

        # Generate rotations
        euler_zyx = self.rotation_rng.generate(size=(num_samples, 3))
        rotation = Rotation3.from_euler_zyx(euler_zyx)
        orientations = rotation.to_quat()

        return num_samples, positions, orientations.numpy()


class PoissonObstacleGenerator(UniformObstacleGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # @timerlog.timer.timed
    def generate_samples(self, density):
        # estimate minimum distance between samples from density
        radius = th.sqrt(th.tensor(2.0 / (th.pi * density))).item()
        lower_bound = self.obstacle_bounds.min[:2]
        upper_bound = self.obstacle_bounds.max[:2]

        # Generate positions from poisson disk sampling
        samples_2d = self.poisson_disk_sampling(radius, lower_bound, upper_bound, k=30)
        num_samples = len(samples_2d)
        z = 0.5 * (self.obstacle_bounds.max[2] - self.obstacle_bounds.min[2])
        positions = th.cat([samples_2d, th.full((num_samples, 1), z)], dim=1)
        positions = std_to_habitat(positions, None)[0]

        # Generate rotations
        euler_zyx = self.rotation_rng.generate(size=(num_samples, 3))
        rotation = Rotation3.from_euler_zyx(euler_zyx)
        orientations = rotation.to_quat()

        return num_samples, positions, orientations.numpy()

    @staticmethod
    def poisson_disk_sampling(radius, lower_bound, upper_bound, k=30):
        """
        Poisson disk sampling in 2D using Bridson's algorithm

        Args:
            radius (float): Minimum distance between points.
            lower_bound (tuple): (x_min, y_min) specifying the lower bound.
            upper_bound (tuple): (x_max, y_max) specifying the upper bound.
            k (int): Number of candidate attempts per active sample before removal.

        Returns:
            A tensor of shape (N, 2) containing sampled 2D positions.
        """
        x_min, y_min = lower_bound
        x_max, y_max = upper_bound
        width, height = x_max - x_min, y_max - y_min

        # Grid cell size (each grid cell contains at most 1 sample)
        cell_size = radius / th.sqrt(th.tensor(2.0))
        grid_width = int(width / cell_size) + 1
        grid_height = int(height / cell_size) + 1

        # Grid to store sample indices (-1 means empty)
        grid = -th.ones((grid_width, grid_height), dtype=th.long)

        # Initial random sample
        first_sample = th.tensor(
            [th.rand(1).item() * width + x_min, th.rand(1).item() * height + y_min]
        )

        samples = [first_sample]
        active_list = [first_sample]

        # Helper function to get grid coordinates
        def grid_coords(point):
            return (point - th.tensor([x_min, y_min])) // cell_size

        # Store the first sample in the grid
        gx, gy = grid_coords(first_sample).long()
        grid[gx, gy] = 0  # Store index in grid

        while active_list:
            idx = th.randint(len(active_list), (1,)).item()
            center = active_list[idx]
            found = False
            for _ in range(k):
                # Generate random point in the ring [r, 2r]
                angle = th.rand(1) * 2 * th.pi
                radius_offset = radius * (1 + th.rand(1))
                new_point = center + radius_offset * th.tensor(
                    [th.cos(angle), th.sin(angle)]
                )
                # Check bounds
                if not (
                    x_min <= new_point[0] <= x_max and y_min <= new_point[1] <= y_max
                ):
                    continue

                # Check neighboring cells for conflicts
                gx, gy = grid_coords(new_point).long()
                x0, x1 = max(gx - 2, 0), min(gx + 3, grid_width)
                y0, y1 = max(gy - 2, 0), min(gy + 3, grid_height)

                valid = True
                for i in range(x0, x1):
                    for j in range(y0, y1):
                        if grid[i, j] != -1:
                            neighbor = samples[grid[i, j]]
                            if th.norm(neighbor - new_point) < radius:
                                valid = False
                                break
                    if not valid:
                        break

                if valid:
                    samples.append(new_point)
                    active_list.append(new_point)
                    grid[gx, gy] = len(samples) - 1
                    found = True

            # Remove if no valid points found
            if not found:
                active_list.pop(idx)

        return th.stack(samples)


class SceneManager:
    obstacle_generator_aliases = {
        "uniform": UniformObstacleGenerator,
        "poisson": PoissonObstacleGenerator,
    }

    def __init__(
        self,
        path: str,
        dataset_path: str = "./datasets/clearnav_dataset",
        spawn_obstacles: bool = False,
        obstacle_generator_class: str = "uniform",
        obstacle_generator_kwargs=None,
        scene_type: str = "json",
        reload_scenes: bool = False,
        load_geodesics: bool = False,
        num_scene: int = 1,
        num_agent_per_scene: Union[int, List[int]] = 1,
        seed: int = 1,
        uav_radius=0.1,
        sensitive_radius=5,
        semantic=False,
        multi_drone: bool = False,
        sensor_settings=None,
        render_settings=None,
        noise_settings=None,
        gpu2gpu=True,
        collision_backend: str = "kdtree",
        cloud_sample_spacing: float = 0.02,
        cloud_boundary_sample_spacing: float = 0.05,
        cloud_boundary_mode: str = "analytic",
        cloud_max_radius: Optional[float] = None,
        env_device=th.device("cpu"),
    ):
        if sensor_settings is None:
            raise ValueError("No sensor settings provided")

        self.dataset_path = os.path.abspath(dataset_path)
        self._scene_dataset_config_path = [
            os.path.join(self.dataset_path, file)
            for file in os.listdir(self.dataset_path)
            if file.endswith(".scene_dataset_config.json")
        ][0]
        self.scene_path = (
            path
            if os.path.isabs(path)
            else os.path.abspath(os.path.join(self.dataset_path, path))
        )
        self.reload_scenes = reload_scenes
        self.num_scene = num_scene
        self.num_agent_per_scene = num_agent_per_scene
        self.num_agent = self.num_agent_per_scene * self.num_scene
        self.seed = seed
        # If habitat is built --with-cuda flag, we can enable this to avoid
        # copying image from gpu -> cpu when we get observations
        self.gpu2gpu = gpu2gpu
        self.sensor_settings = sensor_settings
        self.render_settings = render_settings
        self.noise_settings = noise_settings
        self.drone_radius = uav_radius
        self.sensitive_radius = sensitive_radius
        self.spawn_obstacles = spawn_obstacles
        if spawn_obstacles:
            generator_class = self.obstacle_generator_aliases[obstacle_generator_class]
            self.obstacle_generator = generator_class(**obstacle_generator_kwargs)
            self._obstacles: List[ManagedRigidObject] = [
                None for _ in range(self.num_scene)
            ]
        self.load_geodesics = load_geodesics
        self.geodesics: List[np.lib.npyio.NpyFile] = [
            None for _ in range(self.num_scene)
        ]

        # collision backend: "kdtree" queries habitat's closest-collision
        # API per agent; "cloud" samples obstacle surfaces into a GPU point
        # cloud at load time and answers queries with cdist (plus an analytic
        # AABB boundary). The cloud backend is what the reference
        # implementation trains with (see base config collision_backend).
        self._collision_backend = collision_backend
        self._cloud_sample_spacing = cloud_sample_spacing
        self._cloud_boundary_sample_spacing = cloud_boundary_sample_spacing
        self._cloud_boundary_mode = cloud_boundary_mode
        self._cloud_max_radius = (
            cloud_max_radius if cloud_max_radius is not None else sensitive_radius
        )
        self._collision_clouds: List[Optional[th.Tensor]] = [
            None for _ in range(self.num_scene)
        ]
        self._mesh_cache: Dict[str, object] = {}
        # torch device for env-owned resources (e.g. geodesic grids and
        # collision clouds) so they stay on the same device as the env state
        self._env_device = env_device

        self._data_loader = SimpleDataLoader(
            ChildrenPathDataset(self.scene_path), batch_size=num_scene, shuffle=True
        )
        self._scene_loader = cycle(self._data_loader)
        self.scene_paths: List[str] = [None for _ in range(num_scene)]
        self.scenes: List[habitat_sim.Simulator] = [None for _ in range(num_scene)]
        self.agents: List[List[habitat_sim.Agent]] = [[] for _ in range(num_scene)]

        self._scene_bounds = [None for _ in range(num_scene)]
        self.is_multi_drone = multi_drone

        self._obj_mgrs = None

        if self.render_settings is not None:
            assert "object_path" in self.render_settings
            self._robot_path = str(
                Path(self.render_settings.get("object_path", None)).resolve()
            )
            assert os.path.exists(self._robot_path), self._robot_path
            self.render_settings["line_width"] = self.render_settings.get(
                "line_width", 1.0
            )
            self.render_settings["axes"] = self.render_settings.get("axes", False)
            self.render_settings["trajectory"] = self.render_settings.get(
                "trajectory", False
            )
            self.render_settings["sensor_type"] = self.render_settings.get(
                "sensor_type", "color"
            )
            self.render_settings["mode"] = self.render_settings.get("mode", "fix")
            self.render_settings["view"] = self.render_settings.get("view", "near")
            self.render_settings["resolution"] = self.render_settings.get(
                "resolution", [256, 256]
            )
            self.render_settings["position"] = self.render_settings.get(
                "position", None
            )
            self._render_camera = [None for _ in range(num_scene)]
            self._line_renders: habitat_sim.gfx.DebugLineRender = [
                None for _ in range(num_scene)
            ]

        if self.render_settings is not None or multi_drone:
            self._obj_mgrs: RigidObjectManager = [None for _ in range(num_scene)]
            self._objects = [
                [None for _ in range(num_agent_per_scene)] for _ in range(num_scene)
            ]

        self.trajectory = [
            [[] for _ in range(num_agent_per_scene)] for _ in range(num_scene)
        ]
        self._collision_point = [
            [None for _ in range(num_agent_per_scene)] for _ in range(num_scene)
        ]
        self._is_out_bounds = [
            [False for _ in range(num_agent_per_scene)] for _ in range(num_scene)
        ]

    @property
    def collision_backend(self) -> str:
        return self._collision_backend

    def get_scene_bounds(self, scene_id: int = 0) -> tuple[th.Tensor, th.Tensor]:
        """Returns (min_pos, max_pos) of the scene bounding box in canonical ENU."""
        bb = self._scene_bounds[scene_id]
        scene_min_hab = np.array([bb.min.x, bb.min.y, bb.min.z])
        scene_max_hab = np.array([bb.max.x, bb.max.y, bb.max.z])
        c0, _ = habitat_to_std(scene_min_hab, None)
        c1, _ = habitat_to_std(scene_max_hab, None)
        c0 = c0.squeeze()
        c1 = c1.squeeze()
        scene_min = th.minimum(c0, c1)
        scene_max = th.maximum(c0, c1)
        return scene_min, scene_max

    def _resolve_template_mesh(self, obj):
        """Resolve a placed rigid object's template mesh to a cached trimesh.Trimesh."""
        attrs = obj.creation_attributes
        cache_key = attrs.handle
        if cache_key in self._mesh_cache:
            return self._mesh_cache[cache_key]
        asset_name = attrs.collision_asset_handle or attrs.render_asset_handle
        if not asset_name:
            warnings.warn(f"[cloud] template '{cache_key}' has no mesh asset, skipping")
            return None
        file_dir = attrs.file_directory
        glb_path = os.path.join(file_dir, asset_name) if file_dir else asset_name
        if not os.path.exists(glb_path):
            warnings.warn(f"[cloud] mesh asset not found: {glb_path}, skipping")
            return None
        mesh = trimesh.load(glb_path, force="mesh")
        self._mesh_cache[cache_key] = mesh
        return mesh

    def _build_obstacle_cloud(self, scene_id: int) -> np.ndarray:
        """Sample surface points from every placed rigid-object mesh in the scene."""
        scene = self.scenes[scene_id]
        rigid_mgr = scene.get_rigid_object_manager()
        handles = rigid_mgr.get_object_handles()
        spacing = self._cloud_sample_spacing

        chunks = []
        for handle in handles:
            obj = rigid_mgr.get_object_by_handle(handle)
            if obj is None or not obj.is_alive:
                continue
            mesh = self._resolve_template_mesh(obj)
            if mesh is None or mesh.area <= 0:
                continue
            scale = np.array(obj.scale, dtype=np.float32)
            scaled_local = np.array(mesh.vertices, dtype=np.float32) * scale
            scaled_area = float(
                trimesh.Trimesh(
                    vertices=scaled_local, faces=mesh.faces, process=False
                ).area
            )
            if scaled_area <= 0:
                continue
            n = max(1, int(scaled_area / (spacing ** 2)))
            scaled_mesh = trimesh.Trimesh(
                vertices=scaled_local, faces=mesh.faces, process=False
            )
            local_pts = np.array(scaled_mesh.sample(n), dtype=np.float32)
            transform = np.array(obj.transformation, dtype=np.float32)
            R = transform[:3, :3]
            t = transform[:3, 3]
            hab_pts = local_pts @ R.T + t
            std_pts, _ = habitat_to_std(hab_pts, None)
            chunks.append(std_pts.numpy() if isinstance(std_pts, th.Tensor) else std_pts)

        if not chunks:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def _build_boundary_cloud(self, scene_id: int) -> np.ndarray:
        """Lay a regular grid of points on the 6 AABB faces of the scene bounds."""
        b_min_t, b_max_t = self.get_scene_bounds(scene_id)
        b_min = b_min_t.numpy() if isinstance(b_min_t, th.Tensor) else np.array(b_min_t, dtype=np.float32)
        b_max = b_max_t.numpy() if isinstance(b_max_t, th.Tensor) else np.array(b_max_t, dtype=np.float32)
        bsp = self._cloud_boundary_sample_spacing

        chunks = []
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            lo = np.array([b_min[o] for o in others], dtype=np.float32)
            hi = np.array([b_max[o] for o in others], dtype=np.float32)
            n0 = max(2, int(np.ceil((hi[0] - lo[0]) / bsp)))
            n1 = max(2, int(np.ceil((hi[1] - lo[1]) / bsp)))
            g0 = np.linspace(lo[0], hi[0], n0)
            g1 = np.linspace(lo[1], hi[1], n1)
            gg0, gg1 = np.meshgrid(g0, g1, indexing="ij")
            for face_val in (b_min[axis], b_max[axis]):
                pts = np.zeros((n0 * n1, 3), dtype=np.float32)
                pts[:, axis] = face_val
                pts[:, others[0]] = gg0.ravel()
                pts[:, others[1]] = gg1.ravel()
                chunks.append(pts)

        return np.concatenate(chunks, axis=0).astype(np.float32)

    def _build_collision_cloud(self, scene_id: int) -> None:
        """Build the GPU collision surface-point cloud for a scene."""
        scene = self.scenes[scene_id]
        if scene is None:
            self._collision_clouds[scene_id] = None
            return

        obstacle_pts = self._build_obstacle_cloud(scene_id)
        if self._cloud_boundary_mode == "points":
            boundary_pts = self._build_boundary_cloud(scene_id)
            cloud_np = np.concatenate(
                [obstacle_pts, boundary_pts], axis=0
            ).astype(np.float32)
            n_bound = len(boundary_pts)
        else:
            cloud_np = obstacle_pts.astype(np.float32)
            n_bound = 0
        cloud = th.as_tensor(cloud_np, device=self._env_device)
        self._collision_clouds[scene_id] = cloud
        if DEBUG:
            print(
                f"[cloud] scene {scene_id}: {len(obstacle_pts)} obstacle + "
                f"{n_bound} boundary ({self._cloud_boundary_mode}) = "
                f"{cloud.shape[0]} points"
            )

    def _analytic_boundary(
        self, pts: th.Tensor, b_min: th.Tensor, b_max: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        """Closed-form AABB boundary distance."""
        with th.no_grad():
            d_neg = pts - b_min
            d_pos = b_max - pts
            d_per_axis = th.minimum(d_neg, d_pos)
            interior_dis = d_per_axis.min(dim=1).values
            nearest_axis = d_per_axis.argmin(dim=1)

            clamped = th.clamp(pts, b_min, b_max)
            exterior_dis = (clamped - pts).norm(dim=1)
            is_out = exterior_dis > 0

            m = pts.shape[0]
            ar = th.arange(m, device=pts.device)
            is_min_face = d_neg[ar, nearest_axis] <= d_pos[ar, nearest_axis]
            face_val = th.where(
                is_min_face, b_min[nearest_axis], b_max[nearest_axis]
            )
            face_pt = pts.clone()
            face_pt[ar, nearest_axis] = face_val
            boundary_pt = th.where(is_out.unsqueeze(1), clamped, face_pt)
            boundary_dis = th.where(is_out, exterior_dis, interior_dis)
            boundary_pt = boundary_pt.detach()
        return boundary_pt, boundary_dis

    def get_collision_query(
        self,
        positions_std: th.Tensor,
        scene_ids: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """GPU cdist collision query over the per-scene surface-point cloud."""
        device = self._env_device
        positions_std = positions_std.to(device).float()
        if not isinstance(scene_ids, th.Tensor):
            scene_ids = th.as_tensor(scene_ids, device=device)
        else:
            scene_ids = scene_ids.to(device)

        sort_idx = th.argsort(scene_ids)
        sorted_sids = scene_ids[sort_idx]
        sorted_pos = positions_std[sort_idx]

        unique_sids, counts = th.unique(sorted_sids, return_counts=True)
        blocks = th.split_with_sizes(sorted_pos, counts.tolist())

        point_parts, dis_parts, out_parts = [], [], []
        for sid, pts in zip(unique_sids.tolist(), blocks):
            sid = int(sid)
            cloud = self._collision_clouds[sid]
            b_min, b_max = self.get_scene_bounds(sid)
            b_min = b_min.to(device).float()
            b_max = b_max.to(device).float()
            out_mask = ((pts < b_min) | (pts > b_max)).any(dim=1)

            has_obs = cloud is not None and cloud.shape[0] > 0
            analytic_bnd = self._cloud_boundary_mode == "analytic"

            if not has_obs and not analytic_bnd:
                point_parts.append(pts.detach())
                dis_parts.append(
                    th.full((pts.shape[0],), float("inf"), device=device)
                )
                out_parts.append(out_mask)
                continue

            if has_obs:
                with th.no_grad():
                    d = th.cdist(pts, cloud)
                    min_idx = d.argmin(dim=1)
                    nearest = cloud[min_idx]
                    obs_dis = d.min(dim=1).values
                nearest = nearest.detach().float()
            else:
                nearest = None
                obs_dis = th.full(
                    (pts.shape[0],), float("inf"), device=device
                )

            if analytic_bnd:
                bnd_pt, bnd_dis = self._analytic_boundary(pts, b_min, b_max)
                if nearest is not None:
                    use_boundary = bnd_dis < obs_dis
                    nearest = th.where(
                        use_boundary.unsqueeze(1), bnd_pt, nearest
                    )
                else:
                    nearest = bnd_pt

            if self.is_multi_drone and pts.shape[0] > 1:
                pts_d = pts.detach()
                with th.no_grad():
                    pair = th.cdist(pts_d, pts_d)
                    m = pair.shape[0]
                    pair[th.arange(m, device=device), th.arange(m, device=device)] = (
                        float("inf")
                    )
                    pair_min, pair_idx = pair.min(dim=1)
                    surf_dis = (nearest - pts_d).norm(dim=1)
                    override = pair_min < surf_dis
                drone_nearest = pts_d[pair_idx]
                nearest = th.where(override.unsqueeze(1), drone_nearest, nearest)

            point_parts.append(nearest)
            dis_parts.append((nearest - pts).norm(dim=1))
            out_parts.append(out_mask)

        point_sorted = th.cat(point_parts, dim=0)
        dis_sorted = th.cat(dis_parts, dim=0)
        out_sorted = th.cat(out_parts, dim=0)

        inv_idx = th.argsort(sort_idx)
        return point_sorted[inv_idx], dis_sorted[inv_idx], out_sorted[inv_idx]

    def get_pose(self, indices: Union[List, int] = None):
        """get position and rotation of agents by indice"""
        if indices is None:
            position = np.empty((self.num_agent_per_scene * self.num_scene, 3))
            rotation = np.empty((self.num_agent_per_scene * self.num_scene, 4))
            for scene_id in range(self.num_scene):
                for agent_id in range(self.num_agent_per_scene):
                    state = self.agents[scene_id][agent_id].get_state()
                    std_pos, std_ori = habitat_to_std(
                        state.position, state.rotation.components
                    )
                    position[scene_id * self.num_agent_per_scene + agent_id] = std_pos
                    rotation[scene_id * self.num_agent_per_scene + agent_id] = std_ori

            return position, rotation
        else:
            if not hasattr(indices, "__iter__"):
                indices = [indices]

            position = np.empty((len(indices), 3))
            rotation = np.empty((len(indices), 4))
            for i, indice in enumerate(indices):
                scene_id = indice // self.num_agent_per_scene
                agent_id = indice % self.num_agent_per_scene
                state = self.agents[scene_id][agent_id].get_state()
                position[i] = state.position
                rotation[i] = state.rotation.components
            position, rotation = habitat_to_std(position, rotation)

            return position, rotation

    # @timerlog.timer.timed
    def set_pose(self, position, rotation):
        """set position and rotation of agents in each scene"""
        assert len(position) == len(rotation) == self.num_agent

        hab_pos, hab_ori = std_to_habitat(position, rotation)
        drone_id = 0
        for scene_id in range(self.num_scene):
            for agent_id in range(self.num_agent_per_scene):
                # self.agents[scene_id][agent_id].set_state(
                #     habitat_sim.AgentState(
                #         position=hab_pos[drone_id], rotation=quaternion.from_float_array(hab_ori[drone_id])
                #     )
                # )
                self.agents[scene_id][agent_id].scene_node.translation = hab_pos[
                    drone_id
                ]
                self.agents[scene_id][agent_id].scene_node.rotation = mn.Quaternion(
                    mn.Vector3(hab_ori[drone_id][1:]), hab_ori[drone_id][0]
                )

                # disable updating trajectory for now
                # self.trajectory[scene_id][agent_id].insert(0,
                #     np.hstack([hab_pos[drone_id], hab_ori[drone_id]])
                # )
                drone_id += 1
                if self.is_multi_drone:
                    self._objects[scene_id][
                        agent_id
                    ].root_scene_node.transformation = self.agents[scene_id][
                        agent_id
                    ].scene_node.transformation
                    # self._objects[scene_id][agent_id].scene_node.translation = hab_pos[drone_id]
                    # self._objects[scene_id][agent_id].scene_node.rotation = mn.Quaternion(mn.Vector3(hab_ori[drone_id][1:]), hab_ori[drone_id][0])
        if self._collision_backend == "kdtree":
            self._update_collision_infos()

        if self._obj_mgrs is not None:
            # set the pose of objects or agents in the scene
            for scene_id in range(self.num_scene):
                for agent_id in range(self.num_agent_per_scene):
                    obj = self._objects[scene_id][agent_id]
                    if obj is not None:
                        obj.root_scene_node.transformation = self.agents[scene_id][
                            agent_id
                        ].scene_node.transformation

    @timerlog.timer.timed
    def get_observation(self, indices: Optional[int] = None):
        obses = []
        if indices is None:
            for scene in self.scenes:
                agent_observations = scene.get_sensor_observations(
                    list(range(self.num_agent_per_scene))
                )
                obses += list(agent_observations.values())
        else:
            for index in indices:
                agent_id = index % self.num_agent_per_scene
                scene_id = index // self.num_agent_per_scene
                obses.append(
                    self.scenes[scene_id].get_sensor_observations(int(agent_id))
                )
        return obses

    # @timerlog.timer.timed
    def _update_collision_infos(self, indices: Optional[List] = None):
        """update the collision distance of each agent"""

        indices = np.arange(self.num_agent) if indices is None else indices
        indices = [indices] if not hasattr(indices, "__iter__") else indices

        for indice in indices:
            scene_id = indice // self.num_agent_per_scene
            agent_id = indice % self.num_agent_per_scene

            col_record = self.scenes[scene_id].get_closest_collision_point(
                pt=self.agents[scene_id][agent_id].scene_node.translation,
                max_search_radius=self.sensitive_radius,
            )
            self._collision_point[scene_id][agent_id] = col_record.hit_pos
            self._is_out_bounds[scene_id][agent_id] = col_record.is_out_bound

        if self.is_multi_drone:
            for scene_id in range(self.num_scene):
                positions = np.array(
                    [agent.state.position for agent in self.agents[scene_id]]
                )
                cur_dis = np.linalg.norm(
                    positions
                    - np.array(
                        [
                            self._collision_point[scene_id][agent_id]
                            for agent_id in range(self.num_agent_per_scene)
                        ]
                    ),
                    axis=1,
                )
                rela_dis = np.diag(
                    np.full(self.num_agent_per_scene, np.inf, dtype=np.float32)
                )
                for i in range(self.num_agent_per_scene):
                    j = i + 1
                    rela_dis[i, j:] = np.linalg.norm(positions[j:] - positions[i])
                rela_dis += rela_dis.T
                min_rela_dis, min_indices = (
                    np.min(rela_dis, axis=1),
                    np.argmin(rela_dis, axis=1),
                )
                is_rela_dis_less = min_rela_dis < cur_dis
                for agent_id in np.arange(self.num_agent_per_scene)[is_rela_dis_less]:
                    self._collision_point[scene_id][agent_id] = positions[
                        min_indices[agent_id]
                    ]

    def get_point_is_collision(
        self,
        std_positions: Optional[th.tensor] = None,
        scene_id: Optional[int] = None,
        uav_radius: Optional[float] = None,
        hab_positions: Optional[np.ndarray] = None,
    ):
        """
        search within uav_radius for obstacles
        input positions in either std_positions or hab_positions
        returns tensor of input len
        """
        uav_radius = self.drone_radius if uav_radius is None else uav_radius

        assert scene_id is not None
        if self._collision_backend == "cloud":
            n = std_positions.shape[0]
            sids = th.full(
                (n,), int(scene_id), dtype=th.long, device=self._env_device
            )
            _pt, dis, out = self.get_collision_query(std_positions, sids)
            return (dis < uav_radius) & ~out
        if hab_positions is None:
            hab_positions, _ = std_to_habitat(std_positions, None)
        min_distance = np.empty(len(hab_positions), dtype=np.float32)
        is_in_bounds = np.empty(len(hab_positions), dtype=bool)
        for indice, hab_position in enumerate(hab_positions):
            col_record = self.scenes[scene_id].get_closest_collision_point(
                pt=hab_position.reshape(3, 1), max_search_radius=uav_radius
            )
            min_distance[indice] = np.linalg.norm((col_record.hit_pos - hab_position))
            is_in_bounds[indice] = not col_record.is_out_bound
        return th.as_tensor((min_distance < uav_radius) & is_in_bounds)

    def get_collision_point(self, indices=None):
        if indices is None:
            return habitat_to_std(
                np.array(self._collision_point).reshape((-1, 3)), None
            )[0]
        else:
            return habitat_to_std(
                np.array(self._collision_point).reshape((-1, 3))[indices], None
            )[0]

    def render(
        self,
        is_draw_axes: bool = False,
        points: Optional[th.Tensor] = None,
        lines: Optional[th.Tensor] = None,
        curves: Optional[th.Tensor] = None,
        c_curves: Optional[th.Tensor] = None,
    ):
        """render for visualization and debugging purposes"""

        # draw lines in local coordinate of agent_s or objects
        def draw_axes(sim, translation, axis_len=1.0):
            lr = sim.get_debug_line_render()
            x_axis, _ = std_to_habitat(th.Tensor([axis_len, 0.0, 0.0]), None)
            y_axis, _ = std_to_habitat(th.Tensor([0.0, axis_len, 0.0]), None)
            z_axis, _ = std_to_habitat(th.Tensor([0.0, 0.0, axis_len]), None)
            lr.draw_transformed_line(translation, mn.Vector3(x_axis), red)
            lr.draw_transformed_line(translation, mn.Vector3(y_axis), green)
            lr.draw_transformed_line(translation, mn.Vector3(z_axis), blue)

        if self.render_settings is None:
            raise EnvironmentError("render settings is not set")

        # output images
        render_imgs = []

        # debug
        if DEBUG or is_draw_axes or self.render_settings["axes"]:
            for scene_id in range(self.num_scene):
                draw_axes(self.scenes[scene_id], origin, axis_len=1)

        if points is not None:
            for scene_id in range(self.num_scene):
                scene_points = std_to_habitat(points[scene_id], None)[0]
                for indice, point in enumerate(scene_points):
                    self._line_renders[scene_id].draw_circle(
                        mn.Vector3(point[:3]), radius=0.25, color=ColorSet3[indice]
                    )

        if lines is not None:
            for scene_id in range(self.num_scene):
                for line_id in range(len(lines[scene_id])):
                    line = std_to_habitat(lines[scene_id][line_id], None)[0]
                    self._line_renders[scene_id].draw_transformed_line(
                        mn.Vector3(line[0]), mn.Vector3(line[1]), ColorSet3[line_id]
                    )

        if curves is not None:
            for scene_id in range(self.num_scene):
                for curve in curves[scene_id]:
                    curve = std_to_habitat(curve, None)[0]
                    curve = [mn.Vector3(point) for point in curve]
                    self._line_renders[0].draw_path_with_endpoint_circles(
                        curve, 0.1, white
                    )

        # draw the axes of agents
        if self.render_settings["axes"]:
            for scene_id in range(self.num_scene):
                for agent_id in range(self.num_agent_per_scene):
                    # self._line_renders[scene_id].push_transform(self.agents[scene_id][agent_id].scene_node.transformation)
                    self._line_renders[scene_id].push_transform(
                        self._objects[scene_id][agent_id].transformation
                    )
                    draw_axes(self.scenes[scene_id], origin, axis_len=1)
                    self._line_renders[scene_id].pop_transform()

        # draw the trajectory of agents
        if self.render_settings["trajectory"]:
            for scene_id in range(self.num_scene):
                for agent_id in range(self.num_agent_per_scene):
                    # traj = [mn.Vector3(point[:3]) for point in self.trajectory[scene_id][agent_id]]
                    # if len(traj) > 1:
                    #     self._line_renders[0].draw_path_with_endpoint_circles(
                    #         traj, 0.1, white)
                    for line_id in np.arange(
                        len(self.trajectory[scene_id][agent_id]) - 1
                    ):
                        self._line_renders[scene_id].draw_transformed_line(
                            self.trajectory[scene_id][agent_id][line_id][:3],
                            self.trajectory[scene_id][agent_id][line_id + 1][:3],
                            color_consequence(factor=line_id / 10),
                        )
                    # trajectory_data = np.array(self.trajectory[scene_id][agent_id])
                    # self._line_renders[scene_id].draw_transformed_line(trajectory_data[:3], self.trajectory[scene_id][agent_id][i+1][:3], white)

        # set the render camera pose
        if self.render_settings["mode"] == "follow":
            if (
                self.render_settings["view"] == "back"
                or self.render_settings["view"] == "near"
            ):
                if self.render_settings["view"] == "back":
                    rela_pos = eye_pos_follow_back
                elif self.render_settings["view"] == "near":
                    rela_pos = eye_pos_follow_near
                for scene_id in range(self.num_scene):
                    if self.render_settings["position"] is None:
                        obj = self.agents[scene_id][0].get_state().position
                    else:
                        obj = mn.Vector3(
                            *std_to_habitat(
                                th.tensor(self.render_settings["position"]), None
                            )[0]
                        )
                    camera_pose = calc_camera_transform(
                        eye_translation=(
                            self.agents[scene_id][0].scene_node.transformation
                            * mn.Vector4(rela_pos, 1)
                        ).xyz,
                        lookat=obj,
                    )
                    self._render_camera[scene_id].set_state(
                        habitat_sim.AgentState(
                            rotation=R.from_matrix(
                                np.array(camera_pose)[:3, :3]
                            ).as_quat(),
                            position=camera_pose.translation,
                        )
                    )
            else:
                for scene_id in range(self.num_scene):
                    pos, quat = (
                        self.agents[scene_id][0].get_state().position,
                        self.agents[scene_id][0].get_state().rotation,
                    )
                    camera_pose = calc_camera_transform(
                        eye_translation=eye_pos_follow_back + pos, lookat=pos
                    )
                    self._render_camera[scene_id].set_state(
                        habitat_sim.AgentState(
                            rotation=R.from_matrix(
                                np.array(camera_pose)[:3, :3]
                            ).as_quat(),
                            position=camera_pose.translation,
                        )
                    )

        elif self.render_settings["mode"] == "fix":
            if self.render_settings["view"] == "top":
                # fix the camera at the center top of the scene to observe the whole scene
                for scene_id in range(self.num_scene):
                    if self.render_settings["position"] is None:
                        scene_aabb = self._scene_bounds[scene_id]
                        scene_center = (scene_aabb.min + scene_aabb.max) / 2
                        scene_height = (
                            scene_aabb.max[1] - scene_aabb.min[1]
                        ) + scene_aabb.max[1] * 2
                    else:
                        scene_center = std_to_habitat(
                            th.tensor(self.render_settings["position"]), None
                        )[0][0]
                        scene_height = scene_center[1]

                    self._render_camera[scene_id].set_state(
                        habitat_sim.AgentState(
                            rotation=R.from_euler(
                                "zyx", [90, 0, -90], degrees=True
                            ).as_quat(),
                            position=mn.Vector3(
                                scene_center[0], scene_height, scene_center[2]
                            ),
                        )
                    )

            elif self.render_settings["view"] == "near":
                # fix the camera at third person view to observe the agent
                obj = (
                    origin
                    if self.render_settings["position"] is None
                    else std_to_habitat(self.render_settings["position"], None)[
                        0
                    ].squeeze()
                )
                for scene_id in range(self.num_scene):
                    camera_pose = calc_camera_transform(
                        eye_translation=eye_pos_near + obj, lookat=obj
                    )
                    self._render_camera[scene_id].set_state(
                        habitat_sim.AgentState(
                            rotation=R.from_matrix(
                                np.array(camera_pose)[:3, :3]
                            ).as_quat(),
                            position=camera_pose.translation,
                        )
                    )

            elif self.render_settings["view"] == "side":
                hab_position = std_to_habitat(self.render_settings["position"], None)[0]
                for scene_id in range(self.num_scene):
                    self._render_camera[scene_id].set_state(
                        habitat_sim.AgentState(
                            rotation=R.from_euler(
                                "zyx", [-0, -90, 0], degrees=True
                            ).as_quat(),
                            # position=mn.Vector3(scene_center[0], scene_height, scene_center[2])
                            position=mn.Vector3(np.squeeze(hab_position)),
                        )
                    )

            elif self.render_settings["view"] == "back":
                # fix the camera at third person view to observe the agent
                obj = (
                    origin
                    if self.render_settings["position"] is None
                    else std_to_habitat(self.render_settings["position"], None)[0]
                )
                for scene_id in range(self.num_scene):
                    camera_pose = calc_camera_transform(
                        eye_translation=eye_pos_back + obj, lookat=obj
                    )
                    self._render_camera[scene_id].set_state(
                        habitat_sim.AgentState(
                            rotation=R.from_matrix(
                                np.array(camera_pose)[:3, :3]
                            ).as_quat(),
                            position=camera_pose.translation,
                        )
                    )
            elif self.render_settings["view"] == "custom":
                position = std_to_habitat(self.render_settings["position"], None)[0]
                for scene_id in range(self.num_scene):
                    camera_pose = calc_camera_transform(
                        eye_translation=position[0], lookat=position[1]
                    )
                    self._render_camera[scene_id].set_state(
                        habitat_sim.AgentState(
                            rotation=R.from_matrix(
                                np.array(camera_pose)[:3, :3]
                            ).as_quat(),
                            position=camera_pose.translation,
                        )
                    )
            else:
                raise ValueError("Invalid render position.")

        else:
            raise ValueError("Invalid render mode.")

        # get images from render cameras
        for scene_id in range(self.num_scene):
            render_imgs.append(
                self.scenes[scene_id].get_sensor_observations(self.num_agent_per_scene)[
                    "render"
                ]
            )

        return render_imgs

    def _load_render_camera(self) -> habitat_sim.agent.AgentConfiguration:
        """create an Agent Configuration for the render agent"""
        sensor_cfgs_list = []
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = "render"
        sensor_spec.resolution = self.render_settings["resolution"]
        sensor_spec.position = mn.Vector3([0, 0, 0])
        sensor_str = self.render_settings["sensor_type"]
        if "depth" in sensor_str:
            sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        elif "color" in sensor_str:
            sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        else:
            raise ValueError(f"Unsupported sensor type {sensor_str}")
        sensor_cfgs_list.append(sensor_spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration(
            radius=0.01, height=0.01, sensor_specifications=sensor_cfgs_list
        )
        return agent_cfg

    def _load_geodesic(self, scene_path, scene_id):
        with open(scene_path, "r") as f:
            scene_cfg = json.load(f)

        geodesic_path = scene_cfg.get("user_defined", {}).get("geodesic_path", "")
        full_geodesic_path = os.path.join(self.dataset_path, geodesic_path)
        if geodesic_path != "" and os.path.exists(full_geodesic_path):
            data = np.load(full_geodesic_path)
            self.geodesics[scene_id] = {
                key: th.from_numpy(data[key]).to(self._env_device)
                for key in data.files
            }
            colorlog.log.info(f"Loaded geodesics from {geodesic_path}")
        else:
            # generate geodesics on the spot
            colorlog.log.critical("Generating geodesics")
            raise NotImplementedError

    @timerlog.timer.timed
    def load_scenes(self, indices: Optional[List] = None):
        """
        load scenes and auto switch to next minibatch of scenes
        """
        scene_paths = next(
            self._scene_loader
        )  # returns a batch of num_scenes (or smaller)
        # extend scene_paths until we reach len num_scenes
        cycle_i = 0
        while len(scene_paths) < self.num_scene:
            scene_paths.append(scene_paths[cycle_i])
            cycle_i += 1

        if indices is None:
            indices = th.arange(self.num_scene)

        for scene_id in indices:
            scene_path = scene_paths[scene_id]
            self.scene_paths[scene_id] = scene_path
            cfg = self._load_cfg(scene_path)
            if self.scenes[scene_id] is None:
                # load new scene
                self.scenes[scene_id] = habitat_sim.Simulator(cfg)
                self.scenes[scene_id].seed(self.seed)

                if self.load_geodesics:
                    print("LOADING SCENE")
                    print(scene_path)
                    self._load_geodesic(scene_path, scene_id)
            else:
                # remove old objects
                rigid_mgr = self.scenes[scene_id].get_rigid_object_manager()
                objs = rigid_mgr.remove_all_objects()

                if self.reload_scenes:
                    # reconfigure existing scene
                    self.scenes[scene_id].reconfigure(cfg)
                    self.scenes[scene_id].recompute_mesh_kdtree()
                    if self.load_geodesics:
                        self._load_geodesic(scene_path, scene_id)

            if self.spawn_obstacles:
                self._obstacles[scene_id] = (
                    self.obstacle_generator.add_obstacles_to_scene(
                        self.scenes[scene_id]
                    )
                )

            self._scene_bounds[scene_id] = (
                self.scenes[scene_id]
                .get_active_scene_graph()
                .get_root_node()
                .cumulative_bb
            )
            if self._collision_backend == "cloud":
                self._build_collision_cloud(scene_id)
            # get agent handles in each scene
            self.agents[scene_id] = [
                self.scenes[scene_id].get_agent(agent_id)
                for agent_id in range(self.num_agent_per_scene)
            ]

            # get render agent handles in each scene
            if self.render_settings is not None:
                self._render_camera[scene_id] = self.scenes[scene_id].get_agent(
                    self.num_agent_per_scene
                )
                # create line renders and object managers
                self._obj_mgrs[scene_id] = self.scenes[
                    scene_id
                ].get_rigid_object_manager()
                self._line_renders[scene_id] = self.scenes[
                    scene_id
                ].get_debug_line_render()
                self._line_renders[scene_id].set_line_width(
                    self.render_settings["line_width"]
                )
                # create objects in each scene
                drone_template_id = self._load_drone_template(self.scenes[scene_id])
                for agent_id in range(self.num_agent_per_scene):
                    self._objects[scene_id][agent_id] = self._obj_mgrs[
                        scene_id
                    ].add_object_by_template_id(drone_template_id)

            if self.is_multi_drone:
                if self._objects[scene_id][0] is None:
                    if self._obj_mgrs[scene_id] is None:
                        self._obj_mgrs[scene_id] = self.scenes[
                            scene_id
                        ].get_rigid_object_manager()
                    drone_template_id = self._load_drone_template(
                        self.scenes[scene_id]
                    )
                    for agent_id in range(self.num_agent_per_scene):
                        self._objects[scene_id][agent_id] = self._obj_mgrs[
                            scene_id
                        ].add_object_by_template_id(drone_template_id)

    def _load_cfg(self, scene_path: str) -> habitat_sim.Simulator:
        """load single scene with agents"""
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.scene_dataset_config_file = self._scene_dataset_config_path
        if self.gpu2gpu:
            sim_cfg.gpu_device_id = int(os.environ.get("HABITAT_GPU_ID", "0"))
        sim_cfg.enable_physics = True
        sim_cfg.use_semantic_textures = False
        sim_cfg.load_semantic_mesh = False
        sim_cfg.force_separate_semantic_scene_graph = False
        sim_cfg.create_renderer = False
        sim_cfg.enable_gfx_replay_save = True
        sim_cfg.leave_context_with_background_renderer = True
        sim_cfg.random_seed = self.seed
        sim_cfg.requires_textures = self.render_settings is not None

        cfg = habitat_sim.Configuration(
            sim_cfg=sim_cfg,
            agents=self._load_agents(self.num_agent_per_scene),
        )
        return cfg

    def _load_drone_template(self, scene) -> int:
        template_mgr = scene.get_object_template_manager()
        template_ids = template_mgr.load_configs(self._robot_path)
        assert template_ids and template_ids[0] >= 0, (
            f"failed to load drone object template: {self._robot_path}"
        )
        return template_ids[0]

    def reset_agents(
        self,
        std_positions: th.Tensor,
        std_orientations: th.Tensor,
        indices: Optional[th.Tensor] = None,
    ):
        """external interference to reset all the agents to the initial state"""
        hab_positions, hab_orientations = std_to_habitat(
            std_positions, std_orientations
        )
        for indice, hab_position, hab_orientation in zip(
            np.arange(self.num_agent) if indices is None else indices,
            hab_positions,
            hab_orientations,
        ):
            scene_id = indice // self.num_agent_per_scene
            agent_id = indice % self.num_agent_per_scene
            self._reset_agent(scene_id, agent_id, hab_position, hab_orientation)
        if self._collision_backend == "kdtree":
            self._update_collision_infos(indices=indices)

    def _reset_agent(
        self,
        scene_id: int,
        agent_id: int,
        position: np.ndarray,
        orientation: np.ndarray,
    ):
        """reset the agent"""
        self.trajectory[scene_id][agent_id] = []
        self.agents[scene_id][agent_id].set_state(
            habitat_sim.AgentState(
                position=position,
                rotation=quaternion.from_float_array(orientation),
            )
        )

    def _load_agents(
        self, num_agent: int
    ) -> List[habitat_sim.agent.AgentConfiguration]:
        """create num_agent configurations"""
        agent_cfgs_list = []
        for i in range(num_agent):
            agent_cfgs_list.append(self._load_agent())

        # add render agent
        if self.render_settings is not None:
            agent_cfgs_list.append(self._load_render_camera())

        return agent_cfgs_list

    def _load_agent(self) -> habitat_sim.agent.AgentConfiguration:
        """load single agent configuration"""
        sensor_cfgs_list = self._load_sensor()
        agent_cfg = habitat_sim.agent.AgentConfiguration(
            radius=0.1, height=0.1, sensor_specifications=sensor_cfgs_list
        )
        return agent_cfg

    def _load_sensor(self) -> List[habitat_sim.sensor.SensorSpec]:
        """load sensors configuration of each agent"""
        sensor_cfgs_list = []
        for i, sensor_cfg in enumerate(self.sensor_settings):
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_str = sensor_cfg["sensor_type"]
            if "depth" in sensor_str:
                sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            elif "color" in sensor_str:
                sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            else:
                raise ValueError(f"Unsupported sensor type {sensor_str}")
            sensor_spec.uuid = sensor_cfg["uuid"]
            sensor_spec.resolution = sensor_cfg.get("resolution", [128, 128])
            orientation = std_to_habitat(
                th.tensor(sensor_cfg.get("orientation", [0.0, 0, 0])), None
            )[0]
            position = std_to_habitat(
                th.tensor(sensor_cfg.get("position", [0.0, 0, 0])), None
            )[0]
            sensor_spec.orientation = mn.Vector3(*orientation)
            sensor_spec.position = mn.Vector3(*position)
            sensor_spec.far = sensor_cfg.get("far", 20.0)
            sensor_spec.near = sensor_cfg.get("near", 0.01)
            sensor_spec.hfov = sensor_cfg.get("hfov", 89)
            # If habitat is built --with-cuda flag, we can enable this to avoid
            # copying image from gpu -> cpu when we get observations
            sensor_spec.gpu2gpu_transfer = self.gpu2gpu
            if self.noise_settings is not None:
                if sensor_spec.uuid in self.noise_settings.keys():
                    sensor_spec.noise_model = self.noise_settings[sensor_spec.uuid].get(
                        "model", "None"
                    )
                    sensor_spec.noise_model_kwargs = self.noise_settings[
                        sensor_spec.uuid
                    ].get("kwargs", {})
            sensor_cfgs_list.append(sensor_spec)

        return sensor_cfgs_list

    def close(self):
        """
        release resources held by the simulator
        found through testing that closing in reversed order prevents
        GL::Context::Current(): no current context errors
        """
        for scene_id in reversed(range(self.num_scene)):
            if self.scenes[scene_id] is not None:
                self.scenes[scene_id].close()
                self.scenes[scene_id] = None

    def fmm_3d(self, occupancy, target_ijk, grid_resolution):
        # compute truncated distance from obstacles
        dist_outside = grid_resolution * scipy.ndimage.distance_transform_edt(
            occupancy == 0
        )
        tsdf = np.clip(dist_outside, 0.0, 1.0)

        # speed is a piecewise function related to distance to nearest obstacle
        ROBOT_RADIUS = 0.3
        SAFE_RADIUS = 0.6
        SLOW_SPEED = 0.1
        # above SAFE_RADIUS, speed is 1
        speed = np.ones_like(tsdf)

        # between ROBOT_RADIUS and SAFE_RADIUS, linearly interpolate speed
        m = (SAFE_RADIUS - SLOW_SPEED) / (SAFE_RADIUS - ROBOT_RADIUS)
        b = SLOW_SPEED - m * ROBOT_RADIUS
        mask1 = (ROBOT_RADIUS < tsdf) & (tsdf <= SAFE_RADIUS)
        speed[mask1] = m * tsdf[mask1] + b

        # below ROBOT_RADIUS, set speed to SLOW_SPEED
        mask2 = (0 < tsdf) & (tsdf <= ROBOT_RADIUS)
        speed[mask2] = SLOW_SPEED
        speed[occupancy == 1] = 0

        # target point (source of wavefront)
        phi = np.ones_like(occupancy, dtype=np.float32)
        phi[target_ijk] = -1

        # compute travel cost from target
        travel_time = skfmm.travel_time(phi, speed, dx=grid_resolution)  # FMM
        return travel_time

    def calculate_gradient(self, costs, resolution):
        """
        Use finite differences to calculate gradients at each pixel in cost map
        Inputs:
            costs: np.ndarray - scalar cost field
            resolution: float - size of each grid cell in cost array
        Output:
            gradient vector field of same shape as costs
        """
        gradient = np.zeros((*costs.shape, 3), dtype=np.float32)
        finite = np.isfinite(costs)

        def finite_difference(axis):
            # Roll neighbors
            plus = np.roll(costs, -1, axis=axis)
            minus = np.roll(costs, 1, axis=axis)

            plus_valid = np.isfinite(plus)
            minus_valid = np.isfinite(minus)

            grad = np.zeros_like(costs, dtype=np.float32)

            # Central difference where both are valid
            central_mask = finite & plus_valid & minus_valid
            grad[central_mask] = (plus[central_mask] - minus[central_mask]) / (
                2 * resolution
            )

            # Forward difference where only plus is valid
            forward_mask = finite & plus_valid & ~minus_valid
            grad[forward_mask] = (plus[forward_mask] - costs[forward_mask]) / resolution

            # Backward difference where only minus is valid
            backward_mask = finite & ~plus_valid & minus_valid
            grad[backward_mask] = (
                costs[backward_mask] - minus[backward_mask]
            ) / resolution

            # Else: leave as 0 (already initialized)
            return grad

        dx = finite_difference(axis=0)
        dy = finite_difference(axis=1)
        dz = finite_difference(axis=2)

        gradient[..., 0] = -dx
        gradient[..., 1] = -dy
        gradient[..., 2] = -dz
        gradient[~finite] = 0.0

        # normalize gradients
        norms = np.linalg.norm(gradient, axis=-1, keepdims=True) + 1e-8
        gradient /= norms
        return gradient

    def generate_geodesic(
        self, scene_id: int, target: np.array, grid_resolution: float = 0.25
    ):
        scene = self.scenes[scene_id]
        bb = scene.get_active_scene_graph().get_root_node().cumulative_bb
        bb_habitat = th.tensor(
            [[bb.min[0], bb.min[1], bb.min[2]], [bb.max[0], bb.max[1], bb.max[2]]]
        )
        bb_std = habitat_to_std(bb_habitat, None)[0]
        bb_std = bb_std.sort(dim=0)[0]  # flips coords so min < max

        xs = np.arange(bb_std[0, 0], bb_std[1, 0] + grid_resolution, grid_resolution)
        ys = np.arange(bb_std[0, 1], bb_std[1, 1] + grid_resolution, grid_resolution)
        zs = np.arange(bb_std[0, 2], bb_std[1, 2] + grid_resolution, grid_resolution)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

        # batched closest point queries
        pts = th.from_numpy(np.stack([X, Y, Z], axis=-1).reshape(-1, 3)).float()
        pts_habitat = std_to_habitat(pts, None)[0].astype(np.float32)
        col_records = scene.get_closest_collision_point_batch(
            pts=pts_habitat,
            max_search_radius=grid_resolution,
            num_threads=-1,
            print_threads=True,
        )
        closest_pts_habitat = np.array([record.hit_pos for record in col_records])
        hit_dist = np.linalg.norm(closest_pts_habitat - pts_habitat, axis=1)
        obs_dists = hit_dist.reshape(*X.shape)
        occupancy = obs_dists < grid_resolution

        target_ijk = (
            int((target[0].item() - bb_std[0, 0]) // grid_resolution),
            int((target[1].item() - bb_std[0, 1]) // grid_resolution),
            int((target[2].item() - bb_std[0, 2]) // grid_resolution),
        )
        assert target_ijk[0] >= 0 and target_ijk[1] >= 0 and target_ijk[2] >= 0, (
            "invalid target point given"
        )
        costs = self.fmm_3d(occupancy, target_ijk, grid_resolution)
        gradients = self.calculate_gradient(costs, grid_resolution)

        scene_path = self.scene_paths[scene_id]
        rel_path = os.path.dirname(
            os.path.relpath(scene_path, self.dataset_path).replace(
                "configs", "geodesics"
            )
        )
        filename = os.path.basename(scene_path).split(".")[0] + ".npz"
        rel_geodesic_path = os.path.join(rel_path, filename)
        abs_geodesic_path = os.path.join(self.dataset_path, rel_geodesic_path)
        os.makedirs(os.path.dirname(abs_geodesic_path), exist_ok=True)

        colorlog.log.info(f"Saved geodesics to {abs_geodesic_path}")
        np.savez(
            abs_geodesic_path,
            occupancy=occupancy,
            target=target_ijk,
            costs=costs,
            gradients=gradients,
            grid_resolution=grid_resolution,
            bb_std=bb_std,
        )

        # modify scene_instance.json to add geodesic_path
        with open(scene_path, "r") as f:
            scene_json = json.load(f)

        if "user_defined" in scene_json:
            scene_json["user_defined"]["geodesic_path"] = rel_geodesic_path
        else:
            scene_json["user_defined"] = {"geodesic_path": rel_geodesic_path}

        with open(scene_path, "w") as f:
            json.dump(scene_json, f, indent=4)

        colorlog.log.info(f"Updated {scene_path}")
        return abs_geodesic_path

    def trilinear_interpolate(self, grid, coords):
        """
        Args:
            grid: (X, Y, Z, ?) tensor
            coords: (N, 3) tensor of float coords in (x, y, z) format
        Returns:
            interpolated: (N, ?) tensor
        """
        X, Y, Z, _ = grid.shape
        grid = th.as_tensor(grid, device=coords.device)

        # Clamp to valid range
        _max = th.tensor([X - 1.001, Y - 1.001, Z - 1.001], device=coords.device)
        coords_clamped = th.clamp(coords, min=th.zeros(3, device=coords.device), max=_max)

        x, y, z = coords_clamped.unbind(1)

        x0 = x.floor().long()
        y0 = y.floor().long()
        z0 = z.floor().long()

        x1 = th.clamp(x0 + 1, max=X - 1)
        y1 = th.clamp(y0 + 1, max=Y - 1)
        z1 = th.clamp(z0 + 1, max=Z - 1)

        # Get fractional parts
        dx = (x - x0.float()).unsqueeze(-1)
        dy = (y - y0.float()).unsqueeze(-1)
        dz = (z - z0.float()).unsqueeze(-1)

        # Gather values at 8 surrounding corners
        def gather(z_idx, y_idx, x_idx):
            return grid[x_idx, y_idx, z_idx]

        # cZYX
        c000 = gather(z0, y0, x0)
        c001 = gather(z0, y0, x1)
        c010 = gather(z0, y1, x0)
        c011 = gather(z0, y1, x1)
        c100 = gather(z1, y0, x0)
        c101 = gather(z1, y0, x1)
        c110 = gather(z1, y1, x0)
        c111 = gather(z1, y1, x1)

        # interpolate at dx coord to get cZY
        c00 = c000 * (1 - dx) + c001 * dx
        c01 = c010 * (1 - dx) + c011 * dx
        c10 = c100 * (1 - dx) + c101 * dx
        c11 = c110 * (1 - dx) + c111 * dx

        # interpolate at dy coord to get cZ
        c0 = c00 * (1 - dy) + c01 * dy
        c1 = c10 * (1 - dy) + c11 * dy

        # interpolate at dz coord to get c
        c = c0 * (1 - dz) + c1 * dz
        return c

    def interpolate_geodesic(self, scene_id, pts, gradient=True):
        geodesic = self.geodesics[scene_id]
        grid_resolution = geodesic["grid_resolution"]
        bb_std = geodesic["bb_std"]
        if gradient:
            grid = geodesic["gradients"]
        else:
            grid = geodesic["costs"].unsqueeze(-1)

        # Print out-of-bounds coords (only when debugging; avoids per-step GPU syncs)
        if DEBUG:
            below_min = pts < bb_std[0]  # shape (N, 3)
            above_max = pts > bb_std[1]
            out_of_bounds = below_min | above_max  # shape (N, 3)
            if out_of_bounds.any():
                for i in range(pts.shape[0]):
                    if out_of_bounds[i].any():
                        print(f"Coord {i}: {pts[i].tolist()} is out of bounds")

        # clamp query coords to bounds
        pts = pts.clamp(min=bb_std[0], max=bb_std[1])

        # convert std coordinates to cell index space
        inds = (pts - bb_std[0]) / grid_resolution

        # trilinear interpolation
        return self.trilinear_interpolate(grid, inds)

    @property
    def is_out_bounds(self):
        return th.as_tensor(self._is_out_bounds).reshape(-1)
