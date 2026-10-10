import os
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass
import habitat_sim
import json
import numpy as np
import random
import argparse
import torch as th
from copy import deepcopy

from clearnav.common import std_to_habitat
from clearnav.utils.type import Uniform, Normal
from clearnav.utils.rotation3 import Rotation3

empty_scene = {
    "stage_instance": {
        "template_name": "",
    },
    "object_instances": [],
    "articulated_object_instances": [],
    "default_lighting": "lighting/garage_v1_0",
    # "default_lighting": "default",
    "user_defined": {},
}


def get_all_children_path(root_path):
    if os.path.isdir(root_path):
        file_paths = []
        for root, directories, files in os.walk(root_path):
            for filename in files:
                file_paths.append(str.split(filename, ".")[0])
        return file_paths
    else:
        basename = os.path.basename(root_path)
        directory = os.path.dirname(root_path)
        file_paths = os.listdir(directory)
        matches = [filename for filename in file_paths if basename in filename]
        return matches


class BoxGenerator:
    def __init__(
        self,
        low=[0.0, 0.0, 0.0],
        high=[1.0, 1.0, 1.0],
        dataset_path="./datasets/clearnav_dataset",
        object_set="primitives/medium",
        density=0.2,
        scale_rng=Uniform([1.0], [0.0]),
        rotation_rng=Uniform([0.0, 0.0, 0.0], [3.0, 3.0, 3.0]),
        seed=None,
    ):
        self.low = th.tensor(low)
        self.high = th.tensor(high)
        self.object_set = os.path.join(dataset_path, f"configs/{object_set}")
        self.object_paths = get_all_children_path(self.object_set)
        self.density = density
        self.scale_rng = scale_rng
        self.rotation_rng = rotation_rng

        self.gen = th.Generator()
        if seed is not None:
            self.gen.manual_seed(seed)

    def is_inside(self, points):
        coord_inside = th.logical_and(self.low <= points, points <= self.high).to(
            dtype=th.bool
        )
        inside = th.all(coord_inside, dim=1)
        return inside

    def sample(self):
        volume = th.prod(th.abs(self.high - self.low))
        num_points = int(volume * self.density)
        positions = self.low + (self.high - self.low) * th.rand(
            (num_points, 3), generator=self.gen
        )

        # generate random orientations (quaternions)
        orientations = []
        for _ in range(num_points):
            u1, u2, u3 = np.random.random(3)
            quat = np.array(
                [
                    np.sqrt(u1) * np.cos(2 * np.pi * u3),
                    np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                    np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                    np.sqrt(u1) * np.sin(2 * np.pi * u3),
                ]
            )
            orientations.append(quat)
        orientations = th.tensor(np.array(orientations))

        # generate sizes
        sizes = self.scale_rng.generate(num_points, self.gen)

        # generate ids
        ids = th.randint(0, len(self.object_paths), (num_points,), generator=self.gen)

        return positions, orientations, sizes, ids


class CylinderGenerator:
    def __init__(
        self,
        base_center=[0.0, 0.0, 0.0],
        radius=3.0,
        height=10.0,
        dataset_path="./datasets/clearnav_dataset",
        object_set="primitives/medium",
        density=0.2,
        scale_rng=Uniform([1.0], [0.0]),
        rotation_rng=Uniform([0.0, 0.0, 0.0], [3.0, 3.0, 3.0]),
        seed=None,
    ):
        self.base_center = th.tensor(base_center)
        self.radius = radius
        self.height = height
        self.object_set = os.path.join(dataset_path, f"configs/{object_set}")
        self.object_paths = get_all_children_path(self.object_set)
        self.density = density
        self.scale_rng = scale_rng
        self.rotation_rng = rotation_rng

        self.gen = th.Generator()
        if seed is not None:
            self.gen.manual_seed(seed)

    def is_inside(self, points):
        from_center = points - self.base_center
        in_radius = (from_center[:, 0:2].norm(dim=1) <= self.radius).to(dtype=th.bool)
        in_height = th.logical_and(
            0.0 <= from_center[:, 2], from_center[:, 2] <= self.height
        ).to(dtype=th.bool)
        inside = in_radius & in_height
        return inside

    def sample(self):
        volume = th.pi * self.radius**2 * self.height
        num_points = int(volume * self.density)

        # sample uniformly in cylinder
        theta = 2.0 * th.pi * th.rand(num_points, generator=self.gen)
        r = self.radius * th.sqrt(th.rand(num_points, generator=self.gen))
        z = self.height * th.rand(num_points, generator=self.gen)
        x = r * th.cos(theta)
        y = r * th.sin(theta)
        positions = th.stack([x, y, z], dim=1) + self.base_center

        # generate random orientations (quaternions)
        # Generate rotations
        euler_zyx = self.rotation_rng.generate(size=(num_points, 3), generator=self.gen)
        rotation = Rotation3.from_euler_zyx(euler_zyx)
        orientations = rotation.to_quat()

        """
        orientations = []
        for _ in range(num_points):
            u1, u2, u3 = np.random.random(3)
            quat = np.array([
                np.sqrt(u1) * np.cos(2 * np.pi * u3),
                np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                np.sqrt(u1) * np.sin(2 * np.pi * u3),

            ])
            orientations.append(quat)
        orientations = th.tensor(np.array(orientations))
        """

        # generate sizes
        if self.scale_rng.mean.shape[0] == 3:
            sizes = self.scale_rng.generate((num_points, 3), self.gen)
        else:
            sizes = self.scale_rng.generate((num_points), self.gen)

        # generate ids
        ids = th.randint(0, len(self.object_paths), (num_points,), generator=self.gen)

        return positions, orientations, sizes, ids


class SceneGenerator:
    def __init__(
        self,
        dataset_path: str,
        num: int,
        name: str,
        stage: str,
        keep_in_bounds: Union[List, BoxGenerator] = BoxGenerator(),
        keep_out_bounds: Union[List, BoxGenerator] = [],
    ) -> None:
        self.dataset_path = dataset_path
        self.num = num
        self.name = name
        self.stage = stage
        self.keep_in_bounds = keep_in_bounds
        self.keep_out_bounds = keep_out_bounds

        self.summary_path = str(
            list(Path(self.dataset_path).glob("*.scene_dataset_config.json"))[0]
        )
        self.save_path = os.path.join(dataset_path, f"configs/{name}")

        self._write_scene_dir_in_summary(name)

    def generate(self):
        return self._create_scene_json()

    def _write_scene_dir_in_summary(self, name):
        # load json file
        with open(self.summary_path, "r") as file:
            summary = json.load(file)

        # write
        if f"configs/{name}" not in summary["scene_instances"]["paths"][".json"]:
            summary["scene_instances"]["paths"][".json"].append(f"configs/{name}")
        with open(self.summary_path, "w") as file:
            json.dump(summary, file, indent=4)

    def _create_scene_json(self):
        scene_save_paths = []
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        for id in range(self.num):
            filename = self.name.split("/")[-1]
            scene_save_path = f"{self.save_path}/{filename}_{id}.scene_instance.json"
            scene_json = deepcopy(empty_scene)
            scene_save_paths.append(scene_save_path)

            # stage
            scene_json["stage_instance"]["template_name"] = self.stage

            # objects
            for generator in self.keep_in_bounds:
                positions, orientations, sizes, ids = generator.sample()
                valid = th.ones(len(positions), dtype=th.bool)
                for keep_out in self.keep_out_bounds:
                    outside = ~keep_out.is_inside(positions)
                    valid = valid & outside

                scene_json["object_instances"].extend(
                    self._create_objects(
                        generator.object_paths,
                        positions[valid],
                        orientations[valid],
                        sizes[valid],
                        ids[valid],
                    )
                )

            # save
            self._save_json_file(scene_save_path, scene_json)
        print(f"{self.num} Files has been save in {os.getcwd()}/{self.save_path}")
        return scene_save_paths

    def _create_objects(self, object_paths, positions, orientations, sizes, ids):
        object_instances = []
        for i in range(len(positions)):
            object_instance = {}
            object_instance["template_name"] = object_paths[ids[i]]
            object_instance["translation"] = std_to_habitat(positions[i], None)[
                0
            ].tolist()
            object_instance["rotation"] = std_to_habitat(None, orientations[i])[
                1
            ].tolist()
            if sizes[i].ndim == 0:
                object_instance["uniform_scale"] = float(sizes[i])
            else:
                object_instance["non_uniform_scale"] = sizes[i].tolist()
            object_instance["motion_type"] = "STATIC"
            object_instance["translation_origin"] = "COM"
            object_instances.append(object_instance)
        return object_instances

    def _save_json_file(self, file_path, data):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)


def parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", "-g", type=int, default=1, help="generate scenes")
    parser.add_argument("--render", "-r", type=int, default=1, help="render scenes")
    parser.add_argument(
        "--quantity", "-q", type=int, default=1, help="generated quantity "
    )
    parser.add_argument("--name", "-n", type=str, default="test_generator", help="name")
    parser.add_argument(
        "--density", "-d", type=float, default=0.25, help="obstacle density"
    )
    return parser


if __name__ == "__main__":
    args = parsers().parse_args()
    dataset_path = "./datasets/clearnav_dataset"
    g = SceneGenerator(
        dataset_path=dataset_path,
        num=args.quantity,
        name=args.name,
        stage="stages/box_2",
        keep_in_bounds=[
            CylinderGenerator(
                base_center=[0.0, 0.0, 0.0],
                radius=12.5,
                height=5.0,
                dataset_path=dataset_path,
                object_set="primitives/medium",
                density=args.density,
                seed=42,
            )
        ],
        keep_out_bounds=[
            CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
        ],
    )

    if args.generate:
        scene_save_paths = g.generate()
    if args.render:
        os.system(
            f"python clearnav/scripts/scene_viewer.py --dataset {g.summary_path} --scene {g.name}"
        )
