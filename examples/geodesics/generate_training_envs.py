import os
from tqdm import trange
from pylogtools import colorlog

from clearnav.envs.scene_generator import (
    BoxGenerator,
    CylinderGenerator,
    SceneGenerator,
)
from clearnav.utils.type import Uniform, Normal, Cylinder
from clearnav.envs.navigation_env import NavigationEnv

DATASET_PATH = "./datasets/clearnav_dataset"
GRID_RESOLUTION = 0.1
NUM_TRAIN = 25
NUM_EVAL = 5
VIS = False


def generate_geodesics(level_name, target_pos=[0.0, 0.0, 1.5]):
    env = NavigationEnv(
        num_envs=1,
        single_env=True,
        visual=True,
        random_kwargs={
            "position": {
                "class": "cylinder",
                "mean": target_pos,
                "half": [12.5, 12.5, 0.0],
            },
            "target": {"class": "uniform", "mean": target_pos, "half": [0.0, 0.0, 0.0]},
        },
        scene_kwargs={
            "path": f"configs/{level_name}",
            "reload_scenes": True,
            "load_geodesics": False,
            "spawn_obstacles": False,
        },
        sensor_kwargs=[
            {"sensor_type": "depth", "uuid": "depth", "resolution": [64, 64]}
        ],
    )

    num_scenes = len(env.scene_manager._data_loader)
    colorlog.log.info(f"Generating geodesics for {num_scenes} scenes")

    for i in trange(num_scenes):
        env.reset()
        env.scene_manager.generate_geodesic(
            scene_id=0,
            target=env.target[0],
            grid_resolution=GRID_RESOLUTION,
        )

    return level_name


if __name__ == "__main__":
    if True:
        level_name = "level_1/ring_walls_small"  # 1.15
        g = SceneGenerator(
            dataset_path=DATASET_PATH,
            num=NUM_TRAIN,
            name=level_name,
            stage="stages/box_2",
            keep_in_bounds=[
                CylinderGenerator(
                    base_center=[0.0, 0.0, 1.5],
                    radius=10.0,
                    height=2.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cube",
                    scale_rng=Uniform([0.2, 5.0, 2.0], [0.0, 0.0, 0.0]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.2, 0.2, 3.14]),
                    density=0.03,
                    seed=42,
                )
            ],
            keep_out_bounds=[
                CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
            ],
        )
        g.generate()
        generate_geodesics(level_name)

        # generate eval set
        level_name = "level_1_eval/ring_walls_small"
        g = SceneGenerator(
            dataset_path=DATASET_PATH,
            num=NUM_EVAL,
            name=level_name,
            stage="stages/box_2",
            keep_in_bounds=[
                CylinderGenerator(
                    base_center=[0.0, 0.0, 1.5],
                    radius=10.0,
                    height=2.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cube",
                    scale_rng=Uniform([0.2, 5.0, 2.0], [0.0, 0.0, 0.0]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.2, 0.2, 3.14]),
                    density=0.03,
                    seed=42,
                )
            ],
            keep_out_bounds=[
                CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
            ],
        )
        g.generate()
        generate_geodesics(level_name)

        if VIS:
            os.system(f"python clearnav/scripts/scene_viewer.py --scene {level_name}")
            os.system(
                f"python examples/geodesics/interactive_flow_field.py --scene {level_name}"
            )

    if True:
        level_name = "level_2/ring_walls_small_dense"  # 1.19
        g = SceneGenerator(
            dataset_path=DATASET_PATH,
            num=NUM_TRAIN,
            name=level_name,
            stage="stages/box_2",
            keep_in_bounds=[
                CylinderGenerator(
                    base_center=[0.0, 0.0, 1.5],
                    radius=10.0,
                    height=2.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cube",
                    scale_rng=Uniform([0.2, 5.0, 2.0], [0.0, 0.0, 0.0]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.2, 0.2, 3.14]),
                    density=0.04,
                    seed=42,
                )
            ],
            keep_out_bounds=[
                CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
            ],
        )
        g.generate()
        generate_geodesics(level_name)

        # generate eval set
        level_name = "level_2_eval/ring_walls_small_dense"
        g = SceneGenerator(
            dataset_path=DATASET_PATH,
            num=NUM_EVAL,
            name=level_name,
            stage="stages/box_2",
            keep_in_bounds=[
                CylinderGenerator(
                    base_center=[0.0, 0.0, 1.5],
                    radius=10.0,
                    height=2.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cube",
                    scale_rng=Uniform([0.2, 5.0, 2.0], [0.0, 0.0, 0.0]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.2, 0.2, 3.14]),
                    density=0.04,
                    seed=42,
                )
            ],
            keep_out_bounds=[
                CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
            ],
        )
        g.generate()
        generate_geodesics(level_name)

        if VIS:
            os.system(f"python clearnav/scripts/scene_viewer.py --scene {level_name}")
            os.system(
                f"python examples/geodesics/interactive_flow_field.py --scene {level_name}"
            )

    if True:
        level_name = "level_3/ring_walls_large"  # 1.53
        g = SceneGenerator(
            dataset_path=DATASET_PATH,
            num=NUM_TRAIN,
            name=level_name,
            stage="stages/box_2",
            keep_in_bounds=[
                CylinderGenerator(
                    base_center=[0.0, 0.0, 1.5],
                    radius=10.0,
                    height=2.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cube",
                    scale_rng=Uniform([0.2, 5.0, 4.0], [0.0, 0.0, 0.0]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.2, 0.2, 3.14]),
                    density=0.03,
                    seed=42,
                )
            ],
            keep_out_bounds=[
                CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
            ],
        )
        g.generate()
        generate_geodesics(level_name)

        # generate eval set
        level_name = "level_3_eval/ring_walls_large"  # 1.53
        g = SceneGenerator(
            dataset_path=DATASET_PATH,
            num=NUM_EVAL,
            name=level_name,
            stage="stages/box_2",
            keep_in_bounds=[
                CylinderGenerator(
                    base_center=[0.0, 0.0, 1.5],
                    radius=10.0,
                    height=2.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cube",
                    scale_rng=Uniform([0.2, 5.0, 4.0], [0.0, 0.0, 0.0]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.2, 0.2, 3.14]),
                    density=0.03,
                    seed=42,
                )
            ],
            keep_out_bounds=[
                CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
            ],
        )
        g.generate()
        generate_geodesics(level_name)

        if VIS:
            os.system(f"python clearnav/scripts/scene_viewer.py --scene {level_name}")
            os.system(
                f"python examples/geodesics/interactive_flow_field.py --scene {level_name}"
            )

    if True:
        level_name = "level_3/ring_cubes_medium"  # 1.17
        g = SceneGenerator(
            dataset_path=DATASET_PATH,
            num=NUM_TRAIN,
            name=level_name,
            stage="stages/box_2",
            keep_in_bounds=[
                CylinderGenerator(
                    base_center=[0.0, 0.0, 0.0],
                    radius=10.0,
                    height=5.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cube",
                    scale_rng=Uniform([1.5], [0.0]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),
                    density=0.03,
                    seed=42,
                )
            ],
            keep_out_bounds=[
                CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
            ],
        )
        g.generate()
        generate_geodesics(level_name)

        # generate eval set
        level_name = "level_3_eval/ring_cubes_medium"
        g = SceneGenerator(
            dataset_path=DATASET_PATH,
            num=NUM_EVAL,
            name=level_name,
            stage="stages/box_2",
            keep_in_bounds=[
                CylinderGenerator(
                    base_center=[0.0, 0.0, 0.0],
                    radius=10.0,
                    height=5.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cube",
                    scale_rng=Uniform([1.5], [0.0]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),
                    density=0.03,
                    seed=42,
                )
            ],
            keep_out_bounds=[
                CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
            ],
        )
        g.generate()
        generate_geodesics(level_name)

        if VIS:
            os.system(f"python clearnav/scripts/scene_viewer.py --scene {level_name}")
            os.system(
                f"python examples/geodesics/interactive_flow_field.py --scene {level_name}"
            )

    if True:
        level_name = "level_1/ring_cubes_large"  # 1.32
        g = SceneGenerator(
            dataset_path=DATASET_PATH,
            num=NUM_TRAIN,
            name=level_name,
            stage="stages/box_2",
            keep_in_bounds=[
                CylinderGenerator(
                    base_center=[0.0, 0.0, 0.0],
                    radius=10.0,
                    height=5.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cube",
                    scale_rng=Uniform([1.875], [0.375]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),
                    density=0.03,
                    seed=42,
                )
            ],
            keep_out_bounds=[
                CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
            ],
        )
        g.generate()
        generate_geodesics(level_name)

        # generate eval set
        level_name = "level_1_eval/ring_cubes_large"
        g = SceneGenerator(
            dataset_path=DATASET_PATH,
            num=NUM_EVAL,
            name=level_name,
            stage="stages/box_2",
            keep_in_bounds=[
                CylinderGenerator(
                    base_center=[0.0, 0.0, 0.0],
                    radius=10.0,
                    height=5.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cube",
                    scale_rng=Uniform([1.875], [0.375]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),
                    density=0.03,
                    seed=42,
                )
            ],
            keep_out_bounds=[
                CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
            ],
        )
        g.generate()
        generate_geodesics(level_name)

        if VIS:
            os.system(f"python clearnav/scripts/scene_viewer.py --scene {level_name}")
            os.system(
                f"python examples/geodesics/interactive_flow_field.py --scene {level_name}"
            )

    if True:
        level_name = "level_2/ring_cylinders_large"  # 1.26
        g = SceneGenerator(
            dataset_path=DATASET_PATH,
            num=NUM_TRAIN,
            name=level_name,
            stage="stages/box_2",
            keep_in_bounds=[
                CylinderGenerator(
                    base_center=[0.0, 0.0, 0.0],
                    radius=10.0,
                    height=5.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cylinder_vertical",
                    scale_rng=Uniform([0.75, 8.0, 0.75], [0.5, 0.0, 0.5]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),
                    density=0.03,
                    seed=42,
                )
            ],
            keep_out_bounds=[
                CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
            ],
        )
        g.generate()
        generate_geodesics(level_name)

        # generate eval set
        level_name = "level_2_eval/ring_cylinders_large"  # 1.26
        g = SceneGenerator(
            dataset_path=DATASET_PATH,
            num=NUM_EVAL,
            name=level_name,
            stage="stages/box_2",
            keep_in_bounds=[
                CylinderGenerator(
                    base_center=[0.0, 0.0, 0.0],
                    radius=10.0,
                    height=5.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/unit/unit_cylinder_vertical",
                    scale_rng=Uniform([0.75, 8.0, 0.75], [0.5, 0.0, 0.5]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),
                    density=0.03,
                    seed=42,
                )
            ],
            keep_out_bounds=[
                CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
            ],
        )
        g.generate()
        generate_geodesics(level_name)

        if VIS:
            os.system(f"python clearnav/scripts/scene_viewer.py --scene {level_name}")
            os.system(
                f"python examples/geodesics/interactive_flow_field.py --scene {level_name}"
            )

    if True:
        level_name = "level_4/ring_primitives_medium"
        g = SceneGenerator(
            dataset_path=DATASET_PATH,
            num=NUM_TRAIN,
            name=level_name,
            stage="stages/box_2",
            keep_in_bounds=[
                CylinderGenerator(
                    base_center=[0.0, 0.0, 0.0],
                    radius=10.0,
                    height=5.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/medium",
                    scale_rng=Uniform([1.0], [0.2]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [3.0, 3.0, 3.0]),
                    density=0.13,
                    seed=42,
                )
            ],
            keep_out_bounds=[
                CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
            ],
        )
        g.generate()
        generate_geodesics(level_name)

        # generate eval set
        level_name = "level_4_eval/ring_primitives_medium"
        g = SceneGenerator(
            dataset_path=DATASET_PATH,
            num=NUM_EVAL,
            name=level_name,
            stage="stages/box_2",
            keep_in_bounds=[
                CylinderGenerator(
                    base_center=[0.0, 0.0, 0.0],
                    radius=10.0,
                    height=5.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/medium",
                    scale_rng=Uniform([1.0], [0.2]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [3.0, 3.0, 3.0]),
                    density=0.13,
                    seed=42,
                )
            ],
            keep_out_bounds=[
                CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
            ],
        )
        g.generate()
        generate_geodesics(level_name)

        if VIS:
            os.system(f"python clearnav/scripts/scene_viewer.py --scene {level_name}")
            os.system(
                f"python examples/geodesics/interactive_flow_field.py --scene {level_name}"
            )

    if True:
        level_name = "level_4/ring_primitives_medium_and_small"
        g = SceneGenerator(
            dataset_path=DATASET_PATH,
            num=NUM_TRAIN,
            name=level_name,
            stage="stages/box_2",
            keep_in_bounds=[
                CylinderGenerator(
                    base_center=[0.0, 0.0, 0.0],
                    radius=10.0,
                    height=5.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/medium",
                    scale_rng=Uniform([1.0], [0.2]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [3.0, 3.0, 3.0]),
                    density=0.06,
                    seed=42,
                ),
                CylinderGenerator(
                    base_center=[0.0, 0.0, 0.0],
                    radius=10.0,
                    height=5.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/small",
                    scale_rng=Uniform([1.0], [0.2]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [3.0, 3.0, 3.0]),
                    density=0.1,
                    seed=42,
                ),
            ],
            keep_out_bounds=[
                CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
            ],
        )
        g.generate()
        generate_geodesics(level_name)

        # generate eval set
        level_name = "level_4_eval/ring_primitives_medium_and_small"
        g = SceneGenerator(
            dataset_path=DATASET_PATH,
            num=NUM_EVAL,
            name=level_name,
            stage="stages/box_2",
            keep_in_bounds=[
                CylinderGenerator(
                    base_center=[0.0, 0.0, 0.0],
                    radius=10.0,
                    height=5.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/medium",
                    scale_rng=Uniform([1.0], [0.2]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [3.0, 3.0, 3.0]),
                    density=0.06,
                    seed=42,
                ),
                CylinderGenerator(
                    base_center=[0.0, 0.0, 0.0],
                    radius=10.0,
                    height=5.0,
                    dataset_path=DATASET_PATH,
                    object_set="primitives/small",
                    scale_rng=Uniform([1.0], [0.2]),
                    rotation_rng=Uniform([0.0, 0.0, 0.0], [3.0, 3.0, 3.0]),
                    density=0.1,
                    seed=42,
                ),
            ],
            keep_out_bounds=[
                CylinderGenerator(base_center=[0.0, 0.0, 0.0], radius=2.5, height=5.0)
            ],
        )
        g.generate()
        generate_geodesics(level_name)

        if VIS:
            os.system(f"python clearnav/scripts/scene_viewer.py --scene {level_name}")
            os.system(
                f"python examples/geodesics/interactive_flow_field.py --scene {level_name}"
            )
