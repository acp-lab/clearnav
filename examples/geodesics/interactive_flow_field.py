import argparse
import ctypes
import sys
import os

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import magnum as mn
import torch as th
import numpy as np
import yaml

from scipy.spatial.transform import Rotation as R
import quaternion
from habitat_sim.agent import AgentState

from habitat_sim.utils.settings import default_sim_settings
from clearnav.scripts.scene_viewer import HabitatSimInteractiveViewer
from clearnav.common import habitat_to_std, std_to_habitat
from clearnav.envs.env_aliases import env_aliases


opacity = 1.0
cyan = mn.Color4(0.0, 0.5, 1.0, opacity)
orange = mn.Color4(1.0, 1.0, 0.0, opacity)


def color_consequence(color1=orange, color2=cyan, factor=1):
    factor = np.array(factor).clip(min=0.0, max=1.0)
    return color1 * (1 - factor) + factor * color2


class FlowFieldInteractiveViewer(HabitatSimInteractiveViewer):
    def __init__(self, sim_settings, env_config):
        self.env_config = env_config
        super().__init__(sim_settings)

    def reset_agent(self):
        agent = self.sim.get_agent(0)

        # set position
        pos = std_to_habitat(th.tensor([-1.5, 0.0, 20.0]), None)[0]
        # agent_node.translation = pos

        # set orientation
        r_scipy = R.from_euler("y", 90, degrees=True)
        quat = r_scipy.as_quat()
        quat = np.roll(quat, 1)  # reorder from (x, y, z, w) -> (w, x, y, z)
        hab_quat = std_to_habitat(None, th.as_tensor(quat))[1]

        # Create agent state
        state = AgentState()
        state.position = pos
        state.rotation = quaternion.from_float_array(hab_quat)
        agent.set_state(state)
        print("Set starting state")

    def reconfigure_sim(self):
        super().reconfigure_sim()

        # get the current scene in the viewer
        md = self.sim.metadata_mediator
        cur_scene = self.sim_settings["scene"]
        scene_handles = md.get_scene_handles()
        try:
            cur_scene_path = [
                handle for handle in scene_handles if cur_scene in handle
            ][0]
            cur_scene_path = os.path.abspath(cur_scene_path)
        except:
            raise ValueError(f"Scene: '{cur_scene}' not found in dataset")

        # create new env using current scene
        env_config["env"]["scene_kwargs"]["path"] = cur_scene_path
        env_class = env_aliases[self.env_config["env_class"]]
        env = env_class(**env_config["env"])

        # generate trails
        env.reset()
        self.trails = generate_flow_trails(env, env.position, length=150, step_size=0.1)
        self.target = std_to_habitat(env.target[0], None)[0]
        env.close()

        # set starting position
        self.reset_agent()

    def debug_draw(self):
        """
        Additional draw commands to be called during draw_event.
        """
        if not (hasattr(self, "trails") and hasattr(self, "target")):
            return

        debug_line_render = self.sim.get_debug_line_render()

        # draw trails with line segments
        for line_id in range(len(self.trails)):
            for j in range(len(self.trails[line_id]) - 1):
                start = self.trails[line_id][j]
                end = self.trails[line_id][j + 1]
                factor = j / len(self.trails[line_id])
                color = color_consequence(orange, cyan, factor)
                debug_line_render.draw_transformed_line(start, end, color)

        # draw target as circle
        debug_line_render.draw_circle(mn.Vector3(self.target), radius=0.25, color=cyan)


def parse_args():
    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument(
        "--scene",
        default="configs/ring_level3/ring_level3_5",
        type=str,
        help="scene/stage file to load",
    )
    parser.add_argument(
        "--dataset",
        default="./datasets/clearnav_dataset/clearnav_dataset.scene_dataset_config.json",
        type=str,
        metavar="DATASET",
        help='dataset configuration file to use (default: "./datasets/clearnav_dataset/clearnav_dataset.scene_dataset_config.json")',
    )
    parser.add_argument(
        "--width",
        default=800,
        type=int,
        help="Horizontal resolution of the window.",
    )
    parser.add_argument(
        "--height",
        default=600,
        type=int,
        help="Vertical resolution of the window.",
    )
    parser.add_argument(
        "--cfg_file", type=str, default="examples/navigation/train_cfg/nav_ring.yaml"
    )
    args = parser.parse_args()

    if args.width < 1:
        parser.error("width must be a positive non-zero integer.")
    if args.height < 1:
        parser.error("height must be a positive non-zero integer.")

    return args


def generate_flow_trails(env, points, length=100, step_size=0.2):
    # move points along gradient field and record the trails
    trails = th.zeros((len(points), length, 3))
    trails[:, 0] = points
    for i in range(1, length):
        gradient = env.geodesic_gradient(trails[:, i - 1])
        trails[:, i] = trails[:, i - 1] + step_size * gradient

    # convert to habitat coords
    trails_habitat = np.zeros((len(points), length, 3))
    for i in range(len(points)):
        trails_habitat[i] = std_to_habitat(trails[i], None)[0]

    return trails_habitat


if __name__ == "__main__":
    args = parse_args()

    # Setting up sim_settings
    sim_settings = default_sim_settings
    sim_settings["scene"] = args.scene
    sim_settings["scene_dataset_config_file"] = args.dataset
    sim_settings["window_width"] = args.width
    sim_settings["window_height"] = args.height
    sim_settings["default_agent_navmesh"] = False

    with open(args.cfg_file, "r") as file:
        env_config = yaml.safe_load(file)

    # setup env config
    # larger batch size slows down rendering
    bs = 50
    env_config["env"]["num_envs"] = bs
    env_config["env"]["single_env"] = True
    env_config["env"]["scene_kwargs"]["load_geodesics"] = True

    # start the application
    FlowFieldInteractiveViewer(sim_settings, env_config).exec()
