#!/usr/bin/env python3
import faulthandler

faulthandler.enable()

import cv2
import os
import yaml
import time
import torch as th
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from tqdm import trange
from copy import deepcopy
from pylogtools import timerlog

timerlog.timer.print_logs(False)

from clearnav.envs.env_aliases import env_aliases
from clearnav.policies.policy_aliases import policy_aliases
from clearnav.policies.multi_input_policy import MultiInputPolicy
from clearnav.common import observation_to_device, rgba2rgb


def main(args):
    with open(args.cfg_file, "r") as file:
        config = yaml.safe_load(file)

    if args.policy_cfg_file:
        with open(args.policy_cfg_file, "r") as file:
            policy_config = yaml.safe_load(file)
        config.update(policy_config)

        if "update_env_kwargs" in policy_config:
            for k, v in policy_config["update_env_kwargs"].items():
                config["env"][k] = v

    eval_config = deepcopy(config["env"])
    eval_config["num_envs"] = args.num_envs

    env_class = env_aliases[config["env_class"]]
    env = env_class(requires_grad=False, **eval_config)

    policy_class = policy_aliases[config["policy_class"]]
    policy_kwargs = config["policy"]
    if policy_class == MultiInputPolicy:
        observation_space = env.observation_space
        policy = policy_class(observation_space, **policy_kwargs)
    else:
        policy = policy_class(**policy_kwargs)
    policy.load(args.weight)
    policy.eval()

    def create_name(run_path, ext=".mp4"):
        index = 1
        while True:
            path = f"{run_path}_{index}"
            if not os.path.exists(path + ext):
                break
            index += 1
        return path

    if args.save_path is None:
        save_path = create_name(args.cfg_file.split(".")[0])

    e = Evaluate(env, policy)

    if args.run_name is None:
        run_name = create_name(args.weight.split(".")[0])
    df = e.run_rollouts(args.num_rollouts, run_name)

    write_header = not os.path.exists(save_path + ".csv")
    df.to_csv(save_path + ".csv", float_format="%.3f", mode="a", header=write_header)
    print(f"wrote stats to {save_path}.csv")
    env.close()


class Evaluate:
    def __init__(self, env, policy, render_kwargs=None):
        self.env = env
        self.policy = policy
        self.render_kwargs = render_kwargs or {}

    @th.no_grad()
    def run_rollouts(self, num_rollouts=1, run_name=0, render=False):
        all_stats = {
            "avg_speed": [],
            "max_speed": [],
            "max_acceleration": [],
            "collision_count": [],
            "success_count": [],
            "timeout_count": [],
            "avg_reward": [],
            "duration": [],
            "steps": [],
            "path_length": [],
            "avg_yaw_rate": [],
            "max_yaw_rate": [],
            "avg_min_obstacle_distance": [],
            "avg_control_effort": [],
            "last_action_x": [],
            "last_action_y": [],
            "last_action_z": [],
            "last_action_yaw": [],
            "last_position_x": [],
            "last_position_y": [],
            "last_position_z": [],
            "last_velocity_x": [],
            "last_velocity_y": [],
            "last_velocity_z": [],
        }

        for _ in trange(num_rollouts):
            self.env.reset()
            batch_stats = self.single_rollout(render=render)
            for key, value in all_stats.items():
                value.extend(batch_stats[key])

        # convert lists to tensors
        all_stats = {key: th.tensor(value) for key, value in all_stats.items()}

        # summarize the rollout stats in a dataframe
        num_trials = num_rollouts * self.env.num_envs
        rows = [
            {
                "success_rate": all_stats["success_count"].sum().item() / num_trials,
                "collision_rate": all_stats["collision_count"].sum().item()
                / num_trials,
                "timeout_rate": all_stats["timeout_count"].sum().item() / num_trials,
                "num_trials": num_trials,
                "avg_speed": all_stats["avg_speed"].mean().item(),
                "max_speed": all_stats["max_speed"].mean().item(),
                "max_acceleration": all_stats["max_acceleration"].mean().item(),
                "avg_reward": all_stats["avg_reward"].mean().item(),
                "avg_duration": all_stats["duration"].mean().item(),
                "std_duration": all_stats["duration"].std().item(),
                "avg_steps": all_stats["steps"].mean().item(),
                "std_steps": all_stats["steps"].std().item(),
                "avg_path_length": all_stats["path_length"].mean().item(),
                "std_path_length": all_stats["path_length"].std().item(),
                "avg_control_effort": all_stats["avg_control_effort"].mean().item(),
                "avg_yaw_rate": all_stats["avg_yaw_rate"].mean().item(),
                "max_yaw_rate": all_stats["max_yaw_rate"].mean().item(),
                "avg_min_obstacle_distance": all_stats["avg_min_obstacle_distance"]
                .mean()
                .item(),
                "last_action_x": all_stats["last_action_x"].mean().item(),
                "last_action_y": all_stats["last_action_y"].mean().item(),
                "last_action_z": all_stats["last_action_z"].mean().item(),
                "last_action_yaw": all_stats["last_action_yaw"].mean().item(),
                "last_position_x": all_stats["last_position_x"].mean().item(),
                "last_position_y": all_stats["last_position_y"].mean().item(),
                "last_position_z": all_stats["last_position_z"].mean().item(),
                "last_velocity_x": all_stats["last_velocity_x"].mean().item(),
                "last_velocity_y": all_stats["last_velocity_y"].mean().item(),
                "last_velocity_z": all_stats["last_velocity_z"].mean().item(),
            }
        ]
        df = pd.DataFrame(rows, index=[run_name])
        return df

    @th.no_grad()
    def single_rollout(self, render=False):
        # For each agent in the batch
        agent_logs = [
            {
                "position": [],
                "velocity": [],
                "speed": [],
                "acceleration": [],
                "jerk": [],
                "yaw_rate": [],
                "obstacle_distance": [],
                "success": 0,
                "collision": 0,
                "timeout": 0,
                "avg_reward": 0,
                "duration": 0,
                "steps": 0,
                "last_action_x": 0,
                "last_action_y": 0,
                "last_action_z": 0,
                "last_action_yaw": 0,
                "last_position_x": 0,
                "last_position_y": 0,
                "last_position_z": 0,
                "last_velocity_x": 0,
                "last_velocity_y": 0,
                "last_velocity_z": 0,
            }
            for _ in range(self.env.num_envs)
        ]

        eval_info_id_list = [i for i in range(self.env.num_envs)]

        latent_state = th.zeros(
            (self.env.num_envs, self.policy.latent_dim), device=self.policy.device
        )
        while True:
            obs = observation_to_device(self.env.get_observation(), self.policy.device)
            if type(self.policy) == MultiInputPolicy:
                if self.policy.is_recurrent:
                    action, latent_state = self.policy(obs, latent_state)
                else:
                    action = self.policy(obs)
            else:
                action = self.policy(obs["state"])
            obs, reward, terminated, infos = self.env.step(action, is_test=True)

            if render:
                self.render_kwargs["points"] = th.cat(
                    [self.env.position.unsqueeze(1), self.env.target.unsqueeze(1)],
                    dim=1,
                )
                # reshape and concatenate visual obs along the width dimension
                B, C, H, W = obs["depth"].shape
                obs_grid = (
                    obs["depth"].permute(2, 0, 3, 1).reshape(H, B * W, C).cpu().numpy()
                )
                obs_grid = (obs_grid - obs_grid.min()) / (
                    obs_grid.max() - obs_grid.min()
                )
                cv2.imshow("agent cams", obs_grid)

                render_obs = rgba2rgb(
                    self.env.scene_manager.render(**self.render_kwargs)
                )
                render_grid = np.hstack(render_obs)
                cv2.imshow("render cams", render_grid)
                cv2.waitKey(1)

            # log
            for index in reversed(eval_info_id_list):
                if not terminated[index]:
                    # metrics to log at every step
                    agent_logs[index]["speed"].append(
                        self.env.velocity[index].norm().item()
                    )
                    agent_logs[index]["acceleration"].append(
                        self.env.acceleration[index].norm().item()
                    )
                    agent_logs[index]["jerk"].append(self.env.jerk[index].norm().item())
                    agent_logs[index]["yaw_rate"].append(
                        self.env.omega[index][2].item()
                    )
                    agent_logs[index]["obstacle_distance"].append(
                        self.env.collision_dis[index].item()
                    )

                    # these metrics aren't logged directly, but are used to compute other metrics
                    agent_logs[index]["position"].append(self.env.position[index])

                else:
                    # metrics to log only once at the end of the episode
                    eval_info_id_list.remove(index)

                    agent_logs[index]["collision"] = (
                        self.env.is_collision[index].int().item()
                    )
                    agent_logs[index]["success"] = int(infos[index]["is_success"])
                    agent_logs[index]["timeout"] = int(
                        not agent_logs[index]["collision"]
                        and not agent_logs[index]["success"]
                    )
                    agent_logs[index]["avg_reward"] = float(
                        infos[index]["episode_avg_step_reward"]
                    )
                    agent_logs[index]["duration"] = float(
                        infos[index]["episode_duration"]
                    )
                    agent_logs[index]["steps"] = float(infos[index]["episode_length"])

                    # last state action
                    action = action.cpu()
                    agent_logs[index]["last_action_x"] = action[index][0].item()
                    agent_logs[index]["last_action_y"] = action[index][1].item()
                    agent_logs[index]["last_action_z"] = action[index][2].item()
                    agent_logs[index]["last_action_yaw"] = (
                        action[index][3].item() if action.shape[1] >= 4 else 0.0
                    )
                    agent_logs[index]["last_position_x"] = self.env.position[index][
                        0
                    ].item()
                    agent_logs[index]["last_position_y"] = self.env.position[index][
                        1
                    ].item()
                    agent_logs[index]["last_position_z"] = self.env.position[index][
                        2
                    ].item()
                    agent_logs[index]["last_velocity_x"] = self.env.velocity[index][
                        0
                    ].item()
                    agent_logs[index]["last_velocity_y"] = self.env.velocity[index][
                        1
                    ].item()
                    agent_logs[index]["last_velocity_z"] = self.env.velocity[index][
                        2
                    ].item()

                    # path length
                    points = th.stack(agent_logs[index]["position"])
                    path_length = (points[1:] - points[:-1]).norm(dim=1).sum()
                    agent_logs[index]["path_length"] = path_length.item()

                    # control effort
                    jerk = th.tensor(agent_logs[index]["jerk"])
                    total_control_effort = (jerk**2).sum() * self.env.dynamics.ctrl_dt
                    avg_control_effort = total_control_effort / len(jerk)
                    agent_logs[index]["avg_control_effort"] = avg_control_effort.item()

            if terminated.all():
                break

        batch_stats = {
            "avg_speed": [
                th.tensor(agent["speed"]).mean().item() for agent in agent_logs
            ],
            "max_speed": [
                th.tensor(agent["speed"]).max().item() for agent in agent_logs
            ],
            "max_acceleration": [
                th.tensor(agent["acceleration"]).max().item() for agent in agent_logs
            ],
            "avg_yaw_rate": [
                th.tensor(agent["yaw_rate"]).mean().item() for agent in agent_logs
            ],
            "max_yaw_rate": [
                th.tensor(agent["yaw_rate"]).max().item() for agent in agent_logs
            ],
            "avg_min_obstacle_distance": [
                th.tensor(agent["obstacle_distance"]).min().item()
                for agent in agent_logs
            ],
            "collision_count": [agent["collision"] for agent in agent_logs],
            "success_count": [agent["success"] for agent in agent_logs],
            "timeout_count": [agent["timeout"] for agent in agent_logs],
            "avg_reward": [agent["avg_reward"] for agent in agent_logs],
            "duration": [agent["duration"] for agent in agent_logs],
            "steps": [agent["steps"] for agent in agent_logs],
            "path_length": [agent["path_length"] for agent in agent_logs],
            "avg_control_effort": [agent["avg_control_effort"] for agent in agent_logs],
            "last_action_x": [agent["last_action_x"] for agent in agent_logs],
            "last_action_y": [agent["last_action_y"] for agent in agent_logs],
            "last_action_z": [agent["last_action_z"] for agent in agent_logs],
            "last_action_yaw": [agent["last_action_yaw"] for agent in agent_logs],
            "last_position_x": [agent["last_position_x"] for agent in agent_logs],
            "last_position_y": [agent["last_position_y"] for agent in agent_logs],
            "last_position_z": [agent["last_position_z"] for agent in agent_logs],
            "last_velocity_x": [agent["last_velocity_x"] for agent in agent_logs],
            "last_velocity_y": [agent["last_velocity_y"] for agent in agent_logs],
            "last_velocity_z": [agent["last_velocity_z"] for agent in agent_logs],
        }
        return batch_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="examples/navigation/eval_cfg/nav_level1.yaml")
    parser.add_argument("--policy_cfg_file", type=str, default="examples/navigation/policy_cfg/small_yaw.yaml")
    parser.add_argument("--weight", type=str, default=None, help="trained weight name")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_rollouts", type=int, default=5)
    args = parser.parse_args()
    main(args)
