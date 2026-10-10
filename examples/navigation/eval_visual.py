#!/usr/bin/env python3

import os
import cv2
import yaml
import time
import torch as th
import torch.nn.functional as F
import numpy as np
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
from pylogtools import timerlog

timerlog.timer.print_logs(False)

from clearnav.envs.env_aliases import env_aliases
from clearnav.policies.policy_aliases import policy_aliases
from clearnav.policies.multi_input_policy import MultiInputPolicy
from clearnav.common import observation_to_device
from clearnav.common import std_to_habitat, obs_list2array, rgba2rgb


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
    eval_config["single_env"] = False
    eval_config["scene_kwargs"]["load_geodesics"] = False

    render_kwargs = {}

    eval_config["scene_kwargs"]["render_settings"] = {
        "mode": "follow",
        "view": "back",
        "sensor_type": "color",
        "resolution": [args.res, args.res],
        "axes": True,
        "trajectory": False,
        "object_path": "./datasets/clearnav_dataset/configs/agents/DJI_Mavic_Mini_2.object_config.json",
        "line_width": 2.0,
    }
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

    def create_name(run_path):
        index = 1
        while True:
            path = f"{run_path}_{index}"
            if not os.path.exists(path + ".mp4"):
                break
            index += 1
        return path

    if args.save_name is None:
        save_name = create_name(args.weight.split(".")[0] + "_eval")

    evaluate = Evaluate(
        env=env,
        policy=policy,
        save_name=save_name,
        save=True,
        show=args.render,
        render_kwargs=render_kwargs,
        res=args.res,
        num_cols=args.num_envs,
    )
    evaluate.run_rollouts(args.num_rollouts)


class Evaluate:
    def __init__(
        self,
        env,
        policy,
        save_name,
        save,
        show=True,
        render_kwargs={},
        res=512,
        num_rows=1,
        num_cols=4,
    ):
        """
        renders the first num_cols * num_rows environments
        """
        assert num_rows * num_cols == env.num_envs

        self.env = env
        self.policy = policy

        self.save_name = save_name
        self.save = save
        self.show = show
        self.render_kwargs = render_kwargs
        self.res = res
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.render_grid = np.zeros((num_rows * res, num_cols * res, 3), dtype=np.uint8)
        self.collided = th.zeros(self.env.num_envs, dtype=th.bool)
        self.first_collision = th.zeros(self.env.num_envs, dtype=th.bool)
        self.done = th.zeros(self.env.num_envs, dtype=th.bool)
        self.all_frames = []

    @th.no_grad()
    def run_rollouts(self, num_rollouts=1):
        for _ in range(num_rollouts):
            self.env.reset()
            # add start and target position to render_kwargs
            self.render_kwargs["points"] = th.cat(
                [self.env.position.unsqueeze(1), self.env.target.unsqueeze(1)], dim=1
            )  # (B, 2, 3)
            self.single_rollout()

        if self.save:
            self.save_video()
        self.env.close()
        cv2.destroyAllWindows()

    def save_video(self):
        print("Processing video")
        fps = int(1.0 / self.env.dynamics.ctrl_dt)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
        video_width = self.num_cols * self.res
        video_height = self.num_rows * self.res
        out = cv2.VideoWriter(
            f"{self.save_name}.mp4", fourcc, fps, (video_width, video_height)
        )
        for img in self.all_frames:
            out.write(img)
        out.release()
        print(f"Saved video to {self.save_name}.mp4")

    @th.no_grad()
    def single_rollout(self):
        self.render_grid = np.zeros(
            (self.num_rows * self.res, self.num_cols * self.res, 3), dtype=np.uint8
        )
        self.collided = th.zeros(self.env.num_envs, dtype=th.bool)
        self.first_collision = th.zeros(self.env.num_envs, dtype=th.bool)
        self.done = th.zeros(self.env.num_envs, dtype=th.bool)

        positions = []
        positions = th.zeros((self.env.num_envs, self.env.max_episode_steps, 3))
        quaternions = th.zeros((self.env.num_envs, self.env.max_episode_steps, 4))
        obs_quaternions = th.zeros((self.env.num_envs, self.env.max_episode_steps, 4))
        velocities = th.zeros((self.env.num_envs, self.env.max_episode_steps, 3))
        obs_velocities = th.zeros((self.env.num_envs, self.env.max_episode_steps, 3))
        moving_avg_velocities = th.zeros(
            (self.env.num_envs, self.env.max_episode_steps, 3)
        )
        accelerations = th.zeros((self.env.num_envs, self.env.max_episode_steps, 3))
        jerks = th.zeros((self.env.num_envs, self.env.max_episode_steps, 3))
        actions = None
        terminations = th.zeros((self.env.num_envs, self.env.max_episode_steps))
        target_dist = th.zeros((self.env.num_envs, self.env.max_episode_steps))
        target_vel = th.zeros((self.env.num_envs, self.env.max_episode_steps))
        loss_metrics = [{} for _ in range(self.env.num_envs)]

        latent_state = th.zeros(
            (self.env.num_envs, self.policy.latent_dim), device=self.policy.device
        )
        i = 0
        while True:
            start = time.time()
            obs = observation_to_device(self.env.get_observation(), self.policy.device)
            if type(self.policy) == MultiInputPolicy:
                if self.policy.is_recurrent:
                    action, latent_state = self.policy(obs, latent_state)
                else:
                    action = self.policy(obs)
            else:
                action = self.policy(obs["state"])
            obs, reward, terminated, infos = self.env.step(action, is_test=True)
            self.first_collision = self.env.is_collision & ~self.collided
            self.collided = self.collided | self.env.is_collision

            if self.show or self.save_video:
                render_list = rgba2rgb(
                    self.env.scene_manager.render(**self.render_kwargs)
                )
                self.render(render_list, obs, terminated)

                if self.show:
                    cv2.imshow("render cams", self.render_grid)
                    elapsed = time.time() - start
                    wait_ms = max(int(1000 * (self.env.dynamics.ctrl_dt - elapsed)), 1)
                    cv2.waitKey(wait_ms)

            for j in range(self.env.num_envs):
                if terminated[j]:
                    continue
                for metric, value in infos[j]["loss_metrics"].items():
                    if i == 0:
                        loss_metrics[j][metric] = th.zeros(self.env.max_episode_steps)
                    loss_metrics[j][metric][i] = value.item()

            i += 1
            if terminated.all():
                break

    def _preprocess_depth(self, depth: th.Tensor):
        """From ImageExtractor.preprocess_depth()"""
        depth = depth.float()
        depth[depth < 1e-6] = 1e-6
        inv_depth = 1.0 / depth

        if hasattr(self.policy.feature_extractor, "input_max_pool_H_W"):
            # apply max pooling
            # note this preserves closer objects, since we use inverted depth
            H, W = self.policy.feature_extractor.input_max_pool_H_W
            inv_depth = F.adaptive_max_pool2d(inv_depth, (H, W))
            pass
        return inv_depth

    def render(self, render_list, observations, dones):
        obs_list = th.split(observations["depth"], 1, dim=0)  # (B, C, H, W)
        obs_list = [obs.squeeze(0).permute(1, 2, 0).cpu().numpy() for obs in obs_list]
        for i, (render, depth, done) in enumerate(zip(render_list, obs_list, dones)):
            row = i // self.num_cols
            col = i % self.num_cols

            # visualize downsampled observation
            # (H, W, C) -> (C, H, W)
            obs = th.from_numpy(np.transpose(depth, (2, 0, 1)))
            obs = self._preprocess_depth(obs).numpy()
            # (C, H, W) -> (H, W, C)
            obs = np.transpose(obs, (1, 2, 0))

            # rescale for visualization purposes
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
            depth = depth.astype(np.uint8)
            obs = (obs - obs.min()) / (obs.max() - obs.min()) * 255
            obs = obs.astype(np.uint8)

            if self.first_collision[i]:
                img = self.inset_image(render, depth, inset_scale=1)
                top = depth.shape[0] * 2
                img = self.inset_image(render, obs, inset_scale=1, top=top)
                img = self.tint_red(img)
                self.render_grid[
                    row * self.res : (row + 1) * self.res,
                    col * self.res : (col + 1) * self.res,
                ] = img

            if not done:
                img = self.inset_image(render, depth, inset_scale=1)
                top = depth.shape[0] * 2
                img = self.inset_image(render, obs, inset_scale=1, top=top)
                self.render_grid[
                    row * self.res : (row + 1) * self.res,
                    col * self.res : (col + 1) * self.res,
                ] = img
        self.all_frames.append(np.copy(self.render_grid))

    @staticmethod
    def tint_red(image, alpha=0.3):
        red_overlay = np.zeros_like(image)
        red_overlay[..., 2] = 255  # bgr
        tinted = (1 - alpha) * image + alpha * red_overlay
        return np.clip(tinted, 0, 255).astype(np.uint8)

    @staticmethod
    def inset_image(
        image: np.ndarray, inset: np.ndarray, inset_scale=1.0, top=0, left=0
    ):
        h, w, c = inset.shape
        new_h = h * inset_scale
        new_w = w * inset_scale
        if c == 1 and image.shape[2] == 3:
            inset = np.repeat(inset, repeats=3, axis=2)
        inset_scaled = cv2.resize(
            inset, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )
        image[top : top + new_h, left : left + new_w] = inset_scaled
        return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="examples/navigation/eval_cfg/nav_level1.yaml")
    parser.add_argument("--policy_cfg_file", type=str, default="examples/navigation/policy_cfg/small_yaw.yaml")
    parser.add_argument("--weight", type=str, default=None, help="trained weight name")
    parser.add_argument("--render", action="store_true", help="Show observations")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_rollouts", type=int, default=10)
    parser.add_argument("--res", type=int, default=256)
    args = parser.parse_args()
    main(args)
