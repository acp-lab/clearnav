import os
import sys
import time
from typing import Any, Union, Optional, Type, Dict, TypeVar, ClassVar, List

import torch as th
import numpy as np
from tqdm import tqdm

from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common import logger

from pylogtools import timerlog

timerlog.timer.print_logs(False)

from .debug import check_none_parameters, get_network_statistics, compute_gradient_norm
from .mlp_policy import MlpPolicy
from ..common import observation_to_device, rgba2rgb, ExitCode
from clearnav.scripts.eval_logger import Evaluate


class BPTT:
    """
    Back Propagation Through Time (BPTT).
    Pass in an env, eval_env, and policy
    Call learn() and get back the trained policy
    Gradients are computed across multiple timesteps and loss is computed at final time step.
    """

    def __init__(
        self,
        policy: Union[MlpPolicy],
        env: GymEnv,
        eval_envs: List[GymEnv] = None,
        eval_csvs: List[str] = None,
        learning_rate_init: float = 3e-4,
        learning_rate_final: float = 0.0,
        weight_decay: float = 0.0,
        run_name: Optional[str] = "BPTT",
        logging_dir: Optional[str] = "./saved",
        horizon: int = 32,
        gamma: float = 0.99,
        iterations: int = 1000,
        log_interval: int = 1,
        eval_at_end_only: bool = False,
        early_stop_reward_threshold: float = -th.inf,
        checkpoint_interval: int = 1000,
        device: Optional[Union[str, th.device]] = "cpu",
    ):
        self.logging_dir = os.path.abspath(logging_dir)
        self.run_path = self._create_run_path(run_name)

        self.env = env
        self.eval_envs = eval_envs or []
        self.eval_csvs = []
        if eval_csvs:
            for csv_path in eval_csvs:
                self.eval_csvs.append(
                    os.path.join(self.run_path, os.path.basename(csv_path))
                )
        assert len(self.eval_envs) == len(self.eval_csvs)
        self.device = th.device(device)
        self.policy = policy.to(self.device)

        # training parameters
        self.iterations = iterations
        self.horizon = horizon
        self.gamma = gamma
        self.optimizer = th.optim.AdamW(
            self.policy.parameters(), lr=learning_rate_init, weight_decay=weight_decay
        )
        self.lr_schedule = th.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=iterations, eta_min=learning_rate_final
        )

        # training buffers used for logging
        self.log_interval = log_interval
        self.eval_at_end_only = eval_at_end_only
        self.early_stop_reward_threshold = early_stop_reward_threshold
        self.checkpoint_interval = checkpoint_interval

        self.whitelisted_tensorboard_keys = [
            "avg_reward",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "avg_steps",
            "avg_path_length",
            "avg_speed",
            "max_speed",
            "max_acceleration",
            "max_yaw_rate",
            "avg_min_obstacle_distance",
        ]

        self.whitelisted_csv_keys = [
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "num_trials",
            "avg_speed",
            "max_speed",
            "avg_path_length",
            "avg_control_effort",
            "max_acceleration",
            "avg_yaw_rate",
            "max_yaw_rate",
            "avg_reward",
            "avg_min_obstacle_distance",
            "scene",
        ]

    def _create_run_path(self, run_name="BPTT"):
        index = 1
        run_name = run_name or "BPTT"
        while True:
            path = f"{self.logging_dir}/{run_name}_{index}"
            if not os.path.exists(path):
                break
            index += 1
        return path

    def save(self, filepath=None):
        filepath = filepath or self.run_path + ".pth"
        print(f"Saving to {filepath}")
        self.policy.save(filepath)

    def _train_iteration(self, latent_state, render) -> tuple:
        """One iteration: rollout the policy over the horizon, backprop,
        and take an optimizer step.  Returns (latent_state, metrics)."""
        assert self.horizon >= 1, "horizon must be greater than 1"
        loss = 0.0
        reward_metric_sums = {}
        fps_start = time.time_ns()
        discount_factor = th.ones(
            self.env.num_envs, dtype=th.float32, device=self.device
        )

        for _ in range(self.horizon):
            obs = self.env.get_observation()
            obs = observation_to_device(obs, self.policy.device)
            if type(self.policy) == MlpPolicy:
                actions = self.policy(obs["state"])
            else:
                if self.policy.is_recurrent:
                    actions, latent_state = self.policy(obs, latent_state)
                else:
                    actions = self.policy(obs)

            # step
            obs, reward, done, info = self.env.step(actions, is_test=False)
            reward = reward.to(self.device)
            done = done.to(self.device).to(th.bool)
            loss = loss + -1.0 * reward * discount_factor

            # for logging
            for _k, _v in self.env._last_loss_metrics.items():
                reward_metric_sums[_k] = reward_metric_sums.get(_k, 0.0) + _v.mean()

            # if done, reset discount factor and latents
            discount_factor = discount_factor * self.gamma * ~done + done
            latent_state = latent_state * ~done.unsqueeze(1)

        total_steps = self.env.num_envs * self.horizon
        time_elapsed = max(
            (time.time_ns() - fps_start) / 1e9, sys.float_info.epsilon
        )
        fps = int(total_steps / time_elapsed)

        # backprop
        loss = loss / self.horizon  # average across the rollout
        loss = loss.mean()  # average across the batch
        self.optimizer.zero_grad()
        loss.backward()

        # log total gradient magnitude and clip to prevent exploding gradients
        max_norm = 5.0
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.policy.parameters(), max_norm=max_norm
        )

        # update policy
        self.optimizer.step()
        self.lr_schedule.step()

        metrics = {
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "fps": fps,
            "steps": self.horizon,
        }
        for _k, _s in reward_metric_sums.items():
            metrics[_k] = float(_s / self.horizon)
        return latent_state, metrics

    def _eval(self, iter, start_iter, render) -> bool:
        """Run eval rollouts on every eval env, log to tensorboard + csv.

        Returns ``True`` if the early-stop reward threshold is breached.
        """
        self.policy.eval()
        for eval_env, csv_file in zip(self.eval_envs, self.eval_csvs):
            # evaluate policy in eval_env with multiple rollouts
            e = Evaluate(eval_env, self.policy)
            index = start_iter + iter
            df = e.run_rollouts(num_rollouts=5, run_name=index, render=render)

            # add a column to log the scene
            df["scene"] = os.path.basename(self.env.scene_manager.scene_path)

            # log the df to tensorboard
            basename = os.path.basename(csv_file).split(".")[0]
            self.df_to_tensorboard(self._logger, df, prefix=basename)

            # save df to csv
            write_header = not os.path.exists(csv_file)
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            df.to_csv(
                csv_file,
                float_format="%.3f",
                mode="a",
                header=write_header,
                columns=self.whitelisted_csv_keys,
            )
            print(f"wrote stats to {csv_file}")

            # check if we should early stop
            last_avg_reward = df["avg_reward"].iloc[-1]
            if last_avg_reward < self.early_stop_reward_threshold:
                print("REWARD FELL BELOW EARLY STOP THRESHOLD")
                print(f"Last reward: {last_avg_reward}")
                return True
        return False

    def _log_train_metrics(self, logger, metrics: dict) -> None:
        """Surface reward loss-metrics (incl. visibility-gated terms) in logs."""
        for key, val in metrics.items():
            if key in ("loss", "grad_norm", "fps", "steps"):
                continue
            logger.record(f"train/{key}", float(val))

    def learn(self, render=False, start_iter=0) -> ExitCode:
        """
        Train policy using BPTT, return True when finished training
        """
        assert self.env is not None

        self.policy.train()
        check_none_parameters(self.policy)
        self.start_time = time.time_ns()
        exit_code = ExitCode.ERROR
        self._logger = logger.configure(self.run_path, ["stdout", "tensorboard"])

        try:
            self.env.reset()
            episode_steps = 0
            latent_state = th.zeros(
                (self.env.num_envs, self.policy.latent_dim), device=self.policy.device
            )
            iter = 0
            for iter in tqdm(range(self.iterations)):
                timerlog.timer.tic("iteration")

                self.policy.train()

                # reset agents once they have max_episode_steps experience
                if episode_steps >= self.env.max_episode_steps:
                    self.env.reset()
                    episode_steps = 0

                latent_state, metrics = self._train_iteration(latent_state, render)
                episode_steps += metrics["steps"]

                # detach gradients
                self.env.detach()
                latent_state = latent_state.clone().detach()
                timerlog.timer.toc("iteration")

                timerlog.timer.tic("eval")
                if iter % self.log_interval == 0:
                    early_stop = False
                    if not self.eval_at_end_only:
                        early_stop = self._eval(iter, start_iter, render)

                    # log and dump iter to tensorboard
                    self._logger.record(
                        "train/learning_rate", self.lr_schedule.get_last_lr()[0]
                    )
                    self._logger.record("train/loss", metrics["loss"])
                    self._logger.record("train/grad_norm", metrics["grad_norm"])
                    self._logger.record("train/steps_per_second", metrics["fps"])
                    self._log_train_metrics(self._logger, metrics)
                    self._logger.dump(start_iter + iter)

                    timerlog.timer.toc("eval")
                    timerlog.timer.print_summary()
                    timerlog.timer.clear_history()

                    if early_stop:
                        exit_code = ExitCode.EARLY_STOP
                        break
                if iter > 0 and iter % self.checkpoint_interval == 0:
                    self.save(self.run_path + "_iteration_" + str(iter) + ".pth")
            else:
                exit_code = ExitCode.SUCCESS
                if self.eval_at_end_only and self.eval_envs:
                    self._eval(self.iterations - 1, start_iter, render)
                    self._logger.dump(start_iter + self.iterations - 1)
        except KeyboardInterrupt:
            self.save(self.run_path + "_iteration_" + str(iter) + ".pth")
            exit_code = ExitCode.KEYBOARD_INTERRUPT
        finally:
            for eval_env in self.eval_envs:
                eval_env.close()
            self.env.close()
        return exit_code

    def df_to_tensorboard(self, logger, df, prefix="eval"):
        keys = df.columns.tolist()
        for key in keys:
            if key not in self.whitelisted_tensorboard_keys:
                continue
            logger.record(f"{prefix}/{key}", df[key].iloc[-1])
