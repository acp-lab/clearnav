#!/usr/bin/env python3
import faulthandler

faulthandler.enable()

import sys
import yaml
import torch as th
import argparse
from copy import deepcopy

from clearnav.policies.bptt_algorithm import BPTT
from clearnav.envs.env_aliases import env_aliases
from clearnav.policies.policy_aliases import policy_aliases
from clearnav.policies.multi_input_policy import MultiInputPolicy
from clearnav.common import ExitCode
from clearnav.scripts.eval_logger import Evaluate


def main(args):
    # load training env
    with open(args.cfg_file, "r") as file:
        config = yaml.safe_load(file)
    env_class = env_aliases[config["env_class"]]
    env = env_class(requires_grad=True, **config["env"])

    # load eval envs
    eval_envs = []
    if args.eval_configs is not None:
        for cfg_file in args.eval_configs:
            with open(cfg_file, "r") as file:
                eval_config = yaml.safe_load(file)

            if args.render:
                eval_config["scene_kwargs"]["render_settings"] = {
                    "mode": "follow",
                    "view": "back",
                    "sensor_type": "color",
                    "resolution": [512, 512],
                    "axes": True,
                    "trajectory": False,
                    "object_path": "./datasets/clearnav_dataset/configs/agents/DJI_Mavic_Mini_2.object_config.json",
                    "line_width": 2.0,
                }

            env_class = env_aliases[config["env_class"]]
            eval_env = env_class(requires_grad=False, **eval_config["env"])
            eval_envs.append(eval_env)

    # load policy
    policy_class = policy_aliases[config["policy_class"]]
    policy_kwargs = config["policy"]

    if policy_class == MultiInputPolicy:
        policy = policy_class(env.observation_space, **policy_kwargs)
    else:
        policy = policy_class(**policy_kwargs)

    if args.weight is not None:
        policy.load(args.weight)
    elif config.get("weights_file", None):
        policy.load(config["weights_file"])

    # setup trainer
    trainer = BPTT(
        env=env,
        eval_envs=eval_envs,
        eval_csvs=args.eval_csvs,
        policy=policy,
        run_name=args.run_name,
        logging_dir=args.logging_root,
        **config["train_bptt"],
    )

    # train
    exit_code = trainer.learn(args.render, args.start_iter)

    # save model
    print("Done training. Saving model")
    trainer.save()
    sys.exit(exit_code.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_file", type=str, default="examples/hovering/bptt_hover_1.yaml"
    )
    parser.add_argument("--logging_root", type=str, default="examples/hovering/saved")
    parser.add_argument("--run_name", type=str)
    parser.add_argument(
        "--start_iter", type=int, default=0, help="start index to log to df"
    )
    parser.add_argument(
        "--weight", type=str, default=None, help="pre-trained model weights file"
    )
    parser.add_argument("--render", action="store_true", help="Show observations")
    parser.add_argument(
        "--eval_configs",
        nargs="+",
        type=str,
        default=None,
        help="list of eval env paths",
    )
    parser.add_argument(
        "--eval_csvs",
        nargs="+",
        type=str,
        default=None,
        help="list of paths to write eval stats",
    )
    args = parser.parse_args()
    main(args)
