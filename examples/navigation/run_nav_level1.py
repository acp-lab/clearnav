from clearnav.scripts.runner import run_experiment

if __name__ == "__main__":
    config_keys = (
        "env.scene_kwargs.load_geodesics",
        "env.scene_kwargs.path",
        "train_bptt.iterations",
    )

    # for tensorboard
    run_params = {
        "level0": (False, "configs/box_2", 500),
        "level1": (True, "configs/level_1", 20000),
    }
    base_config_files = [
        "examples/navigation/train_cfg/nav_empty.yaml",
        "examples/navigation/train_cfg/nav_levelX.yaml",
    ]
    run_experiment(
        script="clearnav/scripts/train_bptt.py",
        experiment_dir="examples/navigation/logs/level1",
        config_keys=config_keys,
        run_params=run_params,
        base_config_files=base_config_files,
        policy_config_file="examples/navigation/policy_cfg/small_yaw.yaml",
        eval_configs=[
            "examples/navigation/eval_cfg/nav_level1.yaml",
        ],
        eval_csvs=[
            "examples/navigation/logs/level1/nav_level_1.csv",
        ],
        curriculum=True,
        max_retries=5,
    )
