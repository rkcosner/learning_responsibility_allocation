import argparse
import json
import os

from tbsim.utils.experiment_utils import (
    launch_experiments_ngc,
    upload_codebase_to_ngc_workspace,
    read_evaluation_configs
)
from tbsim.configs.eval_configs import EvaluationConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="(optional) path to a config json to launch the experiment with"
    )

    parser.add_argument(
        "--config_dir",
        type=str,
        default="experiments/",
        help="directory to read config files from."
    )

    parser.add_argument(
        "--ngc_config",
        type=str,
        help="path to your ngc config file",
        default="ngc/ngc_config.json"
    )

    parser.add_argument(
        "--script_path",
        type=str,
        default="scripts/evaluate.py"
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="vae-sweep"
    )

    parser.add_argument(
        "--ngc_instance",
        type=str,
        default="dgx1v.16g.1.norm"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False
    )

    args = parser.parse_args()
    if args.config_file is not None:
        cfg = EvaluationConfig()
        ext_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**ext_cfg)
        cfgs = [cfg]
        cfg_fns = [args.config_file]
    else:
        cfgs, cfg_fns = read_evaluation_configs(args.config_dir)

    ngc_cfg = json.load(open(args.ngc_config, "r"))
    ngc_cfg["wandb_apikey"] = os.environ["WANDB_APIKEY"]
    ngc_cfg["wandb_project_name"] = args.wandb_project_name
    ngc_cfg["instance"] = args.ngc_instance

    script_command = [
        "python",
        args.script_path,
        "--results_dir",
        ngc_cfg["result_dir"],
        "--dataset_path",
        ngc_cfg["dataset_path"]
    ]

    res = input("upload codebase to ngc workspace? (y/n)")
    if res == "y":
        print("uploading codebase ... (this may take a while)")
        upload_codebase_to_ngc_workspace(ngc_cfg)
    launch_experiments_ngc(script_command, cfgs, cfg_fns, ngc_config=ngc_cfg, dry_run=args.dry_run)