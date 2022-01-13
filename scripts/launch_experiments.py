import argparse
import json
import os

from tbsim.utils.experiment_utils import read_configs, launch_experiments_ngc, launch_experiments_local
from tbsim.configs.registry import get_registered_experiment_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="(optional) create experiment config from a preregistered name (see configs/registry.py)"
    )

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
        "--ngc",
        action="store_true",
        help="(optional) if launching experiment on ngc"
    )

    parser.add_argument(
        "--ngc_config",
        type=str,
        help="path to ngc config"
    )

    parser.add_argument(
        "--script_path",
        type=str,
        default="scripts/train_l5kit.py"
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

    args = parser.parse_args()
    if args.config_name is not None:
        cfg = get_registered_experiment_config(args.config_name)
        cfgs = [cfg]
        cfg_fn = os.path.join(args.config_dir, "/{}.json".format(cfg.name))
        cfg.dump(filename=cfg_fn)
        cfg_fns = [cfg_fn]
    elif args.config_file is not None:
        ext_cfg = json.load(open(args.config_file, "r"))
        cfg = get_registered_experiment_config(ext_cfg["registered_name"])
        cfg.update(**ext_cfg)
        cfgs = [cfg]
        cfg_fns = [args.config_file]
    else:
        cfgs, cfg_fns = read_configs(args.config_dir)

    if args.ngc:
        ngc_cfg = json.load(open(args.ngc_config, "r"))
        ngc_cfg["wandb_apikey"] = os.environ["WANDB_APIKEY"]
        ngc_cfg["wandb_project_name"] = args.wandb_project_name
        ngc_cfg["instance"] = args.ngc_instance
        launch_experiments_ngc(args.script_path, cfgs, cfg_fns, ngc_config=ngc_cfg)
    else:
        launch_experiments_local(args.script_path, cfgs, cfg_fns)