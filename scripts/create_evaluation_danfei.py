"""A script for launching training runs on NGC"""
import argparse
import os.path

import yaml
from tbsim.configs.eval_configs import EvaluationConfig

from tbsim.utils.experiment_utils import create_evaluation_configs, ParamSearchPlan, ParamRange, Param


def configs_to_search(base_cfg):
    """Override this with your hyperparameter search plan"""
    plan = ParamSearchPlan()
    base_cfg.seed_each_episode = [0, 1, 2, 3, 4]
    plan.extend(plan.compose_cartesian([
        ParamRange("num_episode_repeats", alias="ep", range=[5]),
        # ParamRange("policy.diversification_clearance", alias="div", range=[3])
    ]))
    return plan.generate_configs(base_cfg=base_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used as the template for parameter tuning"
    )

    parser.add_argument(
        "--config_dir",
        type=str,
        default="experiments/test/",
        help="directory for saving generated config files."
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix of the experiment names"
    )

    parser.add_argument(
        "--eval_class",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location of each model",
        default=None,
        required=True
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["nusc", "l5kit"],
        help="Which env to run evaluation in",
        required=True
    )

    parser.add_argument(
        "--ckpt_root_dir",
        type=str,
        default=None,
        help="root directory of checkpoints",
        required=False
    )

    args = parser.parse_args()
    cfg = EvaluationConfig()

    prefix = os.path.basename(args.ckpt_yaml)[:-5]
    if args.prefix is not None:
        prefix = args.prefix + "_" + prefix

    cfg.eval_class = args.eval_class
    cfg.env = args.env
    if args.ckpt_root_dir is not None:
        cfg.ckpt_root_dir = args.ckpt_root_dir
    with open(args.ckpt_yaml, "r") as f:
        ckpt_info = yaml.safe_load(f)
        cfg.ckpt.update(**ckpt_info)

    create_evaluation_configs(
        configs_to_search,
        args.config_dir,
        cfg,
        prefix=prefix,
        delete_config_dir=False
    )
