"""A script for launching training runs on NGC"""
import argparse

from tbsim.utils.experiment_utils import create_evaluation_configs, ParamSearchPlan, ParamRange, Param


def configs_to_search(base_cfg):
    """Override this with your hyperparameter search plan"""
    plan = ParamSearchPlan()

    plan.extend(plan.compose_cartesian([
        ParamRange("eval_class", alias="eval", range=["HierAgentAware", "Hierarchical"]),
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

    args = parser.parse_args()

    create_evaluation_configs(
        configs_to_search,
        args.config_file,
        args.config_dir,
        args.prefix,
        delete_config_dir=False
    )
