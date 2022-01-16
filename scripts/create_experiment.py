"""A script for launching training runs on NGC"""
import argparse

from tbsim.utils.experiment_utils import create_configs, ParamSearchPlan, ParamRange, Param


def configs_to_search(base_cfg):
    """Override this with your hyperparameter search plan"""
    plan = ParamSearchPlan()
    # plan.add_const_param(Param("algo.dynamics.type", alias="dyn", value="Unicycle"))
    #
    # plan.extend(plan.compose_cartesian([
    #     ParamRange("algo.optim_params.policy.learning_rate.initial", alias="lr", range=[1e-3, 1e-4]),
    #     ParamRange("train.training.batch_size", alias="bs", range=[50, 100]),
    # ]))
    #
    # plan.extend(plan.compose_zip([
    #     ParamRange("algo.optim_params.policy.learning_rate.initial", alias="lr", range=[1e-3, 1e-4]),
    #     ParamRange("algo.model_architecture", alias="arch", range=["resnet50", "resnet18"]),
    # ]))
    # plan.extend(plan.compose_concate([
    #     ParamRange("algo.dynamics.type", alias="dyn", range=[None, "Unicycle"]),
    # ]))

    plan.extend(plan.compose_concate([
        ParamRange("algo.dynamics.type", alias="mapfd", range=[64, 128, 256]),
    ]))
    return plan.generate_configs(base_cfg=base_cfg)


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

    create_configs(
        configs_to_search,
        args.config_name,
        args.config_file,
        args.config_dir,
        args.prefix
    )