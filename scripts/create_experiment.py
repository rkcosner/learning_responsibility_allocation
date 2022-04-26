"""A script for launching training runs on NGC"""
import argparse

from tbsim.utils.experiment_utils import create_configs, ParamSearchPlan, ParamRange, Param


def configs_to_search_nusc(base_cfg):
    """Override this with your hyperparameter search plan"""
    plan = ParamSearchPlan()
    base_cfg.train.training.num_data_workers = 24
    base_cfg.train.validation.num_data_workers = 8

    plan.add_const_param(Param("algo.history_num_frames", alias="ht", value=3))
    plan.add_const_param(Param("algo.future_num_frames", alias="ft", value=10))
    plan.add_const_param(Param("algo.step_time", alias="dt", value=0.5))
    plan.extend(plan.compose_cartesian([
        ParamRange("algo.loss_weights.yaw_reg_loss", alias="yrl", range=[0.01]),
        ParamRange("algo.dynamics.type", alias="dyn", range=[None, "Unicycle"])
    ]))

    return plan.generate_configs(base_cfg=base_cfg)


def configs_to_search_l5kit(base_cfg):
    plan = ParamSearchPlan()
    base_cfg.train.training.num_data_workers = 24
    base_cfg.train.validation.num_data_workers = 8

    plan.extend(plan.compose_zip([
        ParamRange("algo.loss_weights.pixel_ce_loss", alias="clw", range=[0.0, 1.0]),
        ParamRange("algo.loss_weights.pixel_bce_loss", alias="blw", range=[1.0, 0.0]),
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
        "--env",
        type=str,
        choices=["nusc", "l5kit"],
        required=True
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix of the experiment names"
    )

    args = parser.parse_args()
    fn = configs_to_search_nusc if args.env == "nusc" else configs_to_search_l5kit
    prefix = args.env
    prefix += "_" + args.prefix if args.prefix is not None else ""

    create_configs(
        fn,
        args.config_name,
        args.config_file,
        args.config_dir,
        prefix,
        delete_config_dir=False
    )
