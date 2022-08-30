"""A script for launching training runs on NGC"""
import argparse

from tbsim.utils.experiment_utils import create_configs, ParamSearchPlan, ParamRange, Param


def configs_to_search_nusc(base_cfg):
    """Override this with your hyperparameter search plan"""
    plan = ParamSearchPlan()
    base_cfg.train.training.num_data_workers = 1
    base_cfg.train.validation.num_data_workers = 1

    # Unchangning Defaults
    # plan.add_const_param(Param("algo.history_num_frames", alias="ht", value=10))
    # plan.add_const_param(Param("algo.future_num_frames", alias="ft", value=5))
    # plan.add_const_param(Param("algo.step_time", alias="dt", value=0.1))

    plan.add_const_param(Param("algo.loss_weights.constraint_loss", alias="cl", value=1.0))
    plan.add_const_param(Param("algo.loss_weights.sum_resp_loss", alias="srl", value=10))
    plan.add_const_param(Param("algo.constraint_loss.leaky_relu_negative_slope", alias="cl_lrns", value=1e-1))
    plan.add_const_param(Param("algo.max_angle_diff", alias="max_d_angl", value=100))
    # plan.add_const_param(Param("algo.loss_weights.yaw_reg_loss", alias="yrl", value=0.01))
    # plan.add_const_param(Param("algo.dynamics.type", alias="dyn", value=None))
    # plan.add_const_param(Param("train.rollout.enabled", alias="rl", value=False))

    # plan.extend(plan.compose_cartesian([
    #     ParamRange("algo.vae.latent_dim", alias="vae_dim", range=[10,15,20]),
    #     ParamRange("algo.model_architecture", alias="arch", range=["resnet50"]),
    # ]))
    plan.extend(plan.compose_cartesian([
        ParamRange("algo.loss_weights.max_likelihood_loss", alias="mll", range=[1e-3]),
        ParamRange("algo.sum_resp_loss.leaky_relu_negative_slope", alias="srl_lrns", range=[1e-3, 1e-2]),
        ParamRange("algo.cbf.alpha", alias="a", range=[2]),
        ParamRange("algo.cbf.T_horizon", alias="T", range=[4]),
        # ParamRange("algo.perturb.OU.sigma", alias="sigma", range=[1.0,2.0,4.0]),
        # ParamRange("algo.loss_weights.EC_collision_loss", alias="EC_loss", range=[10.0,20.0,40.0]),
        # ParamRange("algo.dynamics.type", alias="dyn", range=[None,"Unicycle"]),
    ]))

    plan.extend(plan.compose_zip([
        ParamRange("algo.cbf.saturate_cbf", alias="sat_cbf", range=[True, False])
    ]))

    return plan.generate_configs(base_cfg=base_cfg)


def configs_to_search_l5kit(base_cfg):
    plan = ParamSearchPlan()
    base_cfg.train.training.num_data_workers = 8
    base_cfg.train.validation.num_data_workers = 8

    plan.extend(plan.compose_cartesian([
        # ParamRange("algo.loss_weights.pixel_ce_loss", alias="clw", range=[0.0, 1.0]),
        # ParamRange("algo.loss_weights.pixel_bce_loss", alias="blw", range=[1.0, 0.0]),
        # ParamRange("algo.vae.latent_dim", alias="vae_dim", range=[5,10,15,20]),
        ParamRange("algo.vae.latent_dim", alias="vae_dim", range=[3,5]),
        ParamRange("algo.model_architecture", alias="arch", range=["resnet50"]),
    ]))

    plan.extend(plan.compose_zip([
        ParamRange("algo.stage", alias="stage", range=[2,3,4]),
        ParamRange("algo.num_frames_per_stage", alias="per_stage", range=[25,18,12]),
        ParamRange("algo.model_architecture", alias="arch", range=["resnet50","resnet50","resnet50"]),
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
