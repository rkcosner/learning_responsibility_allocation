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

    # plan.extend(plan.compose_cartesian([
    #     ParamRange("algo.dynamics.type", alias="dyn", range=[None, "Bicycle", "Unicycle"]),
    #     ParamRange("algo.spatial_softmax.enabled", alias="ssmax", range=[False, True]),
    #     ParamRange("algo.map_feature_dim", alias="mapfd", range=[64])
    # ]))

    # plan.extend(plan.compose_cartesian([
    #     ParamRange("algo.spatial_softmax.enabled", alias="ssmax", range=[True]),
    #     ParamRange("algo.spatial_softmax.kwargs.num_kp", alias="kp", range=[32, 64]),
    #     ParamRange("algo.map_feature_dim", alias="mapfd", range=[64, 256]),
    #     ParamRange("algo.decoder.layer_dims", alias="decmlp", range=[[], [128, 128]])
    # ]))
    # plan.extend(plan.compose_cartesian([
    #     ParamRange("algo.spatial_softmax.enabled", alias="ssmax", range=[False]),
    #     ParamRange("algo.decoder.layer_dims", alias="decmlp", range=[[], [128, 128]])
    # ]))

    # plan.extend(plan.compose_cartesian([
    #     ParamRange("algo.dynamics.type", alias="dyn", range=["Bicycle", "Unicycle"]),
    #     ParamRange("algo.loss_weights.collision_loss", alias="clw", range=[1.0]),
    #     ParamRange("algo.loss_weights.goal_loss", alias="glw", range=[1.0]),
    #     ParamRange("algo.loss_weights.prediction_loss", alias="plw", range=[0.0, 0.1])
    # ]))

    # plan.add_const_param(Param("train.rollout.enabled", alias="rl", value=False))
    #
    # plan.extend(plan.compose_zip([
    #     ParamRange("algo.model_architecture", alias="arch", range=["resnet50", "resnet50"]),
    #     ParamRange("train.training.batch_size", alias="bs", range=[64, 64]),
    #     ParamRange("algo.loss_weights.pixel_ce_loss", alias="pcl", range=[0.0, 1.0]),
    #     ParamRange("algo.loss_weights.pixel_bce_loss", alias="pbl", range=[1.0, 0.0]),
    # ]))

    # plan.extend(plan.compose_cartesian([
    #     ParamRange("algo.dynamics.type", alias="dyn", range=[None, "Bicycle", "Unicycle"]),
    #     ParamRange("algo.decoder.layer_dims", alias="decmlp", range=[[128, 128]])
    # ]))

    plan.add_const_param(
        Param("train.rollout.enabled", alias="rl", value=False))
    plan.extend(plan.compose_cartesian([
        ParamRange("algo.model_architecture",
                   alias="arch", range=["resnet50"]),
        ParamRange("algo.loss_weights.GAN_loss", alias="Gw", range=[0.1, 0.2]),
        ParamRange("algo.use_transformer", alias="trans", range=[True, False]),
        # ParamRange("algo.optim_params.policy.learning_rate.initial", alias="lr", range=[3e-4]),
        ParamRange("algo.loss_weights.lane_reg_loss",
                   alias="lreg", range=[0.5, 1.0]),
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
        args.prefix,
        delete_config_dir=False
    )
