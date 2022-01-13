"""A global registry for looking up named experiment configs"""
from tbsim.configs.base import ExperimentConfig

from tbsim.configs.l5kit_config import (
    L5KitTrainConfig,
    L5KitMixedTrainConfig,
    L5KitEnvConfig,
    L5RasterizedPlanningConfig,
    L5TransformerPredConfig,
    L5KitVectorizedEnvConfig,
    L5KitMixedEnvConfig,
    L5RasterizedVAEConfig
)

EXP_CONFIG_REGISTRY = dict()

EXP_CONFIG_REGISTRY["l5_rasterized_plan"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitEnvConfig(),
    algo_config=L5RasterizedPlanningConfig(),
    registered_name="l5_rasterized_plan"
)

EXP_CONFIG_REGISTRY["l5_rasterized_vae_plan"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitEnvConfig(),
    algo_config=L5RasterizedVAEConfig(),
    registered_name="l5_rasterized_vae_plan"
)

EXP_CONFIG_REGISTRY["l5_mixed_transformer_plan"] = ExperimentConfig(
    train_config=L5KitMixedTrainConfig(),
    env_config=L5KitMixedEnvConfig(),
    algo_config=L5TransformerPredConfig(),
    registered_name="l5_mixed_transformer_plan"
)


def get_registered_experiment_config(registered_name):
    if registered_name not in EXP_CONFIG_REGISTRY.keys():
        raise KeyError("'{}' is not a registered experiment config please choose from {}".format(
            registered_name, list(EXP_CONFIG_REGISTRY.keys())))
    return EXP_CONFIG_REGISTRY[registered_name].clone()