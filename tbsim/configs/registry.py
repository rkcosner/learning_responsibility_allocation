"""A global registry for looking up named experiment configs"""

from tbsim.configs import (
    ExperimentConfig,
    L5KitEnvConfig,
    L5KitTrainConfig,
    L5RasterizedPlanningConfig,
    L5KitMixedTrainConfig,
    L5KitVectorizedEnvConfig,
    L5TransformerPredConfig,
    L5KitMixedEnvConfig
)

EXP_CONFIG_REGISTRY = dict()

EXP_CONFIG_REGISTRY["l5_raster_plan"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitEnvConfig(),
    algo_config=L5RasterizedPlanningConfig(),
    registered_name="l5_raster_plan"
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
    return EXP_CONFIG_REGISTRY[registered_name]