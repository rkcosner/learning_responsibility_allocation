"""A global registry for looking up named experiment configs"""
from tbsim.configs.base import ExperimentConfig

from tbsim.configs.l5kit_config import (
    L5KitTrainConfig,
    L5KitMixedTrainConfig,
    L5KitEnvConfig,
    L5RasterizedPlanningConfig,
    L5RasterizedECConfig,
    SpatialPlannerConfig,
    L5RasterizedGCConfig,
    L5TransformerPredConfig,
    L5TransformerGANConfig,
    L5KitMixedEnvConfig,
    L5KitMixedSemanticMapEnvConfig,
    MARasterizedPlanningConfig,
    L5RasterizedVAEConfig,
    EBMMetricConfig,
    L5RasterizedGANConfig,
    L5RasterizedDiscreteVAEConfig,
    OccupancyMetricConfig
)

from tbsim.configs.nusc_config import (
    NuscTrainConfig,
    NuscEnvConfig
)


EXP_CONFIG_REGISTRY = dict()

# EXP_CONFIG_REGISTRY["l5_rasterized_plan"] = ExperimentConfig(
#     train_config=L5KitTrainConfig(),
#     env_config=L5KitEnvConfig(),
#     algo_config=L5RasterizedPlanningConfig(),
#     registered_name="l5_rasterized_plan",
# )

EXP_CONFIG_REGISTRY["l5_mixed_gc"] = ExperimentConfig(
    train_config=L5KitMixedTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=L5RasterizedGCConfig(),
    registered_name="l5_mixed_gc",
)

EXP_CONFIG_REGISTRY["l5_spatial_planner"] = ExperimentConfig(
    train_config=L5KitMixedTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=SpatialPlannerConfig(),
    registered_name="l5_spatial_planner",
)

EXP_CONFIG_REGISTRY["l5_ma_rasterized_plan"] = ExperimentConfig(
    train_config=L5KitMixedTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=MARasterizedPlanningConfig(),
    registered_name="l5_ma_rasterized_plan"
)

EXP_CONFIG_REGISTRY["l5_mixed_plan"] = ExperimentConfig(
    train_config=L5KitMixedTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=L5RasterizedPlanningConfig(),
    registered_name="l5_mixed_plan",
)

EXP_CONFIG_REGISTRY["l5_gan_plan"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitEnvConfig(),
    algo_config=L5RasterizedGANConfig(),
    registered_name="l5_gan_plan",
)

EXP_CONFIG_REGISTRY["l5_mixed_vae_plan"] = ExperimentConfig(
    train_config=L5KitMixedTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=L5RasterizedVAEConfig(),
    registered_name="l5_mixed_vae_plan",
)

EXP_CONFIG_REGISTRY["l5_mixed_ec_plan"] = ExperimentConfig(
    train_config=L5KitMixedTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=L5RasterizedECConfig(),
    registered_name="l5_mixed_ec_plan",
)

EXP_CONFIG_REGISTRY["l5_mixed_discrete_vae_plan"] = ExperimentConfig(
    train_config=L5KitMixedTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=L5RasterizedDiscreteVAEConfig(),
    registered_name="l5_mixed_discrete_vae_plan",
)

EXP_CONFIG_REGISTRY["l5_mixed_transformer_plan"] = ExperimentConfig(
    train_config=L5KitMixedTrainConfig(),
    env_config=L5KitMixedEnvConfig(),
    algo_config=L5TransformerPredConfig(),
    registered_name="l5_mixed_transformer_plan",
)

EXP_CONFIG_REGISTRY["l5_mixed_transformerGAN_plan"] = ExperimentConfig(
    train_config=L5KitMixedTrainConfig(),
    env_config=L5KitMixedEnvConfig(),
    algo_config=L5TransformerGANConfig(),
    registered_name="l5_mixed_transformerGAN_plan",
)

EXP_CONFIG_REGISTRY["l5_ebm"] = ExperimentConfig(
    train_config=L5KitMixedTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=EBMMetricConfig(),
    registered_name="l5_ebm",
)

EXP_CONFIG_REGISTRY["l5_occupancy"] = ExperimentConfig(
    train_config=L5KitMixedTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=OccupancyMetricConfig(),
    registered_name="l5_occupancy"
)

EXP_CONFIG_REGISTRY["l5_rasterized_ebm"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitEnvConfig(),
    algo_config=EBMMetricConfig(),
    registered_name="l5_rasterized_ebm",
)


EXP_CONFIG_REGISTRY["nusc_rasterized_plan"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=L5RasterizedPlanningConfig(),
    registered_name="nusc_rasterized_plan"
)

EXP_CONFIG_REGISTRY["nusc_spatial_planner"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=SpatialPlannerConfig(),
    registered_name="nusc_spatial_planner"
)

EXP_CONFIG_REGISTRY["nusc_vae_plan"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=L5RasterizedVAEConfig(),
    registered_name="nusc_vae_plan"
)

EXP_CONFIG_REGISTRY["nusc_discrete_vae_plan"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=L5RasterizedDiscreteVAEConfig(),
    registered_name="nusc_discrete_vae_plan"
)

EXP_CONFIG_REGISTRY["nusc_ma_rasterized_plan"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=MARasterizedPlanningConfig(),
    registered_name="nusc_ma_rasterized_plan"
)

EXP_CONFIG_REGISTRY["nusc_rasterized_gc"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=L5RasterizedGCConfig(),
    registered_name="nusc_rasterized_gc"
)



def get_registered_experiment_config(registered_name):
    if registered_name not in EXP_CONFIG_REGISTRY.keys():
        raise KeyError(
            "'{}' is not a registered experiment config please choose from {}".format(
                registered_name, list(EXP_CONFIG_REGISTRY.keys())
            )
        )
    return EXP_CONFIG_REGISTRY[registered_name].clone()
