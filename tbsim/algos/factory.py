"""Factory methods for creating models"""
from tbsim.configs.base import AlgoConfig
from tbsim.algos.l5kit_algos import (
    L5TrafficModel,
    L5TransformerTrafficModel,
    L5TransformerGANTrafficModel,
)
from tbsim.algos.l5kit_algos import (
    L5TrafficModel,
    L5TransformerTrafficModel,
    L5VAETrafficModel,
    L5TrafficModelGC,
    SpatialPlanner,
)


def algo_factory(algo_config: AlgoConfig, modality_shapes, **kwargs):
    """
    A factory for creating training algos

    Args:
        algo_config (AlgoConfig): an algo config object
        modality_shapes (dict): A dictionary of named observation shapes (e.g., rasterized image shape)
        **kwargs: any info needed to create an algo

    Returns:
        algo: pl.LightningModule
    """
    algo_name = algo_config.name

    if algo_name == "l5_rasterized":
        algo = L5TrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "l5_rasterized_gc":
        algo = L5TrafficModelGC(
            algo_config=algo_config, modality_shapes=modality_shapes
        )
    elif algo_name == "l5_rasterized_vae":
        algo = L5VAETrafficModel(
            algo_config=algo_config, modality_shapes=modality_shapes
        )
    elif algo_name == "spatial_planner":
        algo = SpatialPlanner(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "TransformerPred":
        algo = L5TransformerTrafficModel(algo_config=algo_config)
    elif algo_name == "TransformerGAN":
        algo = L5TransformerGANTrafficModel(algo_config=algo_config)
    else:
        raise NotImplementedError("{} is not a valid algorithm" % algo_name)
    return algo
