"""Factory methods for creating models"""
from tbsim.configs.base import AlgoConfig
from tbsim.algos.l5kit_algos import (
    L5TrafficModel,
    L5TransformerTrafficModel,
    L5TransformerGANTrafficModel,
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
    elif algo_name == "TransformerPred":
        algo = L5TransformerTrafficModel(algo_config=algo_config)
    elif algo_name == "TransformerGAN":
        algo = L5TransformerGANTrafficModel(algo_config=algo_config)
    return algo
