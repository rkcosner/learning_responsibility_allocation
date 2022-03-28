"""Factory methods for creating models"""
from pytorch_lightning import LightningDataModule
from tbsim.configs.base import ExperimentConfig

from tbsim.algos.l5kit_algos import (
    L5TrafficModel,
    L5TransformerTrafficModel,
    L5TransformerGANTrafficModel,
    L5VAETrafficModel,
    L5TrafficModelGC,
    SpatialPlanner,
)

from tbsim.algos.multiagent_algos import (
    MATrafficModel,
    MAGANTrafficModel
)

from tbsim.algos.selfplay_algos import (
    SelfPlayHierarchical
)

from tbsim.algos.metric_algos import (
    EBMMetric
)

def algo_factory(config: ExperimentConfig, modality_shapes: dict, data_module: LightningDataModule, **kwargs):
    """
    A factory for creating training algos

    Args:
        config (ExperimentConfig): an ExperimentConfig object,
        modality_shapes (dict): a dictionary that maps observation modality names to shapes
        data_module (LightningDataModule): (optional) a pytorch_lightning data_module object
        **kwargs: any info needed to create an algo

    Returns:
        algo: pl.LightningModule
    """
    algo_config = config.algo
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
    elif algo_name == "ma_rasterized":
        if algo_config.use_GAN:
            algo = MAGANTrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
        else:
            algo = MATrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "TransformerPred":
        algo = L5TransformerTrafficModel(algo_config=algo_config)
    elif algo_name == "TransformerGAN":
        algo = L5TransformerGANTrafficModel(algo_config=algo_config)
    elif algo_name == "sp_hierarchical":
        algo = SelfPlayHierarchical(cfg=config, data_module=data_module)
    elif algo_name == "l5_ebm":
        algo = EBMMetric(algo_config=algo_config, modality_shapes=modality_shapes)
    else:
        raise NotImplementedError("{} is not a valid algorithm" % algo_name)
    return algo
