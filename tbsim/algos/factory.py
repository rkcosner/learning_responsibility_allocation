"""Factory methods for creating models"""
from pytorch_lightning import LightningDataModule
from tbsim.configs.base import ExperimentConfig

from tbsim.algos.l5kit_algos import (
    BehaviorCloning,
    TransformerTrafficModel,
    TransformerGANTrafficModel,
    VAETrafficModel,
    DiscreteVAETrafficModel,
    BehaviorCloningGC,
    SpatialPlanner,
    GANTrafficModel,
    BehaviorCloningEC,
    TreeVAETrafficModel,
)

from tbsim.algos.multiagent_algos import (
    MATrafficModel,
    HierarchicalAgentAwareModel
)

from tbsim.algos.metric_algos import (
    OccupancyMetric
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
        algo = BehaviorCloning(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "l5_rasterized_gc":
        algo = BehaviorCloningGC(
            algo_config=algo_config, modality_shapes=modality_shapes
        )
    elif algo_name == "l5_rasterized_vae":
        algo = VAETrafficModel(
            algo_config=algo_config, modality_shapes=modality_shapes
        )
    elif algo_name == "l5_rasterized_discrete_vae":
        algo = DiscreteVAETrafficModel(
            algo_config=algo_config, modality_shapes=modality_shapes
        )
    elif algo_name == "l5_rasterized_tree_vae":
        algo = TreeVAETrafficModel(
            algo_config=algo_config, modality_shapes=modality_shapes
        )
    elif algo_name == "l5_rasterized_ec":
        algo = BehaviorCloningEC(
            algo_config=algo_config, modality_shapes=modality_shapes
        )
    elif algo_name == "spatial_planner":
        algo = SpatialPlanner(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "hier_agent_aware":
        algo = HierarchicalAgentAwareModel(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "occupancy":
        algo = OccupancyMetric(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "ma_rasterized":
        algo = MATrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "TransformerPred":
        algo = TransformerTrafficModel(algo_config=algo_config)
    elif algo_name == "TransformerGAN":
        algo = TransformerGANTrafficModel(algo_config=algo_config)
    elif algo_name == "gan":
        algo = GANTrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    else:
        raise NotImplementedError("{} is not a valid algorithm" % algo_name)
    return algo
