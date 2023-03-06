"""A global registry for looking up named experiment configs"""
from tbsim.configs.base import ExperimentConfig

from tbsim.configs.l5kit_config import (
    L5KitTrainConfig,
    L5KitMixedEnvConfig,
    L5KitMixedSemanticMapEnvConfig,
)

from tbsim.configs.nusc_config import (
    NuscTrainConfig,
    NuscEnvConfig
)

from tbsim.configs.algo_config import (
    ResponsibilityConfig,
)


EXP_CONFIG_REGISTRY = dict()


EXP_CONFIG_REGISTRY["nusc_resp"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=ResponsibilityConfig(),
    registered_name="nusc_resp"
)



def get_registered_experiment_config(registered_name):
    registered_name = backward_compatible_translate(registered_name)

    if registered_name not in EXP_CONFIG_REGISTRY.keys():
        raise KeyError(
            "'{}' is not a registered experiment config please choose from {}".format(
                registered_name, list(EXP_CONFIG_REGISTRY.keys())
            )
        )
    return EXP_CONFIG_REGISTRY[registered_name].clone()


def backward_compatible_translate(registered_name):
    """Try to translate registered name to maintain backward compatibility."""
    translation = {
        "l5_mixed_plan": "l5_bc",
        "l5_mixed_gc": "l5_bc_gc",
        "l5_ma_rasterized_plan": "l5_agent_predictor",
        "l5_gan_plan": "l5_gan",
        "l5_mixed_ec_plan": "l5_bc_ec",
        "l5_mixed_vae_plan": "l5_vae",
        "l5_mixed_discrete_vae_plan": "l5_discrete_vae",
        "l5_mixed_tree_vae_plan": "l5_tree_vae",
        "nusc_rasterized_plan": "nusc_bc",
        "nusc_mixed_gc": "nusc_bc_gc",
        "nusc_ma_rasterized_plan": "nusc_agent_predictor",
        "nusc_gan_plan": "nusc_gan",
        "nusc_vae_plan": "nusc_vae",
        "nusc_mixed_tree_vae_plan": "nusc_tree_vae",
    }
    if registered_name in translation:
        registered_name = translation[registered_name]
    return registered_name