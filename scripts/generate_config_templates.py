"""
Helpful script to generate example config files for each algorithm. These should be re-generated
when new config options are added, or when default settings in the config classes are modified.
"""
import os

import tbsim
from tbsim.configs import (
    ExperimentConfig,
    L5KitEnvConfig,
    L5KitTrainConfig,
    L5RasterizedPlanningConfig
)


def main():
    # store template config jsons in this directory
    target_dir = os.path.join(tbsim.__path__[0], "../experiments/templates/")

    # Vanilla Rasterized Planner on L5Kit
    cfg = ExperimentConfig(
        train_config=L5KitTrainConfig(),
        env_config=L5KitEnvConfig(),
        algo_config=L5RasterizedPlanningConfig()
    )
    cfg.dump(filename=os.path.join(target_dir, "l5_raster_plan.json"))

    # Vectorized Transformer on L5Kit
    # TODO

if __name__ == '__main__':
    main()