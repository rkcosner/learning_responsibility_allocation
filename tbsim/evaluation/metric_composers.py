"""A script for evaluating closed-loop simulation"""
from tbsim.algos.l5kit_algos import (
    L5DiscreteVAETrafficModel,
)

from tbsim.algos.metric_algos import (
    OccupancyMetric
)

import tbsim.envs.env_metrics as EnvMetrics

from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.configs.base import ExperimentConfig

from tbsim.utils.experiment_utils import get_checkpoint

try:
    from Pplan.spline_planner import SplinePlanner
    from Pplan.trajectory_tree import TrajTree
except ImportError:
    print("Cannot import Pplan")


class MetricsComposer(object):
    def __init__(self, eval_config, device, ckpt_root_dir="checkpoints/"):
        self.device = device
        self.ckpt_root_dir = ckpt_root_dir
        self.eval_config = eval_config
        self._exp_config = None

    def get_modality_shapes(self, exp_cfg: ExperimentConfig):
        return batch_utils().get_modality_shapes(exp_cfg)

    def get_metrics(self):
        raise NotImplementedError


class CVAEMetrics(MetricsComposer):
    def get_metrics(self, **kwargs):
        # TODO: pass in perturbations through kwargs
        perturbations = None

        ckpt_path, config_path = get_checkpoint(
            ngc_job_id="2873777",  # aaplan_dynUnicycle_yrl0.1_roiFalse_gcTrue_rlayerlayer2_rlFalse
            ckpt_key="iter2000",
            # ngc_job_id=self.eval_config.ckpt.cvae_metric.ngc_job_id,
            # ckpt_key=self.eval_config.ckpt.cvae_metric.ckpt_key,
            ckpt_root_dir=self.eval_config.ckpt_root_dir
        )

        controller_cfg = get_experiment_config_from_file(config_path)
        modality_shapes = batch_utils().get_modality_shapes(controller_cfg)
        CVAE_model = L5DiscreteVAETrafficModel.load_from_checkpoint(
            ckpt_path,
            algo_config=controller_cfg.algo,
            modality_shapes=modality_shapes
        ).to(self.device).eval()
        return EnvMetrics.LearnedCVAENLL(metric_algo=CVAE_model, perturbations=perturbations)


class OccupancyMetrics(MetricsComposer):
    def get_metrics(self, **kwargs):
        pass

