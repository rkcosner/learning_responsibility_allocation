"""A script for evaluating closed-loop simulation"""
from tbsim.algos.l5kit_algos import (
    L5DiscreteVAETrafficModel,
)

from tbsim.algos.metric_algos import (
    OccupancyMetric,
    EBMMetric
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
    def get_metrics(self, perturbations = None, **kwargs):
        # TODO: pass in perturbations through kwargs


        ckpt_path, config_path = get_checkpoint(
            ngc_job_id="2874790",
            ckpt_key="iter27000_ep0_minADE0.61",
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


class learnedEBMMetric(MetricsComposer):
    def get_metrics(self, perturbations = None, **kwargs):
        # TODO: pass in perturbations through kwargs


        ckpt_path, config_path = get_checkpoint(
            ngc_job_id="",
            ckpt_key="",
            ckpt_root_dir=self.eval_config.ckpt_root_dir
        )

        controller_cfg = get_experiment_config_from_file(config_path)
        modality_shapes = batch_utils().get_modality_shapes(controller_cfg)
        ebm_model = EBMMetric.load_from_checkpoint(
            ckpt_path,
            algo_config=controller_cfg.algo,
            modality_shapes=modality_shapes
        ).to(self.device).eval()
        return EnvMetrics.LearnedCVAENLL(metric_algo=ebm_model, perturbations=perturbations)


class OccupancyMetrics(MetricsComposer):
    def get_metrics(self, perturbations = None, **kwargs):
        # TODO: adding checkpoints

        ckpt_path, config_path = get_checkpoint(
            ngc_job_id="2878434",
            ckpt_key="iter74000_ep0_valCELoss1.63",
            ckpt_root_dir=self.eval_config.ckpt_root_dir
        )

        cfg = get_experiment_config_from_file(config_path)

        modality_shapes = batch_utils().get_modality_shapes(cfg)
        occupancy_model = OccupancyMetric.load_from_checkpoint(
            ckpt_path,
            algo_config=cfg.algo,
            modality_shapes=modality_shapes
        ).to(self.device).eval()

        # cfg = get_experiment_config_from_file("/home/yuxiaoc/repos/behavior-generation/experiments/templates/l5_occupancy.json")

        # modality_shapes = batch_utils().get_modality_shapes(cfg)
        # occupancy_model = OccupancyMetric(
        #     algo_config=cfg.algo,
        #     modality_shapes=modality_shapes
        # ).to(self.device).eval()
        return EnvMetrics.Occupancy_likelihood(metric_algo=occupancy_model, perturbations=perturbations)