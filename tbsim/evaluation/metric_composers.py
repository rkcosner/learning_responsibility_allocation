from tbsim.algos.l5kit_algos import (
    L5DiscreteVAETrafficModel,
)

from tbsim.algos.metric_algos import (
    OccupancyMetric,
)

import tbsim.envs.env_metrics as EnvMetrics

from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.configs.base import ExperimentConfig

from tbsim.utils.experiment_utils import get_checkpoint

try:
    from Pplan.Sampling.spline_planner import SplinePlanner
    from Pplan.Sampling.trajectory_tree import TrajTree
except ImportError:
    print("Cannot import Pplan")


class MetricsComposer(object):
    """Wrapper for building learned metrics from trained checkpoints."""
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
    def get_metrics(self, perturbations=None, rolling=False, env="l5kit", **kwargs):
        # TODO (@yuxiao): use ckpt.cvae_metric field in self.eval_config

        if env=="nusc":
            ckpt_path, config_path = get_checkpoint(
                ngc_job_id="1000002",
                ckpt_key="iter13000_ep0_minADE0.97",
                ckpt_root_dir=self.eval_config.ckpt_root_dir
            )
        elif env=="l5kit":
            ckpt_path, config_path = get_checkpoint(
                ngc_job_id="1000001",
                ckpt_key="iter27000_ep0_minADE0.60",
                ckpt_root_dir=self.eval_config.ckpt_root_dir
            )

        controller_cfg = get_experiment_config_from_file(config_path)
        modality_shapes = batch_utils().get_modality_shapes(controller_cfg)
        CVAE_model = L5DiscreteVAETrafficModel.load_from_checkpoint(
            ckpt_path,
            algo_config=controller_cfg.algo,
            modality_shapes=modality_shapes
        ).to(self.device).eval()
        if not rolling:
            return EnvMetrics.LearnedCVAENLL(metric_algo=CVAE_model, perturbations=perturbations)
        else:
            if "rolling_horizon" in kwargs:
                rolling_horizon = kwargs["rolling_horizon"]
            else:
                rolling_horizon = None
            return EnvMetrics.LearnedCVAENLLRolling(metric_algo=CVAE_model, rolling_horizon=rolling_horizon, perturbations=perturbations)


class OccupancyMetrics(MetricsComposer):
    def get_metrics(self, perturbations = None, rolling=False, env="l5kit", **kwargs):
        if env=="nusc":
            ckpt_path, config_path = get_checkpoint(
                ngc_job_id="0000003",
                ckpt_key="iter51000_ep1_posErr0.61",
                ckpt_root_dir=self.eval_config.ckpt_root_dir
            )
        elif env=="l5kit":
            ckpt_path, config_path = get_checkpoint(
                ngc_job_id="0000001",
                ckpt_key="iter62000_ep1_posErr0.71",
                ckpt_root_dir=self.eval_config.ckpt_root_dir
            )

        cfg = get_experiment_config_from_file(config_path)

        modality_shapes = batch_utils().get_modality_shapes(cfg)
        occupancy_model = OccupancyMetric.load_from_checkpoint(
            ckpt_path,
            algo_config=cfg.algo,
            modality_shapes=modality_shapes
        ).to(self.device).eval()

        if not rolling:
            return EnvMetrics.Occupancy_likelihood(metric_algo=occupancy_model, perturbations=perturbations)
        else:
            if "rolling_horizon" in kwargs:
                rolling_horizon = kwargs["rolling_horizon"]
            else:
                rolling_horizon = None
            return EnvMetrics.Occupancy_rolling(metric_algo=occupancy_model, rolling_horizon=rolling_horizon, perturbations=perturbations)

        