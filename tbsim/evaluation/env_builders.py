import argparse
from copy import deepcopy
import numpy as np
import json
import random
import yaml
from collections import defaultdict

import os

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.rasterization import build_rasterizer
from avdata import AgentType, UnifiedDataset

from tbsim.l5kit.vectorizer import build_vectorizer

from tbsim.envs.env_l5kit import EnvL5KitSimulation
from tbsim.envs.env_avdata import EnvUnifiedSimulation
from tbsim.utils.config_utils import translate_l5kit_cfg, translate_avdata_cfg
import tbsim.envs.env_metrics as EnvMetrics


from tbsim.l5kit.l5_ego_dataset import EgoDatasetMixed
from tbsim.utils.vis_utils import build_visualization_rasterizer_l5kit

try:
    from Pplan.spline_planner import SplinePlanner
    from Pplan.trajectory_tree import TrajTree
except ImportError:
    print("Cannot import Pplan")


def create_env_l5kit(
        exp_cfg,
        eval_cfg,
        device,
        skimp_rollout=False,
        compute_metrics=True,
        seed=1
):
    os.environ["L5KIT_DATA_FOLDER"] = eval_cfg.dataset_path

    dm = LocalDataManager(None)
    l5_config = translate_l5kit_cfg(exp_cfg)
    rasterizer = build_rasterizer(l5_config, dm)
    vectorizer = build_vectorizer(l5_config, dm)
    eval_zarr = ChunkedDataset(dm.require(exp_cfg.train.dataset_valid_key)).open()

    env_dataset = EgoDatasetMixed(l5_config, eval_zarr, vectorizer, rasterizer)

    l5_config = deepcopy(l5_config)
    l5_config["raster_params"]["raster_size"] = (500, 500)
    l5_config["raster_params"]["pixel_size"] = (0.2, 0.2)
    l5_config["raster_params"]["ego_center"] = (0.5, 0.5)
    render_rasterizer = build_visualization_rasterizer_l5kit(l5_config, LocalDataManager(None))
    exp_cfg.env.simulation.num_simulation_steps = eval_cfg.num_simulation_steps
    exp_cfg.env.simulation.distance_th_far = 1e+5  # keep controlling everything
    exp_cfg.env.simulation.disable_new_agents = True
    exp_cfg.env.generate_agent_obs = True
    exp_cfg.env.simulation.distance_th_close = 30  # control everything within this bound
    exp_cfg.env.rasterizer.filter_agents_threshold = 0.8  # control everything that's above this confidence threshold

    metrics = dict()
    if compute_metrics:
        gridinfo = {"offset": np.zeros(2), "step": 2.0*np.ones(2)}
        # cvae_metrics = CVAEMetrics(eval_config=eval_cfg, device=device, ckpt_root_dir=eval_cfg.ckpt_root_dir)
        failure_metric = EnvMetrics.CriticalFailure()
        metrics = dict(
            # all_off_road_rate=EnvMetrics.OffRoadRate(),
            # all_collision_rate=EnvMetrics.CollisionRate(),
            # all_occupancy = EnvMetrics.Occupancydistr(gridinfo,sigma=2.0)
            # ego_cvae_metrics=cvae_metrics.get_metrics(),
            ego_occupancy_diversity=EnvMetrics.OccupancyDiversity(gridinfo, sigma=2.0),
            all_occupancy_coverage=EnvMetrics.OccupancyCoverage(gridinfo,failure_metric, sigma=2.0)
            # all_ebm_score=EnvMetrics.LearnedMetric(metric_algo=metric_algo, perturbations=perturbations),
        )

    env = EnvL5KitSimulation(
        exp_cfg.env,
        dataset=env_dataset,
        seed=seed,
        num_scenes=eval_cfg.num_scenes_per_batch,
        prediction_only=False,
        renderer=render_rasterizer,
        metrics=metrics,
        skimp_rollout=skimp_rollout,
    )

    return env


def create_env_nusc(
        exp_cfg,
        eval_cfg,
        device,
        skimp_rollout=False,
        compute_metrics=True,
        seed=1
):
    exp_cfg.unlock()
    exp_cfg.train.dataset_path = eval_cfg.dataset_path
    exp_cfg.env.simulation.num_simulation_steps = eval_cfg.num_simulation_steps
    exp_cfg.env.simulation.start_frame_index = exp_cfg.algo.history_num_frames + 1
    exp_cfg.lock()

    data_cfg = translate_avdata_cfg(exp_cfg)

    future_sec = data_cfg.future_num_frames * data_cfg.step_time
    history_sec = data_cfg.history_num_frames * data_cfg.step_time
    neighbor_distance = data_cfg.max_agents_distance

    kwargs = dict(
        desired_data=["nusc-val"],
        future_sec=(future_sec, future_sec),
        history_sec=(history_sec, history_sec),
        data_dirs={
            "nusc": data_cfg.dataset_path,
            "nusc_mini": data_cfg.dataset_path,
        },
        only_types=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: neighbor_distance),
        incl_map=True,
        map_params={
            "px_per_m": int(1 / data_cfg.pixel_size),
            "map_size_px": data_cfg.raster_size,
            "return_rgb": False,
            "offset_frac_xy": data_cfg.raster_center
        },
        num_workers=os.cpu_count(),
        desired_dt=data_cfg.step_time
    )

    env_dataset = UnifiedDataset(**kwargs)

    metrics = None
    if compute_metrics:
        metrics = dict(
            all_off_road_rate=EnvMetrics.OffRoadRate(),
            all_collision_rate=EnvMetrics.CollisionRate(),
            all_coverage=EnvMetrics.OccupancyCoverage(
                gridinfo={"offset": np.zeros(2), "step": 2.0*np.ones(2)},
                failure_metric=EnvMetrics.CriticalFailure(num_offroad_frames=2)
            ),
            all_failure=EnvMetrics.CriticalFailure(num_offroad_frames=2)
        )

    env = EnvUnifiedSimulation(
        exp_cfg.env,
        dataset=env_dataset,
        seed=seed,
        num_scenes=eval_cfg.num_scenes_per_batch,
        prediction_only=False,
        metrics=metrics
    )

    return env