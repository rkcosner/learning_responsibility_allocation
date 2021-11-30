import torch

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset, filter_agents_by_frames
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer

from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                               DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric)
from l5kit.cle.validators import RangeValidator, ValidationCountingAggregator
from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from l5kit.visualization.visualizer.zarr_utils import simulation_out_to_visualizer_scene
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show
from l5kit.data import MapAPI


def main():
    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = "/home/danfeix/workspace/lfs/lyft/lyft_prediction"
    dm = LocalDataManager(None)
    # get config
    cfg = load_config_data("/home/danfeix/workspace/l5kit/examples/simulation/config.yaml")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    simulation_model_path = "/home/danfeix/workspace/l5kit/checkpoints/sim_pt/simulation_model_20210416_5steps.pt"
    simulation_model = torch.load(simulation_model_path).to(device)
    simulation_model = simulation_model.eval()

    ego_model_path = "/home/danfeix/workspace/l5kit/checkpoints/sim_pt/planning_model_20210421_5steps.pt"
    ego_model = torch.load(ego_model_path).to(device)
    ego_model = ego_model.eval()

    torch.set_grad_enabled(False)

    eval_cfg = cfg["val_data_loader"]
    eval_cfg["key"] = "scenes/validate.zarr"
    rasterizer = build_rasterizer(cfg, dm)
    eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
    eval_dataset = EgoDataset(cfg, eval_zarr, rasterizer)
    print(eval_dataset)

    scenes_to_unroll = [30, 31, 32]

    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=True,
                               distance_th_far=30, distance_th_close=15, num_simulation_steps=50,
                               start_frame_index=0, show_info=True)

    sim_loop = ClosedLoopSimulator(sim_cfg, eval_dataset, device, model_ego=ego_model, model_agents=simulation_model)
    sim_outs = sim_loop.unroll(scenes_to_unroll)

if __name__ == "__main__":
    main()