import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import tbsim.utils.geometry_utils as GeoUtils

from l5kit.data import ChunkedDataset, LocalDataManager
from tbsim.external.l5_ego_dataset import EgoDatasetMixed
from tbsim.external.vectorizer import build_vectorizer
from tbsim.utils.config_utils import (
    translate_l5kit_cfg,
    get_experiment_config_from_file,
)
from l5kit.rasterization import build_rasterizer

import tbsim.utils.planning_utils as PlanUtils

os.environ["L5KIT_DATA_FOLDER"] = "/home/chenyx/repos/l5kit/prediction-dataset"






def test_sample_planner():
    config_file = "/home/chenyx/repos/behavior-generation/experiments/templates/l5_ma_rasterized_plan.json"
    pred_cfg = get_experiment_config_from_file(config_file)

    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = os.path.abspath(pred_cfg.train.dataset_path)
    dm = LocalDataManager(None)
    l5_config = translate_l5kit_cfg(pred_cfg)
    rasterizer = build_rasterizer(l5_config, dm)
    vectorizer = build_vectorizer(l5_config, dm)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_zarr = ChunkedDataset(dm.require(pred_cfg.train.dataset_valid_key)).open()
    train_dataset = EgoDatasetMixed(l5_config, train_zarr, vectorizer, rasterizer)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=2,
        num_workers=1,
    )
    tr_it = iter(train_dataloader)
    batch = next(tr_it)

    for key, obj in batch.items():
        batch[key] = obj.to(device)

    # model(batch)
    N = 10
    ego_trajectories = torch.cat((batch["target_positions"], batch["target_yaws"]), -1)
    ego_trajectories = ego_trajectories.unsqueeze(1).repeat(1, N, 1, 1)
    ego_trajectories += torch.normal(
        torch.zeros_like(ego_trajectories), torch.ones_like(ego_trajectories) *0.5
    )
    agent_trajectories = torch.cat(
        (
            batch["all_other_agents_future_positions"],
            batch["all_other_agents_future_yaws"],
        ),
        -1,
    )
    raw_types = batch["all_other_agents_types"]
    agent_extents = batch["all_other_agents_future_extents"][..., :2].max(dim=-2)[0]
    lane_mask = (batch["image"][:, -3] < 1.0).type(torch.float)
    dis_map = GeoUtils.calc_distance_map(lane_mask)
    col_loss = PlanUtils.get_collision_loss(
        ego_trajectories,
        agent_trajectories,
        batch["extent"][:, :2],
        agent_extents,
        raw_types,
    )
    lane_loss = PlanUtils.get_drivable_area_loss(
        ego_trajectories,
        batch["centroid"],
        batch["yaw"],
        batch["raster_from_world"],
        dis_map,
        batch["extent"][:, :2],
    )
    idx = PlanUtils.ego_sample_planning(
        ego_trajectories,
        agent_trajectories,
        batch["extent"][:, :2],
        agent_extents,
        raw_types,
        batch["centroid"],
        batch["yaw"],
        batch["raster_from_world"],
        dis_map,
        weights={"collision_weight":1.0,"lane_weight":1.0},
    )


if __name__ == "__main__":
    test_sample_planner()
