import numpy as np
import torch
import torch.nn as nn
from tbsim.models.cnn_roi_encoder import obtain_lane_flag
from tbsim.utils.loss_utils import collision_loss
from tbsim.utils.l5_utils import gen_ego_edges
from tbsim.utils.geometry_utils import (
    VEH_VEH_collision,
    VEH_PED_collision,
    PED_VEH_collision,
    PED_PED_collision,
)
import pdb


def get_collision_loss(
    ego_trajectories,
    agent_trajectories,
    ego_extents,
    agent_extents,
    raw_types,
    col_funcs=None,
):
    with torch.no_grad():
        ego_edges, type_mask = gen_ego_edges(
            ego_trajectories, agent_trajectories, ego_extents, agent_extents, raw_types
        )
        if col_funcs is None:
            col_funcs = {
                "VV": VEH_VEH_collision,
                "VP": VEH_PED_collision,
            }
        B, N, T = ego_trajectories.shape[:3]
        col_loss = torch.zeros([B, N]).to(ego_trajectories.device)
        for et, func in col_funcs.items():
            dis = func(
                ego_edges[..., 0:3],
                ego_edges[..., 3:6],
                ego_edges[..., 6:8],
                ego_edges[..., 8:],
            ).min(dim=-1)[0]
            col_loss += torch.mean(
                torch.sigmoid(-dis - 4.0) * type_mask[et].unsqueeze(1), dim=2
            )
    return col_loss


def get_drivable_area_loss(
    ego_trajectories, centroid, scene_yaw, raster_from_agent, dis_map, ego_extents
):
    with torch.no_grad():
        lane_flags = obtain_lane_flag(
            dis_map,
            ego_trajectories[..., :2],
            ego_trajectories[..., 2:],
            raster_from_agent,
            torch.ones(*ego_trajectories.shape[:3]).to(ego_trajectories.device),
            ego_extents.unsqueeze(1).repeat(1, ego_trajectories.shape[1], 1),
            1,
        ).squeeze(-1)
    return lane_flags.mean(dim=-1)


def ego_sample_planning(
    ego_trajectories,
    agent_trajectories,
    ego_extents,
    agent_extents,
    raw_types,
    raster_from_agent,
    dis_map,
    weights,
    likelihood=None,
    col_funcs=None,
):
    col_loss = get_collision_loss(
        ego_trajectories,
        agent_trajectories,
        ego_extents,
        agent_extents,
        raw_types,
        col_funcs,
    )
    lane_loss = get_drivable_area_loss(
        ego_trajectories, raster_from_agent, dis_map, ego_extents
    )
    if likelihood is None:
        total_score = (
            -weights["collision_weight"] * col_loss - weights["lane_weight"] * lane_loss
        )
    else:
        total_score = (
            likelihood
            - weights.collision_weight * col_loss
            - weights.lane_weight * lane_loss
        )
    return torch.argmax(total_score, dim=1)
