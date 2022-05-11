import numpy as np
import torch
import torch.nn as nn
from tbsim.models.cnn_roi_encoder import obtain_lane_flag
from tbsim.utils.loss_utils import collision_loss
from tbsim.utils.l5_utils import gen_ego_edges, gen_EC_edges
from tbsim.utils.geometry_utils import (
    VEH_VEH_collision,
    VEH_PED_collision,
    PED_VEH_collision,
    PED_PED_collision,
)


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
            if dis.nelement()>0:
                col_loss += torch.max(
                    torch.sigmoid(-dis - 4.0) * type_mask[et].unsqueeze(1), dim=2
                )[0]
    return col_loss


def get_drivable_area_loss(
    ego_trajectories, raster_from_agent, dis_map, ego_extents
):
    with torch.no_grad():
        lane_flags = obtain_lane_flag(
            dis_map,
            ego_trajectories[..., :2],
            ego_trajectories[..., 2:],
            raster_from_agent,
            torch.ones(*ego_trajectories.shape[:3]
                       ).to(ego_trajectories.device),
            ego_extents.unsqueeze(1).repeat(1, ego_trajectories.shape[1], 1),
            1,
        ).squeeze(-1)
    return lane_flags.max(dim=-1)[0]


def get_total_distance(ego_trajectories):
    # Assume format [..., T, 3]
    assert ego_trajectories.shape[-1] == 3
    diff = ego_trajectories[..., 1:, :] - ego_trajectories[..., :-1, :]
    dist = torch.norm(diff[..., :2], dim=-1)
    total_dist = torch.sum(dist, dim=-1)
    return total_dist


def ego_sample_planning(
    ego_trajectories,
    agent_trajectories,
    ego_extents,
    agent_extents,
    raw_types,
    raster_from_agent,
    dis_map,
    weights,
    log_likelihood=None,
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
    progress = get_total_distance(ego_trajectories)

    log_likelihood = 0 if log_likelihood is None else log_likelihood
    # print("coll={}, lane={}, progress={}, likelihood={}".format(
    #     torch.max(col_loss) * weights["collision_weight"],
    #     torch.max(lane_loss) * weights["lane_weight"],
    #     torch.max(progress) * weights["progress_weight"],
    #     torch.topk(log_likelihood.reshape(-1), k=10)[0] * weights["likelihood_weight"]
    # ))

    total_score = (
            + weights["likelihood_weight"] * log_likelihood
            + weights["progress_weight"] * progress
            - weights["collision_weight"] * col_loss
            - weights["lane_weight"] * lane_loss
    )

    return torch.argmax(total_score, dim=1)
