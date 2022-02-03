import torch
import torch.nn.functional as F

import tbsim.dynamics as dynamics
import tbsim.utils.tensor_utils as TensorUtils


def get_agent_masks(raw_type):
    """
    PERCEPTION_LABELS = [
    "PERCEPTION_LABEL_NOT_SET",
    "PERCEPTION_LABEL_UNKNOWN",
    "PERCEPTION_LABEL_DONTCARE",
    "PERCEPTION_LABEL_CAR",
    "PERCEPTION_LABEL_VAN",
    "PERCEPTION_LABEL_TRAM",
    "PERCEPTION_LABEL_BUS",
    "PERCEPTION_LABEL_TRUCK",
    "PERCEPTION_LABEL_EMERGENCY_VEHICLE",
    "PERCEPTION_LABEL_OTHER_VEHICLE",
    "PERCEPTION_LABEL_BICYCLE",
    "PERCEPTION_LABEL_MOTORCYCLE",
    "PERCEPTION_LABEL_CYCLIST",
    "PERCEPTION_LABEL_MOTORCYCLIST",
    "PERCEPTION_LABEL_PEDESTRIAN",
    "PERCEPTION_LABEL_ANIMAL",
    "AVRESEARCH_LABEL_DONTCARE",
    ]
    """
    veh_mask = (raw_type >= 3) & (raw_type <= 13)
    ped_mask = (raw_type == 14) | (raw_type == 15)
    # veh_mask = veh_mask | ped_mask
    # ped_mask = ped_mask * 0
    return veh_mask, ped_mask


def get_dynamics_types(veh_mask, ped_mask):
    dyn_type = torch.zeros_like(veh_mask)
    dyn_type += dynamics.DynType.UNICYCLE * veh_mask
    dyn_type += dynamics.DynType.DI * ped_mask
    return dyn_type


def raw_to_features(batch_raw):
    """ map raw src into features of dim 21 """
    raw_type = batch_raw["raw_types"]
    pos = batch_raw["history_positions"]
    vel = batch_raw["history_velocities"]
    yaw = batch_raw["history_yaws"]
    mask = batch_raw["history_availabilities"]

    veh_mask, ped_mask = get_agent_masks(raw_type)

    # all vehicles, cyclists, and motorcyclists
    feature_veh = torch.cat((pos, vel, torch.cos(yaw), torch.sin(yaw)), dim=-1)

    # pedestrians and animals
    ped_feature = torch.cat(
        (pos, vel, vel * torch.sin(yaw), vel * torch.cos(yaw)), dim=-1
    )

    feature = feature_veh * veh_mask.view(
        [*raw_type.shape, 1, 1]
    ) + ped_feature * ped_mask.view([*raw_type.shape, 1, 1])

    type_embedding = F.one_hot(raw_type, 16)

    feature = torch.cat(
        (feature, type_embedding.unsqueeze(-2).repeat(1, 1, feature.size(2), 1)),
        dim=-1,
    )
    feature = feature * mask.unsqueeze(-1)

    return feature


def raw_to_states(batch_raw):
    raw_type = batch_raw["raw_types"]
    pos = batch_raw["history_positions"]
    vel = batch_raw["history_velocities"]
    yaw = batch_raw["history_yaws"]
    avail_mask = batch_raw["history_availabilities"]

    veh_mask, ped_mask = get_agent_masks(raw_type)  # [B, (A)]

    # all vehicles, cyclists, and motorcyclists
    state_veh = torch.cat((pos, vel, yaw), dim=-1)  # [B, (A), T, S]
    # pedestrians and animals
    state_ped = torch.cat((pos, vel * torch.cos(yaw), vel * torch.sin(yaw)), dim=-1)  # [B, (A), T, S]

    state = state_veh * veh_mask.view(
        [*raw_type.shape, 1, 1]
    ) + state_ped * ped_mask.view([*raw_type.shape, 1, 1])  # [B, (A), T, S]

    # Get the current state of the agents
    num = torch.arange(0, avail_mask.shape[-1]).view(1, 1, -1).to(avail_mask.device)
    nummask = num * avail_mask
    last_idx, _ = torch.max(nummask, dim=2)
    curr_state = torch.gather(
        state, 2, last_idx[..., None, None].repeat(1, 1, 1, 4)
    )
    return state, curr_state


def batch_to_raw_ego(data_batch, step_time):
    batch_size = data_batch["history_positions"].shape[0]
    raw_type = torch.ones(batch_size).type(torch.int64).to(data_batch["history_positions"].device)  # [B, T]
    raw_type = raw_type * 3  # index for type PERCEPTION_LABEL_CAR

    src_pos = torch.flip(data_batch["history_positions"], dims=[-2])
    src_yaw = torch.flip(data_batch["history_yaws"], dims=[-2])
    src_mask = torch.flip(data_batch["history_availabilities"], dims=[-1]).bool()

    src_vel = dynamics.Unicycle.calculate_vel(pos=src_pos, yaw=src_yaw, dt=step_time, mask=src_mask)
    src_vel[:, -1] = data_batch["curr_speed"].unsqueeze(-1)

    raw = {
        "history_positions": src_pos,
        "history_velocities": src_vel,
        "history_yaws": src_yaw,
        "raw_types": raw_type,
        "history_availabilities": src_mask,
        "extents": data_batch["extents"]
    }

    raw = TensorUtils.unsqueeze(raw, dim=1)  # Add the agent dimension
    return raw


def batch_to_raw_all_agents(data_batch, step_time):
    raw_type = torch.cat(
        (data_batch["type"].unsqueeze(1), data_batch["all_other_agents_types"]),
        dim=1,
    ).type(torch.int64)

    src_pos = torch.cat(
        (
            data_batch["history_positions"].unsqueeze(1),
            data_batch["all_other_agents_history_positions"],
        ),
        dim=1,
    )
    # history position and yaw need to be flipped so that they go from past to recent
    src_pos = torch.flip(src_pos, dims=[-2])
    src_yaw = torch.cat(
        (
            data_batch["history_yaws"].unsqueeze(1),
            data_batch["all_other_agents_history_yaws"],
        ),
        dim=1,
    )
    src_yaw = torch.flip(src_yaw, dims=[-2])
    src_mask = torch.cat(
        (
            data_batch["history_availabilities"].unsqueeze(1),
            data_batch["all_other_agents_history_availability"],
        ),
        dim=1,
    ).bool()

    src_mask = torch.flip(src_mask, dims=[-1])

    extents = torch.cat(
        (
            data_batch["extent"][..., :2].unsqueeze(1),
            torch.max(data_batch["all_other_agents_history_extents"], dim=-2)[0],
        ),
        dim=1,
    )

    # estimate velocity
    src_vel = dynamics.Unicycle.calculate_vel(
        src_pos, src_yaw, step_time, src_mask
    )
    src_vel[:, 0, -1] = data_batch["curr_speed"].unsqueeze(-1)

    return {
        "history_positions": src_pos,
        "history_velocities": src_vel,
        "history_yaws": src_yaw,
        "raw_types": raw_type,
        "history_availabilities": src_mask,
        "extents": extents
    }


def batch_to_target_all_agents(data_batch):
    pos = torch.cat(
        (
            data_batch["target_positions"].unsqueeze(1),
            data_batch["all_other_agents_future_positions"],
        ),
        dim=1,
    )
    yaw = torch.cat(
        (
            data_batch["target_yaws"].unsqueeze(1),
            data_batch["all_other_agents_future_yaws"],
        ),
        dim=1,
    )
    avails = torch.cat(
        (
            data_batch["target_availabilities"].unsqueeze(1),
            data_batch["all_other_agents_future_availability"],
        ),
        dim=1,
    )

    extents = torch.cat(
        (
            data_batch["extent"][..., :2].unsqueeze(1),
            torch.max(data_batch["all_other_agents_history_extents"], dim=-2)[0],
        ),
        dim=1,
    )

    return {
        "target_positions": pos,
        "target_yaws": yaw,
        "target_availabilities": avails,
        "extents": extents
    }


def generate_edges(
        raw_type,
        extents,
        pos_pred,
        yaw_pred,
):
    veh_mask = (raw_type >= 3) & (raw_type <= 13)
    ped_mask = (raw_type == 14) | (raw_type == 15)

    agent_mask = veh_mask | ped_mask
    edge_types = ["VV", "VP", "PV", "PP"]
    edges = {et: list() for et in edge_types}
    for i in range(agent_mask.shape[0]):
        agent_idx = torch.where(agent_mask[i] != 0)[0]
        edge_idx = torch.combinations(agent_idx, r=2)
        VV_idx = torch.where(
            veh_mask[i, edge_idx[:, 0]] & veh_mask[i, edge_idx[:, 1]]
        )[0]
        VP_idx = torch.where(
            veh_mask[i, edge_idx[:, 0]] & ped_mask[i, edge_idx[:, 1]]
        )[0]
        PV_idx = torch.where(
            ped_mask[i, edge_idx[:, 0]] & veh_mask[i, edge_idx[:, 1]]
        )[0]
        PP_idx = torch.where(
            ped_mask[i, edge_idx[:, 0]] & ped_mask[i, edge_idx[:, 1]]
        )[0]
        if pos_pred.ndim == 4:
            edges_of_all_types = torch.cat(
                (
                    pos_pred[i, edge_idx[:, 0], :],
                    yaw_pred[i, edge_idx[:, 0], :],
                    pos_pred[i, edge_idx[:, 1], :],
                    yaw_pred[i, edge_idx[:, 1], :],
                    extents[i, edge_idx[:, 0]]
                        .unsqueeze(-2)
                        .repeat(1, pos_pred.size(-2), 1),
                    extents[i, edge_idx[:, 1]]
                        .unsqueeze(-2)
                        .repeat(1, pos_pred.size(-2), 1),
                ),
                dim=-1,
            )
            edges["VV"].append(edges_of_all_types[VV_idx])
            edges["VP"].append(edges_of_all_types[VP_idx])
            edges["PV"].append(edges_of_all_types[PV_idx])
            edges["PP"].append(edges_of_all_types[PP_idx])
        elif pos_pred.ndim == 5:

            edges_of_all_types = torch.cat(
                (
                    pos_pred[i, :, edge_idx[:, 0], :],
                    yaw_pred[i, :, edge_idx[:, 0], :],
                    pos_pred[i, :, edge_idx[:, 1], :],
                    yaw_pred[i, :, edge_idx[:, 1], :],
                    extents[i, None, edge_idx[:, 0], None, :].repeat(
                        pos_pred.size(1), 1, pos_pred.size(-2), 1
                    ),
                    extents[i, None, edge_idx[:, 1], None, :].repeat(
                        pos_pred.size(1), 1, pos_pred.size(-2), 1
                    ),
                ),
                dim=-1,
            )
            edges["VV"].append(edges_of_all_types[:, VV_idx])
            edges["VP"].append(edges_of_all_types[:, VP_idx])
            edges["PV"].append(edges_of_all_types[:, PV_idx])
            edges["PP"].append(edges_of_all_types[:, PP_idx])
    if pos_pred.ndim == 4:
        for et, v in edges.items():
            edges[et] = torch.cat(v, dim=0)
    elif pos_pred.ndim == 5:
        for et, v in edges.items():
            edges[et] = torch.cat(v, dim=1)
    return edges


def get_edges_from_batch(data_batch, ego_predictions):
    targets_all = batch_to_target_all_agents(data_batch)
    raw_type = torch.cat(
        (data_batch["type"].unsqueeze(1), data_batch["all_other_agents_types"]),
        dim=1,
    ).type(torch.int64)

    # Use predicted ego position to compute future box edges
    targets_all["target_positions"] [:, 0, :, :] = ego_predictions["positions"]
    targets_all["target_yaws"][:, 0, :, :] = ego_predictions["yaws"]

    pred_edges = generate_edges(
        raw_type, targets_all["extents"],
        pos_pred=targets_all["target_positions"],
        yaw_pred=targets_all["target_yaws"]
    )
    return pred_edges

