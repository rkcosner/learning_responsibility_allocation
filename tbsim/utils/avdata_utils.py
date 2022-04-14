import torch

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.geometry_utils import transform_points_tensor
from tbsim.configs.base import ExperimentConfig


def avdata2posyawspeed(state):
    assert state.shape[-1] == 8  # x, y, vx, vy, ax, ay, sin(heading), cos(heading)
    pos = state[..., :2]
    yaw = torch.atan2(state[..., [-2]], state[..., [-1]])
    speed = torch.norm(state[..., 2:4], dim=-1)
    return pos, yaw, speed


def rasterize_agents(
        maps: torch.Tensor,
        agent_hist_pos: torch.Tensor,
        agent_hist_yaw: torch.Tensor,
        raster_from_agent: torch.Tensor,
        map_res: torch.Tensor,
) -> torch.Tensor:
    """Paint agent histories onto an agent-centric map image"""
    b, a, t, _ = agent_hist_pos.shape
    _, _, h, w = maps.shape
    maps = maps.clone()

    agent_hist_pos = agent_hist_pos.reshape(b, a * t, 2)
    raster_hist_pos = transform_points_tensor(agent_hist_pos, raster_from_agent)
    raster_hist_pos = raster_hist_pos.reshape(b, a, t, 2).permute(0, 2, 1, 3)  # [B, T, A, 2]

    raster_hist_pos_flat = torch.round(raster_hist_pos[..., 1] * w + raster_hist_pos[..., 0]).long()  # [B, T, A]
    raster_hist_pos_flat[raster_hist_pos_flat < 0] = 0  # NaN will be converted to negative indices. Set it to 0 for now. Will correct below

    hist_image = torch.zeros(b, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, T, H * W]

    hist_image.scatter_(dim=2, index=raster_hist_pos_flat[:, :, [0]], src=torch.ones_like(hist_image))  # mark ego with 1.
    hist_image.scatter_(dim=2, index=raster_hist_pos_flat[:, :, 1:], src=torch.ones_like(hist_image) * -1)  # mark other agents with -1
    hist_image[:, :, 0] = 0  # correct the 0th index caused by NaNs

    hist_image = hist_image.reshape(b, t, h, w)

    maps = torch.cat((hist_image, maps), dim=1)  # treat time as extra channels
    return maps


def parse_avdata_batch(batch: dict):
    fut_pos, fut_yaw, _ = avdata2posyawspeed(batch["agent_fut"])
    hist_pos, hist_yaw, hist_speed = avdata2posyawspeed(batch["agent_hist"])
    curr_speed = hist_speed[..., -1]
    curr_state = batch["curr_agent_state"]

    neigh_hist_pos, neigh_hist_yaw, _ = avdata2posyawspeed(batch["neigh_hist"])

    # map-related
    map_res = batch["maps_resolution"][0]
    h, w = batch["maps"].shape[-2:]
    raster_from_agent = torch.Tensor(
        [[map_res, 0, 0.5 * h], [0, map_res, 0.5 * w], [0, 0, 1]]
    ).to(curr_state.device)
    raster_from_agent = TensorUtils.unsqueeze_expand_at(raster_from_agent, size=batch["maps"].shape[0], dim=0)

    agent_hist_pos = torch.cat((hist_pos[:, None], neigh_hist_pos), dim=1)
    agent_hist_yaw = torch.cat((hist_yaw[:, None], neigh_hist_yaw), dim=1)
    maps = rasterize_agents(batch["maps"], agent_hist_pos, agent_hist_yaw, raster_from_agent, map_res)

    d = dict(
        image=maps,
        target_positions=fut_pos,
        target_yaws=fut_yaw,
        target_availabilities=torch.ones_like(fut_pos[..., 0]),
        history_positions=hist_pos,
        history_yaws=hist_yaw,
        history_availabilities=torch.ones_like(hist_pos[..., 0]),
        all_other_agents_history_positions=neigh_hist_pos,
        all_other_agents_history_yaws=neigh_hist_yaw,
        curr_speed=curr_speed,
        centroid=curr_state[..., :2],
        yaw=curr_state[..., -1],
        raster_from_agent=raster_from_agent
    )
    batch = dict(batch)
    batch.update(d)
    batch.pop("agent_name")
    return batch


def maybe_parse_batch(batch):
    """Parse batch to the expected format"""
    if "agent_fut" in batch:  # avdata
        return parse_avdata_batch(batch)
    else:
        return batch


def get_modality_shapes(cfg: ExperimentConfig):
    num_channels = (cfg.algo.history_num_frames + 1) + 7
    h = cfg.env.rasterizer.raster_size
    return dict(image=(num_channels, h, h))