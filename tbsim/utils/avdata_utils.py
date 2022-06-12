import torch
import numpy as np

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.geometry_utils import transform_points_tensor
from tbsim.configs.base import ExperimentConfig


def avdata2posyawspeed(state, nan_to_zero=True):
    """Converts avdata's state format to pos, yaw, and speed. Set Nans to 0s"""
    
    if state.shape[-1] == 7:  # x, y, vx, vy, ax, ay, sin(heading), cos(heading)
        state = torch.cat((state[...,:6],torch.sin(state[...,6:7]),torch.cos(state[...,6:7])),-1)
    else:
        state.shape[-1] == 8
    pos = state[..., :2]
    yaw = torch.atan2(state[..., [-2]], state[..., [-1]])
    speed = torch.norm(state[..., 2:4], dim=-1)
    mask = torch.bitwise_not(torch.max(torch.isnan(state), dim=-1)[0])
    if nan_to_zero:
        pos[torch.bitwise_not(mask)] = 0.
        yaw[torch.bitwise_not(mask)] = 0.
        speed[torch.bitwise_not(mask)] = 0.
    return pos, yaw, speed, mask

def rasterize_agents_scene(
        maps: torch.Tensor,
        agent_hist_pos: torch.Tensor,
        agent_hist_yaw: torch.Tensor,
        agent_mask: torch.Tensor,
        raster_from_agent: torch.Tensor,
        map_res: torch.Tensor,
) -> torch.Tensor:
    """Paint agent histories onto an agent-centric map image"""
    
    b, a, t, _ = agent_hist_pos.shape
    _, _, _, h, w = maps.shape
    maps = maps.clone()
    agent_hist_pos = TensorUtils.unsqueeze_expand_at(agent_hist_pos,a,1)
    raster_hist_pos = transform_points_tensor(agent_hist_pos.reshape(b*a,-1,2), raster_from_agent.reshape(b*a,3,3)).reshape(b,a,a,t,2)
    raster_hist_pos[~agent_mask[:,None].repeat_interleave(a,1)] = 0.0  # Set invalid positions to 0.0 Will correct below
    
    raster_hist_pos[..., 0].clip_(0, (w - 1))
    raster_hist_pos[..., 1].clip_(0, (h - 1))
    raster_hist_pos = torch.round(raster_hist_pos).long()  # round pixels [B, A, A, T, 2]
    raster_hist_pos = raster_hist_pos.transpose(2,3)
    raster_hist_pos_flat = raster_hist_pos[..., 1] * w + raster_hist_pos[..., 0]  # [B, A, T, A]
    hist_image = torch.zeros(b, a, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, A, T, H * W]
    
    ego_mask = torch.zeros_like(raster_hist_pos_flat,dtype=torch.bool)
    ego_mask[:,range(a),:,range(a)]=1
    agent_mask = torch.logical_not(ego_mask)


    hist_image.scatter_(dim=3, index=raster_hist_pos_flat*agent_mask, src=torch.ones_like(hist_image) * -1)  # mark other agents with -1
    hist_image.scatter_(dim=3, index=raster_hist_pos_flat*ego_mask, src=torch.ones_like(hist_image))  # mark ego with 1.
    hist_image[..., 0] = 0  # correct the 0th index from invalid positions
    hist_image[..., -1] = 0  # correct the maximum index caused by out of bound locations

    hist_image = hist_image.reshape(b, a, t, h, w)

    maps = torch.cat((hist_image, maps), dim=2)  # treat time as extra channels
    return maps


def rasterize_agents(
        maps: torch.Tensor,
        agent_hist_pos: torch.Tensor,
        agent_hist_yaw: torch.Tensor,
        agent_mask: torch.Tensor,
        raster_from_agent: torch.Tensor,
        map_res: torch.Tensor,
) -> torch.Tensor:
    """Paint agent histories onto an agent-centric map image"""
    b, a, t, _ = agent_hist_pos.shape
    _, _, h, w = maps.shape
    maps = maps.clone()

    agent_hist_pos = agent_hist_pos.reshape(b, a * t, 2)
    raster_hist_pos = transform_points_tensor(agent_hist_pos, raster_from_agent)
    raster_hist_pos[~agent_mask.reshape(b, a * t)] = 0.0  # Set invalid positions to 0.0 Will correct below
    raster_hist_pos = raster_hist_pos.reshape(b, a, t, 2).permute(0, 2, 1, 3)  # [B, T, A, 2]
    raster_hist_pos[..., 0].clip_(0, (w - 1))
    raster_hist_pos[..., 1].clip_(0, (h - 1))
    raster_hist_pos = torch.round(raster_hist_pos).long()  # round pixels

    raster_hist_pos_flat = raster_hist_pos[..., 1] * w + raster_hist_pos[..., 0]  # [B, T, A]

    hist_image = torch.zeros(b, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, T, H * W]

    hist_image.scatter_(dim=2, index=raster_hist_pos_flat[:, :, 1:], src=torch.ones_like(hist_image) * -1)  # mark other agents with -1
    hist_image.scatter_(dim=2, index=raster_hist_pos_flat[:, :, [0]], src=torch.ones_like(hist_image))  # mark ego with 1.
    hist_image[:, :, 0] = 0  # correct the 0th index from invalid positions
    hist_image[:, :, -1] = 0  # correct the maximum index caused by out of bound locations

    hist_image = hist_image.reshape(b, t, h, w)

    maps = torch.cat((hist_image, maps), dim=1)  # treat time as extra channels
    return maps



def get_drivable_region_map(maps):
    if isinstance(maps, torch.Tensor):
        drivable = torch.amax(maps[..., -7:-4, :, :], dim=-3).bool()
    else:
        drivable = np.amax(maps[..., -7:-4, :, :], axis=-3).astype(np.bool)
    return drivable


def maybe_pad_neighbor(batch):
    """Pad neighboring agent's history to the same length as that of the ego using NaNs"""
    hist_len = batch["agent_hist"].shape[1]
    fut_len = batch["agent_fut"].shape[1]
    b, a, neigh_len, _ = batch["neigh_hist"].shape
    empty_neighbor = a == 0
    if empty_neighbor:
        batch["neigh_hist"] = torch.ones(b, 1, hist_len, batch["neigh_hist"].shape[-1]) * torch.nan
        batch["neigh_fut"] = torch.ones(b, 1, fut_len, batch["neigh_fut"].shape[-1]) * torch.nan
        batch["neigh_types"] = torch.zeros(b, 1)
        batch["neigh_hist_extents"] = torch.zeros(b, 1, hist_len, batch["neigh_hist_extents"].shape[-1])
        batch["neigh_fut_extents"] = torch.zeros(b, 1, fut_len, batch["neigh_hist_extents"].shape[-1])
    elif neigh_len < hist_len:
        hist_pad = torch.ones(b, a, hist_len - neigh_len, batch["neigh_hist"].shape[-1]) * torch.nan
        batch["neigh_hist"] = torch.cat((hist_pad, batch["neigh_hist"]), dim=2)
        hist_pad = torch.zeros(b, a, hist_len - neigh_len, batch["neigh_hist_extents"].shape[-1])
        batch["neigh_hist_extents"] = torch.cat((hist_pad, batch["neigh_hist_extents"]), dim=2)


@torch.no_grad()
def parse_avdata_batch(batch: dict):
    
    if "num_agents" in batch:
        # scene centric

        num_agents = batch["num_agents"]
        fut_pos, fut_yaw, _, fut_mask = avdata2posyawspeed(batch["agent_fut"])
        hist_pos, hist_yaw, hist_speed, hist_mask = avdata2posyawspeed(batch["agent_hist"])

        curr_pos = hist_pos[:,:,-1]
        curr_yaw = hist_yaw[:,:,-1]
        curr_speed = hist_speed[..., -1]
        centered_state = batch["centered_agent_state"]
        centered_yaw = centered_state[:, -1]
        centered_pos = centered_state[:, :2]

        # convert nuscenes types to l5kit types
        agent_type = batch["agent_type"]
        agent_type[agent_type < 0] = 0
        agent_type[agent_type == 1] = 3
        # mask out invalid extents
        agent_hist_extent = batch["agent_hist_extent"]
        agent_hist_extent[torch.isnan(agent_hist_extent)] = 0.


        centered_world_from_agent = torch.inverse(batch["centered_agent_from_world_tf"])



        # map-related
        if batch["maps"] is not None:
            map_res = batch["maps_resolution"][0,0]
            h, w = batch["maps"].shape[-2:]
            # TODO: pass env configs to here
            
            centered_raster_from_agent = torch.Tensor([
                [map_res, 0, 0.25 * w],
                [0, map_res, 0.5 * h],
                [0, 0, 1]
            ]).to(centered_state.device)
            b,a = curr_yaw.shape[:2]
            centered_agent_from_raster,_ = torch.linalg.inv_ex(centered_raster_from_agent)
            
            agents_from_center = (GeoUtils.transform_matrices(-curr_yaw.flatten(),torch.zeros(b*a,2,device=curr_yaw.device))
                                 @GeoUtils.transform_matrices(torch.zeros(b*a,device=curr_yaw.device),-curr_pos.reshape(-1,2))).reshape(*curr_yaw.shape[:2],3,3)
            center_from_agents = GeoUtils.transform_matrices(curr_yaw.flatten(),curr_pos.reshape(-1,2)).reshape(*curr_yaw.shape[:2],3,3)
            raster_from_center = centered_raster_from_agent @ agents_from_center
            center_from_raster = center_from_agents @ centered_agent_from_raster

            raster_from_world = batch["rasters_from_world_tf"]
            world_from_raster,_ = torch.linalg.inv_ex(raster_from_world)
            raster_from_world[torch.isnan(raster_from_world)] = 0.
            world_from_raster[torch.isnan(world_from_raster)] = 0.

            maps = rasterize_agents_scene(
                batch["maps"],
                hist_pos,
                hist_yaw,
                hist_mask,
                raster_from_center,
                map_res
            )
            drivable_map = get_drivable_region_map(batch["maps"])
        else:
            maps = None
            drivable_map = None
            raster_from_agent = None
            agent_from_raster = None
            raster_from_world = None

        extent_scale = 1.0


        d = dict(
            image=maps,
            drivable_map=drivable_map,
            target_positions=fut_pos,
            target_yaws=fut_yaw,
            target_availabilities=fut_mask,
            history_positions=hist_pos,
            history_yaws=hist_yaw,
            history_availabilities=hist_mask,
            curr_speed=curr_speed,
            centroid=curr_pos,
            yaw=curr_yaw,
            type=agent_type,
            extent=agent_hist_extent.max(dim=-2)[0] * extent_scale,
            raster_from_agent=centered_raster_from_agent,
            agent_from_raster=centered_agent_from_raster,
            raster_from_center=raster_from_center,
            center_from_raster=center_from_raster,
            agents_from_center = agents_from_center,
            center_from_agents = center_from_agents,
            raster_from_world=raster_from_world,
            agent_from_world=batch["centered_agent_from_world_tf"],
            world_from_agent=centered_world_from_agent,
        )
    else:
        # agent centric
        maybe_pad_neighbor(batch)
        fut_pos, fut_yaw, _, fut_mask = avdata2posyawspeed(batch["agent_fut"])
        hist_pos, hist_yaw, hist_speed, hist_mask = avdata2posyawspeed(batch["agent_hist"])
        curr_speed = hist_speed[..., -1]
        curr_state = batch["curr_agent_state"]
        curr_yaw = curr_state[:, -1]
        curr_pos = curr_state[:, :2]

        # convert nuscenes types to l5kit types
        agent_type = batch["agent_type"]
        agent_type[agent_type < 0] = 0
        agent_type[agent_type == 1] = 3
        # mask out invalid extents
        agent_hist_extent = batch["agent_hist_extent"]
        agent_hist_extent[torch.isnan(agent_hist_extent)] = 0.

        neigh_hist_pos, neigh_hist_yaw, neigh_hist_speed, neigh_hist_mask = avdata2posyawspeed(batch["neigh_hist"])
        neigh_fut_pos, neigh_fut_yaw, _, neigh_fut_mask = avdata2posyawspeed(batch["neigh_fut"])
        neigh_curr_speed = neigh_hist_speed[..., -1]
        neigh_types = batch["neigh_types"]
        # convert nuscenes types to l5kit types
        neigh_types[neigh_types < 0] = 0
        neigh_types[neigh_types == 1] = 3
        # mask out invalid extents
        neigh_hist_extents = batch["neigh_hist_extents"]
        neigh_hist_extents[torch.isnan(neigh_hist_extents)] = 0.

        world_from_agents = torch.inverse(batch["agents_from_world_tf"])

        # map-related
        if batch["maps"] is not None:
            map_res = batch["maps_resolution"][0]
            h, w = batch["maps"].shape[-2:]
            # TODO: pass env configs to here
            raster_from_agent = torch.Tensor([
                [map_res, 0, 0.25 * w],
                [0, map_res, 0.5 * h],
                [0, 0, 1]
            ]).to(curr_state.device)
            agent_from_raster = torch.inverse(raster_from_agent)
            raster_from_agent = TensorUtils.unsqueeze_expand_at(raster_from_agent, size=batch["maps"].shape[0], dim=0)
            agent_from_raster = TensorUtils.unsqueeze_expand_at(agent_from_raster, size=batch["maps"].shape[0], dim=0)
            raster_from_world = torch.bmm(raster_from_agent, batch["agents_from_world_tf"])

            all_hist_pos = torch.cat((hist_pos[:, None], neigh_hist_pos), dim=1)
            all_hist_yaw = torch.cat((hist_yaw[:, None], neigh_hist_yaw), dim=1)
            all_hist_mask = torch.cat((hist_mask[:, None], neigh_hist_mask), dim=1)
            maps = rasterize_agents(
                batch["maps"],
                all_hist_pos,
                all_hist_yaw,
                all_hist_mask,
                raster_from_agent,
                map_res
            )
            drivable_map = get_drivable_region_map(batch["maps"])
        else:
            maps = None
            drivable_map = None
            raster_from_agent = None
            agent_from_raster = None
            raster_from_world = None

        extent_scale = 1.0
        d = dict(
            image=maps,
            drivable_map=drivable_map,
            target_positions=fut_pos,
            target_yaws=fut_yaw,
            target_availabilities=fut_mask,
            history_positions=hist_pos,
            history_yaws=hist_yaw,
            history_availabilities=hist_mask,
            curr_speed=curr_speed,
            centroid=curr_pos,
            yaw=curr_yaw,
            type=agent_type,
            extent=agent_hist_extent.max(dim=-2)[0] * extent_scale,
            raster_from_agent=raster_from_agent,
            agent_from_raster=agent_from_raster,
            raster_from_world=raster_from_world,
            agent_from_world=batch["agents_from_world_tf"],
            world_from_agent=world_from_agents,
            all_other_agents_history_positions=neigh_hist_pos,
            all_other_agents_history_yaws=neigh_hist_yaw,
            all_other_agents_history_availabilities=neigh_hist_mask,
            all_other_agents_history_availability=neigh_hist_mask,  # dump hack to agree with l5kit's typo ...
            all_other_agents_curr_speed=neigh_curr_speed,
            all_other_agents_future_positions=neigh_fut_pos,
            all_other_agents_future_yaws=neigh_fut_yaw,
            all_other_agents_future_availability=neigh_fut_mask,
            all_other_agents_types=neigh_types,
            all_other_agents_extents=neigh_hist_extents.max(dim=-2)[0] * extent_scale,
            all_other_agents_history_extents=neigh_hist_extents * extent_scale,
            ego_lanes = batch["agent_lanes"]

        )

    batch = dict(batch)
    batch.update(d)
    for k,v in batch.items():
        if isinstance(v,torch.Tensor):
            batch[k]=v.nan_to_num(0)
    batch.pop("agent_name", None)
    batch.pop("robot_fut", None)
    return batch


def get_modality_shapes(cfg: ExperimentConfig):
    num_channels = (cfg.algo.history_num_frames + 1) + 7
    h = cfg.env.rasterizer.raster_size
    return dict(image=(num_channels, h, h))