import torch
from avdata import AgentBatch
from avdata.simulation import SimulationScene
from avdata.data_structures.batch_element import AgentBatchElement
from typing import List


def avdata2posyawspeed(state):
    assert state.shape[-1] == 8  # x, y, vx, vy, ax, ay, sin(heading), cos(heading)
    pos = state[..., :2]
    yaw = torch.atan2(state[..., [-2]], state[..., [-1]])
    speed = torch.norm(state[..., 2:4], dim=-1)
    return pos, yaw, speed


def parse_avdata_batch(batch: dict):
    fut_pos, fut_yaw, _ = avdata2posyawspeed(batch["agent_fut"])
    hist_pos, hist_yaw, hist_speed = avdata2posyawspeed(batch["agent_hist"])
    curr_speed = hist_speed[..., -1]
    curr_state = batch["curr_agent_state"]

    d = dict(
        image=batch["maps"],
        target_positions=fut_pos,
        target_yaws=fut_yaw,
        target_availabilities=torch.ones_like(fut_pos[..., 0]),
        history_positions=hist_pos,
        history_yaws=hist_yaw,
        history_availabilities=torch.ones_like(hist_pos[..., 0]),
        curr_speed=curr_speed,
        centroid=curr_state[..., :2],
        yaw=curr_state[..., -1],
    )
    batch.update(d)
    return batch


def maybe_parse_batch(batch):
    """Parse batch to the expected format"""
    if "agent_fut" in batch:  # avdata
        return parse_avdata_batch(batch)
    else:
        return batch