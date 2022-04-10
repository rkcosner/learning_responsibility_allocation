import torch
from avdata import AgentBatch


def avdata2posyaw(state):
    assert state.shape[-1] == 8  # x, y, vx, vy, ax, ay, sin(heading), cos(heading)
    pos = state[..., :2]
    yaw = torch.arccos(state[..., [-1]])
    speed = torch.norm(state[..., 2:4], dim=-1)
    return pos, yaw, speed


def _parse_avdata_batch(batch: AgentBatch):
    assert isinstance(batch, AgentBatch)
    fut_pos, fut_yaw, _ = avdata2posyaw(batch.agent_fut)
    hist_pos, hist_yaw, hist_speed = avdata2posyaw(batch.agent_hist)
    curr_speed = hist_speed[..., -1]

    d = dict(
        image=batch.maps,
        target_positions=fut_pos,
        target_yaws=fut_yaw,
        target_availabilities=torch.ones_like(fut_pos[..., 0]),
        curr_speed=curr_speed,

    )
    return d


def maybe_parse_batch(batch):
    """Parse batch to the expected format"""
    if isinstance(batch, AgentBatch):
        return _parse_avdata_batch(batch)
    else:
        return batch