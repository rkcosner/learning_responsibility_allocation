import torch 
from tbsim import dynamics as dynamics


def batch_to_raw_all_agents(data_batch, step_time):
    """
        RYAN: 
            stacks ego with other agents history
        arg: 
            - data_batch : all learning data
            - step_time 
        return: 
            Everything is ordered so that the timing index is [0 : newest], [-1 : oldest] 
            - "history_positions"
            - "history_vel"
            - "history_yaws"
            - "raw_types:
            - "history_availabilities"
            - "extents"
    """

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

    src_yaw = torch.cat(
        (
            data_batch["history_yaws"].unsqueeze(1),
            data_batch["all_other_agents_history_yaws"],
        ),
        dim=1,
    )
    
    src_mask = torch.cat(
        (
            data_batch["history_availabilities"].unsqueeze(1),
            data_batch["all_other_agents_history_availability"],
        ),
        dim=1,
    ).bool()


    """
        RYAN: extent seems to be dim=3 in nuscenes and dim=2 in L5, what does that data mean? 
    """
    extents = torch.cat(
        (
            data_batch["extent"][..., :3].unsqueeze(1),
            torch.max(data_batch["all_other_agents_history_extents"], dim=-2)[0],
        ),
        dim=1,
    )

    # estimate velocity
    src_vel = dynamics.Unicycle.calculate_vel(src_pos, src_yaw, step_time, src_mask)
    src_vel[:, 0, -1] = data_batch["curr_speed"].unsqueeze(-1)

    # Flip everything so that 0 is the current velocity and -1 is the oldest
    src_pos = torch.flip(src_pos, dims=[-2])
    src_yaw = torch.flip(src_yaw, dims=[-2])
    src_mask = torch.flip(src_mask, dims=[-1])
    src_vel = torch.flip(src_vel, dims=[-1])

    return {
        "history_positions": src_pos,
        "history_vel" : src_vel, 
        "history_yaws": src_yaw,
        "raw_types": raw_type,
        "history_availabilities": src_mask,
        "extents": extents,
    }