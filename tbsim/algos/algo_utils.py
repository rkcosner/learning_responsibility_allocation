import torch
from torch import optim as optim

import tbsim.utils.tensor_utils as TensorUtils
from tbsim import dynamics as dynamics
from tbsim.utils import l5_utils as L5Utils
from tbsim.utils.geometry_utils import transform_points_tensor
from tbsim.utils.l5_utils import get_last_available_index
from tbsim.utils.loss_utils import goal_reaching_loss, trajectory_loss, collision_loss


def decode_spatial_map(pred_map):
    # decode map as predictions
    b, c, h, w = pred_map.shape
    pixel_logit, pixel_loc_flat = torch.max(torch.flatten(pred_map[:, 0], start_dim=1), dim=1)
    pixel_loc_x = torch.remainder(pixel_loc_flat, w)
    pixel_loc_y = torch.floor(pixel_loc_flat.float() / float(w)).long()
    pixel_loc = torch.stack((pixel_loc_x, pixel_loc_y), dim=1)  # [B, 2]
    local_pred = torch.gather(
        input=torch.flatten(pred_map, 2),  # [B, C, H * W]
        dim=2,
        index=TensorUtils.unsqueeze_expand_at(pixel_loc_flat, size=c, dim=1)[:, :, None]  # [B, C, 1]
    ).squeeze(-1)
    residual_pred = local_pred[:, 1:3]
    yaw_pred = local_pred[:, 3:4]
    return pixel_loc.float(), residual_pred, yaw_pred, pixel_logit


def get_spatial_goal_supervision(data_batch):
    b, _, h, w = data_batch["image"].shape  # [B, C, H, W]

    # use last available step as goal location
    goal_index = get_last_available_index(data_batch["target_availabilities"])[:, None, None]

    # gather by goal index
    goal_pos_agent = torch.gather(
        data_batch["target_positions"],  # [B, T, 2]
        dim=1,
        index=goal_index.expand(-1, 1, data_batch["target_positions"].shape[-1])
    )  # [B, 1, 2]

    goal_yaw_agent = torch.gather(
        data_batch["target_yaws"],  # [B, T, 1]
        dim=1,
        index=goal_index.expand(-1, 1, data_batch["target_yaws"].shape[-1])
    )  # [B, 1, 1]

    # create spatial supervisions
    goal_pos_raster = transform_points_tensor(
        goal_pos_agent,
        data_batch["raster_from_agent"].float()
    ).squeeze(1)  # [B, 2]
    # make sure all pixels are within the raster image
    goal_pos_raster[:, 0] = goal_pos_raster[:, 0].clip(0, w - 1e-5)
    goal_pos_raster[:, 1] = goal_pos_raster[:, 1].clip(0, h - 1e-5)

    goal_pos_pixel = torch.floor(goal_pos_raster).float()  # round down pixels
    goal_pos_residual = goal_pos_raster - goal_pos_pixel  # compute rounding residuals (range 0-1)
    # compute flattened pixel location
    goal_pos_pixel_flat = goal_pos_pixel[:, 1] * w + goal_pos_pixel[:, 0]
    raster_sup_flat = TensorUtils.to_one_hot(goal_pos_pixel_flat.long(), num_class=h * w)
    raster_sup = raster_sup_flat.reshape(b, h, w)
    return {
        "goal_position_residual": goal_pos_residual,  # [B, 2]
        "goal_spatial_map": raster_sup,  # [B, H, W]
        "goal_position_pixel": goal_pos_pixel,  # [B, 2]
        "goal_position_pixel_flat": goal_pos_pixel_flat,  # [B]
        "goal_position": goal_pos_agent.squeeze(1),  # [B, 2]
        "goal_yaw": goal_yaw_agent.squeeze(1),  # [B, 1]
        "goal_index": goal_index.reshape(b)  # [B]
    }


def optimize_trajectories(
        init_u,
        init_x,
        target_trajs,
        target_avails,
        dynamics_model,
        step_time: float,
        data_batch = None,
        goal_loss_weight=1.0,
        traj_loss_weight=0.0,
        coll_loss_weight=0.0,
        num_optim_iterations: int = 50
):
    curr_u = init_u.detach().clone()
    curr_u.requires_grad = True
    action_optim = optim.LBFGS([curr_u], max_iter=20, lr=1.0, line_search_fn='strong_wolfe')

    for oidx in range(num_optim_iterations):
        def closure():
            action_optim.zero_grad()

            # get trajectory with current params
            _, pos, yaw = dynamics.forward_dynamics(
                dyn_model=dynamics_model,
                initial_states=init_x,
                actions=curr_u,
                step_time=step_time
            )
            curr_trajs = torch.cat((pos, yaw), dim=-1)
            # compute trajectory optimization losses
            losses = dict()
            losses["goal_loss"] = goal_reaching_loss(
                predictions=curr_trajs,
                targets=target_trajs,
                availabilities=target_avails
            ) * goal_loss_weight
            losses["traj_loss"] = trajectory_loss(
                predictions=curr_trajs,
                targets=target_trajs,
                availabilities=target_avails
            ) * traj_loss_weight
            if coll_loss_weight > 0:
                assert data_batch is not None
                coll_edges = L5Utils.get_edges_from_batch(
                    data_batch,
                    ego_predictions=dict(positions=pos, yaws=yaw)
                )
                for c in coll_edges:
                    coll_edges[c] = coll_edges[c][:, :target_trajs.shape[-2]]
                vv_edges = dict(VV=coll_edges["VV"])
                if vv_edges["VV"].shape[0] > 0:
                    losses["coll_loss"] = collision_loss(vv_edges) * coll_loss_weight

            total_loss = torch.hstack(list(losses.values())).sum()

            # backprop
            total_loss.backward()
            return total_loss
        action_optim.step(closure)

    final_raw_trajs, final_pos, final_yaw = dynamics.forward_dynamics(
        dyn_model=dynamics_model,
        initial_states=init_x,
        actions=curr_u,
        step_time=step_time
    )
    final_trajs = torch.cat((final_pos, final_yaw), dim=-1)
    losses = dict()
    losses["goal_loss"] = goal_reaching_loss(
        predictions=final_trajs,
        targets=target_trajs,
        availabilities=target_avails
    )
    losses["traj_loss"] = trajectory_loss(
        predictions=final_trajs,
        targets=target_trajs,
        availabilities=target_avails
    )

    return dict(positions=final_pos, yaws=final_yaw), final_raw_trajs, curr_u, losses