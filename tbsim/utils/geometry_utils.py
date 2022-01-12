import collections
import numpy as np
import torch


def get_box_world_coords(pos, yaw, extent):
    corners = (torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * 0.5) * (
        extent.unsqueeze(-2)
    )
    s = torch.sin(yaw).unsqueeze(-1)
    c = torch.cos(yaw).unsqueeze(-1)
    rotM = torch.cat((torch.cat((c, s), dim=-1), torch.cat((-s, c), dim=-1)), dim=-2)
    rotated_corners = (corners + pos.unsqueeze(-2)) @ rotM
    return rotated_corners


def batch_nd_transform_points(points, Mat):
    ndim = Mat.shape[-1] - 1
    Mat = torch.transpose(Mat, -1, -2)
    return (points.unsqueeze(-2) @ Mat[..., :ndim, :ndim]).squeeze(-2) + Mat[
        ..., -1:, :ndim
    ].squeeze(-2)


def PED_PED_collision(p1, p2, S1, S2):
    if isinstance(p1, torch.Tensor):

        return (
            torch.linalg.norm(p1[..., 0:2] - p2[..., 0:2], dim=-1)
            - (S1[..., 0] + S2[..., 0]) / 2
        )

    elif isinstance(p1, np.ndarray):

        return (
            np.linalg.norm(p1[..., 0:2] - p2[..., 0:2], axis=-1)
            - (S1[..., 0] + S2[..., 0]) / 2
        )
    else:
        raise NotImplementedError


def batch_rotate_2D(xy, theta):
    if isinstance(xy, torch.Tensor):
        x1 = xy[..., 0] * torch.cos(theta) - xy[..., 1] * torch.sin(theta)
        y1 = xy[..., 1] * torch.cos(theta) + xy[..., 0] * torch.sin(theta)
        return torch.stack([x1, y1], dim=-1)
    elif isinstance(xy, np.ndarray):
        x1 = xy[..., 0] * np.cos(theta) - xy[..., 1] * np.sin(theta)
        y1 = xy[..., 1] * np.cos(theta) + xy[..., 0] * np.sin(theta)
        return np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1)), axis=-1)


def VEH_VEH_collision(
    p1, p2, S1, S2, alpha=5, return_dis=False, offsetX=1.0, offsetY=0.3
):
    if isinstance(p1, torch.Tensor):
        cornersX = torch.kron(
            S1[..., 0] + offsetX, torch.tensor([0.5, 0.5, -0.5, -0.5]).to(p1.device)
        )
        cornersY = torch.kron(
            S1[..., 1] + offsetY, torch.tensor([0.5, -0.5, 0.5, -0.5]).to(p1.device)
        )
        corners = torch.stack([cornersX, cornersY], dim=-1)
        theta1 = p1[..., 2]
        theta2 = p2[..., 2]
        dx = (p1[..., 0:2] - p2[..., 0:2]).repeat_interleave(4, dim=-2)
        delta_x1 = batch_rotate_2D(corners, theta1.repeat_interleave(4, dim=-1)) + dx
        delta_x2 = batch_rotate_2D(delta_x1, -theta2.repeat_interleave(4, dim=-1))
        dis = torch.maximum(
            torch.abs(delta_x2[..., 0]) - 0.5 * S2[..., 0].repeat_interleave(4, dim=-1),
            torch.abs(delta_x2[..., 1]) - 0.5 * S2[..., 1].repeat_interleave(4, dim=-1),
        ).view(*S1.shape[:-1], 4)
        min_dis, _ = torch.min(dis, dim=-1)

        return min_dis

    elif isinstance(p1, np.ndarray):
        cornersX = np.kron(S1[..., 0] + offsetX, np.array([0.5, 0.5, -0.5, -0.5]))
        cornersY = np.kron(S1[..., 1] + offsetY, np.array([0.5, -0.5, 0.5, -0.5]))
        corners = np.concatenate((cornersX, cornersY), axis=-1)
        theta1 = p1[..., 2]
        theta2 = p2[..., 2]
        dx = (p1[..., 0:2] - p2[..., 0:2]).repeat(4, axis=-2)
        delta_x1 = batch_rotate_2D(corners, theta1.repeat(4, axis=-1)) + dx
        delta_x2 = batch_rotate_2D(delta_x1, -theta2.repeat(4, axis=-1))
        dis = np.maximum(
            np.abs(delta_x2[..., 0]) - 0.5 * S2[..., 0].repeat(4, axis=-1),
            np.abs(delta_x2[..., 1]) - 0.5 * S2[..., 1].repeat(4, axis=-1),
        ).reshape(*S1.shape[:-1], 4)
        min_dis = np.min(dis, axis=-1)
        return min_dis
    else:
        raise NotImplementedError


def VEH_PED_collision(p1, p2, S1, S2):
    if isinstance(p1, torch.Tensor):

        mask = torch.logical_or(
            torch.abs(p1[..., 2]) > 0.1, torch.linalg.norm(p2[..., 2:4], dim=-1) > 0.1
        ).detach()
        theta = p1[..., 2]
        dx = batch_rotate_2D(p2[..., 0:2] - p1[..., 0:2], -theta)

        return torch.maximum(
            torch.abs(dx[..., 0]) - S1[..., 0] / 2 - S2[..., 0] / 2,
            torch.abs(dx[..., 1]) - S1[..., 1] / 2 - S2[..., 0] / 2,
        )
    elif isinstance(p1, np.ndarray):

        theta = p1[..., 2]
        dx = batch_rotate_2D(p2[..., 0:2] - p1[..., 0:2], -theta)
        return np.maximum(
            np.abs(dx[..., 0]) - S1[..., 0] / 2 - S2[..., 0] / 2,
            np.abs(dx[..., 1]) - S1[..., 1] / 2 - S2[..., 0] / 2,
        )
    else:
        raise NotImplementedError


def PED_VEH_collision(p1, p2, S1, S2):
    return VEH_PED_collision(p2, p1, S2, S1)
