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
