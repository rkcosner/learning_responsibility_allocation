import argparse
import h5py
import numpy as np
import torch

import os

from l5kit.data import LocalDataManager
from l5kit.geometry import transform_points
from avdata.simulation.sim_stats import calc_stats

from tbsim.utils.config_utils import translate_l5kit_cfg, translate_avdata_cfg
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.utils.vis_utils import build_visualization_rasterizer_l5kit
from tbsim.utils.vis_utils import COLORS, draw_agent_boxes
from PIL import Image, ImageDraw


def get_l5_rasterizer(dataset_path):
    exp_cfg = get_registered_experiment_config("l5_mixed_plan")
    exp_cfg.dataset_path = dataset_path
    os.environ["L5KIT_DATA_FOLDER"] = dataset_path

    l5_config = translate_l5kit_cfg(exp_cfg)
    l5_config["raster_params"]["raster_size"] = (500, 500)
    l5_config["raster_params"]["pixel_size"] = (0.2, 0.2)
    l5_config["raster_params"]["ego_center"] = (0.5, 0.5)
    render_rasterizer = build_visualization_rasterizer_l5kit(l5_config, LocalDataManager(None))
    return render_rasterizer


def get_state_image_l5(rasterizer, positions, yaws, extents, ras_pos, ras_yaw):
    state_im = rasterizer.rasterize(
        ras_pos,
        ras_yaw
    )

    raster_from_world = rasterizer.render_context.raster_from_world(
        ras_pos,
        ras_yaw
    )
    state_im = draw_agent_boxes(
        state_im,
        positions,
        yaws[:, None],
        extents,
        raster_from_world,
        outline_color=COLORS["agent_contour"],
        fill_color=COLORS["agent_fill"]
    )

    return state_im, raster_from_world


def draw_trajectories(im, trajectories, raster_from_world):
    im = im.copy()
    im = Image.fromarray((im * 255).astype(np.uint8))
    draw = ImageDraw.Draw(im)

    for atraj in trajectories:
        pos_raster = transform_points(atraj[None], raster_from_world)[0]
        for point in pos_raster:
            circle = np.hstack([point - 3, point + 3])
            draw.ellipse(circle.tolist(), fill="#FE5F55", outline="#911A12")

    im = np.asarray(im).astype(np.float32) / 255.
    return im


def visualize_l5(h5f, dataset_path):
    rasterizer = get_l5_rasterizer(dataset_path)

    scene_data = h5f["5232_0"]

    t = 100
    traj_len = 50
    im, raster_from_world = get_state_image_l5(
        rasterizer,
        positions=scene_data["centroid"][:, t],
        yaws=scene_data["yaw"][:, t],
        extents=scene_data["extent"][:, t, :2],
        ras_pos=scene_data["centroid"][0, t],
        ras_yaw=scene_data["yaw"][0, t],
    )

    im = draw_trajectories(im, scene_data["centroid"][:, t:t+traj_len], raster_from_world)

    im = (im * 255).astype(np.uint8)

    Image.fromarray(im).save("im.png")


def get_stats(h5f):
    bins = {
        "velocity": torch.linspace(0, 100, 21),
        "lon_accel": torch.linspace(0, 20, 21),
        "lat_accel": torch.linspace(0, 20, 21),
        "jerk": torch.linspace(0, 20, 21),
    }

    scene_data = h5f["5232_0"]
    sim_pos = scene_data["centroid"]
    sim_yaw = scene_data["yaw"][:][:, None]

    gt_pos = scene_data["gt_centroid"]
    gt_yaw = scene_data["yaw"][:][:, None]

    sim_stats = calc_stats(positions=torch.Tensor(sim_pos), heading=torch.Tensor(sim_yaw), dt=0.1, bins=bins)
    gt_stats = calc_stats(positions=torch.Tensor(gt_pos), heading=torch.Tensor(gt_yaw), dt=0.1, bins=bins)


def main(hdf5_path, dataset_path):
    h5f = h5py.File(hdf5_path, "r")
    # visualize_l5(h5f, dataset_path)
    get_stats(h5f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hdf5_path",
        type=str,
        default=None,
        required=True,
        help="An hdf5 containing the saved rollout info"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    main(args.hdf5_path, args.dataset_path)