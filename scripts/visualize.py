import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import pathlib
import json

import os

from l5kit.data import LocalDataManager
from l5kit.geometry import transform_points
from avdata.simulation.sim_stats import calc_stats

from tbsim.utils.geometry_utils import get_box_world_coords_np
from tbsim.utils.config_utils import translate_l5kit_cfg, translate_avdata_cfg
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.utils.vis_utils import build_visualization_rasterizer_l5kit
from tbsim.utils.vis_utils import COLORS, draw_agent_boxes
from tbsim.configs.eval_configs import EvaluationConfig
import tbsim.utils.tensor_utils as TensorUtils
from PIL import Image, ImageDraw

import matplotlib.collections as mcoll
import matplotlib.patches as patches
import matplotlib.path as mpath


def colorline(
        ax, x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


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


def get_state_image_l5(rasterizer, ras_pos, ras_yaw):
    state_im = rasterizer.rasterize(
        ras_pos,
        ras_yaw
    )

    raster_from_world = rasterizer.render_context.raster_from_world(
        ras_pos,
        ras_yaw
    )

    return state_im, raster_from_world


def draw_trajectories(ax, trajectories, raster_from_world):
    raster_trajs = transform_points(trajectories, raster_from_world)
    for traj in raster_trajs:
        colorline(
            ax,
            traj[..., 0],
            traj[..., 1],
            cmap="viridis",
            # z=np.linspace(1.0, 0.0, len(traj[..., 0],)),
            linewidth=3
        )
    # lplot.set_label("Trajectories")


def draw_agent_boxes_plt(ax, pos, yaw, extent, raster_from_agent, outline_color, fill_color):
    boxes = get_box_world_coords_np(pos, yaw, extent)
    boxes_raster = transform_points(boxes, raster_from_agent)
    boxes_raster = boxes_raster.reshape((-1, 4, 2))
    for b in boxes_raster:
        rect = patches.Polygon(b, fill=True, color=fill_color, zorder=1)
        rect_border = patches.Polygon(b, fill=False, color="grey", zorder=1, linewidth=0.5)
        ax.add_patch(rect)
        ax.add_patch(rect_border)


def draw_scene_data(ax, scene_data, starting_frame, rasterizer):
    t = starting_frame
    traj_len = 200
    state_im, raster_from_world = get_state_image_l5(
        rasterizer,
        ras_pos=scene_data["centroid"][0, t],
        # ras_yaw=scene_data["yaw"][0, t],
        ras_yaw=np.pi
    )
    ax.imshow(state_im)

    draw_trajectories(
        ax,
        trajectories=scene_data["centroid"][:, t:t+traj_len],
        raster_from_world=raster_from_world,
    )

    draw_agent_boxes_plt(
        ax,
        pos=scene_data["centroid"][:, t],
        yaw=scene_data["yaw"][:, [t]],
        extent=scene_data["extent"][:, t, :2],
        raster_from_agent=raster_from_world,
        outline_color=COLORS["agent_contour"],
        fill_color=COLORS["agent_fill"]
    )

    ax.set_xlim([0, state_im.shape[1] - 140])
    ax.set_ylim([80, state_im.shape[0] - 130])
    # ax.set_xlim([0, state_im.shape[1]])
    # ax.set_ylim([0, state_im.shape[0]])
    ax.grid(False)
    ax.axis("off")
    ax.invert_xaxis()


# def visualize_l5_scene(rasterizer, h5f, scene_index, starting_frame, output_dir, gt_h5f=None):
#     fig, axes = plt.subplots(1, 5, figsize=(30, 6))
#     for ep_i in range(5):
#         ax = axes[ep_i]
#         scene_name = "{}_{}".format(scene_index, ep_i)
#         if scene_name not in list(h5f.keys()):
#             continue
#         scene_data = h5f[scene_name]
#         draw_scene_data(ax, scene_data, starting_frame, rasterizer)
#
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     ffn = os.path.join(output_dir, "{}_t{}.png").format(scene_index, starting_frame)
#     plt.savefig(ffn, dpi=400, bbox_inches="tight")
#     plt.close()
#     print("Figure written to {}".format(ffn))


def visualize_l5_scene(rasterizer, h5f, scene_index, starting_frame, output_dir, gt_h5f=None):
    # fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    for ep_i in range(5):
        fig, ax = plt.subplots()
        scene_name = "{}_{}".format(scene_index, ep_i)
        if scene_name not in list(h5f.keys()):
            continue
        scene_data = h5f[scene_name]
        draw_scene_data(ax, scene_data, starting_frame, rasterizer)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ffn = os.path.join(output_dir, "{}_t{}_{}.png").format(scene_index, starting_frame, ep_i)
        plt.savefig(ffn, dpi=400, bbox_inches="tight", pad_inches = 0)
        plt.close()
        print("Figure written to {}".format(ffn))


def main(hdf5_path, dataset_path, output_dir):
    # SOI_l5kit: [1069, 1090, 4558, ]
    rasterizer = get_l5_rasterizer(dataset_path)
    h5f = h5py.File(hdf5_path, "r")
    sids = EvaluationConfig().l5kit.eval_scenes
    sids = [1069]
    for si in sids:
        visualize_l5_scene(rasterizer, h5f, si, 0, output_dir=output_dir)
        # visualize_l5_scene(rasterizer, h5f, si, 50, output_dir=output_dir)
        # visualize_l5_scene(rasterizer, h5f, si, 100, output_dir=output_dir)


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

    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations/"
    )

    args = parser.parse_args()

    main(args.hdf5_path, args.dataset_path, args.output_dir)