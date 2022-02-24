import numpy as np
from l5kit.geometry import transform_points
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from tbsim.utils.tensor_utils import map_ndarray
from tbsim.utils.l5_utils import get_last_available_index
from tbsim.external.vis_rasterizer import VisualizationRasterizer, cv2_subpixel, CV2_SUB_VALUES
from tbsim.utils.geometry_utils import get_box_world_coords_np
from l5kit.rasterization.render_context import RenderContext
from l5kit.configs.config import load_metadata
from PIL import Image, ImageDraw


COLORS = {
    "agent_contour": "#247BA0",
    "agent_fill": "#56B1D8",
    "ego_contour": "#247BA0",
    "ego_fill": "#56B1D8",
}


def agent_to_raster_np(pt_tensor, trans_mat):
    pos_raster = transform_points(pt_tensor[None], trans_mat)[0]
    return pos_raster


def _render_state(
        state_image,
        trans_mat,
        title="",
        text=None,
        pred_actions=None,
        gt_actions=None,
        pred_plan=None,
        gt_plan=None,
        pred_plan_info=None
):
    fig = Figure(figsize=(10, 8), dpi=100)
    canvas = FigureCanvasAgg(fig)
    if text is not None:
        fig.text(x=10, y=10, s=text)
    ax = fig.add_subplot(121)
    ax.imshow(state_image)
    fig.suptitle(title)

    if pred_actions is not None:
        raster_traj = agent_to_raster_np(pred_actions["positions"], trans_mat)
        ax.plot(raster_traj[:-1, 0], raster_traj[:-1, 1], color="red")

    if gt_actions is not None:
        raster_traj = agent_to_raster_np(gt_actions["positions"], trans_mat)
        ax.plot(raster_traj[:, 0], raster_traj[:, 1], color="blue")

    if pred_plan is not None:
        # predicted subgoal
        pos_raster = agent_to_raster_np(pred_plan["positions"], trans_mat)[[-1]]
        ax.scatter(pos_raster[:, 0], pos_raster[:, 1], marker='o', color="red")

    if gt_plan is not None:
        pos_raster = agent_to_raster_np(gt_plan["positions"], trans_mat)[[-1]]
        ax.scatter(pos_raster[:, 0], pos_raster[:, 1], marker='o', color="blue")

    # visualize plan heat map
    if pred_plan_info is not None and "location_map" in pred_plan_info:
        ax = fig.add_subplot(122)
        ax.imshow(pred_plan_info["location_map"])

    canvas.draw()
    im = np.asarray(canvas.buffer_rgba())[:, :, :3]
    return im


def _render_draw(
        state_image,
        trans_mat,
        title="",
        text=None,
        pred_actions=None,
        gt_actions=None,
        pred_plan=None,
        gt_plan=None,
        pred_plan_info=None
):
    im = Image.fromarray((state_image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(im)
    if pred_actions is not None:
        raster_traj = agent_to_raster_np(pred_actions["positions"], trans_mat)
        draw.line(raster_traj[:-1].reshape(-1).tolist(), fill="red", width=3, joint="curve")

    if pred_plan is not None:
        # predicted subgoal
        pos_raster = agent_to_raster_np(pred_plan["positions"], trans_mat)[-1]
        circle = np.hstack([pos_raster - 5, pos_raster + 5])
        draw.ellipse(circle.tolist(), fill="red")

    im = np.asarray(im)

    # visualize plan heat map
    if pred_plan_info is not None and "location_map" in pred_plan_info:
        import matplotlib.pyplot as plt

        # Get the color map by name:
        cm = plt.get_cmap("jet")
        heatmap = pred_plan_info["location_map"]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / heatmap.max()
        heatmap = cm(heatmap)

        heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap = heatmap.resize(size=im.shape[:2])
        heatmap = np.asarray(heatmap)[..., :3]
        im = np.concatenate((im, heatmap), axis=1)

    return im


def draw_agents(image, pos, yaw, extent, raster_from_agent, outline_color, fill_color):
    boxes = get_box_world_coords_np(pos, yaw, extent)
    boxes_raster = transform_points(boxes, raster_from_agent)
    boxes_raster = boxes_raster.reshape((-1, 4, 2)).astype(np.int)

    im = Image.fromarray((image * 255).astype(np.uint8))
    im_draw = ImageDraw.Draw(im)
    for b in boxes_raster:
        im_draw.polygon(xy=b.reshape(-1).tolist(), outline=outline_color, fill=fill_color)

    im = np.asarray(im).astype(np.float32) / 255.
    return im


def _get_state_image(ego_obs, agents_obs, rasterizer):
    state_im = rasterizer.rasterize(
        ego_obs["ego_translation"],
        ego_obs["yaw"]
    )

    raster_from_world = rasterizer.render_context.raster_from_world(
        ego_obs["ego_translation"],
        ego_obs["yaw"]
    )
    raster_from_agent = raster_from_world @ ego_obs["world_from_agent"]
    state_im = draw_agents(
        state_im,
        agents_obs["centroid"],
        agents_obs["yaw"][:, None],
        agents_obs["extent"][:, :2],
        raster_from_world,
        outline_color=COLORS["agent_contour"],
        fill_color=COLORS["agent_fill"]
    )

    state_im = draw_agents(
        state_im,
        ego_obs["centroid"][None],
        ego_obs["yaw"][None, None],
        ego_obs["extent"][None, :2],
        raster_from_world,
        outline_color=COLORS["agent_contour"],
        fill_color=COLORS["agent_fill"]
    )

    return state_im, raster_from_agent


def render_state_l5kit(
        rasterizer: VisualizationRasterizer,
        state_obs,
        action,
        scene_index,
        step_index,
        dataset_scene_index,
        step_metrics=None
):
    agent_scene_index = dataset_scene_index == state_obs["agents"]["scene_index"]
    agent_obs = map_ndarray(state_obs["agents"], lambda x: x[agent_scene_index])
    ego_obs = map_ndarray(state_obs["ego"], lambda x:  x[scene_index])
    # combined_obs = dict()
    # for k in ego_obs.keys():
    #     combined_obs[k] = np.concatenate((ego_obs[k][None], agent_obs[k]), axis=0)

    state_im, raster_from_agent = _get_state_image(ego_obs, agent_obs, rasterizer)
    # state_im = rasterizer.to_rgb(state_obs["image"].transpose(1, 2, 0))
    # raster_from_agent = ego_obs["raster_from_agent"]
    gt_actions = dict(
        positions=ego_obs["target_positions"],
        yaws=ego_obs["target_yaws"]
    )

    slice_idx = lambda x:  x[scene_index]
    if action.ego is not None:
        pred_actions = map_ndarray(action.ego.to_dict(), slice_idx)
        pred_plan = action.ego_info.get("plan", None)
        pred_plan_info = action.ego_info.get("plan_info", None)
    else:
        pred_actions = None
        pred_plan = None
        pred_plan_info = None

    if pred_plan is not None:
        pred_plan = map_ndarray(pred_plan, slice_idx)
        pred_plan_info = map_ndarray(pred_plan_info, slice_idx)

    msg = ""
    if step_metrics is not None:
        for k in step_metrics:
            msg += "{}={}\n".format(k, step_metrics[k])

    im = _render_draw(
        state_image=state_im,
        trans_mat=raster_from_agent,
        pred_actions=pred_actions,
        # gt_actions=gt_actions,
        pred_plan=pred_plan,
        pred_plan_info=pred_plan_info,
        title="scene={}, step={}".format(dataset_scene_index, step_index) + "\n" + msg,
    )

    return im


def build_visualization_rasterizer_l5kit(cfg, dm):
    raster_cfg = cfg["raster_params"]
    dataset_meta_key = raster_cfg["dataset_meta_key"]
    dataset_meta = load_metadata(dm.require(dataset_meta_key))
    world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)

    render_context = RenderContext(
        raster_size_px=np.array(raster_cfg["raster_size"]),
        pixel_size_m=np.array(raster_cfg["pixel_size"]),
        center_in_raster_ratio=np.array(raster_cfg["ego_center"]),
        set_origin_to_bottom=raster_cfg["set_origin_to_bottom"],
    )

    semantic_map_filepath = dm.require(raster_cfg["semantic_map_key"])
    return VisualizationRasterizer(render_context, semantic_map_filepath, world_to_ecef)