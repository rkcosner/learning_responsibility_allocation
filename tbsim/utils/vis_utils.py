import numpy as np
from l5kit.geometry import transform_points
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from tbsim.utils.tensor_utils import map_ndarray
from tbsim.utils.l5_utils import get_last_available_index


def agent_to_raster_np(pt_tensor, trans_mat):
    pos_raster = transform_points(pt_tensor[None], trans_mat)[0]
    return pos_raster


def _render_state(state_image, trans_mat, title="", pred_actions=None, gt_actions=None, pred_plan=None, gt_plan=None, pred_plan_info=None):
    fig = Figure(figsize=(10, 8), dpi=100)
    canvas = FigureCanvasAgg(fig)

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
        # if "predictions_no_res" in pred_plan:
        #     pos_raster = agent_to_raster_np(pred_plan["predictions_no_res"]["positions"], trans_mat)[[-1]]
        #     ax.scatter(pos_raster[:, 0], pos_raster[:, 1], marker='*', color="red")

    if gt_plan is not None:
        pos_raster = agent_to_raster_np(gt_plan["positions"], trans_mat)[[-1]]
        ax.scatter(pos_raster[:, 0], pos_raster[:, 1], marker='o', color="blue")

    # visualize plan heat map
    if "location_map" in pred_plan_info:
        ax = fig.add_subplot(122)
        ax.imshow(pred_plan_info["location_map"])

    canvas.draw()
    im = np.asarray(canvas.buffer_rgba())[:, :, :3]
    return im


def render_state_l5kit(rasterizer, state_obs, action, scene_index, step_index, dataset_scene_index):
    state_obs = state_obs["ego"]
    state_im = rasterizer.to_rgb(state_obs["image"][scene_index].transpose(1, 2, 0))
    slice_idx = lambda x:  x[scene_index]
    pred_actions = map_ndarray(action.ego.to_dict(), slice_idx)
    gt_actions = dict(
        positions=state_obs["target_positions"],
        yaws=state_obs["target_yaws"]
    )

    gt_actions = map_ndarray(gt_actions, slice_idx)
    trans_mat = state_obs["raster_from_agent"][scene_index]
    pred_plan = action.ego_info.get("plan", None)
    pred_plan_info = action.ego_info.get("plan_info", None)
    if pred_plan is not None:
        pred_plan = map_ndarray(pred_plan, slice_idx)
        pred_plan_info = map_ndarray(pred_plan_info, slice_idx)

    im = _render_state(
        state_image=state_im,
        trans_mat=trans_mat,
        pred_actions=pred_actions,
        gt_actions=gt_actions,
        pred_plan=pred_plan,
        pred_plan_info=pred_plan_info,
        title="scene={}, step={}".format(dataset_scene_index, step_index)
    )

    return im