import argparse

from collections import OrderedDict
import os
import torch
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.vectorization.vectorizer_builder import build_vectorizer

from tbsim.algos.l5kit_algos import L5TrafficModel, L5VAETrafficModel, L5TrafficModelGC, SpatialPlanner
from tbsim.algos.multiagent_algos import MATrafficModel
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.envs.env_l5kit import EnvL5KitSimulation, BatchedEnv
from tbsim.utils.config_utils import translate_l5kit_cfg, get_experiment_config_from_file
from tbsim.utils.env_utils import rollout_episodes, PolicyWrapper, OptimController, HierarchicalWrapper, GTPlanner, RolloutWrapper
from tbsim.utils.tensor_utils import to_torch, to_numpy
from tbsim.external.l5_ego_dataset import EgoDatasetMixed
from tbsim.utils.experiment_utils import get_checkpoint
from imageio import get_writer


def run_checkpoint(ckpt_dir="checkpoints/", video_dir="videos/"):

    policy_ckpt_path, policy_config_path = get_checkpoint(
        ngc_job_id="2561797", # gc_dynNone_decmlp128,128
        ckpt_key="iter83999_",
        # ngc_job_id="2580486", # gc_dynNone_decmlp128,128
        # ckpt_key="iter33999_ep0_valLoss",
        ckpt_root_dir=ckpt_dir
    )
    policy_cfg = get_experiment_config_from_file(policy_config_path)

    planner_ckpt_path, planner_config_path = get_checkpoint(
        ngc_job_id="2573128",  # spatial_archresnet50_bs64_pcl1.0_pbl0.0_rlFalse
        ckpt_key="iter55999_",
        ckpt_root_dir=ckpt_dir
    )
    planner_cfg = get_experiment_config_from_file(planner_config_path)

    data_cfg = policy_cfg
    assert data_cfg.env.rasterizer.map_type == "py_semantic"
    os.environ["L5KIT_DATA_FOLDER"] = os.path.abspath("/home/danfeix/workspace/lfs/lyft/lyft_prediction/")
    dm = LocalDataManager(None)
    l5_config = translate_l5kit_cfg(data_cfg)
    rasterizer = build_rasterizer(l5_config, dm)
    vectorizer = build_vectorizer(l5_config, dm)
    eval_zarr = ChunkedDataset(dm.require(data_cfg.train.dataset_valid_key)).open()
    env_dataset = EgoDatasetMixed(l5_config, eval_zarr, vectorizer, rasterizer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modality_shapes = OrderedDict(image=[rasterizer.num_channels()] + data_cfg.env.rasterizer.raster_size)

    planner = SpatialPlanner.load_from_checkpoint(
        planner_ckpt_path,
        algo_config=planner_cfg.algo,
        modality_shapes=modality_shapes,
    ).to(device).eval()

    controller = L5TrafficModelGC.load_from_checkpoint(
        policy_ckpt_path,
        algo_config=policy_cfg.algo,
        modality_shapes=modality_shapes
    ).to(device).eval()

    policy = HierarchicalWrapper(planner, controller)

    # policy = MATrafficModel.load_from_checkpoint(
    #     policy_ckpt_path,
    #     algo_config=policy_cfg.algo,
    #     modality_shapes=modality_shapes
    # ).to(device).eval()

    policy = RolloutWrapper(ego_policy=policy, agents_policy=policy)

    data_cfg.env.simulation.num_simulation_steps = 200
    env = EnvL5KitSimulation(
        data_cfg.env,
        dataset=env_dataset,
        seed=data_cfg.seed,
        num_scenes=3,
        prediction_only=False
    )

    stats, info, renderings = rollout_episodes(
        env,
        policy,
        num_episodes=1,
        n_step_action=10,
        render=True,
        # scene_indices=[11, 16, 35, 38, 45, 58, 150, 152, 154, 156],
        scene_indices=[38, 45, 58]
    )

    for i, scene_images in enumerate(renderings[0]):
        writer = get_writer(os.path.join(video_dir, "{}.mp4".format(i)), fps=10)
        for im in scene_images:
            writer.append_data(im)
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--ngc_job_id",
        type=str,
        default=None,
        help="ngc job ID to fetch checkpoint from."
    )

    parser.add_argument(
        "--ckpt_key",
        type=str,
        default=None,
        help="(Optional) partial string that uniquely identifies a checkpoint, e.g., 'iter17999_ep0_posErr'"
    )

    parser.add_argument(
        "--ckpt_root_dir",
        type=str,
        default="../checkpoints/",
        help="(Optional) directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="(Optional) path to a checkpoint file"
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="(Optional) path to a config file"
    )

    args = parser.parse_args()

    run_checkpoint()