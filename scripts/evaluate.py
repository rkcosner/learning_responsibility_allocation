"""A script for evaluating closed-loop simulation"""
import argparse
from copy import deepcopy
import numpy as np
import json

from collections import OrderedDict
import os
import torch
from torch.utils.data import DataLoader

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer

from tbsim.l5kit.vectorizer import build_vectorizer
from tbsim.algos.l5kit_algos import L5TrafficModel, L5VAETrafficModel, L5TrafficModelGC, SpatialPlanner
from tbsim.algos.multiagent_algos import MATrafficModel
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.configs.eval_configs import EvaluationConfig
from tbsim.envs.env_l5kit import EnvL5KitSimulation, BatchedEnv
from tbsim.utils.config_utils import translate_l5kit_cfg, get_experiment_config_from_file
from tbsim.utils.env_utils import rollout_episodes
from tbsim.policies.wrappers import (
    PolicyWrapper,
    HierarchicalWrapper,
    HierarchicalSamplerWrapper,
    RolloutWrapper,
    SamplingPolicyWrapper
)

from tbsim.utils.tensor_utils import to_torch, to_numpy, map_ndarray
from tbsim.l5kit.l5_ego_dataset import EgoDatasetMixed, EgoReplayBufferMixed, ExperienceIterableWrapper
from tbsim.utils.experiment_utils import get_checkpoint
from tbsim.utils.vis_utils import build_visualization_rasterizer_l5kit
from imageio import get_writer
from tbsim.utils.timer import Timers


class PolicyComposer(object):
    def __init__(self, modality_shapes, device, ckpt_dir="checkpoints/"):
        self.modality_shapes = modality_shapes
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.eval_config = None
        self.policy = None

    def get_policy(self):
        return self.policy


class Hierarchical(PolicyComposer):
    def _get_planner(self):
        planner_ckpt_path, planner_config_path = get_checkpoint(
            ngc_job_id="2573128",  # spatial_archresnet50_bs64_pcl1.0_pbl0.0_rlFalse
            ckpt_key="iter55999_",
            ckpt_root_dir=self.ckpt_dir
        )
        planner_cfg = get_experiment_config_from_file(planner_config_path)
        planner = SpatialPlanner.load_from_checkpoint(
            planner_ckpt_path,
            algo_config=planner_cfg.algo,
            modality_shapes=self.modality_shapes,
        ).to(self.device).eval()
        return planner

    def _get_controller(self):
        policy_ckpt_path, policy_config_path = get_checkpoint(
            # ngc_job_id="2596419",  # gc_clip_regyaw_dynUnicycle_decmlp128,128_decstateTrue_yrl1.0
            # ckpt_key="iter120999_",
            ngc_job_id="2645989",  # aaplan_dynUnicycle_yrl0.1_roiFalse_gcTrue_rlayerlayer2_rlFalse
            ckpt_key="iter92999_",
            ckpt_root_dir=self.ckpt_dir
        )
        policy_cfg = get_experiment_config_from_file(policy_config_path)

        controller = MATrafficModel.load_from_checkpoint(
            policy_ckpt_path,
            algo_config=policy_cfg.algo,
            modality_shapes=self.modality_shapes
        ).to(self.device).eval()
        return controller

    def get_policy(self, **kwargs):
        planner = self._get_planner()
        controller = self._get_controller()
        planner = PolicyWrapper.wrap_planner(planner, mask_drivable=kwargs.get("mask_drivable"), sample=False)
        self.policy = HierarchicalWrapper(planner, controller)
        return self.policy


class HierAgentAware(Hierarchical):
    def __init__(self, modality_shapes, device, ckpt_dir="checkpoints/"):
        super(HierAgentAware, self).__init__(modality_shapes, device, ckpt_dir)

    def _get_predictor(self):
        predictor_ckpt_path, predictor_config_path = get_checkpoint(
            ngc_job_id="2645989",  # aaplan_dynUnicycle_yrl0.1_roiFalse_gcTrue_rlayerlayer2_rlFalse
            ckpt_key="iter92999_",
            ckpt_root_dir=self.ckpt_dir
        )
        predictor_cfg = get_experiment_config_from_file(predictor_config_path)

        predictor = MATrafficModel.load_from_checkpoint(
            predictor_ckpt_path,
            algo_config=predictor_cfg.algo,
            modality_shapes=self.modality_shapes
        ).to(self.device).eval()
        return predictor

    def get_policy(self, **kwargs):
        planner = self._get_planner()
        predictor = self._get_predictor()
        controller = predictor
        plan_sampler = PolicyWrapper.wrap_planner(
            planner,
            mask_drivable=kwargs.get("mask_drivable"),
            sample=True,
            num_plan_samples=kwargs.get("num_plan_samples")
        )
        sampler = HierarchicalSamplerWrapper(plan_sampler, controller)

        self.policy = SamplingPolicyWrapper(ego_action_sampler=sampler, agent_traj_predictor=predictor)
        return self.policy


def create_env(sim_cfg, num_scenes_per_batch, num_simulation_steps=200, skimp_rollout=False, seed=1):

    dm = LocalDataManager(None)
    l5_config = translate_l5kit_cfg(sim_cfg)
    rasterizer = build_rasterizer(l5_config, dm)
    vectorizer = build_vectorizer(l5_config, dm)
    eval_zarr = ChunkedDataset(dm.require(sim_cfg.train.dataset_valid_key)).open()
    env_dataset = EgoDatasetMixed(l5_config, eval_zarr, vectorizer, rasterizer)
    modality_shapes = OrderedDict(image=[rasterizer.num_channels()] + list(sim_cfg.env.rasterizer.raster_size))

    l5_config = deepcopy(l5_config)
    l5_config["raster_params"]["raster_size"] = (1000, 1000)
    l5_config["raster_params"]["pixel_size"] = (0.1, 0.1)
    render_rasterizer = build_visualization_rasterizer_l5kit(l5_config, LocalDataManager(None))
    sim_cfg.env.simulation.num_simulation_steps = num_simulation_steps
    sim_cfg.env.simulation.distance_th_far = 1e+5
    sim_cfg.env.simulation.disable_new_agents = True
    sim_cfg.env.generate_agent_obs = True

    env = EnvL5KitSimulation(
        sim_cfg.env,
        dataset=env_dataset,
        seed=seed,
        num_scenes=num_scenes_per_batch,
        prediction_only=False,
        renderer=render_rasterizer,
        compute_metrics=True,
        skimp_rollout=skimp_rollout
    )
    return env, modality_shapes


def dump_episode_buffer(buffer, scene_index, h5_path):
    import h5py
    h5_file = h5py.File(h5_path, "a")

    for si, scene_buffer in zip(scene_index, buffer):
        for mk in scene_buffer:
            for k in scene_buffer[mk]:
                h5key = "/{}/{}/{}".format(si, mk, k)
                h5_file.create_dataset(h5key, data=scene_buffer[mk][k])
    h5_file.close()
    print("scene {} written to {}".format(scene_index, h5_path))


def run_evaluation(eval_cfg):
    print(eval_cfg)

    # for reproducibility
    torch.manual_seed(eval_cfg.seed)
    np.random.seed(eval_cfg.seed)

    print('save results to {}'.format(eval_cfg.results_dir))
    os.makedirs(eval_cfg.results_dir, exist_ok=True)
    os.makedirs(os.path.join(eval_cfg.results_dir, "videos/"), exist_ok=True)
    os.makedirs(eval_cfg.ckpt_dir, exist_ok=True)

    os.environ["L5KIT_DATA_FOLDER"] = eval_cfg.dataset_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sim_cfg = get_registered_experiment_config("l5_mixed_plan")

    env, modality_shapes = create_env(
        sim_cfg,
        num_scenes_per_batch=eval_cfg.num_scenes_per_batch,
        num_simulation_steps=eval_cfg.num_simulation_steps,
        seed=eval_cfg.seed
    )

    evaluation = eval(eval_cfg.eval_class)(modality_shapes, device, ckpt_dir=eval_cfg.ckpt_dir)
    policy = evaluation.get_policy(**eval_cfg.policy)

    if eval_cfg.ego_only:
        rollout_policy = RolloutWrapper(ego_policy=policy)
    else:
        rollout_policy = RolloutWrapper(ego_policy=policy, agents_policy=policy)

    npr = np.random.RandomState(seed=0)  # TODO: maybe put indices into a file?
    eval_scenes = npr.choice(
        np.arange(env.total_num_scenes),
        size=eval_cfg.num_scenes_to_evaluate,
        replace=False
    )

    iter_i = 0

    result_stats = None
    while iter_i < eval_cfg.num_scenes_to_evaluate:
        stats, info, renderings = rollout_episodes(
            env,
            rollout_policy,
            num_episodes=1,
            n_step_action=eval_cfg.n_step_action,
            render=eval_cfg.render_to_video,
            skip_first_n=1,
            scene_indices=eval_scenes[iter_i: iter_i + eval_cfg.num_scenes_per_batch],
        )
        iter_i += eval_cfg.num_scenes_per_batch

        print(info["scene_index"])
        print(stats)

        if result_stats is None:
            result_stats = stats
            result_stats["scene_index"] = np.array(info["scene_index"])
        else:
            for k in stats:
                result_stats[k] = np.concatenate([result_stats[k], stats[k]], axis=0)
            result_stats["scene_index"] = np.concatenate([result_stats["scene_index"], np.array(info["scene_index"])])

        with open(os.path.join(eval_cfg.results_dir, "stats.json"), "w+") as fp:
            stats_to_write = map_ndarray(result_stats, lambda x: x.tolist())
            json.dump(stats_to_write, fp)

        if eval_cfg.render_to_video:
            for i, scene_images in enumerate(renderings[0]):
                video_dir = os.path.join(eval_cfg.results_dir, "videos/")
                writer = get_writer(os.path.join(
                    video_dir, "{}.mp4".format(info["scene_index"][i])), fps=10)
                for im in scene_images:
                    writer.append_data(im)
                writer.close()

        if eval_cfg.data_to_disk:
            dump_episode_buffer(
                info["buffer"],
                info["scene_index"],
                h5_path=os.path.join(eval_cfg.results_dir, "data.hdf5")
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="A json file containing evaluation configs"
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory to save results and videos"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Root directory of the dataset"
    )

    parser.add_argument(
        "--num_scenes_per_batch",
        type=int,
        default=None,
        help="Number of scenes to run concurrently (to accelerate eval)"
    )

    parser.add_argument(
        "--num_scenes_to_evaluate",
        type=int,
        default=None,
        help="Number of scenes to run in total"
    )

    parser.add_argument(
        "--render",
        action="store_true",
        default=None,
        help="whether to render videos"
    )

    args = parser.parse_args()

    cfg = EvaluationConfig()

    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)

    if args.ckpt_dir is not None:
        cfg.ckpt_dir = args.ckpt_dir

    if args.num_scenes_per_batch is not None:
        cfg.num_scenes_per_batch = args.num_scenes_per_batch

    if args.num_scenes_to_evaluate is not None:
        cfg.num_scenes_to_evaluate = args.num_scenes_to_evaluate

    if args.render is not None:
        cfg.render_to_video = args.render

    if args.results_dir is not None:
        cfg.results_dir = args.results_dir

    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path

    cfg.results_dir = os.path.join(cfg.results_dir, cfg.name)

    cfg.lock()
    run_evaluation(cfg)

