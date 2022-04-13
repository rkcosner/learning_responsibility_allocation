"""A script for evaluating closed-loop simulation"""
import argparse
import shutil
from copy import deepcopy
import numpy as np
import json
import h5py
import random
from collections import defaultdict

from collections import OrderedDict
import os
import torch
from torch.utils.data import DataLoader

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.kinematic import Perturbation

from tbsim.l5kit.vectorizer import build_vectorizer
from tbsim.algos.l5kit_algos import (
    L5TrafficModel,
    L5VAETrafficModel,
    L5TrafficModelGC,
    SpatialPlanner,
    GANTrafficModel,
    L5DiscreteVAETrafficModel
)
from tbsim.algos.metric_algos import EBMMetric
from tbsim.algos.multiagent_algos import MATrafficModel
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.configs.eval_configs import EvaluationConfig
from tbsim.envs.env_l5kit import EnvL5KitSimulation, BatchedEnv
from tbsim.envs.env_avdata import EnvUnifiedSimulation
from tbsim.utils.config_utils import translate_l5kit_cfg, get_experiment_config_from_file, translate_avdata_cfg
from tbsim.utils.env_utils import rollout_episodes
import tbsim.envs.env_metrics as EnvMetrics
from tbsim.policies.hardcoded import ReplayPolicy, GTPolicy
from avdata import AgentBatch, AgentType, UnifiedDataset

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
    def __init__(self, eval_config, modality_shapes, device, ckpt_dir="checkpoints/"):
        self.modality_shapes = modality_shapes
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.eval_config = eval_config

    def get_policy(self):
        raise NotImplementedError


class ReplayAction(PolicyComposer):
    def get_policy(self, **kwargs):
        print("Loading action log from {}".format(self.eval_config.experience_hdf5_path))
        h5 = h5py.File(self.eval_config.experience_hdf5_path, "r")
        return ReplayPolicy(h5, self.device)


class GroundTruth(PolicyComposer):
    def get_policy(self, **kwargs):
        return GTPolicy(device=self.device)


class BC(PolicyComposer):
    def get_policy(self, **kwargs):
        policy_ckpt_path, policy_config_path = get_checkpoint(
            ngc_job_id="2775586",
            ckpt_key="iter73000",
            ckpt_root_dir=self.ckpt_dir,
        )
        policy_cfg = get_experiment_config_from_file(policy_config_path)
        policy = L5TrafficModel.load_from_checkpoint(
            policy_ckpt_path,
            algo_config=policy_cfg.algo,
            modality_shapes=self.modality_shapes,
        ).to(self.device).eval()
        return policy


class TrafficSim(PolicyComposer):
    def get_policy(self, **kwargs):
        policy_ckpt_path, policy_config_path = get_checkpoint(
            ngc_job_id="",
            ckpt_key="",
            ckpt_root_dir=self.ckpt_dir,
        )
        policy_cfg = get_experiment_config_from_file(policy_config_path)
        policy = L5VAETrafficModel.load_from_checkpoint(
            policy_ckpt_path,
            algo_config=policy_cfg.algo,
            modality_shapes=self.modality_shapes,
        ).to(self.device).eval()
        return policy


class TPP(PolicyComposer):
    def get_policy(self, **kwargs):
        policy_ckpt_path, policy_config_path = get_checkpoint(
            ngc_job_id="",
            ckpt_key="",
            ckpt_root_dir=self.ckpt_dir,
        )
        policy_cfg = get_experiment_config_from_file(policy_config_path)
        policy = L5DiscreteVAETrafficModel.load_from_checkpoint(
            policy_ckpt_path,
            algo_config=policy_cfg.algo,
            modality_shapes=self.modality_shapes,
        ).to(self.device).eval()
        policy = PolicyWrapper.wrap_controller(policy, sample=True)
        return policy


class GAN(PolicyComposer):
    def get_policy(self, **kwargs):
        policy_ckpt_path = "/home/danfeix/workspace/tbsim/gan_trained_models/test/run0/checkpoints/iter11999_ep0_egoADE1.41.ckpt"
        policy_config_path = "/home/danfeix/workspace/tbsim/gan_trained_models/test/run0/config.json"
        policy_cfg = get_experiment_config_from_file(policy_config_path)
        policy = GANTrafficModel.load_from_checkpoint(
            policy_ckpt_path,
            algo_config=policy_cfg.algo,
            modality_shapes=self.modality_shapes,
        ).to(self.device).eval()
        return policy


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
            ngc_job_id="2732861",  # aaplan_dynUnicycle_yrl0.1_roiFalse_gcTrue_rlayerlayer2_rlFalse
            ckpt_key="iter20999",
            ckpt_root_dir=self.ckpt_dir
        )
        policy_cfg = get_experiment_config_from_file(policy_config_path)
        policy_cfg.lock()
        assert policy_cfg.algo.goal_conditional

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
    def _get_predictor(self):
        predictor_ckpt_path, predictor_config_path = get_checkpoint(
            ngc_job_id="2732861",  # aaplan_dynUnicycle_yrl0.1_roiFalse_gcTrue_rlayerlayer2_rlFalse
            ckpt_key="iter20999",
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
            num_plan_samples=kwargs.get("num_plan_samples"),
            clearance=3,
        )
        sampler = HierarchicalSamplerWrapper(plan_sampler, controller)

        self.policy = SamplingPolicyWrapper(ego_action_sampler=sampler, agent_traj_predictor=predictor)
        return self.policy


class RandomPerturbation(object):
    def __init__(self, std: np.ndarray):
        assert std.shape == (3,) and np.all(std >= 0)
        self.std = std

    def perturb(self, obs):
        obs = dict(obs)
        target_traj = np.concatenate((obs["target_positions"], obs["target_yaws"]), axis=-1)
        std = np.ones_like(target_traj) * self.std[None, :]
        noise = np.random.normal(np.zeros_like(target_traj), std)
        target_traj += noise
        obs["target_positions"] = target_traj[..., :2]
        obs["target_yaws"] = target_traj[..., :1]
        return obs


def create_env_l5kit(
        eval_cfg,
        device,
        skimp_rollout=False,
        compute_metrics=True,
        seed=1
):
    os.environ["L5KIT_DATA_FOLDER"] = eval_cfg.dataset_path
    sim_cfg = get_registered_experiment_config("l5_mixed_plan")
    # sim_cfg.algo.step_time = 0.2
    # sim_cfg.algo.future_num_frames = 20

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
    l5_config["raster_params"]["ego_center"] = (0.5, 0.5)
    render_rasterizer = build_visualization_rasterizer_l5kit(l5_config, LocalDataManager(None))
    sim_cfg.env.simulation.distance_th_far = 1e+5  # keep controlling everything
    sim_cfg.env.simulation.disable_new_agents = True
    sim_cfg.env.generate_agent_obs = True
    sim_cfg.env.simulation.distance_th_close = 30  # control everything within this bound
    sim_cfg.env.rasterizer.filter_agents_threshold = 0.8  # control everything that's above this confidence threshold

    metrics = dict()
    if compute_metrics:
        ckpt_path, cfg_path = get_checkpoint("2761440", "69999_")
        metric_cfg = get_experiment_config_from_file(cfg_path, locked=True)
        metric_algo = EBMMetric.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            algo_config=metric_cfg.algo,
            modality_shapes=modality_shapes
        ).eval().to(device)

        base_noise = np.array(eval_cfg.perturb.std)
        perturbations = dict()
        # perturbations = {
        #     "p=0.0": RandomPerturbation(std=base_noise * 0.0),
        #     "p=0.1": RandomPerturbation(std=base_noise * 0.1),
        #     "p=0.2": RandomPerturbation(std=base_noise * 0.2),
        #     "p=0.5": RandomPerturbation(std=base_noise * 0.5),
        #     "p=1.0": RandomPerturbation(std=base_noise * 1.0)
        # }

        metrics = dict(
            all_off_road_rate=EnvMetrics.OffRoadRate(),
            all_collision_rate=EnvMetrics.CollisionRate(),
            all_ebm_score=EnvMetrics.LearnedMetric(metric_algo=metric_algo, perturbations=perturbations)
        )

    env = EnvL5KitSimulation(
        sim_cfg.env,
        dataset=env_dataset,
        seed=seed,
        num_scenes=eval_cfg.num_scenes_per_batch,
        prediction_only=False,
        renderer=render_rasterizer,
        metrics=metrics,
        skimp_rollout=skimp_rollout,
    )

    eval_scenes = [9058, 5232, 14153, 8173, 10314, 7027, 9812, 1090, 9453, 978, 10263, 874, 5563, 9613, 261, 2826, 2175, 9977, 6423, 1069, 1836, 8198, 5034, 6016, 2525, 927, 3634, 11806, 4911, 6192, 11641, 461, 142, 15493, 4919, 8494, 14572, 2402, 308, 1952, 13287, 15614, 6529, 12, 11543, 4558, 489, 6876, 15279, 6095, 5877, 8928, 10599, 16150, 11296, 9382, 13352, 1794, 16122, 12429, 15321, 8614, 12447, 4502, 13235, 2919, 15893, 12960, 7043, 9278, 952, 4699, 768, 13146, 8827, 16212, 10777, 15885, 11319, 9417, 14092, 14873, 6740, 11847, 15331, 15639, 11361, 14784, 13448, 10124, 4872, 3567, 5543, 2214, 7624, 10193, 7297, 1308, 3951, 14001]
    return env, modality_shapes, eval_scenes, sim_cfg


def create_env_nusc(
        eval_cfg,
        device,
        skimp_rollout=False,
        compute_metrics=True,
        seed=1
):
    sim_cfg = get_registered_experiment_config("nusc_rasterized_plan")
    sim_cfg.algo.step_time = 0.5
    sim_cfg.algo.future_num_frames = 10
    sim_cfg.train.dataset_path = eval_cfg.dataset_path
    sim_cfg.env.simulation.num_simulation_steps = eval_cfg.num_simulation_steps
    sim_cfg.env.simulation.start_frame_index = sim_cfg.algo.history_num_frames + 1
    sim_cfg.lock()

    data_cfg = translate_avdata_cfg(sim_cfg)

    future_sec = data_cfg.future_num_frames * data_cfg.step_time
    history_sec = data_cfg.history_num_frames * data_cfg.step_time
    neighbor_distance = data_cfg.max_agents_distance
    kwargs = dict(
        desired_data=["nusc-val"],
        future_sec=(future_sec, future_sec),
        history_sec=(history_sec, history_sec),
        data_dirs={
            "nusc": data_cfg.dataset_path,
            "nusc_mini": data_cfg.dataset_path,
        },
        only_types=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: neighbor_distance),
        incl_map=True,
        map_params={
            "px_per_m": int(1 / data_cfg.pixel_size),
            "map_size_px": data_cfg.raster_size,
        },
        verbose=True,
        num_workers=os.cpu_count(),
    )
    print(kwargs)

    env_dataset = UnifiedDataset(**kwargs)

    env = EnvUnifiedSimulation(
        sim_cfg.env,
        dataset=env_dataset,
        seed=seed,
        num_scenes=eval_cfg.num_scenes_per_batch,
        prediction_only=False
    )

    modality_shapes = dict(image=(7 + data_cfg.history_num_frames + 1, data_cfg.pixel_size, data_cfg.pixel_size))
    eval_scenes = [0, 1, 2, 3, 4]

    return env, modality_shapes, eval_scenes, sim_cfg


def run_evaluation(eval_cfg, save_cfg, skimp_rollout, compute_metrics, data_to_disk, render_to_video):
    print(eval_cfg)

    # for reproducibility
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(eval_cfg.seed)
    torch.cuda.manual_seed(eval_cfg.seed)

    # basic setup
    print('saving results to {}'.format(eval_cfg.results_dir))
    os.makedirs(eval_cfg.results_dir, exist_ok=True)
    os.makedirs(os.path.join(eval_cfg.results_dir, "videos/"), exist_ok=True)
    os.makedirs(eval_cfg.ckpt_dir, exist_ok=True)
    if save_cfg:
        json.dump(eval_cfg, open(os.path.join(eval_cfg.results_dir, "config.json"), "w+"))
    if data_to_disk and os.path.exists(eval_cfg.experience_hdf5_path):
        os.remove(eval_cfg.experience_hdf5_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create env
    if eval_cfg.env == "nusc":
        env_func = create_env_nusc
    elif eval_cfg.env == 'l5kit':
        env_func = create_env_l5kit
    else:
        raise NotImplementedError("{} is not a valid env".format(eval_cfg.env))

    env, modality_shapes, eval_scenes, sim_cfg = env_func(
        eval_cfg,
        device=device,
        skimp_rollout=skimp_rollout,
        compute_metrics=compute_metrics,
        seed=eval_cfg.seed
    )

    # create policy and rollout wrapper
    evaluation = eval(eval_cfg.eval_class)(eval_cfg, modality_shapes, device, ckpt_dir=eval_cfg.ckpt_dir)
    policy = evaluation.get_policy(**eval_cfg.policy)

    if eval_cfg.env == "nusc":
        rollout_policy = RolloutWrapper(agents_policy=policy)
    elif eval_cfg.ego_only:
        rollout_policy = RolloutWrapper(ego_policy=policy)
    else:
        rollout_policy = RolloutWrapper(ego_policy=policy, agents_policy=policy)

    # eval loop
    obs_to_torch = eval_cfg.eval_class not in ["GroundTruth", "ReplayAction"]

    result_stats = None
    scene_i = 0
    parallel_count = 0

    while scene_i < eval_cfg.num_scenes_to_evaluate:
        scene_indices = eval_scenes[scene_i: scene_i + eval_cfg.num_scenes_per_batch]
        if eval_cfg.parallel_simulation:
            parallel_count += 1
            if parallel_count == eval_cfg.num_parallel:
                parallel_count = 0
                scene_i += eval_cfg.num_scenes_per_batch
        else:
            scene_i += eval_cfg.num_scenes_per_batch

        stats, info, renderings = rollout_episodes(
            env,
            rollout_policy,
            num_episodes=1,
            n_step_action=eval_cfg.n_step_action,
            render=render_to_video,
            skip_first_n=eval_cfg.skip_first_n if eval_cfg.env != "nusc" else 0,
            scene_indices=scene_indices,
            obs_to_torch=obs_to_torch
        )

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

        if render_to_video:
            for i, scene_images in enumerate(renderings[0]):
                video_dir = os.path.join(eval_cfg.results_dir, "videos/")
                writer = get_writer(os.path.join(
                    video_dir, "{}.mp4".format(info["scene_index"][i])), fps=10)
                print("video to {}".format(os.path.join(
                    video_dir, "{}.mp4".format(info["scene_index"][i]))))
                for im in scene_images:
                    writer.append_data(im)
                writer.close()

        if data_to_disk:
            dump_episode_buffer(
                info["buffer"],
                info["scene_index"],
                h5_path=eval_cfg.experience_hdf5_path
            )


def dump_episode_buffer(buffer, scene_index, h5_path):
    h5_file = h5py.File(h5_path, "a")

    for si, scene_buffer in zip(scene_index, buffer):
        for mk in scene_buffer:
            for k in scene_buffer[mk]:
                h5key = "/{}/{}/{}".format(si, mk, k)
                h5_file.create_dataset(h5key, data=scene_buffer[mk][k])
    h5_file.close()
    print("scene {} written to {}".format(scene_index, h5_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="A json file containing evaluation configs"
    )

    parser.add_argument(
        "--eval_class",
        type=str,
        default=None,
        help="Optionally specify the evaluation class through argparse"
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--results_root_dir",
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
        "--render",
        action="store_true",
        default=False,
        help="whether to render videos"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="evaluate_rollout",
        choices=["record_rollout", "evaluate_replay", "evaluate_rollout"],
        required=True
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    cfg = EvaluationConfig()

    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)

    if args.eval_class is not None:
        cfg.eval_class = args.eval_class

    if args.ckpt_dir is not None:
        cfg.ckpt_dir = args.ckpt_dir

    if args.num_scenes_per_batch is not None:
        cfg.num_scenes_per_batch = args.num_scenes_per_batch

    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path

    if cfg.name is None:
        cfg.name = cfg.eval_class

    if args.prefix is not None:
        cfg.name = args.prefix + cfg.name

    if args.seed is not None:
        cfg.seed = args.seed

    if args.results_root_dir is not None:
        cfg.results_dir = os.path.join(args.results_root_dir, cfg.name)
    cfg.experience_hdf5_path = os.path.join(cfg.results_dir, "data.hdf5")

    data_to_disk = False
    skimp_rollout = False
    compute_metrics = False

    if args.mode == "record_rollout":
        data_to_disk = True
        skimp_rollout = True
        compute_metrics = False
    elif args.mode == "evaluate_replay":
        cfg.eval_class = "ReplayAction"
        cfg.n_step_action = 1
        data_to_disk = False
        skimp_rollout = False
        compute_metrics = True
    elif args.mode == "evaluate_rollout":
        data_to_disk = False
        skimp_rollout = False
        compute_metrics = True
    # compute_metrics = False
    cfg.lock()
    run_evaluation(
        cfg,
        save_cfg=args.mode == "record_rollout",
        data_to_disk=data_to_disk,
        skimp_rollout=skimp_rollout,
        compute_metrics=compute_metrics,
        render_to_video=args.render
    )
