"""A script for evaluating closed-loop simulation"""
import argparse
from copy import deepcopy
import numpy as np
import json
import random
import yaml
from collections import defaultdict

import os
import torch

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.rasterization import build_rasterizer
from avdata import AgentType, UnifiedDataset

from tbsim.l5kit.vectorizer import build_vectorizer
from tbsim.algos.l5kit_algos import (
    L5TrafficModel,
    L5VAETrafficModel,
    L5TrafficModelGC,
    SpatialPlanner,
    GANTrafficModel,
    L5DiscreteVAETrafficModel,
    L5ECTrafficModel
)
from tbsim.algos.metric_algos import EBMMetric
from tbsim.utils.batch_utils import set_global_batch_type, batch_utils
from tbsim.algos.multiagent_algos import MATrafficModel
from tbsim.configs.eval_configs import EvaluationConfig
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.envs.env_l5kit import EnvL5KitSimulation
from tbsim.envs.env_avdata import EnvUnifiedSimulation
from tbsim.utils.config_utils import translate_l5kit_cfg, get_experiment_config_from_file, translate_avdata_cfg
from tbsim.utils.env_utils import rollout_episodes
import tbsim.envs.env_metrics as EnvMetrics
from tbsim.policies.hardcoded import ReplayPolicy, GTPolicy, EC_sampling_controller
from tbsim.configs.base import ExperimentConfig

from tbsim.policies.wrappers import (
    PolicyWrapper,
    HierarchicalWrapper,
    HierarchicalSamplerWrapper,
    RolloutWrapper,
    SamplingPolicyWrapper,
    Pos2YawWrapper
)

from tbsim.utils.tensor_utils import map_ndarray
from tbsim.l5kit.l5_ego_dataset import EgoDatasetMixed
from tbsim.utils.experiment_utils import get_checkpoint
from tbsim.utils.vis_utils import build_visualization_rasterizer_l5kit
from imageio import get_writer

try:
    from Pplan.spline_planner import SplinePlanner
    from Pplan.trajectory_tree import TrajTree
except ImportError:
    print("Cannot import Pplan")


class PolicyComposer(object):
    def __init__(self, eval_config, device, ckpt_root_dir="checkpoints/"):
        self.device = device
        self.ckpt_root_dir = ckpt_root_dir
        self.eval_config = eval_config
        self._exp_config = None

    def get_modality_shapes(self, exp_cfg: ExperimentConfig):
        return batch_utils().get_modality_shapes(exp_cfg)

    def get_policy(self):
        raise NotImplementedError


class ReplayAction(PolicyComposer):
    def get_policy(self, **kwargs):
        print("Loading action log from {}".format(self.eval_config.experience_hdf5_path))
        import h5py
        h5 = h5py.File(self.eval_config.experience_hdf5_path, "r")
        if self.eval_config.env == "nusc":
            exp_cfg = get_registered_experiment_config("nusc_rasterized_plan")
        elif self.eval_config.env == "l5kit":
            exp_cfg = get_registered_experiment_config("l5_mixed_plan")
        else:
            raise NotImplementedError("invalid env {}".format(self.eval_config.env))
        return ReplayPolicy(h5, self.device), exp_cfg


class GroundTruth(PolicyComposer):
    def get_policy(self, **kwargs):
        if self.eval_config.env == "nusc":
            exp_cfg = get_registered_experiment_config("nusc_rasterized_plan")
        elif self.eval_config.env == "l5kit":
            exp_cfg = get_registered_experiment_config("l5_mixed_plan")
        else:
            raise NotImplementedError("invalid env {}".format(self.eval_config.env))
        return GTPolicy(device=self.device), exp_cfg


class BC(PolicyComposer):
    def get_policy(self, **kwargs):
        policy_ckpt_path, policy_config_path = get_checkpoint(
            ngc_job_id=self.eval_config.ckpt.policy.ngc_job_id,
            ckpt_key=self.eval_config.ckpt.policy.ckpt_key,
            ckpt_root_dir=self.ckpt_root_dir,
        )
        policy_cfg = get_experiment_config_from_file(policy_config_path)
        policy = L5TrafficModel.load_from_checkpoint(
            policy_ckpt_path,
            algo_config=policy_cfg.algo,
            modality_shapes=self.get_modality_shapes(policy_cfg),
        ).to(self.device).eval()
        return policy, policy_cfg.clone()


class TrafficSim(PolicyComposer):
    def get_policy(self, **kwargs):
        policy_ckpt_path, policy_config_path = get_checkpoint(
            ngc_job_id=self.eval_config.ckpt.policy.ngc_job_id,
            ckpt_key=self.eval_config.ckpt.policy.ckpt_key,
            ckpt_root_dir=self.ckpt_root_dir,
        )
        policy_cfg = get_experiment_config_from_file(policy_config_path)
        policy = L5VAETrafficModel.load_from_checkpoint(
            policy_ckpt_path,
            algo_config=policy_cfg.algo,
            modality_shapes=self.get_modality_shapes(policy_cfg),
        ).to(self.device).eval()
        return policy, policy_cfg.clone()


class TPP(PolicyComposer):
    def get_policy(self, **kwargs):
        policy_ckpt_path, policy_config_path = get_checkpoint(
            ngc_job_id=self.eval_config.ckpt.policy.ngc_job_id,
            ckpt_key=self.eval_config.ckpt.policy.ckpt_key,
            ckpt_root_dir=self.ckpt_root_dir,
        )
        policy_cfg = get_experiment_config_from_file(policy_config_path)
        policy = L5DiscreteVAETrafficModel.load_from_checkpoint(
            policy_ckpt_path,
            algo_config=policy_cfg.algo,
            modality_shapes=self.get_modality_shapes(policy_cfg),
        ).to(self.device).eval()
        policy = PolicyWrapper.wrap_controller(policy, sample=True)
        return policy, policy_cfg.clone()


class GAN(PolicyComposer):
    def get_policy(self, **kwargs):
        policy_ckpt_path, policy_config_path = get_checkpoint(
            ngc_job_id=self.eval_config.ckpt.policy.ngc_job_id,
            ckpt_key=self.eval_config.ckpt.policy.ckpt_key,
            ckpt_root_dir=self.ckpt_root_dir,
        )
        policy_cfg = get_experiment_config_from_file(policy_config_path)
        policy = GANTrafficModel.load_from_checkpoint(
            policy_ckpt_path,
            algo_config=policy_cfg.algo,
            modality_shapes=self.get_modality_shapes(policy_cfg),
        ).to(self.device).eval()
        return policy, policy_cfg.clone()


class Hierarchical(PolicyComposer):
    def _get_planner(self):
        planner_ckpt_path, planner_config_path = get_checkpoint(
            ngc_job_id=self.eval_config.ckpt.planner.ngc_job_id,
            ckpt_key=self.eval_config.ckpt.planner.ckpt_key,
            ckpt_root_dir=self.ckpt_root_dir,
        )
        planner_cfg = get_experiment_config_from_file(planner_config_path)
        planner = SpatialPlanner.load_from_checkpoint(
            planner_ckpt_path,
            algo_config=planner_cfg.algo,
            modality_shapes=self.get_modality_shapes(planner_cfg),
        ).to(self.device).eval()
        return planner, planner_cfg.clone()

    def _get_gt_planner(self):
        return GTPolicy(device=self.device), None

    def _get_gt_controller(self):
        return GTPolicy(device=self.device), None

    def _get_controller(self):
        policy_ckpt_path, policy_config_path = get_checkpoint(
            ngc_job_id=self.eval_config.ckpt.policy.ngc_job_id,
            ckpt_key=self.eval_config.ckpt.policy.ckpt_key,
            ckpt_root_dir=self.ckpt_root_dir,
        )
        policy_cfg = get_experiment_config_from_file(policy_config_path)
        policy_cfg.lock()

        controller = MATrafficModel.load_from_checkpoint(
            policy_ckpt_path,
            algo_config=policy_cfg.algo,
            modality_shapes=self.get_modality_shapes(policy_cfg),
        ).to(self.device).eval()
        return controller, policy_cfg.clone()

    def get_policy(self, **kwargs):
        planner, _ = self._get_planner()
        controller, exp_cfg = self._get_controller()
        planner = PolicyWrapper.wrap_planner(planner, mask_drivable=kwargs.get("mask_drivable"), sample=False)
        policy = HierarchicalWrapper(planner, controller)
        return policy, exp_cfg


class HierAgentAware(Hierarchical):
    def _get_predictor(self):
        predictor_ckpt_path, predictor_config_path = get_checkpoint(
            ngc_job_id=self.eval_config.ckpt.predictor.ngc_job_id,
            ckpt_key=self.eval_config.ckpt.predictor.ckpt_key,
            ckpt_root_dir=self.ckpt_root_dir
        )
        predictor_cfg = get_experiment_config_from_file(predictor_config_path)

        predictor = MATrafficModel.load_from_checkpoint(
            predictor_ckpt_path,
            algo_config=predictor_cfg.algo,
            modality_shapes=self.get_modality_shapes(predictor_cfg),
        ).to(self.device).eval()
        return predictor, predictor_cfg.clone()

    def get_policy(self, **kwargs):
        planner, _ = self._get_planner()
        predictor, exp_cfg = self._get_predictor()
        controller = predictor
        plan_sampler = PolicyWrapper.wrap_planner(
            planner,
            mask_drivable=kwargs.get("mask_drivable"),
            sample=True,
            num_plan_samples=kwargs.get("num_plan_samples"),
            clearance=kwargs.get("diversification_clearance"),
        )
        sampler = HierarchicalSamplerWrapper(plan_sampler, controller)

        policy = SamplingPolicyWrapper(ego_action_sampler=sampler, agent_traj_predictor=predictor)
        return policy, exp_cfg


class HierAgentAwareCVAE(Hierarchical):
    def _get_controller(self):
        controller_ckpt_path, controller_config_path = get_checkpoint(
            ngc_job_id=self.eval_config.ckpt.policy.ngc_job_id,
            ckpt_key=self.eval_config.ckpt.policy.ckpt_key,
            # ngc_job_id="2792906",  # aaplan_dynUnicycle_yrl0.1_roiFalse_gcTrue_rlayerlayer2_rlFalse
            # ckpt_key="iter13000",
            ckpt_root_dir=self.ckpt_root_dir
        )
        controller_cfg = get_experiment_config_from_file(controller_config_path)

        controller = L5DiscreteVAETrafficModel.load_from_checkpoint(
            controller_ckpt_path,
            algo_config=controller_cfg.algo,
            modality_shapes=self.get_modality_shapes(controller_cfg),
        ).to(self.device).eval()
        return controller, controller_cfg.clone()

    def _get_predictor(self):
        predictor_ckpt_path, predictor_config_path = get_checkpoint(
            ngc_job_id=self.eval_config.ckpt.predictor.ngc_job_id,
            ckpt_key=self.eval_config.ckpt.predictor.ckpt_key,
            # ngc_job_id="2732861",  # aaplan_dynUnicycle_yrl0.1_roiFalse_gcTrue_rlayerlayer2_rlFalse
            # ckpt_key="iter20999",
            ckpt_root_dir=self.ckpt_root_dir
        )
        predictor_cfg = get_experiment_config_from_file(predictor_config_path)

        predictor = MATrafficModel.load_from_checkpoint(
            predictor_ckpt_path,
            algo_config=predictor_cfg.algo,
            modality_shapes=self.get_modality_shapes(predictor_cfg),
        ).to(self.device).eval()
        return predictor, predictor_cfg.clone()

    def get_policy(self, **kwargs):
        planner, _ = self._get_planner()
        predictor, _ = self._get_predictor()
        controller, exp_cfg = self._get_controller()
        controller = PolicyWrapper.wrap_controller(
            controller,
            sample=True,
            num_action_samples=kwargs.get("num_action_samples")
        )

        sampler = HierarchicalWrapper(planner, controller)

        policy = SamplingPolicyWrapper(ego_action_sampler=sampler, agent_traj_predictor=predictor)
        return policy, exp_cfg


class AgentAwareEC(Hierarchical):
    def _get_EC_predictor(self):
        EC_ckpt_path, EC_config_path = get_checkpoint(
            ngc_job_id=self.eval_config.ckpt.policy.ngc_job_id,
            ckpt_key=self.eval_config.ckpt.policy.ckpt_key,
            # ngc_job_id="2596419",  # gc_clip_regyaw_dynUnicycle_decmlp128,128_decstateTrue_yrl1.0
            # ckpt_key="iter120999_",
            # ngc_job_id="2783997",  # aaplan_dynUnicycle_yrl0.1_roiFalse_gcTrue_rlayerlayer2_rlFalse
            # ckpt_key="iter33000",
            ckpt_root_dir=self.ckpt_root_dir
        )
        EC_cfg = get_experiment_config_from_file(EC_config_path)

        EC_model = L5ECTrafficModel.load_from_checkpoint(
            EC_ckpt_path,
            algo_config=EC_cfg.algo,
            modality_shapes=self.get_modality_shapes(EC_cfg),
        ).to(self.device).eval()
        return EC_model, EC_cfg.clone()

    def get_policy(self, **kwargs):
        planner, _ = self._get_planner()
        EC_model, exp_cfg = self._get_EC_predictor()
        ego_sampler = SplinePlanner(self.device, N_seg=planner.algo_config.future_num_frames+1)
        agent_planner = planner
        policy = EC_sampling_controller(
            ego_sampler=ego_sampler,EC_model=EC_model, agent_planner=agent_planner, device=self.device)
        return policy, exp_cfg


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
        exp_cfg,
        eval_cfg,
        device,
        skimp_rollout=False,
        compute_metrics=True,
        seed=1
):
    os.environ["L5KIT_DATA_FOLDER"] = eval_cfg.dataset_path

    dm = LocalDataManager(None)
    l5_config = translate_l5kit_cfg(exp_cfg)
    rasterizer = build_rasterizer(l5_config, dm)
    vectorizer = build_vectorizer(l5_config, dm)
    eval_zarr = ChunkedDataset(dm.require(exp_cfg.train.dataset_valid_key)).open()

    env_dataset = EgoDatasetMixed(l5_config, eval_zarr, vectorizer, rasterizer)

    l5_config = deepcopy(l5_config)
    l5_config["raster_params"]["raster_size"] = (500, 500)
    l5_config["raster_params"]["pixel_size"] = (0.2, 0.2)
    l5_config["raster_params"]["ego_center"] = (0.5, 0.5)
    render_rasterizer = build_visualization_rasterizer_l5kit(l5_config, LocalDataManager(None))
    exp_cfg.env.simulation.num_simulation_steps = eval_cfg.num_simulation_steps
    exp_cfg.env.simulation.distance_th_far = 1e+5  # keep controlling everything
    exp_cfg.env.simulation.disable_new_agents = True
    exp_cfg.env.generate_agent_obs = True
    exp_cfg.env.simulation.distance_th_close = 30  # control everything within this bound
    exp_cfg.env.rasterizer.filter_agents_threshold = 0.8  # control everything that's above this confidence threshold

    metrics = dict()
    if compute_metrics:
        modality_shapes = batch_utils().get_modality_shapes(exp_cfg)
        # ckpt_path, cfg_path = get_checkpoint("2761440", "69999_")
        # metric_cfg = get_experiment_config_from_file(cfg_path, locked=True)
        # metric_algo = EBMMetric.load_from_checkpoint(
        #     checkpoint_path=ckpt_path,
        #     algo_config=metric_cfg.algo,
        #     modality_shapes=modality_shapes
        # ).eval().to(device)

        gridinfo = {"offset":np.zeros(2),"step":2.0*np.ones(2)}
        metrics = dict(
            # all_off_road_rate=EnvMetrics.OffRoadRate(),
            # all_collision_rate=EnvMetrics.CollisionRate(),
            # all_occupancy = EnvMetrics.Occupancydistr(gridinfo,sigma=2.0)
            ego_occupancy_diversity = EnvMetrics.OccupancyDiversity(gridinfo,sigma=2.0)
            # all_ebm_score=EnvMetrics.LearnedMetric(metric_algo=metric_algo, perturbations=perturbations),
        )

    env = EnvL5KitSimulation(
        exp_cfg.env,
        dataset=env_dataset,
        seed=seed,
        num_scenes=eval_cfg.num_scenes_per_batch,
        prediction_only=False,
        renderer=render_rasterizer,
        metrics=metrics,
        skimp_rollout=skimp_rollout,
    )

    return env


def create_env_nusc(
        exp_cfg,
        eval_cfg,
        device,
        skimp_rollout=False,
        compute_metrics=True,
        seed=1
):
    exp_cfg.unlock()
    exp_cfg.train.dataset_path = eval_cfg.dataset_path
    exp_cfg.env.simulation.num_simulation_steps = eval_cfg.num_simulation_steps
    exp_cfg.env.simulation.start_frame_index = exp_cfg.algo.history_num_frames + 1
    exp_cfg.lock()

    data_cfg = translate_avdata_cfg(exp_cfg)

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
            "return_rgb": False,
            "offset_frac_xy": data_cfg.raster_center
        },
        num_workers=os.cpu_count(),
        desired_dt=data_cfg.step_time
    )
    print(kwargs)

    env_dataset = UnifiedDataset(**kwargs)

    metrics = None
    if compute_metrics:
        metrics = dict(
            all_off_road_rate=EnvMetrics.OffRoadRate(),
            all_collision_rate=EnvMetrics.CollisionRate(),
        )

    env = EnvUnifiedSimulation(
        exp_cfg.env,
        dataset=env_dataset,
        seed=seed,
        num_scenes=eval_cfg.num_scenes_per_batch,
        prediction_only=False,
        metrics=metrics
    )

    return env


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
    os.makedirs(eval_cfg.ckpt_root_dir, exist_ok=True)
    if save_cfg:
        json.dump(eval_cfg, open(os.path.join(eval_cfg.results_dir, "config.json"), "w+"))
    if data_to_disk and os.path.exists(eval_cfg.experience_hdf5_path):
        os.remove(eval_cfg.experience_hdf5_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create env
    if eval_cfg.env == "nusc":
        env_func = create_env_nusc
        set_global_batch_type("avdata")
    elif eval_cfg.env == 'l5kit':
        env_func = create_env_l5kit
        set_global_batch_type("l5kit")
    else:
        raise NotImplementedError("{} is not a valid env".format(eval_cfg.env))

    # create policy and rollout wrapper
    evaluation = eval(eval_cfg.eval_class)(eval_cfg, device, ckpt_root_dir=eval_cfg.ckpt_root_dir)
    policy, exp_config = evaluation.get_policy(**eval_cfg.policy)

    if eval_cfg.policy.pos_to_yaw:
        policy = Pos2YawWrapper(
            policy,
            dt=exp_config.algo.step_time,
            yaw_correction_speed=eval_cfg.policy.yaw_correction_speed
        )

    if eval_cfg.env == "nusc":
        rollout_policy = RolloutWrapper(agents_policy=policy)
    elif eval_cfg.ego_only:
        rollout_policy = RolloutWrapper(ego_policy=policy)
    else:
        rollout_policy = RolloutWrapper(ego_policy=policy, agents_policy=policy)

    print(exp_config.algo)

    env = env_func(
        exp_config,
        eval_cfg,
        device=device,
        skimp_rollout=skimp_rollout,
        compute_metrics=compute_metrics,
        seed=eval_cfg.seed
    )
    if False:
        env_delayed_start = deepcopy(env)
        ckpt_path, config_path = get_checkpoint(
            ngc_job_id="2780940",  # aaplan_dynUnicycle_yrl0.1_roiFalse_gcTrue_rlayerlayer2_rlFalse
            ckpt_key="iter43000",
            ckpt_root_dir=eval_cfg.ckpt_dir
        )
        modality_shapes = batch_utils().get_modality_shapes(exp_config)
        controller_cfg = get_experiment_config_from_file(config_path)
        CVAE_model = L5DiscreteVAETrafficModel.load_from_checkpoint(
                ckpt_path,
                algo_config=controller_cfg.algo,
                modality_shapes=modality_shapes
            ).to(device).eval()
        perturbations = None
        env_delayed_start._metrics = dict(all_CVAE_score = EnvMetrics.LearnedCVAENLL(metric_algo=CVAE_model, perturbations=perturbations))
    else:
        env_delayed_start = None
    # eval loop
    obs_to_torch = eval_cfg.eval_class not in ["GroundTruth", "ReplayAction"]

    result_stats = None
    scene_i = 0
    eval_scenes = eval_cfg.eval_scenes

    while scene_i < eval_cfg.num_scenes_to_evaluate:
        scene_indices = eval_scenes[scene_i: scene_i + eval_cfg.num_scenes_per_batch]
        scene_i += eval_cfg.num_scenes_per_batch

        stats, info, renderings = rollout_episodes(
            env,
            rollout_policy,
            num_episodes=1,
            n_step_action=eval_cfg.n_step_action,
            render=render_to_video,
            skip_first_n=eval_cfg.skip_first_n,
            scene_indices=scene_indices,
            obs_to_torch=obs_to_torch
        )
        if env_delayed_start is not None:
            Ts = np.random.choice(np.arange(eval_cfg.skip_first_n,200-exp_config.algo.future_num_frames-1),3,replace=False)
            metric_trials = dict()
            for T in Ts:
                stats_trial_i, _, _ = rollout_episodes(
                    env_delayed_start,
                    rollout_policy,
                    num_episodes=1,
                    n_step_action=eval_cfg.n_step_action,
                    render=False,
                    skip_first_n=T,
                    scene_indices=scene_indices,
                    obs_to_torch=obs_to_torch,
                    horizon=T+exp_config.algo.future_num_frames
                )
                for met in stats_trial_i:
                    if met not in ["ego_ADE", "ego_FDE"]:
                        if met not in metric_trials:
                            metric_trials[met]=[]

                        metric_trials[met].append(stats_trial_i[met])
            for met in metric_trials:
                metric_trials[met] = np.stack(metric_trials[met],0).mean(0)
            stats.update(metric_trials)


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
    import h5py
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
        "--env",
        type=str,
        choices=["nusc", "l5kit"],
        help="Which env to run evaluation in"
    )

    parser.add_argument(
        "--ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location of each model",
        default=None
    )

    parser.add_argument(
        "--eval_class",
        type=str,
        default=None,
        help="Optionally specify the evaluation class through argparse"
    )

    parser.add_argument(
        "--ckpt_root_dir",
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

    if args.ckpt_root_dir is not None:
        cfg.ckpt_root_dir = args.ckpt_root_dir

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

    if args.env is not None:
        cfg.env = args.env
    else:
        assert cfg.env is not None

    cfg.experience_hdf5_path = os.path.join(cfg.results_dir, "data.hdf5")

    for k in cfg[cfg.env]:  # copy env-specific config to the global-level
        cfg[k] = cfg[cfg.env][k]
    cfg.pop("nusc")
    cfg.pop("l5kit")

    if args.ckpt_yaml is not None:
        with open(args.ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)

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
        save_cfg=True,
        data_to_disk=data_to_disk,
        skimp_rollout=skimp_rollout,
        compute_metrics=compute_metrics,
        render_to_video=args.render
    )
