import os
from collections import OrderedDict
import torch
import json
import numpy as np

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.vectorization.vectorizer_builder import build_vectorizer


from tbsim.external.l5_ego_dataset import EgoDatasetMixed
from tbsim.algos.l5kit_algos import L5TrafficModel
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.envs.env_l5kit import EnvL5KitSimulation, BatchedEnv
from tbsim.utils.config_utils import translate_l5kit_cfg, get_experiment_config_from_file
from tbsim.utils.env_utils import rollout_episodes, RolloutWrapper
from tbsim.utils.tensor_utils import to_torch

def rollout(env, policy, scene_indices = (), num_episodes=1):
    stats = {}
    info = {}
    is_batched_env = isinstance(env, BatchedEnv)

    for ei in range(num_episodes):
        env.reset(scene_indices=scene_indices)

        done = env.is_done()
        while not done:
            obs = env.get_observation()
            obs = to_torch(obs, device=policy.device)
            print('step')

            action = policy.get_action(obs, sample=False)
            # action = dict(
            #     ego=dict(
            #         positions=obs["ego"]["target_positions"],
            #         yaws=obs["ego"]["target_yaws"]
            #     )
            # )
            env.step(action, num_steps_to_take=10)
            done = env.is_done()

        metrics = env.get_metrics()
        for k, v in metrics.items():
            if k not in stats:
                stats[k] = []
            if is_batched_env:
                stats[k] = np.concatenate([stats[k], v], axis=0)
            else:
                stats[k].append(v)

        env_info = env.get_info()
        for k, v in env_info.items():
            if k not in info:
                info[k] = []
            if is_batched_env:
                info[k].extend(v)
            else:
                info[k].append(v)

    return stats, info


def main():
    # set env variable for data

    os.environ["L5KIT_DATA_FOLDER"] = os.path.abspath("/home/danfeix/workspace/lfs/lyft/lyft_prediction/")
    # config_path = "/home/danfeix/workspace/tbsim/checkpoints/rasterized_currspeed_dynUnicycle/run0/config.json"
    # ckpt_path = "/home/danfeix/workspace/tbsim/checkpoints/rasterized_currspeed_dynUnicycle/run0/checkpoints/iter43999_ep1_valLoss0.57.ckpt"
    model_class = L5TrafficModel
    # cfg = get_experiment_config_from_file(config_path)
    # cfg = get_experiment_config_from_file("experiments/mapfd/rasterized_nd_mapfd128.json")
    cfg = get_registered_experiment_config("l5_mixed_plan")
    cfg.algo.dynamics.type = "Unicycle"
    dm = LocalDataManager(None)
    l5_config = translate_l5kit_cfg(cfg)
    rasterizer = build_rasterizer(l5_config, dm)
    vectorizer = build_vectorizer(l5_config, dm)
    modality_shapes = OrderedDict(image=(rasterizer.num_channels(), 224, 224))

    eval_zarr = ChunkedDataset(dm.require(cfg.train.dataset_valid_key)).open()
    raster_dataset = EgoDataset(l5_config, eval_zarr, rasterizer)
    # mixed_dataset = EgoDatasetMixed(l5_config, eval_zarr, vectorizer, rasterizer)

    raster_env = EnvL5KitSimulation(cfg.env, dataset=raster_dataset, seed=cfg.seed, num_scenes=1)
    # mixed_env = EnvL5KitSimulation(cfg.env, dataset=mixed_dataset, seed=cfg.seed, num_scenes=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_class(algo_config=cfg.algo, modality_shapes=modality_shapes)
    # model = model_class.load_from_checkpoint(
    #     checkpoint_path=ckpt_path,
    #     algo_config=cfg.algo,
    #     modality_shapes=modality_shapes,
    # )
    model.to(device)

    stats, info = rollout(raster_env, policy=model, scene_indices=(10,))
    print(stats)


if __name__ == "__main__":
    main()