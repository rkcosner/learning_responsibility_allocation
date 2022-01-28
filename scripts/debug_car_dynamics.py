"""
A simple script to debug different types of car dynamics
"""
import os
from collections import OrderedDict
import torch
import json
import numpy as np
import tqdm

import torch.optim as optim

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.vectorization.vectorizer_builder import build_vectorizer


from tbsim.external.l5_ego_dataset import EgoDatasetMixed
from tbsim.algos.l5kit_algos import L5TrafficModel
from tbsim.models.l5kit_models import forward_dynamics
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.envs.env_l5kit import EnvL5KitSimulation, BatchedEnv
from tbsim.utils.config_utils import translate_l5kit_cfg, get_experiment_config_from_file
from tbsim.datasets.l5kit_datamodules import L5MixedDataModule
from tbsim.utils.tensor_utils import to_torch, to_device


def test_dynamics(dyn_type):
    cfg = get_registered_experiment_config("l5_mixed_plan")
    cfg.train.dataset_path = "/home/danfeix/workspace/lfs/lyft/lyft_prediction/"
    l5_config = translate_l5kit_cfg(cfg)
    datamodule = L5MixedDataModule(l5_config=l5_config, train_config=cfg.train, mode="ego")
    datamodule.setup()
    dl = datamodule.train_dataloader()

    cfg.algo.dynamics.type = dyn_type
    cfg.train.training.batch_size = 1
    algo = L5TrafficModel(algo_config=cfg.algo, modality_shapes=datamodule.modality_shapes).cuda()
    model = algo.nets["policy"]
    dyn = model.traj_decoder.dyn

    for batch in iter(dl):
        batch = to_device(batch, algo.device)
        with torch.no_grad():
            preds = model.forward(batch)
        pred_controls = preds["controls"]
        curr_states = preds["curr_states"]
        pred_controls.requires_grad = True
        bike_optim = optim.LBFGS([pred_controls], max_iter=20, lr=1.0, line_search_fn='strong_wolfe')

        pbar_optim = tqdm.tqdm(range(50))
        for oidx in pbar_optim:
            def closure():
                bike_optim.zero_grad()

                # get trajectory with current params
                _, pos, yaw = forward_dynamics(
                    dyn, initial_states=curr_states, actions=pred_controls, step_time=cfg.algo.step_time
                )
                preds["trajectories"] = torch.cat((pos, yaw), dim=-1)
                preds["predictions"]["positions"] = pos
                preds["predictions"]["yaws"] = yaw

                # measure error from GT pos and heading
                losses = model.compute_losses(preds, batch)
                loss = losses["goal_loss"]

                # Metrics to log to the tqdm progress bar
                progress_bar_metrics = {}
                # Keep track of metrics over whole epoch
                for k, v in losses.items():
                    progress_bar_metrics[k] = v.item()
                pbar_optim.set_postfix(progress_bar_metrics)

                # backprop
                loss.backward()
                return loss
            bike_optim.step(closure)

        _, final_pos, final_yaw = forward_dynamics(
            dyn, initial_states=curr_states, actions=pred_controls, step_time=cfg.algo.step_time
        )
        final_preds = dict(
            trajectories=torch.cat((final_pos, final_yaw), dim=-1),
            predictions=dict(
                positions=final_pos,
                yaws=final_yaw
            )
        )
        metrics = algo._compute_metrics(pred_batch=final_preds, data_batch=batch)
        print(metrics)

if __name__ == "__main__":
    test_dynamics("Bicycle")
