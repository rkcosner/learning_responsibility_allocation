import numpy as np
import argparse
import torch
import sys
import os
import time
import json
import psutil
import imageio

from torch import nn, optim
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory
from robomimic.utils.log_utils import PrintLogger, DataLogger
from robomimic.algo import RolloutPolicy, PolicyAlgo
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.log_utils as LogUtils

import tbsim.utils.train_utils as TrainUtils
from tbsim.algos.l5kit_algos import L5TrafficModel
from tbsim.configs import ExperimentConfig, L5KitEnvConfig, L5KitTrainConfig, L5RasterizedPlanningConfig



def rollout_scenes(
        policy,
        envs,
        horizon,
        use_goals=False,
        num_episodes=None,
        render=False,
        video_dir=None,
        video_path=None,
        epoch=None,
        video_skip=5,
        terminate_on_success=False,
        verbose=False,
):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        num_episodes (int): number of rollout episodes per environment

        render (bool): if True, render the rollout to the screen

        video_dir (str): if not None, dump rollout videos to this directory (one per environment)

        video_path (str): if not None, dump a single rollout video for all environments

        epoch (int): epoch number (used for video naming)

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        verbose (bool): if True, print results of each rollout

    Returns:
        all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...)
            averaged across all rollouts

        video_paths (dict): path to rollout videos for each environment
    """
    assert isinstance(policy, RolloutPolicy)

    all_rollout_logs = OrderedDict()

    # handle paths and create writers for video writing
    assert (video_path is None) or (video_dir is None), "rollout_with_stats: can't specify both video path and dir"
    write_video = (video_path is not None) or (video_dir is not None)
    video_paths = OrderedDict()
    video_writers = OrderedDict()
    if video_path is not None:
        # a single video is written for all envs
        video_paths = { k : video_path for k in envs }
        video_writer = imageio.get_writer(video_path, fps=20)
        video_writers = { k : video_writer for k in envs }
    if video_dir is not None:
        # video is written per env
        video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4"
        video_paths = { k : os.path.join(video_dir, "{}{}".format(k, video_str)) for k in envs }
        video_writers = { k : imageio.get_writer(video_paths[k], fps=20) for k in envs }

    for env_name, env in envs.items():
        env_video_writer = None
        if write_video:
            print("video writes to " + video_paths[env_name])
            env_video_writer = video_writers[env_name]

        print("rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
            env.name, horizon, use_goals, num_episodes,
        ))
        rollout_logs = []
        iterator = range(num_episodes)
        if not verbose:
            iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)

        num_success = 0
        for ep_i in iterator:
            rollout_timestamp = time.time()
            rollout_info = run_rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                render=render,
                use_goals=use_goals,
                video_writer=env_video_writer,
                video_skip=video_skip,
                terminate_on_success=terminate_on_success,
            )
            rollout_info["time"] = time.time() - rollout_timestamp
            rollout_logs.append(rollout_info)
            num_success += rollout_info["Success_Rate"]
            if verbose:
                print("Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success))
                print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        # average metric across all episodes
        rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
        rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
        rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes
        all_rollout_logs[env_name] = rollout_logs_mean

    if video_path is not None:
        # close video writer that was used for all envs
        video_writer.close()

    return all_rollout_logs, video_paths


def translate_l5kit_cfg(config):
    """

    Args:
        config (BaseConfig): robomimic config

    Returns:
        cfg for l5kit
    """
    rcfg = dict()
    rcfg["raster_params"] = config.env.rasterizer
    rcfg["raster_params"]["dataset_meta_key"] = config.train.dataset_meta_key
    rcfg["model_params"] = config.algo
    return rcfg


def main(config):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(
        exp_name=config.name,
        output_dir=config.root_dir,
        save_checkpoints=config.train.save.enabled,
        auto_remove_exp_dir=True
    )

    if config.train.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation types (e.g. detecting image observations)

    # make sure the dataset exists
    os.environ["L5KIT_DATA_FOLDER"] = os.path.abspath(config.train.dataset_path)
    data_logger = DataLogger(log_dir, log_tb=config.train.logging.log_tb)
    l5_config = translate_l5kit_cfg(config)

    dm = LocalDataManager(None)
    rasterizer = build_rasterizer(l5_config, dm)

    # ===== INIT DATASET
    train_zarr = ChunkedDataset(dm.require(config.train.dataset_train_key)).open()
    trainset = AgentDataset(l5_config, train_zarr, rasterizer)

    train_loader = DataLoader(
        dataset=trainset,
        shuffle=True,
        batch_size=config.train.training.batch_size,
        num_workers=config.train.training.num_data_workers,
        drop_last=True
    )

    valid_zarr = ChunkedDataset(dm.require(config.train.dataset_valid_key)).open()
    validset = AgentDataset(l5_config, valid_zarr, rasterizer)
    valid_loader = DataLoader(
        dataset=validset,
        shuffle=True,
        batch_size=config.train.validation.batch_size,
        num_workers=config.train.validation.num_data_workers,
        drop_last=True
    )

    modality_shapes = OrderedDict(image=(rasterizer.num_channels(), 224, 224))
    model = L5TrafficModel(
        algo_config=config.algo,
        modality_shapes=modality_shapes,
        device=TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    )

    # main training loop
    best_valid_loss = None
    best_return = -np.inf if config.train.rollout.enabled else None
    best_success_rate = -np.inf if config.train.rollout.enabled else None
    last_ckpt_time = time.time()

    train_num_steps = config.train.training.epoch_every_n_steps
    valid_num_steps = config.train.validation.epoch_every_n_steps

    for epoch in range(1, config.train.num_epochs + 1):  # epoch numbers start at 1
        step_log = TrainUtils.run_epoch(model=model, data_loader=train_loader, epoch=epoch, num_steps=train_num_steps)
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.train.save.enabled:
            time_check = (config.train.save.every_n_seconds is not None) and \
                         (time.time() - last_ckpt_time > config.train.save.every_n_seconds)
            epoch_check = (config.train.save.every_n_epochs is not None) and \
                          (epoch > 0) and (epoch % config.train.save.every_n_epochs == 0)
            epoch_list_check = (epoch in config.train.save.epochs)
            should_save_ckpt = (time_check or epoch_check or epoch_list_check)
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

            print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))

        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)

        # Evaluate the model on validation set
        if config.train.validation.enabled:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps)
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Valid/{}".format(k), v, epoch)

            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.train.save.enabled and config.train.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Evaluate the model by by running rollouts

        # do rollouts at fixed rate or if it's time to save a new ckpt
        video_paths = None
        rollout_check = (epoch % config.train.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time")
        if config.train.rollout.enabled and (epoch > config.train.rollout.warmstart) and rollout_check:

            # wrap model as a RolloutPolicy to prepare for rollouts
            rollout_model = RolloutPolicy(model)

            num_episodes = config.train.rollout.n
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                env=env,
                horizon=config.train.rollout.horizon,
                num_episodes=num_episodes,
                render=False,
                video_dir=video_dir if config.train.render_video else None,
                epoch=epoch,
                terminate_on_success=config.train.rollout.terminate_on_success,
            )

            # summarize results from rollouts to tensorboard and terminal
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                    else:
                        data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

                print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                print('Env: {}'.format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            # checkpoint and video saving logic
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.train.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.train.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (config.train.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

        # Only keep saved videos if the ckpt should be saved (but not because of validation score)
        should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.train.keep_all_videos
        if video_paths is not None and not should_save_video:
            for env_name in video_paths:
                os.remove(video_paths[env_name])

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta={},
                shape_meta={},
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth")
            )

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # terminate logging
    data_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset root path"
    )

    args = parser.parse_args()

    config = ExperimentConfig(
        train_config=L5KitTrainConfig(),
        env_config=L5KitEnvConfig(),
        algo_config=L5RasterizedPlanningConfig()
    )

    if args.name is not None:
        config.name = args.name

    if args.dataset_path is not None:
        config.train.dataset_path = args.dataset_path

    config.lock()
    main(config)

