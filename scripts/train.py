import numpy as np
import argparse
import torch
import sys
import os
import time
import json
import psutil
from typing import Dict, List, Optional

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
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import tbsim.utils.train_utils as TrainUtils
from tbsim.algos.l5kit_algos import L5TrafficModel, L5TrafficModelPL
from tbsim.configs import ExperimentConfig, L5KitEnvConfig, L5KitTrainConfig, L5RasterizedPlanningConfig



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


class L5DataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset_train_key,
            dataset_valid_key,
            train_batch_size,
            valid_batch_size,
            train_n_workers,
            valid_n_workers,
            l5_config,
            data_manager,
            rasterizer,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.train_n_workers = train_n_workers
        self.valid_n_workers = valid_n_workers
        self.dataset_train_key = dataset_train_key
        self.dataset_valid_key = dataset_valid_key
        self.dm = data_manager
        self.l5_config = l5_config
        self.rasterizer = rasterizer
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.train_set = None
        self.valid_set = None

    def setup(self, stage: Optional[str] = None) -> None:
        train_zarr = ChunkedDataset(self.dm.require(self.dataset_train_key)).open()
        self.train_set = AgentDataset(self.l5_config, train_zarr, self.rasterizer)
        valid_zarr = ChunkedDataset(self.dm.require(self.dataset_valid_key)).open()
        self.valid_set = AgentDataset(self.l5_config, valid_zarr, self.rasterizer)

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_set,
            shuffle=True,
            batch_size=self.train_batch_size,
            num_workers=self.train_n_workers,
            drop_last=True
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            dataset=self.valid_set,
            shuffle=True,
            batch_size=self.valid_batch_size,
            num_workers=self.valid_n_workers,
            drop_last=True
        )
        return val_loader


def main_pl(config):
    pl.seed_everything(config.seed)
    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    root_dir, log_dir, ckpt_dir, video_dir, version_key = TrainUtils.get_exp_dir(
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

    # make sure the dataset exists
    os.environ["L5KIT_DATA_FOLDER"] = os.path.abspath(config.train.dataset_path)

    l5_config = translate_l5kit_cfg(config)

    dm = LocalDataManager(None)
    rasterizer = build_rasterizer(l5_config, dm)

    # Dataset
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

    # Model
    modality_shapes = OrderedDict(image=(rasterizer.num_channels(), 224, 224))
    model = L5TrafficModelPL(
        algo_config=config.algo,
        modality_shapes=modality_shapes,
    )

    # Checkpointing
    ckpt_interval = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='iter={step}_ep={epoch}_ADE={val/metrics_ADE:.2f}',
        auto_insert_metric_name=False,
        monitor="val/metrics_ADE",
        save_top_k=10,
        mode="max",
        every_n_train_steps=config.train.save.every_n_steps,
        verbose=True,
    )

    # Logging
    tb_logger = TensorBoardLogger(save_dir=root_dir, version=version_key, name=None, sub_dir="logs/")
    print("Tensorboard event will be saved at {}".format(tb_logger.log_dir))

    # Environment


    trainer = pl.Trainer(
        default_root_dir=root_dir,
        gpus=config.devices.num_gpus,
        # checkpointing
        enable_checkpointing=config.train.save.enabled,
        callbacks=[ckpt_interval],
        # logging
        logger=tb_logger,
        flush_logs_every_n_steps=config.train.logging.flush_every_n_steps,
        log_every_n_steps=config.train.logging.log_every_n_steps,
        # training
        max_steps=config.train.training.num_steps,
        # validation
        val_check_interval=config.train.validation.every_n_steps,
        limit_val_batches=config.train.validation.num_steps_per_epoch
    )

    trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=valid_loader)



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

    train_data_iter = TrainUtils.infinite_iter(data_loader=train_loader)
    step = 0
    while step < config.train.num_steps:  # epoch numbers start at 1
        step_log = TrainUtils.run_training_steps(
            model=model, data_iter=train_data_iter, step=step, num_steps=train_num_steps)
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
    main_pl(config)

