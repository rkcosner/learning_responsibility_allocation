import argparse
import sys
import os
from torch.utils.data import DataLoader
from collections import OrderedDict

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from tbsim.utils.log_utils import PrintLogger
import tbsim.utils.train_utils as TrainUtils
from tbsim.algos.l5kit_algos import L5TrafficModel
from tbsim.configs import ExperimentConfig, L5KitEnvConfig, L5KitTrainConfig, L5RasterizedPlanningConfig
from tbsim.envs.env_l5kit import EnvL5KitSimulation
from tbsim.utils.env_utils import RolloutCallback



def translate_l5kit_cfg(cfg):
    """

    Args:
        cfg (ExperimentConfig): an ExperimentConfig instance

    Returns:
        cfg for l5kit
    """
    rcfg = dict()
    rcfg["raster_params"] = cfg.env.rasterizer
    rcfg["raster_params"]["dataset_meta_key"] = cfg.train.dataset_meta_key
    rcfg["model_params"] = cfg.algo
    return rcfg


def main(cfg):
    pl.seed_everything(cfg.seed)
    print("\n============= New Training Run with Config =============")
    print(cfg)
    print("")
    root_dir, log_dir, ckpt_dir, video_dir, version_key = TrainUtils.get_exp_dir(
        exp_name=cfg.name,
        output_dir=cfg.root_dir,
        save_checkpoints=cfg.train.save.enabled,
        auto_remove_exp_dir=True
    )

    if cfg.train.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # Dataset

    l5_config = translate_l5kit_cfg(cfg)

    os.environ["L5KIT_DATA_FOLDER"] = os.path.abspath(cfg.train.dataset_path)
    dm = LocalDataManager(None)
    rasterizer = build_rasterizer(l5_config, dm)

    train_zarr = ChunkedDataset(dm.require(cfg.train.dataset_train_key)).open()
    trainset = AgentDataset(l5_config, train_zarr, rasterizer)

    train_loader = DataLoader(
        dataset=trainset,
        shuffle=True,
        batch_size=cfg.train.training.batch_size,
        num_workers=cfg.train.training.num_data_workers,
        drop_last=True
    )
    print(trainset)

    valid_zarr = ChunkedDataset(dm.require(cfg.train.dataset_valid_key)).open()
    validset = AgentDataset(l5_config, valid_zarr, rasterizer)
    valid_loader = DataLoader(
        dataset=validset,
        shuffle=True,
        batch_size=cfg.train.validation.batch_size,
        num_workers=cfg.train.validation.num_data_workers,
        drop_last=True
    )
    print(validset)

    # Environment for close-loop evaluation
    env_dataset = EgoDataset(l5_config, valid_zarr, rasterizer)
    env = EnvL5KitSimulation(cfg.env, dataset=env_dataset, seed=cfg.seed, num_scenes=config.train.rollout.num_episodes)
    rollout_callback = RolloutCallback(
        env=env,
        num_episodes=1,  # all scenes run in parallel
        every_n_steps=config.train.rollout.every_n_steps,
        warm_start_n_steps=config.train.rollout.warm_start_n_steps
    )


    # Model
    modality_shapes = OrderedDict(image=(rasterizer.num_channels(), 224, 224))
    model = L5TrafficModel(
        algo_config=cfg.algo,
        modality_shapes=modality_shapes,
    )
    print(model)

    # Checkpointing
    ckpt_ade_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='iter={step}_ep={epoch}_ADE={val/metrics_ADE:.2f}',
        # explicitly spell out metric names, otherwise PL parses '/' in metric names to directories
        auto_insert_metric_name=False,
        save_top_k=cfg.train.save.best_k,  # save the best k models
        monitor="val/metrics_ADE",
        mode="min",
        every_n_train_steps=cfg.train.save.every_n_steps,
        verbose=True,
    )

    # Logging
    tb_logger = TensorBoardLogger(save_dir=root_dir, version=version_key, name=None, sub_dir="logs/")
    print("Tensorboard event will be saved at {}".format(tb_logger.log_dir))

    # Train
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        gpus=cfg.devices.num_gpus,
        # checkpointing
        enable_checkpointing=cfg.train.save.enabled,
        callbacks=[
            ckpt_ade_callback,  # checkpoint for model with best validation ADE
            rollout_callback  # for running evaluation rollout
        ],
        # logging
        logger=tb_logger,
        flush_logs_every_n_steps=cfg.train.logging.flush_every_n_steps,
        log_every_n_steps=cfg.train.logging.log_every_n_steps,
        # training
        max_steps=cfg.train.training.num_steps,
        # validation
        val_check_interval=cfg.train.validation.every_n_steps,
        limit_val_batches=cfg.train.validation.num_steps_per_epoch
    )

    trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=valid_loader)


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

