import argparse
import sys
import os
import json
from torch.utils.data import DataLoader
from collections import OrderedDict
import pdb

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import (
    AgentDataset,
    EgoDataset,
    EgoDatasetVectorized,
    EgoDatasetMixed,
)
from l5kit.rasterization import build_rasterizer
from l5kit.vectorization.vectorizer_builder import build_vectorizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from tbsim.utils.log_utils import PrintLogger
import tbsim.utils.train_utils as TrainUtils
from tbsim.algos.l5kit_algos import L5TrafficModel, L5TransformerTrafficModel
from tbsim.configs import (
    ExperimentConfig,
    L5KitEnvConfig,
    L5KitTrainConfig,
    L5RasterizedPlanningConfig,
    L5KitVectorizedEnvConfig,
    L5KitMixedEnvConfig,
    L5TransformerPredConfig,
)
from tbsim.envs.env_l5kit import EnvL5KitSimulation
from tbsim.utils.env_utils import RolloutCallback
from tbsim.utils.config_utils import translate_l5kit_cfg


def main(cfg):
    pl.seed_everything(cfg.seed)

    print("\n============= New Training Run with Config =============")
    print(cfg)
    print("")
    root_dir, log_dir, ckpt_dir, video_dir, version_key = TrainUtils.get_exp_dir(
        exp_name=cfg.name,
        output_dir=cfg.root_dir,
        save_checkpoints=cfg.train.save.enabled,
        auto_remove_exp_dir=True,
    )

    if cfg.train.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    train_callbacks = []

    # Dataset
    l5_config = translate_l5kit_cfg(cfg)

    os.environ["L5KIT_DATA_FOLDER"] = os.path.abspath(cfg.train.dataset_path)
    dm = LocalDataManager(None)

    train_zarr = ChunkedDataset(dm.require(cfg.train.dataset_train_key)).open()
    vectorizer = build_vectorizer(l5_config, dm)
    rasterizer = build_rasterizer(l5_config, dm)
    trainset = EgoDatasetMixed(l5_config, train_zarr, vectorizer, rasterizer)
    train_loader = DataLoader(
        dataset=trainset,
        shuffle=True,
        batch_size=cfg.train.training.batch_size,
        num_workers=cfg.train.training.num_data_workers,
        drop_last=True,
    )
    print("\n============= Training Dataset =============")
    print(trainset)

    valid_zarr = None
    if cfg.train.validation.enabled:
        valid_zarr = ChunkedDataset(dm.require(cfg.train.dataset_valid_key)).open()
        validset = EgoDatasetMixed(l5_config, valid_zarr, vectorizer, rasterizer)
        valid_loader = DataLoader(
            dataset=validset,
            shuffle=True,
            batch_size=cfg.train.validation.batch_size,
            num_workers=cfg.train.validation.num_data_workers,
            drop_last=True,
        )
        print("\n============= Validation Dataset =============")
        print(validset)

    # Environment for close-loop evaluation
    if cfg.train.rollout.enabled:
        if valid_zarr is None:
            valid_zarr = ChunkedDataset(dm.require(cfg.train.dataset_valid_key)).open()
        env_dataset = EgoDatasetMixed(l5_config, valid_zarr, vectorizer, rasterizer)
        env = EnvL5KitSimulation(
            cfg.env,
            dataset=env_dataset,
            seed=cfg.seed,
            num_scenes=cfg.train.rollout.num_episodes,
        )

        # Run rollout at regular intervals

        rollout_callback = RolloutCallback(
            env=env,
            num_episodes=1,  # all scenes run in parallel
            every_n_steps=cfg.train.rollout.every_n_steps,
            warm_start_n_steps=cfg.train.rollout.warm_start_n_steps,
        )
        train_callbacks.append(rollout_callback)

    # Model
    cfg.algo["tgt_mask_N"] = len(train_loader)
    model = L5TransformerTrafficModel(
        algo_config=cfg.algo,
    )

    # Checkpointing
    ckpt_ade_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="iter={step}_ep={epoch}_simADE={rollout/metrics_ego_ADE:.2f}",
        # explicitly spell out metric names, otherwise PL parses '/' in metric names to directories
        auto_insert_metric_name=False,
        save_top_k=cfg.train.save.best_k,  # save the best k models
        monitor="rollout/metrics_ego_ADE",
        mode="min",
        every_n_train_steps=cfg.train.save.every_n_steps,
        verbose=True,
    )

    ckpt_loss_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="iter={step}_ep={epoch}_valLoss={val/losses_prediction_loss:.2f}",
        # explicitly spell out metric names, otherwise PL parses '/' in metric names to directories
        auto_insert_metric_name=False,
        save_top_k=cfg.train.save.best_k,  # save the best k models
        monitor="val/losses_prediction_loss",
        mode="min",
        every_n_train_steps=cfg.train.save.every_n_steps,
        verbose=True,
    )

    train_callbacks.extend([ckpt_loss_callback, ckpt_ade_callback])

    # Logging
    tb_logger = TensorBoardLogger(
        save_dir=root_dir, version=version_key, name=None, sub_dir="logs/"
    )
    print("Tensorboard event will be saved at {}".format(tb_logger.log_dir))

    # Train
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        gpus=cfg.devices.num_gpus,
        # checkpointing
        enable_checkpointing=cfg.train.save.enabled,
        # logging
        logger=tb_logger,
        flush_logs_every_n_steps=cfg.train.logging.flush_every_n_steps,
        log_every_n_steps=cfg.train.logging.log_every_n_steps,
        # training
        max_steps=cfg.train.training.num_steps,
        # validation
        val_check_interval=cfg.train.validation.every_n_steps,
        limit_val_batches=cfg.train.validation.num_steps_per_epoch,
        # all callbacks
        callbacks=train_callbacks,
    )

    trainer.fit(
        model=model, train_dataloader=train_loader, val_dataloaders=valid_loader
    )


def test_Vec_l5_env(cfg):

    l5_config = translate_l5kit_cfg(cfg)

    os.environ["L5KIT_DATA_FOLDER"] = os.path.abspath(cfg.train.dataset_path)
    dm = LocalDataManager(None)
    valid_zarr = ChunkedDataset(dm.require(cfg.train.dataset_valid_key)).open()
    vectorizer = build_vectorizer(l5_config, dm)
    env_dataset = EgoDatasetVectorized(l5_config, valid_zarr, vectorizer)

    # rasterizer = build_rasterizer(l5_config, dm)
    # env_dataset = EgoDataset(l5_config, valid_zarr, rasterizer)
    env = EnvL5KitSimulation(
        cfg.env,
        dataset=env_dataset,
        seed=cfg.seed,
        num_scenes=cfg.train.rollout.num_episodes,
    )

    env.reset()

    ac = {"ego": env.get_random_ego_actions()}
    env.step(ac)
    print(env.get_metrics())


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
        help="(optional) if provided, override the dataset root path",
    )

    args = parser.parse_args()

    default_config = ExperimentConfig(
        train_config=L5KitTrainConfig(),
        env_config=L5KitMixedEnvConfig(),
        algo_config=L5TransformerPredConfig(),
    )

    if args.name is not None:
        default_config.name = args.name

    if args.dataset_path is not None:
        default_config.train.dataset_path = args.dataset_path

    if args.config is not None:
        # Update default config with external json file
        ext_cfg = json.load(open(args.config, "r"))
        default_config.update(**ext_cfg)

    default_config.lock()  # Make config read-only
    # Rasterized_main(default_config)
    main(default_config)
    # test_Vec_l5_env(default_config)
