import argparse
import sys
import os
import json

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from tbsim.utils.log_utils import PrintLogger
import tbsim.utils.train_utils as TrainUtils
from tbsim.envs.env_l5kit import EnvL5KitSimulation
from tbsim.utils.env_utils import RolloutCallback
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.datasets.factory import datamodule_factory
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.algos.factory import algo_factory


def main(cfg, auto_remove_exp_dir=False, debug=False):
    pl.seed_everything(cfg.seed)

    print("\n============= New Training Run with Config =============")
    print(cfg)
    print("")
    root_dir, log_dir, ckpt_dir, video_dir, version_key = TrainUtils.get_exp_dir(
        exp_name=cfg.name,
        output_dir=cfg.root_dir,
        save_checkpoints=cfg.train.save.enabled,
        auto_remove_exp_dir=auto_remove_exp_dir,
    )
    # Save experiment config to the training dir
    cfg.dump(os.path.join(root_dir, version_key, "config.json"))

    # Save experiment config to the training dir
    cfg.dump(os.path.join(root_dir, version_key, "config.json"))

    if cfg.train.logging.terminal_output_to_txt and not debug:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    train_callbacks = []

    # Training Parallelism
    assert cfg.train.parallel_strategy in [
        "dp",
        "ddp_spawn",
        None,
    ]  # TODO: look into other strategies
    if not cfg.devices.num_gpus > 1:
        # Override strategy when training on a single GPU
        with cfg.train.unlocked():
            cfg.train.parallel_strategy = None
    if cfg.train.parallel_strategy in ["ddp_spawn"]:
        with cfg.train.training.unlocked():
            cfg.train.training.batch_size = int(
                cfg.train.training.batch_size / cfg.devices.num_gpus
            )
        with cfg.train.validation.unlocked():
            cfg.train.validation.batch_size = int(
                cfg.train.validation.batch_size / cfg.devices.num_gpus
            )

    # Dataset
    datamodule = datamodule_factory(
        cls_name=cfg.train.datamodule_class, config=cfg
    )
    datamodule.setup()

    # Environment for close-loop evaluation
    if cfg.train.rollout.enabled:
        env = EnvL5KitSimulation(
            cfg.env,
            dataset=datamodule.env_dataset,
            seed=cfg.seed,
            num_scenes=cfg.train.rollout.num_scenes,
        )
        # Run rollout at regular intervals
        rollout_callback = RolloutCallback(
            env=env,
            num_episodes=cfg.train.rollout.num_episodes,
            every_n_steps=cfg.train.rollout.every_n_steps,
            n_step_action=cfg.train.rollout.n_step_action,
            warm_start_n_steps=cfg.train.rollout.warm_start_n_steps,
            verbose=False,
        )
        train_callbacks.append(rollout_callback)

    # Model
    model = algo_factory(
        config=cfg,
        modality_shapes=datamodule.modality_shapes,
        data_module=datamodule
    )

    # Checkpointing
    if cfg.train.validation.enabled and cfg.train.save.save_best_validation:
        assert (
            cfg.train.save.every_n_steps > cfg.train.validation.every_n_steps
        ), "checkpointing frequency needs to be greater than validation frequency"
        for metric_name, metric_key in model.checkpoint_monitor_keys.items():
            print(
                "Monitoring metrics {} under alias {}".format(metric_key, metric_name)
            )
            ckpt_valid_callback = pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="iter{step}_ep{epoch}_%s{%s:.2f}" % (metric_name, metric_key),
                # explicitly spell out metric names, otherwise PL parses '/' in metric names to directories
                auto_insert_metric_name=False,
                save_top_k=cfg.train.save.best_k,  # save the best k models
                monitor=metric_key,
                mode="min",
                every_n_train_steps=cfg.train.save.every_n_steps,
                verbose=True,
            )
            train_callbacks.append(ckpt_valid_callback)

    if cfg.train.rollout.enabled and cfg.train.save.save_best_rollout:
        assert (
            cfg.train.save.every_n_steps > cfg.train.rollout.every_n_steps
        ), "checkpointing frequency needs to be greater than rollout frequency"
        ckpt_rollout_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="iter{step}_ep{epoch}_simADE{rollout/metrics_ego_ADE:.2f}",
            # explicitly spell out metric names, otherwise PL parses '/' in metric names to directories
            auto_insert_metric_name=False,
            save_top_k=cfg.train.save.best_k,  # save the best k models
            monitor="rollout/metrics_ego_ADE",
            mode="min",
            every_n_train_steps=cfg.train.save.every_n_steps,
            verbose=True,
        )
        train_callbacks.append(ckpt_rollout_callback)

    # Logging
    assert not (cfg.train.logging.log_tb and cfg.train.logging.log_wandb)
    logger = None
    if debug:
        print("Debugging mode, suppress logging.")
    elif cfg.train.logging.log_tb:
        logger = TensorBoardLogger(
            save_dir=root_dir, version=version_key, name=None, sub_dir="logs/"
        )
        print("Tensorboard event will be saved at {}".format(logger.log_dir))
    elif cfg.train.logging.log_wandb:
        assert (
            "WANDB_APIKEY" in os.environ
        ), "Set api key by `export WANDB_APIKEY=<your-apikey>`"
        apikey = os.environ["WANDB_APIKEY"]
        wandb.login(key=apikey)
        logger = WandbLogger(
            name=cfg.name, project=cfg.train.logging.wandb_project_name
        )
        # record the entire config on wandb
        logger.experiment.config.update(cfg.to_dict())
        logger.watch(model=model)
    else:
        print("WARNING: not logging training stats")
        
    # Train
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        # checkpointing
        enable_checkpointing=cfg.train.save.enabled,
        # logging
        logger=logger,
        flush_logs_every_n_steps=cfg.train.logging.flush_every_n_steps,
        log_every_n_steps=cfg.train.logging.log_every_n_steps,
        # training
        max_steps=cfg.train.training.num_steps,
        # validation
        val_check_interval=cfg.train.validation.every_n_steps,
        limit_val_batches=cfg.train.validation.num_steps_per_epoch,
        # all callbacks
        callbacks=train_callbacks,
        # device & distributed training setup
        gpus=cfg.devices.num_gpus,
        strategy=cfg.train.parallel_strategy,
        # setting for overfit debugging
        # limit_val_batches=0,
        # overfit_batches=2
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="(optional) create experiment config from a preregistered name (see configs/registry.py)",
    )
    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
        help="(optional) if provided, override the wandb project name defined in the config",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset root path",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory of training output (checkpoints, visualization, tensorboard log, etc.)",
    )

    parser.add_argument(
        "--remove_exp_dir",
        action="store_true",
        help="Whether to automatically remove existing experiment directory of the same name (remember to set this to "
        "True to avoid unexpected stall when launching cloud experiments).",
    )

    parser.add_argument(
        "--on_ngc",
        action="store_true",
        help="whether running the script on ngc (this will change some behaviors like avoid writing into dataset)"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Debug mode, suppress wandb logging, etc."
    )

    args = parser.parse_args()

    if args.config_name is not None:
        default_config = get_registered_experiment_config(args.config_name)
    elif args.config_file is not None:
        # Update default config with external json file
        default_config = get_experiment_config_from_file(args.config_file)
    else:
        raise Exception(
            "Need either a config name or a json file to create experiment config"
        )

    if args.name is not None:
        default_config.name = args.name

    if args.dataset_path is not None:
        default_config.train.dataset_path = args.dataset_path

    if args.output_dir is not None:
        default_config.root_dir = os.path.abspath(args.output_dir)

    if args.wandb_project_name is not None:
        default_config.train.logging.wandb_project_name = args.wandb_project_name

    default_config.train.on_ngc = args.on_ngc

    if args.debug:
        # Test policy rollout
        default_config.train.rollout.every_n_steps = 100
        default_config.train.rollout.num_episodes = 1

    default_config.lock()  # Make config read-only
    main(default_config, auto_remove_exp_dir=args.remove_exp_dir, debug=args.debug)
