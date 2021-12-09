"""
This file contains several utility functions used to define the main training loop. It
mainly consists of functions to assist with logging, rollouts, and the @run_epoch function,
which is the core training logic for models in this repository.
"""
import os
import time
import datetime
import shutil

import tbsim


def infinite_iter(data_loader):
    """
    Get an infinite generator
    Args:
        data_loader (DataLoader): data loader to iterate through

    """
    c_iter = iter(data_loader)
    while True:
        try:
            data = next(c_iter)
        except StopIteration:
            c_iter = iter(data_loader)
            data = next(c_iter)
        yield data


def get_exp_dir(exp_name, output_dir, save_checkpoints=True, auto_remove_exp_dir=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        exp_name (str): name of the experiment
        output_dir (str): output directory of the experiment
        save_checkpoints (bool): if save checkpoints
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.

    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    t_now = time.time()
    version_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = output_dir
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to tbsim module location
        base_output_dir = os.path.join(tbsim.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, exp_name)
    if os.path.exists(base_output_dir):
        if not auto_remove_exp_dir:
            ans = input("WARNING: model directory ({}) already exists! \noverwrite? (y/n)\n".format(base_output_dir))
        else:
            ans = "y"
        if ans == "y":
            print("REMOVING")
            shutil.rmtree(base_output_dir)

    # only make model directory if model saving is enabled
    ckpt_dir = None
    if save_checkpoints:
        ckpt_dir = os.path.join(base_output_dir, version_str, "models")
        os.makedirs(ckpt_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, version_str, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, version_str, "videos")
    os.makedirs(video_dir)
    return base_output_dir, log_dir, ckpt_dir, video_dir, version_str