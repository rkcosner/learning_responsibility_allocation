import json
import os
import itertools
import sys
from collections import namedtuple
from typing import List
from glob import glob
import subprocess
import shutil
from pathlib import Path
import pdb

import tbsim
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.configs.config import Dict
from tbsim.configs.eval_configs import EvaluationConfig
from tbsim.configs.base import ExperimentConfig


class Param(namedtuple("Param", "config_var alias value")):
    pass


class ParamRange(namedtuple("Param", "config_var alias range")):
    def linearize(self):
        return [Param(self.config_var, self.alias, v) for v in self.range]

    def __len__(self):
        return len(self.range)


class ParamConfig(object):
    def __init__(self, params: List[Param] = None):
        self.params = []
        self.aliases = []
        self.config_vars = []
        print(params)
        if params is not None:
            for p in params:
                self.add(p)

    def add(self, param: Param):
        assert param.config_var not in self.config_vars
        assert param.alias not in self.aliases
        self.config_vars.append(param.config_var)
        self.aliases.append(param.alias)
        self.params.append(param)

    def __str__(self):
        char_to_remove = [" ", "(", ")", ";", "[", "]"]
        name = []
        for p in self.params:
            v_str = str(p.value)
            for c in char_to_remove:
                v_str = v_str.replace(c, "")
            name.append(p.alias + v_str)

        return "_".join(name)

    def generate_config(self, base_cfg: Dict):
        cfg = base_cfg.clone()
        for p in self.params:
            var_list = p.config_var.split(".")
            c = cfg
            # traverse the indexing list
            for v in var_list[:-1]:
                assert v in c, "{} is not a valid config variable".format(
                    p.config_var)
                c = c[v]
            assert var_list[-1] in c, "{} is not a valid config variable".format(
                p.config_var
            )
            c[var_list[-1]] = p.value
        cfg.name = str(self)
        return cfg


class ParamSearchPlan(object):
    def __init__(self):
        self.param_configs = []
        self.const_params = []

    def add_const_param(self, param: Param):
        self.const_params.append(param)

    def add(self, param_config: ParamConfig):
        for c in self.const_params:
            param_config.add(c)
        self.param_configs.append(param_config)

    def extend(self, param_configs: List[ParamConfig]):
        for pc in param_configs:
            self.add(pc)

    @staticmethod
    def compose_concate(param_ranges: List[ParamRange]):
        pcs = []
        for pr in param_ranges:
            for p in pr.linearize():
                pcs.append(ParamConfig([p]))
        return pcs

    @staticmethod
    def compose_cartesian(param_ranges: List[ParamRange]):
        """Cartesian product among parameters"""
        prs = [pr.linearize() for pr in param_ranges]
        return [ParamConfig(pr) for pr in itertools.product(*prs)]

    @staticmethod
    def compose_zip(param_ranges: List[ParamRange]):
        l = len(param_ranges[0])
        assert all(
            len(pr) == l for pr in param_ranges
        ), "All param_range must be the same length"
        prs = [pr.linearize() for pr in param_ranges]
        return [ParamConfig(prz) for prz in zip(*prs)]

    def generate_configs(self, base_cfg: Dict):
        """
        Generate configs from the parameter search plan, also rename the experiment by generating the correct alias.
        """
        return [pc.generate_config(base_cfg) for pc in self.param_configs]


def create_configs(
    configs_to_search_fn,
    config_name,
    config_file,
    config_dir,
    prefix,
    delete_config_dir=True,
):
    if config_name is not None:
        cfg = get_registered_experiment_config(config_name)
        print("Generating configs for {}".format(config_name))
    elif config_file is not None:
        # Update default config with external json file
        ext_cfg = json.load(open(config_file, "r"))
        cfg = get_registered_experiment_config(ext_cfg["registered_name"])
        cfg.update(**ext_cfg)
        print("Generating configs with {} as template".format(config_file))
    else:
        raise FileNotFoundError("No base config is provided")

    configs = configs_to_search_fn(base_cfg=cfg)
    for c in configs:
        pfx = "{}_".format(prefix) if prefix is not None else ""
        c.name = pfx + c.name
    config_fns = []

    if delete_config_dir and os.path.exists(config_dir):
        shutil.rmtree(config_dir)
    os.makedirs(config_dir, exist_ok=True)
    for c in configs:
        fn = os.path.join(config_dir, "{}.json".format(c.name))
        config_fns.append(fn)
        print("Saving config to {}".format(fn))
        c.dump(fn)

    return configs, config_fns


def read_configs(config_dir):
    configs = []
    config_fns = []
    for cfn in glob(config_dir + "/*.json"):
        print(cfn)
        config_fns.append(cfn)
        ext_cfg = json.load(open(cfn, "r"))
        c = get_registered_experiment_config(ext_cfg["registered_name"])
        c.update(**ext_cfg)
        configs.append(c)
    return configs, config_fns


def create_evaluation_configs(
        configs_to_search_fn,
        config_file,
        config_dir,
        prefix,
        delete_config_dir=True,
):
    if config_file is not None:
        # Update default config with external json file
        ext_cfg = json.load(open(config_file, "r"))
        cfg = EvaluationConfig()
        cfg.update(**ext_cfg)
        print("Generating configs with {} as template".format(config_file))
    else:
        cfg = EvaluationConfig()

    configs = configs_to_search_fn(base_cfg=cfg)
    for c in configs:
        pfx = "{}_".format(prefix) if prefix is not None else ""
        c.name = pfx + c.name
    config_fns = []

    if delete_config_dir and os.path.exists(config_dir):
        shutil.rmtree(config_dir)
    os.makedirs(config_dir, exist_ok=True)
    for c in configs:
        fn = os.path.join(config_dir, "{}.json".format(c.name))
        config_fns.append(fn)
        print("Saving config to {}".format(fn))
        c.dump(fn)

    return configs, config_fns


def read_evaluation_configs(config_dir):
    configs = []
    config_fns = []
    for cfn in glob(config_dir + "/*.json"):
        print(cfn)
        config_fns.append(cfn)
        c = EvaluationConfig()
        ext_cfg = json.load(open(cfn, "r"))
        c.update(**ext_cfg)
        configs.append(c)
    return configs, config_fns


def launch_experiments_ngc(
    script_command: list, cfgs: List[Dict], cfg_paths: List[str], ngc_config: dict, dry_run=False
):
    """
    Launch one or more experiments on NGC
    Args:
        script_command (list): the complete python command to run with arguments
            (excluding --config_file, which will be filled by the function)
        cfgs (List[Dict]): list of configs to launch experiments with
        cfg_paths (List[str]): list of path to the config files (for copying to NGC workspace)
        ngc_config (dict): ngc experiment configuration
        dry_run (bool): whether to only print the commands instead of running them

    Returns:
        None
    """
    for cfg, cpath in zip(cfgs, cfg_paths):
        ngc_cpath = os.path.join(
            ngc_config["workspace_mounting_point_local"], "tbsim/", cpath
        )
        ngc_cdir = Path(ngc_cpath).parent
        os.makedirs(ngc_cdir, exist_ok=True)
        print("copying {} to {}".format(cpath, ngc_cpath))
        shutil.copy(cpath, ngc_cpath)

        py_cmd = [
            "export WANDB_APIKEY={};".format(ngc_config["wandb_apikey"]),
            "cd {}/tbsim;".format(ngc_config["workspace_mounting_point"]),
            "pip install -e .; pip install numpy==1.21.4;",
        ]
        py_cmd.extend(script_command)
        py_cmd.extend(["--config_file", cpath])
        py_cmd = " ".join(py_cmd)
        cmd = [
            "ngc",
            "batch",
            "run",
            "--instance",
            ngc_config["instance"],
            # "ml-model." prefix a the naming convention required by NGC
            # see https://confluence.nvidia.com/display/GWE/5.+Job+naming+and+categorization
            "--name",
            "ml-model." + cfg.name,
            "--image",
            ngc_config["docker_image"],
            "--datasetid",
            "{}:{}".format(
                ngc_config["dataset_id"], ngc_config["dataset_mounting_point"]
            ),
            "--workspace",
            "{}:{}".format(
                ngc_config["workspace_id"], ngc_config["workspace_mounting_point"]
            ),
            "--result",
            ngc_config["result_dir"],
            "--total-runtime",
            ngc_config["total_runtime"],
            "--commandline",
            py_cmd,
        ]
        print(cmd)
        if not dry_run:
            subprocess.run(cmd)


def upload_codebase_to_ngc_workspace(ngc_config):
    """
    Upload local codebase to NGC workspace
    Args:
        ngc_config (dict): NGC config

    """
    ngc_path = os.path.join(
        ngc_config["workspace_mounting_point_local"], "tbsim/")
    local_path = Path(tbsim.__path__[0]).parent
    assert os.path.exists(ngc_path), "please mount NGC path first"
    dir_list = ["scripts/", "tbsim/"]
    for d in dir_list:
        print("uploading {}".format(d))
        shutil.copytree(
            os.path.join(local_path, d), os.path.join(ngc_path, d), dirs_exist_ok=True
        )
    file_list = ["setup.py"]
    for f in file_list:
        print("uploading {}".format(f))
        shutil.copy(os.path.join(local_path, f), os.path.join(ngc_path, f))


def launch_experiments_local(script_path, cfgs, cfg_paths, extra_args=[]):
    for cfg, cpath in zip(cfgs, cfg_paths):
        cmd = ["python", script_path, "--config_file", cpath] + extra_args
        subprocess.run(cmd)


def get_results_info_ngc(ngc_job_id):
    cmd = ["ngc", "result", "info", str(ngc_job_id), "--files"]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    outs, errs = process.communicate()
    if len(errs) > 0:
        print(str(errs))
        return None
    outs = str(outs).split("\\n")
    ckpt_paths = [l.strip(" ") for l in outs if l.endswith(".ckpt")]
    cfg_path = [l.strip(" ") for l in outs if l.endswith(".json")]
    assert len(cfg_path) == 1
    cfg_path = cfg_path[0]
    job_name = cfg_path.split("/")[1]
    return ckpt_paths, cfg_path, job_name


def _download_from_ngc(ngc_job_id, paths_to_download, target_dir, tmp_dir="/tmp"):
    cmd = ["ngc", "result", "download", str(ngc_job_id)]
    print("Downloading: ")
    for fp in paths_to_download:
        print(fp)
        cmd.extend(["--file", fp])

    cmd.extend(["--dest", tmp_dir])
    if os.path.exists(os.path.join(tmp_dir, ngc_job_id + "/")):
        print("tmp folder with ngc job ID exists, removing ...")
        shutil.rmtree(
            os.path.join(tmp_dir, ngc_job_id + "/")
        )  # otherwise ngc renames the downloaded folder
    subprocess.run(cmd)

    os.makedirs(target_dir, exist_ok=True)

    for fp in paths_to_download:
        src_path = os.path.join(tmp_dir, ngc_job_id + fp)
        shutil.move(src_path, target_dir)

    shutil.rmtree(os.path.join(tmp_dir, ngc_job_id + "/"))


def download_checkpoints_from_ngc(
    ngc_job_id, ckpt_root_dir, ckpt_path_func=None, tmp_dir="/tmp"
):
    assert os.path.exists(ckpt_root_dir)
    ckpt_paths, cfg_path, job_name = get_results_info_ngc(ngc_job_id)

    if ckpt_path_func is None:
        def ckpt_path_func(x): return x
    to_download = ckpt_path_func(ckpt_paths)
    to_download.append(cfg_path)
    ckpt_target_dir = os.path.join(
        ckpt_root_dir, "{}_{}".format(job_name, ngc_job_id))

    _download_from_ngc(ngc_job_id, to_download,
                       ckpt_target_dir, tmp_dir=tmp_dir)
    return ckpt_target_dir


def get_local_checkpoint_dir(ngc_job_id, ckpt_root_dir):
    for p in glob(ckpt_root_dir + "/*"):
        if str(ngc_job_id) == p.split("_")[-1] or str(ngc_job_id) == p.split("/")[-1]:
            return p
    return None


def get_checkpoint(
    ngc_job_id, ckpt_key, ckpt_root_dir="checkpoints/", download_tmp_dir="/tmp"
):
    def ckpt_path_func(paths): return [p for p in paths if ckpt_key in p]
    local_dir = get_local_checkpoint_dir(ngc_job_id, ckpt_root_dir)
    if local_dir is None:
        print("checkpoint does not exist, downloading ...")
        ckpt_dir = download_checkpoints_from_ngc(
            ngc_job_id=ngc_job_id,
            ckpt_root_dir=ckpt_root_dir,
            ckpt_path_func=ckpt_path_func,
            tmp_dir=download_tmp_dir,
        )
    else:
        ckpt_paths = glob(local_dir + "/*.ckpt")
        if len(ckpt_path_func(ckpt_paths)) == 0:
            print("checkpoint does not exist, downloading ...")
            ckpt_dir = download_checkpoints_from_ngc(
                ngc_job_id=ngc_job_id,
                ckpt_root_dir=ckpt_root_dir,
                ckpt_path_func=ckpt_path_func,
                tmp_dir=download_tmp_dir,
            )
        else:
            ckpt_dir = local_dir
    ckpt_paths = ckpt_path_func(glob(ckpt_dir + "/*.ckpt"))
    assert len(ckpt_paths) > 0, "Could not find a checkpoint that has key {}".format(
        ckpt_key
    )
    assert len(ckpt_paths) == 1, "More than one checkpoint found"
    cfg_path = os.path.join(ckpt_dir, "config.json")
    return ckpt_paths[0], cfg_path


if __name__ == "__main__":
    print(get_checkpoint("2546043", ckpt_key="iter87999_"))
