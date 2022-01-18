import json
import os
import itertools
from collections import namedtuple
from typing import List
from glob import glob
import subprocess
import shutil
from pathlib import Path

from tbsim.configs.registry import get_registered_experiment_config
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
        return '_'.join([p.alias + str(p.value) for p in self.params])

    def generate_config(self, base_cfg: ExperimentConfig):
        cfg = base_cfg.clone()
        for p in self.params:
            var_list = p.config_var.split(".")
            c = cfg
            # traverse the indexing list
            for v in var_list[:-1]:
                assert v in c, "{} is not a valid config variable".format(p.config_var)
                c = c[v]
            assert var_list[-1] in c, "{} is not a valid config variable".format(p.config_var)
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
        assert all(len(pr) == l for pr in param_ranges), "All param_range must be the same length"
        prs = [pr.linearize() for pr in param_ranges]
        return [ParamConfig(prz) for prz in zip(*prs)]

    def generate_configs(self, base_cfg: ExperimentConfig):
        """
        Generate configs from the parameter search plan, also rename the experiment by generating the correct alias.
        """
        return [pc.generate_config(base_cfg) for pc in self.param_configs]


def create_configs(configs_to_search_fn, config_name, config_file, config_dir, prefix, delete_config_dir=True):
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
    config_fns = []

    if delete_config_dir and os.path.exists(config_dir):
        shutil.rmtree(config_dir)
    os.makedirs(config_dir, exist_ok=True)
    for c in configs:
        pfx = "{}_".format(prefix) if prefix is not None else ""
        fn = os.path.join(config_dir, "{}{}.json".format(pfx, c.name))
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


def launch_experiments_ngc(script_path, cfgs, cfg_paths, ngc_config):
    for cfg, cpath in zip(cfgs, cfg_paths):
        ngc_cpath = os.path.join(ngc_config["workspace_mounting_point_local"], "tbsim/", cpath)
        ngc_cdir = Path(ngc_cpath).parent
        os.makedirs(ngc_cdir, exist_ok=True)
        print("copying {} to {}".format(cpath, ngc_cpath))
        shutil.copy(cpath, ngc_cpath)

        py_cmd = [
            "export WANDB_APIKEY={};".format(ngc_config["wandb_apikey"]),
            "cd {}/tbsim;".format(ngc_config["workspace_mounting_point"]),
            "pip install -e .; pip install numpy==1.21.4;"
        ]
        py_cmd.extend([
            "python", script_path,
            "--config_file", cpath,
            "--output_dir", ngc_config["result_dir"],
            "--dataset_path", ngc_config["dataset_path"],
            "--wandb_project_name", ngc_config["wandb_project_name"],
            "--remove_exp_dir"
        ])
        py_cmd = " ".join(py_cmd)
        cmd = [
            "ngc", "batch", "run",
            "--instance", ngc_config["instance"],
            "--name", cfg.name,
            "--image", ngc_config["docker_image"],
            "--datasetid", "{}:{}".format(ngc_config["dataset_id"], ngc_config["dataset_mounting_point"]),
            "--workspace", "{}:{}".format(ngc_config["workspace_id"], ngc_config["workspace_mounting_point"]),
            "--result", ngc_config["result_dir"],
            "--total-runtime", ngc_config["total_runtime"],
            "--commandline", py_cmd
        ]
        print(cmd)
        subprocess.run(cmd)


def launch_experiments_local(script_path, cfgs, cfg_paths, extra_args = []):
    for cfg, cpath in zip(cfgs, cfg_paths):
        cmd = ["python", script_path, "--config_file", cpath] + extra_args
        subprocess.run(cmd)
