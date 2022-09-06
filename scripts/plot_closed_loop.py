import argparse
import h5py
import json
from tbsim.configs.algo_config import ResponsibilityConfig
from tbsim.configs.base import AlgoConfig
from tbsim.utils.trajdata_utils import parse_trajdata_batch
import torch
import importlib

from tqdm import tqdm 
import matplotlib.pyplot as plt

from tbsim.algos.factory import algo_factory
from tbsim.utils.trajdata_utils import parse_trajdata_batch
from tbsim.evaluation.env_builders import EnvNuscBuilder, EnvL5Builder
from tbsim.safety_funcs.cbfs import BackupBarrierCBF, unicycle_dynamics
from tbsim.configs.eval_config import EvaluationConfig
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.safety_funcs.utils import * 
from tbsim.utils.geometry_utils import (
    VEH_VEH_distance, 
    VEH_VEH_collision
)

from tbsim.safety_funcs.debug_utils import * 

type_names = ["HierAgentAwareCBFQP_split_responsibility"]#["HierAgentAwareCBFQP", "HierAgentAwareCBFQP_split_responsibility", "HierAgentAwareCBFQP_fullresp_worst_case"]

if __name__=="__main__": 
    for type_name in type_names:

        filename = "/home/rkcosner/Documents/tbsim/results/idling_closed_loop/" + type_name+"/data.hdf5"

        # Load Evaluation Scene
        set_global_batch_type("trajdata")
        file = open("/home/rkcosner/Documents/tbsim/results/idling_closed_loop/"+type_name+"/config.json")
        eval_cfg = EvaluationConfig()
        external_cfg = json.load(file)
        eval_cfg.update(**external_cfg)
        device = "cuda"
        policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
        composer_class = getattr(policy_composers, eval_cfg.eval_class)
        composer = composer_class(eval_cfg, device, ckpt_root_dir=eval_cfg.ckpt_root_dir)
        policy, exp_config = composer.get_policy()
        # Set to scene_centric
        exp_config.algo.scene_centric = True
        env_builder = EnvNuscBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device)
        env = env_builder.get_env(split_ego=False,parse_obs=True)


        # # Load Gamma Model
        # file = open("/home/rkcosner/Documents/tbsim/checkpoints/braking_checkpoint/run5/config.json")#open("/home/rkcosner/Documents/tbsim/resp_trained_models/test/run15/config.json")
        # algo_cfg = AlgoConfig()
        # algo_cfg.algo = ResponsibilityConfig()
        # external_algo_cfg = json.load(file)
        # algo_cfg.update(**external_algo_cfg)
        # algo_cfg.algo.update(**external_algo_cfg["algo"])
        # device = "cpu" 
        # modality_shapes = dict()
        # gamma_algo = algo_factory(algo_cfg, modality_shapes)
        # checkpoint_path = "/home/rkcosner/Documents/tbsim/checkpoints/braking_checkpoint/run5/iter10000_ep1_valLoss0.00.ckpt"
        # checkpoint = torch.load(checkpoint_path)
        # gamma_algo.load_state_dict(checkpoint["state_dict"])
        # gamma_net = gamma_algo.nets["policy"]

        # Load CBF
        cbf = BackupBarrierCBF(T_horizon = 4, 
                            alpha=2,
                            veh_veh=True, 
                            saturate_cbf=True, 
                            backup_controller_type="idle"
                            )

        # Iterate through evaluation recordings
        with h5py.File(filename, "r") as file: 
            for scene_name in tqdm(file.keys()):
                extent              = torch.tensor(file[scene_name]["extent"])                  # [A, T, 3]
                action_pos          = torch.tensor(file[scene_name]["action_positions"])        # [A, T, 1, 2]
                # action_smpl_pos     = torch.tensor(file[scene_name]["action_sample_positions"]) # [A, T, 50, 20, 2]
                # action_smpl_yaw     = torch.tensor(file[scene_name]["action_sample_yaws"])      # [A, T, 50, 20, 1]
                action_yaws         = torch.tensor(file[scene_name]["action_yaws"])             # [A, T]
                centroid            = torch.tensor(file[scene_name]["centroid"])                # [A, T, 2]
                yaw                 = torch.tensor(file[scene_name]["yaw"])                     # [A, T]
                scene_index         = torch.tensor(file[scene_name]["scene_index"])             # [A, T]
                track_id            = torch.tensor(file[scene_name]["track_id"])                # [A, T]
                world_from_agent    = torch.tensor(file[scene_name]["world_from_agent"])        # [A, T, 3, 3]

                # Get Relevant Data and Fit Bezier Curves
                T = int(centroid.shape[1])
                # N_repeated_indeces = 5
                h_vals = []
                for idx in range(T): 
                    batch = {
                        "history_positions"         : centroid[None, :,idx:idx+5,:], 
                        "history_yaws"              : yaw[None,:,idx:idx+5,None], 
                        "curr_speed"                : yaw[None,:,-1:]*0,
                        "history_availabilities"    : (0*yaw[None,:,idx:idx+5] + 1).int(), 
                        "dt"                        : 0.5*torch.ones(1), 
                        "extent"                    : extent[None,:, 0, :]
                    }
                    substeps = 2
                    batch = scene_centric_batch_to_raw(batch, BEZ_TEST=False, substeps = substeps)

                    dt = batch["dt"][0].item()
                    data = cbf.process_batch(batch)
                    h_vals.append(cbf(data))
                h_vals = torch.cat(h_vals, axis = 0 )

                A = extent.shape[0]
                plt.figure()
                for i in range(A-1): 
                    if i < A-2: 
                        plt.plot(h_vals[:,i], 'k')
                    else: 
                        plt.plot(h_vals[:,i], 'b')
                plt.savefig("./closed_loop_plots/" +type_name +  scene_name+"h_vals.png")
                plt.close()

                plt.figure()
                for i in range(A): 
                    plt.plot(centroid[i,:,0], centroid[i,:,1])
                plt.savefig("./closed_loop_plots/" + type_name + scene_name+"trajectories.png")
                plt.close()

        print("Done")