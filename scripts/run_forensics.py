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
from tbsim.safety_funcs.cbfs import BackupBarrierCBF
from tbsim.configs.eval_config import EvaluationConfig
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.safety_funcs.utils import * 
from tbsim.utils.geometry_utils import (
    VEH_VEH_distance, 
    VEH_VEH_collision
)

from tbsim.safety_funcs.debug_utils import * 

filename = "/home/rkcosner/Documents/tbsim/results/ForForensics/data.hdf5"

if __name__=="__main__": 

    # Load Evaluation Scene
    set_global_batch_type("trajdata")
    file = open("/home/rkcosner/Documents/tbsim/results/ForForensics/config.json")
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


    # Load Gamma Model
    file = open("/home/rkcosner/Documents/tbsim/checkpoints/braking_checkpoint/run3/config.json")#open("/home/rkcosner/Documents/tbsim/resp_trained_models/test/run15/config.json")
    algo_cfg = AlgoConfig()
    algo_cfg.algo = ResponsibilityConfig()
    external_algo_cfg = json.load(file)
    algo_cfg.update(**external_algo_cfg)
    algo_cfg.algo.update(**external_algo_cfg["algo"])
    device = "cpu" 
    modality_shapes = dict()
    gamma_algo = algo_factory(algo_cfg, modality_shapes)
    checkpoint_path = "/home/rkcosner/Documents/tbsim/checkpoints/braking_checkpoint/run3/iter9000_ep1_valLoss0.00.ckpt"
    checkpoint = torch.load(checkpoint_path)
    gamma_algo.load_state_dict(checkpoint["state_dict"])
    gamma_net = gamma_algo.nets["policy"]

    # Load CBF
    cbf = BackupBarrierCBF(T_horizon = algo_cfg.algo.cbf.T_horizon, 
                        alpha=algo_cfg.algo.cbf.alpha, 
                        veh_veh=algo_cfg.algo.cbf.veh_veh, 
                        saturate_cbf=algo_cfg.algo.cbf.saturate_cbf, 
                        backup_controller_type=algo_cfg.algo.cbf.backup_controller_type
                        )

    # Iterate through evaluation recordings
    with h5py.File(filename, "r") as file: 
        for scene_name in tqdm(file.keys()):
            extent              = torch.tensor(file[scene_name]["extent"])                  # [A, T, 3]
            action_pos          = torch.tensor(file[scene_name]["action_positions"])        # [A, T, 1, 2]
            action_smpl_pos     = torch.tensor(file[scene_name]["action_sample_positions"]) # [A, T, 50, 20, 2]
            action_smpl_yaw     = torch.tensor(file[scene_name]["action_sample_yaws"])      # [A, T, 50, 20, 1]
            action_yaws         = torch.tensor(file[scene_name]["action_yaws"])             # [A, T]
            centroid            = torch.tensor(file[scene_name]["centroid"])                # [A, T, 2]
            yaw                 = torch.tensor(file[scene_name]["yaw"])                     # [A, T]
            scene_index         = torch.tensor(file[scene_name]["scene_index"])             # [A, T]
            track_id            = torch.tensor(file[scene_name]["track_id"])                # [A, T]
            world_from_agent    = torch.tensor(file[scene_name]["world_from_agent"])        # [A, T, 3, 3]


            # Get Relevant Data and Fit Bezier Curves
            T = int(centroid.shape[1])
            N_repeated_indeces = 5
            indices = torch.arange(0, T, N_repeated_indeces)
            batch = {
                "history_positions"         : centroid[:,indices,:], 
                "history_yaws"              : yaw[:,indices,None], 
                "curr_speed"                : yaw[:,-1:]*0,
                "history_availabilities"    : (0*yaw[:,indices] + 1).int(), 
                "dt"                        : 0.5*torch.ones(1) 
            }
            substeps = 1
            batch = scene_centric_batch_to_raw(batch, BEZ_TEST=False, substeps = substeps)

            # Get Minimum Distance from Ego Vehicle In Trajectory
            A = batch["states"].shape[0]-1
            p1 = batch["states"][0:1,None,:,[0,1,3]].repeat_interleave(A, axis=1)
            S1 = extent[0:1,None,indices,:].repeat_interleave(A, axis=1)
            S1 = S1.repeat_interleave(substeps, axis=-2)
            p2 = batch["states"][None, 1:,:,[0,1,3]]
            S2 = extent[None,1:,indices,:]
            S2 = S2.repeat_interleave(substeps, axis=-2)
            dist = VEH_VEH_collision(p1, p2, S1, S2)#VEH_VEH_distance(p1, p2, S1, S2)
            dist_agents = dist.amin(-1)

            if dist_agents.min() > 0 : 
                continue
            
            # Find index of collision agent
            agent_idx = torch.where(dist_agents[0]<0)
            agent_idx = agent_idx[0].item()
            plt.figure()
            corners = [[0.5,0.5],[0.5,-0.5],[-0.5,0.5], [-0.5,-0.5]]
            for i, corner in enumerate(corners): 
                plt.plot(batch["states"][0,:,0]+corner[0] * S1[0,0,:,0], batch["states"][0,:,1]+corner[1] * S1[0,0,:,1],"b")
                plt.plot(batch["states"][agent_idx+1,:,0]+corner[0] * S2[0,agent_idx,:,0], batch["states"][agent_idx+1,:,1]+corner[1] * S2[0,agent_idx,:,1], "r")
            plt.savefig("collision_plot.png")
            plt.close()


            # Find Time index of Collision
            dist_time = dist.amin(-2)
            time_idxs = torch.where(dist_time[0] < 0)
            time_idxs = time_idxs[0]

            # Set Up Environment
            scene_indices = [scene_index[0,0].item()]
            env.scene_indices = scene_indices
            env.reset(scene_indices=scene_indices, start_frame_index=None)
            env.update_random_seed(1)
            simscene = env._current_scenes[0]


            # Run through sim
            states = batch["states"]
            inputs = batch["inputs"]
            A = states.shape[0]
            T = states.shape[1] 
            ego_states = []
            ego_imgs = []
            col_states = []
            col_imgs = []
            print("Replaying Experiment")
            for time in tqdm(range(T)): 

                # Set up scene
                scene_action = dict()
                for a, agent in enumerate(simscene.agents):
                    # print("agent name: ", agent, " \t agent id: ", a)
                    scene_action[agent.name] =  np.array(states[a, time, :])
                simscene.step(scene_action)
                obs = parse_trajdata_batch(simscene.get_obs())
                
                # Get network inputs
                col_states.append(states[None, agent_idx+1, time,:])
                ego_states.append(states[None, 0,time,:])
                ego_imgs.append(obs["image"][None, 0,-8:,...])
                col_imgs.append(obs["image"][None, agent_idx+1,-8:,...])
                # if time in time_idxs: 
                #     collision_flag = True
                # else: 
                #     collision_flag=False
                # test_img = plot_eval_img(obs,0, collision_flag)

            col_states = torch.cat(col_states, axis=0)
            ego_states = torch.cat(ego_states, axis=0)
            col_imgs = torch.cat(col_imgs, axis=0)
            ego_imgs = torch.cat(ego_imgs, axis=0)

            # Relative Pose 
            col_states[...,0:2] -= ego_states[...,0:2]
            ego_states[...,0:2] -= ego_states[...,0:2]

            # Build Rotation Matrices and Rotate Relative positions
            ego_from_world = torch.zeros(T,2,2)
            ego_from_world[:,0,0] = torch.cos(ego_states[...,3])
            ego_from_world[:,0,1] = -torch.sin(ego_states[...,3])
            ego_from_world[:,1,0] = torch.sin(ego_states[...,3])
            ego_from_world[:,1,1] = torch.cos(ego_states[...,3])
            col_states[...,3]   -= ego_states[...,3]
            ego_states[...,3]   -= ego_states[...,3]
            col_states[...,0:2]  = torch.bmm(ego_from_world,col_states[...,0:2,None])[...,0]

            agents_from_center = torch.zeros(T,3,3)
            agents_from_center[:,0,0] =  torch.cos(col_states[...,3])
            agents_from_center[:,0,1] = -torch.sin(col_states[...,3])
            agents_from_center[:,1,0] =  torch.sin(col_states[...,3])
            agents_from_center[:,1,1] =  torch.cos(col_states[...,3])
            agents_from_center[:,0,2] =  ( col_states[:,0] * torch.cos(col_states[...,3])  + col_states[:,1] * torch.sin(col_states[...,3]) ) 
            agents_from_center[:,1,2] =  (-col_states[:,0] * torch.sin(col_states[...,3])  + col_states[:,1] * torch.cos(col_states[...,3]) ) 
            agents_from_center[:,2,2] =  1.0



            input_batch = {
                "states" : torch.cat([ego_states[:,None,None,:], col_states[:,None,None,:]], axis=1), 
                "image"  : torch.cat([col_imgs[:,None,...], ego_imgs[:,None,...]], axis=1), 
                "agents_from_center" : torch.cat([agents_from_center[:,None,...],agents_from_center[:,None,...]], axis=1), 
                "extent" : torch.repeat_interleave(extent[None, 0:2, 0,:], repeats = T,axis=0),
                "dt"     : torch.ones(ego_states.shape[0])*0.5
            }

            gammas = gamma_net(input_batch)

            ego_gammas = gammas["gammas_A"][:,0,0,0]
            col_gammas = gammas["gammas_B"][:,0,0,0]

            data = cbf.process_batch(input_batch)
            data.requires_grad = True
            h_vals = cbf(data) 
            dhdx, = torch.autograd.grad(h_vals, inputs = data, grad_outputs = torch.ones_like(h_vals), create_graph=True)

            dhdxA = dhdx[:,:,0:4]
            dhdxB = dhdx[:,:,4:8]
            (fA, fB, gA, gB) = gamma_net.unicycle_dynamics(ego_states, col_states)
            LfhA = torch.bmm(dhdxA, fA).squeeze() 
            LfhB = torch.bmm(dhdxB, fB).squeeze()
            LghA = torch.bmm(dhdxA, gA)
            LghB = torch.bmm(dhdxB, gB) 

            natural_dynamics  = (LfhA + LfhB + cbf.alpha * h_vals[:,0])
            natural_dynamics  /= 2.0 
            # natural_dynamics  /= 20 # TODO There is some scaling issue going wrong somewhere!!!! 
            
            ego_inputs = batch["inputs"][0,:,:,None]
            col_inputs = batch["inputs"][agent_idx+1,:,:,None]

            LghAuA = torch.bmm(LghA, ego_inputs).squeeze()
            LghBuB = torch.bmm(LghB, col_inputs).squeeze()

            constraintA = natural_dynamics + LghAuA - ego_gammas
            constraintB = natural_dynamics + LghBuB - col_gammas

            plt.close()
            plt.figure()
            plt.plot(constraintA.cpu().detach().numpy())
            plt.plot(constraintB.cpu().detach().numpy())
            plt.savefig("responsibility_vis.png")

            plt.close()
            plt.figure()
            plt.plot(LghAuA.cpu().detach().numpy())
            plt.plot(LghBuB.cpu().detach().numpy())
            plt.savefig("Lghus.png")

            plt.close()
            plt.figure()
            plt.plot(ego_gammas.cpu().detach().numpy())
            plt.plot(col_gammas.cpu().detach().numpy())
            plt.savefig("gammas.png")

            plt.close()
            plt.figure()
            plt.plot(h_vals.cpu().detach().numpy())
            plt.savefig("h_vals.png")

            plt.close()
            fig, axs = plt.subplots(2,1)
            axs[0].plot(ego_inputs[:,0].cpu().detach().numpy())
            axs[0].plot(col_inputs[:,0].cpu().detach().numpy())
            axs[1].plot(ego_inputs[:,1].cpu().detach().numpy())
            axs[1].plot(col_inputs[:,1].cpu().detach().numpy())
            plt.savefig("inputs.png")

            breakpoint()


        print("Done")