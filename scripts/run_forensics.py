import argparse
import h5py
import torch

filename = "/home/rkcosner/Documents/tbsim/results/HierAgentAware/data.hdf5"

if __name__=="__main__": 
    with h5py.File(filename, "r") as file: 
        for scene_name in file.keys():
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
            
            

            breakpoint()
        print("helloworld")