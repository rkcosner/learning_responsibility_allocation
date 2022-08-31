from os import device_encoding
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Dict
from copy import deepcopy
import cvxpy as cp

from tbsim.envs.base import BatchedEnv, BaseEnv
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.dynamics as dynamics
from tbsim.utils.l5_utils import get_current_states
from tbsim.utils.batch_utils import batch_utils
from tbsim.algos.algo_utils import optimize_trajectories
from tbsim.utils.geometry_utils import transform_points_tensor, calc_distance_map
from l5kit.geometry import transform_points
from tbsim.utils.timer import Timers
from tbsim.utils.planning_utils import ego_sample_planning
from tbsim.policies.common import Action, Plan
from tbsim.policies.base import Policy
from tbsim.utils.ftocp import FTOCP
from tbsim.safety_funcs.utils import scene_centric_batch_to_raw
from tbsim.safety_funcs.cbfs import BackupBarrierCBF
import matplotlib.pyplot as plt

try:
    from Pplan.Sampling.spline_planner import SplinePlanner
    from Pplan.Sampling.trajectory_tree import TrajTree
except ImportError:
    print("Cannot import Pplan")

import tbsim.utils.planning_utils as PlanUtils
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.timer import Timers


class OptimController(Policy):
    """An optimization-based controller"""

    def __init__(
            self,
            dynamics_type,
            dynamics_kwargs,
            step_time: float,
            optimizer_kwargs=None,
    ):
        self.step_time = step_time
        self.optimizer_kwargs = dict() if optimizer_kwargs is None else optimizer_kwargs
        if dynamics_type in ["Unicycle", dynamics.DynType.UNICYCLE]:
            self.dyn = dynamics.Unicycle(
                "dynamics",
                max_steer=dynamics_kwargs["max_steer"],
                max_yawvel=dynamics_kwargs["max_yawvel"],
                acce_bound=dynamics_kwargs["acce_bound"],
            )
        elif dynamics_type in ["Bicycle", dynamics.DynType.BICYCLE]:
            self.dyn = dynamics.Bicycle(
                acc_bound=dynamics_kwargs["acce_bound"],
                ddh_bound=dynamics_kwargs["ddh_bound"],
                max_hdot=dynamics_kwargs["max_yawvel"],
                max_speed=dynamics_kwargs["max_speed"],
            )
        else:
            raise NotImplementedError(
                "dynamics type {} is not implemented", dynamics_type
            )

    def eval(self):
        pass

    def get_action(self, obs, plan: Plan, init_u=None, **kwargs) -> Tuple[Action, Dict]:
        target_pos = plan.positions
        target_yaw = plan.yaws
        target_avails = plan.availabilities
        device = target_pos.device
        num_action_steps = target_pos.shape[-2]
        init_x = get_current_states(obs, dyn_type=self.dyn.type())
        if init_u is None:
            init_u = torch.randn(
                *init_x.shape[:-1], num_action_steps, self.dyn.udim
            ).to(device)
        if target_avails is None:
            target_avails = torch.ones(target_pos.shape[:-1]).to(device)
        targets = torch.cat((target_pos, target_yaw), dim=-1)
        assert init_u.shape[-2] == num_action_steps
        predictions, raw_traj, final_u, losses = optimize_trajectories(
            init_u=init_u,
            init_x=init_x,
            target_trajs=targets,
            target_avails=target_avails,
            dynamics_model=self.dyn,
            step_time=self.step_time,
            data_batch=obs,
            **self.optimizer_kwargs
        )
        action = Action(**predictions)
        return action, {}


class GTPolicy(Policy):
    def __init__(self, device):
        super(GTPolicy, self).__init__(device)

    def eval(self):
        pass

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        action = Action(
            positions=TensorUtils.to_torch(obs["target_positions"], device=self.device),
            yaws=TensorUtils.to_torch(obs["target_yaws"], device=self.device),
        )
        return action, {}

    def get_plan(self, obs, **kwargs) -> Tuple[Plan, Dict]:
        plan = Plan(
            positions=TensorUtils.to_torch(obs["target_positions"], device=self.device),
            yaws=TensorUtils.to_torch(obs["target_yaws"], device=self.device),
            availabilities=TensorUtils.to_torch(obs["target_availabilities"], self.device),
        )
        
        return plan, {}


class ReplayPolicy(Policy):
    def __init__(self, action_log, device):
        super(ReplayPolicy, self).__init__(device)
        self.action_log = action_log

    def eval(self):
        pass

    def get_action(self, obs, step_index=None, **kwargs) -> Tuple[Action, Dict]:
        assert step_index is not None
        scene_index = TensorUtils.to_numpy(obs["scene_index"]).astype(np.int64).tolist()
        track_id = TensorUtils.to_numpy(obs["track_id"]).astype(np.int64).tolist()
        pos = []
        yaw = []
        for si, ti in zip(scene_index, track_id):
            scene_log = self.action_log[str(si)]
            if ti == -1:  # ego
                pos.append(scene_log["ego_action"]["positions"][step_index, 0])
                yaw.append(scene_log["ego_action"]["yaws"][step_index, 0])
            else:
                scene_track_id = scene_log["agents_obs"]["track_id"][0]
                agent_ind = np.where(ti == scene_track_id)[0][0]
                pos.append(scene_log["agents_action"]["positions"][step_index, agent_ind])
                yaw.append(scene_log["agents_action"]["yaws"][step_index, agent_ind])

        # stack and create the temporal dimension
        pos = np.stack(pos, axis=0)[:, None, :]
        yaw = np.stack(yaw, axis=0)[:, None, :]

        action = Action(
            positions=pos,
            yaws=yaw
        )
        return action, {}


class EC_sampling_controller(Policy):
    def __init__(self,ego_sampler,EC_model,agent_planner=None,device="cpu"):
        self.ego_sampler = ego_sampler
        self.EC_model = EC_model
        self.agent_planner = agent_planner
        self.device = device
        self.timer = Timers()
    
    def eval(self):
        self.EC_model.eval()
        if self.agent_planner is not None:
            self.agent_planner.eval()
    
    def get_action(self,obs,**kwargs)-> Tuple[Action, Dict]:
        assert "agent_obs" in kwargs
        agent_obs = kwargs["agent_obs"]
        #TODO: prediction without GC
        if self.agent_planner is not None:
            agent_plan = self.agent_planner(agent_obs)
            agent_plan = torch.cat((agent_plan["predictions"]["positions"],agent_plan["predictions"]["yaws"]),-1).squeeze(1)
        else:
            agent_plan=None
        self.timer.tic("sampling")
        tf = 5
        T = obs["target_positions"].shape[-2]
        bs = obs["history_positions"].shape[0]
        agent_size = agent_obs["image"].shape[0]
        ego_trajs = list()
        #TODO: paralellize this process
        for i in range(bs):
            vel = obs["curr_speed"][i]
            traj0 = torch.tensor([[0., 0., vel, 0, 0., 0., 0.]]).to(vel.device)
            lanes = TensorUtils.to_numpy(obs["ego_lanes"][i])
            lanes = np.concatenate((lanes[...,:2],np.arctan2(lanes[...,3:],lanes[...,2:3])),-1)
            lanes = np.split(lanes,obs["ego_lanes"].shape[1])
            lanes = [lane[0] for lane in lanes if not (lane==0).all()]
            
            def expand_func(x): return self.ego_sampler.gen_trajectory_batch(
                    x, tf, lanes)
            x0 = TrajTree(traj0, None, 0)
            x0.grow_tree(expand_func, 1)
            leaves = x0.get_all_leaves()
            
            if len(leaves) > 0:
                ego_trajs_i = torch.stack([leaf.total_traj for leaf in leaves], 0)
                ego_trajs_i = ego_trajs_i[...,1:,[0,1,4]]
            else:
                ego_trajs_i = torch.cat((obs["target_positions"][i],obs["target_yaws"][i]),-1).unsqueeze(0)
            ego_trajs.append(ego_trajs_i)

        self.timer.toc("sampling")
        self.timer.tic("prediction")
        N = max(ego_trajs_i.shape[0] for ego_trajs_i in ego_trajs)
        cond_traj = torch.zeros([agent_size,N,T,3]).to(obs["speed"].device)
        ego_traj_samples = torch.zeros([bs,N,T,3])
        for i in range(bs):
            ego_traj_samples[i,:ego_trajs[i].shape[0]] = ego_trajs[i]
            agent_idx = torch.where(agent_obs["scene_index"]==obs["scene_index"][i])[0]
            ego_pos_world = GeoUtils.batch_nd_transform_points(ego_trajs[i][...,:2],obs["world_from_agent"][i].unsqueeze(0))
            ego_pos_agent = GeoUtils.batch_nd_transform_points(
                ego_pos_world.tile(agent_idx.shape[0],1,1,1),agent_obs["agent_from_world"][agent_idx,None,None]
                )

            ego_yaw_agent = (ego_trajs[i][...,2:]+obs["yaw"][i]).tile(agent_idx.shape[0],1,1,1)-agent_obs["yaw"][agent_idx].reshape(-1,1,1,1)
            cond_traj[agent_idx,:ego_trajs[i].shape[0]] = torch.cat((ego_pos_agent,ego_yaw_agent),-1)

        EC_pred = self.EC_model.get_EC_pred(agent_obs,cond_traj,agent_plan)
        self.timer.toc("prediction")
        self.timer.tic("planning")
        if "drivable_map" in obs:
            drivable_map = obs["drivable_map"].float()
        else:
            drivable_map = batch_utils().get_drivable_region_map(obs["image"]).float()
        dis_map = calc_distance_map(drivable_map)
        
        opt_traj = list()
        for i in range(bs):
            agent_idx = torch.where(agent_obs["scene_index"]==obs["scene_index"][i])[0]
            N_i = ego_trajs[i].shape[0]

            agent_pos_world = GeoUtils.batch_nd_transform_points(EC_pred["EC_trajectories"][agent_idx,:N_i,...,:2],agent_obs["world_from_agent"][agent_idx,None,None])
            agent_pos_ego = GeoUtils.batch_nd_transform_points(agent_pos_world,obs["agent_from_world"][i].unsqueeze(0))
            agent_yaw_ego =EC_pred["EC_trajectories"][agent_idx,:N_i,...,2:]+agent_obs["yaw"][agent_idx].reshape(-1,1,1,1)-obs["yaw"][i]
            agent_traj = torch.cat((agent_pos_ego,agent_yaw_ego),-1)

            idx = PlanUtils.ego_sample_planning(
                ego_trajs[i].unsqueeze(0),
                agent_traj.transpose(0,1).unsqueeze(0),
                obs["extent"][i:i+1, :2],
                agent_obs["extent"][None,agent_idx,:2],
                agent_obs["type"][None,agent_idx],
                obs["raster_from_world"][i].unsqueeze(0),                
                dis_map[i].unsqueeze(0),
                weights={"collision_weight": 1.0, "lane_weight": 1.0,"likelihood_weight":0.0,"progress_weight":0.0},
            )[0]

            opt_traj.append(ego_trajs[i][idx])
        self.timer.toc("planning")
        print(self.timer)
        opt_traj = torch.stack(opt_traj,0)
        action = Action(positions=opt_traj[...,:2],yaws=opt_traj[...,2:])
        action_info = dict()
        action_info["action_samples"] = {"positions":ego_traj_samples[...,:2],"yaws":ego_traj_samples[...,2:]}
        return action, action_info
        

class ContingencyPlanner(Policy):
    def __init__(self,ego_sampler,predictor,config,agent_planner=None,device="cpu"):
        self.ego_sampler = ego_sampler
        self.predictor = predictor
        self.agent_planner = agent_planner
        self.device = device
        self.config = config
        self.stage = config.stage
        self.num_frames_per_stage = config.num_frames_per_stage
        self.step_time = config.step_time
        self.tf = self.stage*self.num_frames_per_stage*self.step_time
        self.timer = Timers()
    
    def eval(self):
        self.predictor.eval()
        if self.agent_planner is not None:
            self.agent_planner.eval()

    @staticmethod
    def get_scene_obs(ego_obs,agent_obs,goal=None):
        # turn the observations into scene-centric form

        centered_raster_from_agent = ego_obs["raster_from_agent"][0]
        centered_agent_from_raster,_ = torch.linalg.inv_ex(centered_raster_from_agent)
        num_agents = [sum(agent_obs["scene_index"]==i)+1 for i in ego_obs["scene_index"]]
        num_agents = torch.tensor(num_agents,device=ego_obs["scene_index"].device)
        hist_pos_b = list()
        hist_yaw_b=list()
        hist_avail_b = list()
        fut_pos_b = list()
        fut_yaw_b = list()
        fut_avail_b = list()
        agent_from_center_b = list()
        center_from_agent_b = list()
        raster_from_center_b = list()
        center_from_raster_b = list()
        raster_from_world_b = list()
        maps_b = list()
        curr_speed_b = list()
        type_b = list()
        extents_b = list()
        scene_goal_b = list()


        for i,scene_idx in enumerate(ego_obs["scene_index"]):
            agent_idx = torch.where(agent_obs["scene_index"]==scene_idx)[0]
            if goal is not None:
                scene_goal_b.append(torch.cat((torch.zeros_like(goal[0:1]),goal[agent_idx]),0))
            center_from_agent = ego_obs["agent_from_world"][i].unsqueeze(0) @ agent_obs["world_from_agent"][agent_idx]
            center_from_agents = torch.cat((torch.eye(3,device=center_from_agent.device).unsqueeze(0),center_from_agent),0)

            hist_pos_raw = torch.cat((ego_obs["history_positions"][i:i+1],agent_obs["history_positions"][agent_idx]),0)
            hist_yaw_raw = torch.cat((ego_obs["history_yaws"][i:i+1],agent_obs["history_yaws"][agent_idx]),0)
            
            agents_hist_avail = torch.cat((ego_obs["history_availabilities"][i:i+1],agent_obs["history_availabilities"][agent_idx]),0)
            agents_hist_pos = GeoUtils.transform_points_tensor(hist_pos_raw,center_from_agents)*agents_hist_avail.unsqueeze(-1)
            agents_hist_yaw = (hist_yaw_raw+torch.cat((ego_obs["yaw"][i:i+1],agent_obs["yaw"][agent_idx]),0)[:,None,None]-ego_obs["yaw"][i])*agents_hist_avail.unsqueeze(-1)

            hist_pos_b.append(agents_hist_pos)
            hist_yaw_b.append(agents_hist_yaw)
            hist_avail_b.append(agents_hist_avail)
            if agent_obs["target_availabilities"].shape[1]<ego_obs["target_availabilities"].shape[1]:
                pad_shape=(agent_obs["target_availabilities"].shape[0],ego_obs["target_availabilities"].shape[1]-agent_obs["target_availabilities"].shape[1])
                agent_obs["target_availabilities"] = torch.cat((agent_obs["target_availabilities"],torch.zeros(pad_shape,device=agent_obs["target_availabilities"].device)),1)
            agents_fut_avail = torch.cat((ego_obs["target_availabilities"][i:i+1],agent_obs["target_availabilities"][agent_idx]),0)
            if agent_obs["target_positions"].shape[1]<ego_obs["target_positions"].shape[1]:
                pad_shape=(agent_obs["target_positions"].shape[0],ego_obs["target_positions"].shape[1]-agent_obs["target_positions"].shape[1],*agent_obs["target_positions"].shape[2:])
                agent_obs["target_positions"] = torch.cat((agent_obs["target_positions"],torch.zeros(pad_shape,device=agent_obs["target_positions"].device)),1)
                pad_shape=(agent_obs["target_yaws"].shape[0],ego_obs["target_yaws"].shape[1]-agent_obs["target_yaws"].shape[1],*agent_obs["target_yaws"].shape[2:])
                agent_obs["target_yaws"] = torch.cat((agent_obs["target_yaws"],torch.zeros(pad_shape,device=agent_obs["target_yaws"].device)),1)
            fut_pos_raw = torch.cat((ego_obs["target_positions"][i:i+1],agent_obs["target_positions"][agent_idx]),0)
            fut_yaw_raw = torch.cat((ego_obs["target_yaws"][i:i+1],agent_obs["target_yaws"][agent_idx]),0)
            agents_fut_pos = GeoUtils.transform_points_tensor(fut_pos_raw,center_from_agents)*agents_fut_avail.unsqueeze(-1)
            agents_fut_yaw = (fut_yaw_raw+torch.cat((ego_obs["yaw"][i:i+1],agent_obs["yaw"][agent_idx]),0)[:,None,None]-ego_obs["yaw"][i])*agents_fut_avail.unsqueeze(-1)
            fut_pos_b.append(agents_fut_pos)
            fut_yaw_b.append(agents_fut_yaw)
            fut_avail_b.append(agents_fut_avail)

            curr_yaw = agents_hist_yaw[:,-1]
            curr_pos = agents_hist_pos[:,-1]
            agents_from_center = GeoUtils.transform_matrices(-curr_yaw.flatten(),torch.zeros_like(curr_pos))@GeoUtils.transform_matrices(torch.zeros_like(curr_yaw).flatten(),-curr_pos)
                                 
            # raster_from_center = centered_raster_from_agent @ agents_from_center
            center_from_raster = center_from_agents @ centered_agent_from_raster

            # raster_from_world = torch.cat((ego_obs["raster_from_world"][i:i+1],agent_obs["raster_from_world"][agent_idx]),0)


            agent_from_center_b.append(agents_from_center)
            center_from_agent_b.append(center_from_agents)
            # raster_from_center_b.append(raster_from_center)
            center_from_raster_b.append(center_from_raster)
            # raster_from_world_b.append(raster_from_world)

            maps = torch.cat((ego_obs["image"][i:i+1],agent_obs["image"][agent_idx]),0)
            curr_speed = torch.cat((ego_obs["curr_speed"][i:i+1],agent_obs["curr_speed"][agent_idx]),0)
            agents_type = torch.cat((ego_obs["type"][i:i+1],agent_obs["type"][agent_idx]),0)
            agents_extent = torch.cat((ego_obs["extent"][i:i+1],agent_obs["extent"][agent_idx]),0)
            maps_b.append(maps)
            curr_speed_b.append(curr_speed)
            type_b.append(agents_type)
            extents_b.append(agents_extent)
        if goal is not None:
            scene_goal = pad_sequence(scene_goal_b,batch_first=True,padding_value=0)
        else:
            scene_goal = None
        
        scene_obs = dict(
        num_agents=num_agents,
        image=pad_sequence(maps_b,batch_first=True,padding_value=0),
        target_positions=pad_sequence(fut_pos_b,batch_first=True,padding_value=0),
        target_yaws=pad_sequence(fut_yaw_b,batch_first=True,padding_value=0),
        target_availabilities=pad_sequence(fut_avail_b,batch_first=True,padding_value=0),
        history_positions=pad_sequence(hist_pos_b,batch_first=True,padding_value=0),
        history_yaws=pad_sequence(hist_yaw_b,batch_first=True,padding_value=0),
        history_availabilities=pad_sequence(hist_avail_b,batch_first=True,padding_value=0),
        curr_speed=pad_sequence(curr_speed_b,batch_first=True,padding_value=0),
        centroid=ego_obs["centroid"],
        yaw=ego_obs["yaw"],
        agent_type=pad_sequence(type_b,batch_first=True,padding_value=0),
        extent=pad_sequence(extents_b,batch_first=True,padding_value=0),
        raster_from_agent=ego_obs["raster_from_agent"],
        agent_from_raster=centered_agent_from_raster,
        # raster_from_center=pad_sequence(raster_from_center_b,batch_first=True,padding_value=0),
        # center_from_raster=pad_sequence(center_from_raster_b,batch_first=True,padding_value=0),
        agents_from_center=pad_sequence(agent_from_center_b,batch_first=True,padding_value=0),
        center_from_agents=pad_sequence(center_from_agent_b,batch_first=True,padding_value=0),
        # raster_from_world=pad_sequence(raster_from_world_b,batch_first=True,padding_value=0),
        agent_from_world=ego_obs["agent_from_world"],
        world_from_agent=ego_obs["world_from_agent"],
        )
        return scene_obs, scene_goal
        
    def get_ego_samples(self,obs):
        self.timer.tic("sampling")
        bs = obs["history_positions"].shape[0]
        ego_trajs = list()
        #TODO: paralellize this process
        ego_nodes_by_stage = list()
        T = self.stage*self.num_frames_per_stage
        for i in range(bs):
            vel = obs["curr_speed"][i]
            traj0 = torch.tensor([[0.0, 0.0, vel, 0, 0.0, 0., 0.]]).to(vel.device)
            if "ego_lanes" in obs:
                lanes = TensorUtils.to_numpy(obs["ego_lanes"][i])
                if lanes.shape[-1]==4:
                    lanes = np.concatenate((lanes[...,:2],np.arctan2(lanes[...,3:],lanes[...,2:3])),-1)

                lanes = [lanes[i] for i,lane in enumerate(lanes) if lane.any()]
            else:
                lanes = None
            
            def expand_func(x): return self.ego_sampler.gen_trajectory_batch(
                    x, self.step_time*self.num_frames_per_stage, lanes,N=self.num_frames_per_stage+1,max_children=15)
            
            x0 = TrajTree(traj0, None, 0)
            x0.grow_tree(expand_func, self.stage)
            
            nodes,_ = TrajTree.get_nodes_by_level(x0,depth=self.stage)
            if len(nodes[0])==0:
                ego_nodes_by_stage.append(None)
            else:
                ego_nodes_by_stage.append(nodes)

            if len(nodes[self.stage]) > 0:
                ego_trajs_i = torch.stack([leaf.total_traj for leaf in nodes[self.stage]], 0).float()
                ego_trajs_i = ego_trajs_i[...,1:,[0,1,4]] #TODO: make it less hacky
            else:
                ego_trajs_i = torch.cat((obs["target_positions"][i,:T],obs["target_yaws"][i,:T]),-1).unsqueeze(0)
            ego_trajs.append(ego_trajs_i)
        
        self.timer.toc("sampling")
        return ego_trajs,ego_nodes_by_stage
    def get_prediction(self,obs,scene_obs,scene_goal,ego_trajs):
        bs = obs["history_positions"].shape[0]
        self.timer.tic("prediction")
        Ns = max(ego_trajs_i.shape[0] for ego_trajs_i in ego_trajs)
        T = self.stage*self.num_frames_per_stage
        ego_traj_samples = torch.zeros([bs,Ns,T,3],dtype=torch.float32,device=obs["curr_speed"].device)
        cond_idx = list()
        for i in range(bs):
            cond_idx.append(0)
            ego_traj_samples[i,:ego_trajs[i].shape[0]] = ego_trajs[i]
        
        agent_pred = self.predictor.predict(scene_obs,cond_traj=ego_traj_samples,cond_idx=cond_idx,goal=scene_goal)
        
        self.timer.toc("prediction")
        return agent_pred, ego_traj_samples
    
    def get_action(self,obs,**kwargs)-> Tuple[Action, Dict]:
        assert "agent_obs" in kwargs
        agent_obs = kwargs["agent_obs"]
        T = self.stage*self.num_frames_per_stage
        bs = obs["history_positions"].shape[0]
        if self.agent_planner is not None:
            obs_keys = ["image","drivable_map","agent_from_raster"]
            combined_obs={k:torch.cat((obs[k],agent_obs[k]),0) for k in obs_keys}
            combined_plan,info = self.agent_planner.get_plan(combined_obs)
            ego_likelihood = info["location_map"][:bs]

            
            agent_plan = torch.cat((combined_plan.positions[bs:],combined_plan.yaws[bs:]),-1)
            agent_plan = agent_plan[...,-1,:]

        else:
            agent_plan=None
            ego_likelihood = None
        scene_obs,scene_goal = self.get_scene_obs(obs,agent_obs,agent_plan)
        ego_trajs,ego_nodes_by_stage = self.get_ego_samples(obs)
        agent_pred, ego_traj_samples = self.get_prediction(obs,scene_obs,scene_goal,ego_trajs)
        self.timer.tic("planning")
        if "drivable_map" in obs:
            drivable_map = obs["drivable_map"].float()
        else:
            drivable_map = batch_utils().get_drivable_region_map(obs["image"]).float()
        dis_map = calc_distance_map(drivable_map)
        opt_traj = list()
        print(kwargs["step_index"])
        for i in range(bs):
            N_i = ego_trajs[i].shape[0]
            Na_i = scene_obs["num_agents"][i]
            agent_traj_local = agent_pred["trajectories"][i,:N_i,:,1:Na_i].float()
            prob = agent_pred["p"][i,:N_i]
            if ego_nodes_by_stage[i] is not None:
                motion_policy = PlanUtils.contingency_planning(ego_nodes_by_stage[i],
                                                            obs["extent"][i,:2],
                                                            agent_traj_local,
                                                            prob,
                                                            scene_obs["extent"][i,1:Na_i,:2],
                                                            scene_obs["agent_type"][i,1:Na_i],
                                                            obs["raster_from_agent"][i],
                                                            dis_map[i],
                                                            {"collision_weight": 10.0, "lane_weight": 1.0, "progress_weight": 0.3,"likelihood_weight": 0.10},
                                                            self.num_frames_per_stage,
                                                            self.predictor.algo_config.vae.latent_dim,
                                                            self.config.step_time,
                                                            log_likelihood=ego_likelihood[i],
                                                            pert_std = 0.2
                                                            )
                
            else:
                raise ValueError("ego plan is empty!")
                            
            opt_traj.append(motion_policy.get_plan(None,self.stage*self.num_frames_per_stage))

        self.timer.toc("planning")
        print(self.timer)
        opt_traj = torch.stack(opt_traj,0)
        action = Action(positions=opt_traj[...,:2],yaws=opt_traj[...,2:])
        action_info = dict()
        for i in range(bs):
            perm = torch.randperm(ego_trajs[i].shape[0],device=ego_traj_samples.device)
            ego_traj_samples[i,:ego_trajs[i].shape[0]] = ego_traj_samples[i,perm]

        action_info["action_samples"] = {"positions":ego_traj_samples[...,:2],"yaws":ego_traj_samples[...,2:]}
        return action, action_info

class CBFQPController(Policy):
    def __init__(self, device, step_time, predictor, solver="casadi", *args, **kwargs):
        super().__init__(device, *args, **kwargs)
        self.device = device
        self.step_time = step_time
        self.predictor = predictor
        self.solver=solver
        self.cbf = BackupBarrierCBF(T_horizon=4, alpha=3, veh_veh=True, saturate_cbf=True, backup_controller_type="idle")
        self.firstStep = True
        self.ego_vel = 7 

    def eval(self):
        self.predictor.eval()
    
    def get_action(self, obs_dict, **kwargs):
        num_steps_to_act_on = 5
        dt = obs_dict["dt"][0].item()

        # Is there a coarse plan? 
        if "coarse_plan" in kwargs:
            coarse_plan = kwargs["coarse_plan"]
            ref_positions = coarse_plan.positions
            ref_yaws = coarse_plan.yaws
            mask = torch.ones_like(ref_positions[...,0],dtype=np.bool)
            ref_vel = dynamics.Unicycle.calculate_vel(ref_positions,ref_yaws,self.step_time,mask)
            xref = torch.cat((ref_positions,ref_vel,ref_yaws),-1)
        else:
            xref = None
        

        states = []
        if False: # Set Up Sim 
            self.firstStep = False
            ego_yaw = obs_dict["curr_agent_state"][0, -1]
            ego_pos = obs_dict["curr_agent_state"][0, :2]
            ego_vel = obs_dict["curr_speed"][0]
            agent_yaw = obs_dict["curr_agent_state"][1, -1]
            agent_pos = obs_dict["curr_agent_state"][1, :2]
            agent_vel = obs_dict["curr_speed"][1]

            dist = 0

            from_agent_to_ego = [dist*torch.cos(ego_yaw)+ego_pos[0] - agent_pos[0], dist*torch.sin(ego_yaw)+ego_pos[1] - agent_pos[1], ego_yaw  - agent_yaw]

            agent_from_ego = torch.tensor(
                        [
                            [torch.cos(agent_yaw - ego_yaw), torch.sin(agent_yaw - ego_yaw)],
                            [-torch.sin(agent_yaw - ego_yaw), torch.cos(agent_yaw - ego_yaw)],
                        ], 
                        device = xref.device
                    )

            diff_agent_ref = agent_from_ego @ torch.tensor(from_agent_to_ego[0:2], device = xref.device)

            plan = 0 * xref 
            plan[1,:,0:2] = diff_agent_ref
            # plan[1,:,2] = ego_vel
            # plan[1,:,3] = ego_yaw - agent_yaw
            print(plan[1,0,:])
            print(ego_pos)
            print(agent_pos)

            action = Action(positions=plan[...,:2], yaws=plan[...,3:])
            return action, {}


        else: # Run Sim 
            for idx in range(xref.shape[0]):
                state_veh_idx = []
                curr_yaw = obs_dict["curr_agent_state"][idx, -1]
                curr_pos = obs_dict["curr_agent_state"][idx, :2]
                curr_vel = obs_dict["curr_speed"][idx]

                # Default Ego Controller
                time_steps = torch.linspace(1,20,20, device = curr_pos.device)
                if idx == 0 : 
                    v_des = 7 
                    vs = torch.linspace(curr_vel.item()+(v_des-curr_vel.item())/20 ,v_des,20, device = curr_pos.device)
                    # thetas = torch.linspace((obs_dict["curr_agent_state"][2, -1]-curr_yaw.item())/20, obs_dict["curr_agent_state"][2, -1]-curr_yaw.item(), 20, device = curr_pos.device ) # track heading of forward vehicle
                    xs = []
                    for i in range(20): 
                        xs.append(sum(vs[0:i+1])*dt)
                    xs = torch.tensor(xs)
                    xref[0,:,0] =  xs
                    xref[0,:,1] = 0
                    xref[0,:,2] = vs
                    xref[0,:,3] = 0
                elif idx == 2: 
                    # default other agent
                    if True:
                        v = 3
                        vs = v*torch.ones(20, device = curr_pos.device)
                        xs = []
                        for i in range(20): 
                            xs.append(sum(vs[0:i+1])*dt)
                        xs = torch.tensor(xs)
                        xref[idx,:,0] = xs
                        xref[idx,:,1] = 0
                        xref[idx,:,2] = vs
                        xref[idx,:,3] = 0
                        self.firstStep=False
                    else: 
                        v_des = 0
                        vs = torch.linspace(curr_vel.item()+(v_des-curr_vel.item())/20 ,v_des,20, device = curr_pos.device)
                        xs = []
                        for i in range(20): 
                            xs.append(sum(vs[0:i+1])*dt)
                        xs = torch.tensor(xs)
                        xref[0,:,0] =  xs
                        xref[0,:,1] = 0
                        xref[0,:,2] = vs
                        xref[0,:,3] = 0

                # Get current state and rotation matrix
                curr_state = curr_pos.tolist()
                curr_state.append(curr_vel.item())
                curr_state.append(curr_yaw.item())
                state_veh_idx.append(curr_state)
                world_from_agent = torch.tensor(
                        [
                            [torch.cos(curr_yaw), torch.sin(curr_yaw)],
                            [-torch.sin(curr_yaw), torch.cos(curr_yaw)],
                        ], 
                        device = xref.device
                    )
                for t_step in range(xref.shape[1]): 
                    next_state = torch.zeros(4)
                    next_state[:2] = coarse_plan.positions[idx, t_step] @ world_from_agent + curr_pos
                    next_state[2] = curr_vel + xref[idx, t_step,2]
                    next_state[3] = curr_yaw + coarse_plan.yaws[idx, t_step, 0]
                    state_veh_idx.append(next_state.tolist())
                states.append(state_veh_idx)
            
            states = torch.tensor(states)
            coarse_batch = {
                "states" : states[None,...], 
                "extent" : obs_dict["extent"][None,...], 
                "dt"     : obs_dict["dt"],
            }

            yaw_rate = [xref[0,0,3].item()]
            for i in range(1, xref.shape[1]): yaw_rate.append((xref[0,i,3] - xref[0,i-1,3] ).item())
            yaw_rate = torch.tensor(yaw_rate)/dt

            accel = [xref[0,0,2]-obs_dict["curr_speed"][0]]
            for i in range(1, xref.shape[1]): accel.append((xref[0,i,2] - xref[0,i-1,2] ).item())
            accel = torch.tensor(accel)/dt

            # Build CBFQP for n time steps
            safe_inputs = []
            for t_step in range(num_steps_to_act_on): # TODO: figure out where this variable is coming from and point it here
                # Current Batch 
                current_batch ={
                    "states" : (coarse_batch["states"][...,t_step:t_step+1,:]).to(coarse_batch["extent"].device), 
                    "extent" : obs_dict["extent"][None,...], 
                    "dt"     : obs_dict["dt"], 
                }
                # Set up problem 
                ego_u = cp.Variable(2)
                ego_des_input = np.array([accel[t_step], yaw_rate[t_step]])
                objective = cp.Minimize((ego_des_input[0] - ego_u[0])**2 + 100*(ego_des_input[1] - ego_u[1])**2)
                constraints = []
                with torch.enable_grad():
                    data = self.cbf.process_batch(current_batch)
                    data.requires_grad_()
                    h_vals = self.cbf(data)
                    LfhA, LfhB, LghA, LghB = self.cbf.get_barrier_bits(h_vals, data)
                h_vals = h_vals[0].cpu().detach().numpy()
                print("h_val = ", h_vals[1])
                LfhA = LfhA.cpu().detach().numpy()
                LfhB = LfhB.cpu().detach().numpy()
                LghA = LghA.cpu().detach().numpy()
                LghB = LghB.cpu().detach().numpy()
                for i in range(len(LfhA)): 
                    # worst case is positive 1/2 g acceleration, negative 1 g braking and 4 second u turn 
                    sign_agent_vel = torch.sign(current_batch["states"][0,i+1,0,2]).item()
                    if False: # Model worst case other agents 
                        if False: # worst case other agents
                            worst_case = [[4.5*sign_agent_vel,np.pi/4], [4.5*sign_agent_vel,-np.pi/4], [-9*sign_agent_vel,np.pi/4], [-9*sign_agent_vel,-np.pi/4]] 
                        else: # ego_only 
                            worst_case = [[0,0]]
                        for input in worst_case:
                            u_worst_case = np.array(input)
                            constraints.append(LfhA[i] + LfhB[i] + LghA[i]@ego_u + LghB[i]@u_worst_case >= -self.cbf.alpha * h_vals[i])
                    else: # even split decentralized
                        constraints.append(1.0/2*(LfhA[i] + LfhB[i]) + LghA[i]@ego_u >= -1.0/2*self.cbf.alpha*h_vals[i])


                sign_ego_vel = torch.sign(current_batch["states"][0,0,0,2]).item()
                prob = cp.Problem(objective, constraints)
                try:
                    result = prob.solve()
                    if prob.status == "optimal" or prob.status == "optimal_inaccurate": 
                        ego_safe_input = ego_u.value
                    else: 
                        print(prob.status)
                        ego_safe_input = np.array([-sign_ego_vel*9,0])
                except:
                    ego_safe_input = np.array([-sign_ego_vel*9,0])
                
                #saturate: 
                if ego_safe_input[0] >=9: ego_safe_input[0] = 9
                if ego_safe_input[0] <=-9: ego_safe_input[0] = -9
                if ego_safe_input[1] >= np.pi/4: ego_safe_input[1] = np.pi/4
                if ego_safe_input[1] <= -np.pi/4: ego_safe_input[1] = -np.pi/4

                print("des: ", ego_des_input, "\t safe: ", ego_safe_input, "\t norm: ", np.linalg.norm(ego_des_input-ego_safe_input))
                coarse_batch["states"][0,0,t_step+1,:] = torch.tensor([
                    coarse_batch["states"][0,0,t_step,0] + coarse_batch["states"][0,0,t_step,2]*torch.cos(coarse_batch["states"][0,0,t_step,3])*dt, 
                    coarse_batch["states"][0,0,t_step,1] + coarse_batch["states"][0,0,t_step,2]*torch.sin(coarse_batch["states"][0,0,t_step,3])*dt, 
                    coarse_batch["states"][0,0,t_step,2] + ego_safe_input[0]*dt, 
                    coarse_batch["states"][0,0,t_step,3] + ego_safe_input[1]*dt   
                ])
                safe_inputs.append(ego_safe_input)

            plan = xref.clone()
            v = states[0,0,2]
            theta = 0 
            plan[0,0,0] =     v*np.cos(theta)*dt      
            plan[0,0,1] =     v*np.sin(theta)*dt      
            plan[0,0,2] = v + safe_inputs[0][0]*dt
            plan[0,0,3] =     safe_inputs[0][1]*dt
            for i in range(num_steps_to_act_on-1): 
                plan[0,i+1,0]  =  plan[0,i,0] + plan[0,i,2]*torch.cos(plan[0,i,3])*dt      
                plan[0,i+1,1]  =  plan[0,i,1] + plan[0,i,2]*torch.sin(plan[0,i,3])*dt      
                plan[0,i+1,2]  =  plan[0,i,2] + safe_inputs[i+1][0]*dt
                plan[0,i+1,3]  =  plan[0,i,3] + safe_inputs[i+1][1]*dt  

        action = Action(positions=plan[...,:2], yaws=plan[...,3:])
        return action, {}


class ModelPredictiveController(Policy):
    def __init__(self, device, step_time, predictor, solver="casadi", *args, **kwargs):
        super().__init__(device, *args, **kwargs)
        self.device = device
        self.step_time = step_time
        self.predictor = predictor
        self.solver=solver


    def eval(self):
        self.predictor.eval()
    
    def get_action(self, obs_dict, **kwargs):
        bs,horizon = obs_dict["target_positions"].shape[:2]

        agent_preds, _ = self.predictor.get_prediction(
            obs_dict)
        agent_preds = TensorUtils.to_numpy(agent_preds.to_dict())
        obs_np = TensorUtils.to_numpy(obs_dict)
        plan = list()

        if "coarse_plan" in kwargs:
            coarse_plan = kwargs["coarse_plan"]
            
            ref_positions = TensorUtils.to_numpy(coarse_plan.positions)
            ref_yaws = TensorUtils.to_numpy(coarse_plan.yaws)
            mask = np.ones_like(ref_positions[...,0],dtype=np.bool)
            ref_vel = dynamics.Unicycle.calculate_vel(ref_positions,ref_yaws,self.step_time,mask)
            xref = np.concatenate((ref_positions,ref_vel,ref_yaws),-1)
        else:
            xref = None
        x0 = batch_utils().get_current_states(obs_dict,dyn_type=dynamics.Unicycle)
        x0 = TensorUtils.to_numpy(x0)
        for i in range(bs):
            planner = FTOCP(horizon, 1, self.step_time, obs_dict["extent"][i,1].item(), obs_dict["extent"][i,0].item())
            
            
            x0_i = x0_i=x0[i]
            if xref is not None:
                xref_i = xref[i]
                xref_extended = np.concatenate((x0_i[None,:],xref_i))
                uref_i = dynamics.Unicycle.inverse_dyn(xref_extended[:-1],xref_extended[1:],self.step_time)
                xref_i = np.clip(xref_i,planner.x_lb,planner.x_ub)
                uref_i = np.clip(uref_i,planner.u_lb,planner.u_ub)
                x0_guess = np.concatenate((x0_i[np.newaxis,:],xref_i.repeat(planner.M,axis=0)),0).flatten()
                u0_guess = uref_i.repeat(planner.M,axis=0).flatten()
                planner.xGuessTot = np.concatenate((x0_guess,u0_guess,np.zeros(planner.N*planner.M)))

            else:
                vdes = np.clip(x0_i[2],2.0,25.0)
                if "ego_lanes" in obs_dict and not (obs_dict["ego_lanes"][i,0]==0).all():
                    lane = TensorUtils.to_numpy(obs_dict["ego_lanes"][i,0])
                    lane = np.concatenate([lane[:,:2],np.arctan2(lane[:,3],lane[:,2])[:,np.newaxis]],-1)
                    xref_i = PlanUtils.obtain_ref(lane, x0_i[0:2], vdes, horizon, self.step_time)
                else:
                    s1 = (vdes * np.arange(1, horizon + 1) * self.step_time).reshape(-1,1)
                    xref_i = np.concatenate((np.cos(x0_i[3])*s1,np.sin(x0_i[3])*s1,np.zeros([s1.shape[0],2])),-1)
                    xref_i = xref+x0_i[np.newaxis,:]

            agent_idx = np.where(obs_np["all_other_agents_types"][i]>0)[0]
            ypreds = agent_preds["positions"][i,agent_idx,np.newaxis]
            agent_extent = np.max(obs_np["all_other_agents_history_extents"][i,agent_idx,:,:2],axis=-2)
            planner.buildandsolve(x0_i, ypreds, agent_extent, xref_i, np.ones(1))
            xplan = planner.xSol[1:].reshape((horizon, 4))
            plan.append(TensorUtils.to_torch(xplan,device=self.device))
        plan = torch.stack(plan,0)
        action = Action(positions=plan[...,:2],yaws=plan[...,3:])
        breakpoint()
        return action, {}


class HierSplineSamplingPolicy(Policy):
    def __init__(self, device, step_time, predictor,ego_sampler=None, *args, **kwargs):
        super().__init__(device, *args, **kwargs)
        self.device = device
        if ego_sampler is not None:
            self.ego_sampler = ego_sampler
        else:
            self.ego_sampler = self.get_ego_sampler()
        self.step_time = step_time
        self.predictor = predictor
    
    def get_ego_sampler(self):
        ego_sampler = SplinePlanner(self.device)
        return ego_sampler
    def eval(self):
        self.predictor.eval()
    
    def get_action(self, obs_dict, **kwargs):
        bs,horizon = obs_dict["target_positions"].shape[:2]

        agent_preds, _ = self.predictor.get_prediction(obs_dict)
        ego_trajs = list()
        #TODO: paralellize this process
        for i in range(bs):
            vel = obs_dict["curr_speed"][i]

            traj0 = torch.tensor([[0., 0., vel, 0, 0., 0., 0.]]).to(vel.device)
            lanes = TensorUtils.to_numpy(obs_dict["ego_lanes"][i])
            lanes = np.concatenate((lanes[...,:2],np.arctan2(lanes[...,3:],lanes[...,2:3])),-1)
            lanes = np.split(lanes,obs_dict["ego_lanes"].shape[1])
            lanes = [lane[0] for lane in lanes if not (lane==0).all()]
            
            def expand_func(x): return self.ego_sampler.gen_trajectory_batch(
                    x, self.step_time*horizon, lanes,N=horizon+1)
            x0 = TrajTree(traj0, None, 0)
            x0.grow_tree(expand_func, 1)
            nodes,_ = TrajTree.get_nodes_by_level(x0,depth=1)
            leaves = nodes[1]
            
            if len(leaves) > 0:
                ego_trajs_i = torch.stack([leaf.total_traj for leaf in leaves], 0)
                ego_trajs_i = ego_trajs_i[...,1:,[0,1,4]]
            else:
                ego_trajs_i = torch.cat((obs_dict["target_positions"][i],obs_dict["target_yaws"][i]),-1).unsqueeze(0)
            ego_trajs.append(ego_trajs_i)
        if "drivable_map" in obs_dict:
            drivable_map = obs_dict["drivable_map"].float()
        else:    
            drivable_map = batch_utils().get_drivable_region_map(obs_dict["image"]).float()
        dis_map = calc_distance_map(drivable_map)
        plan = list()
        for i in range(bs):
            
            agent_idx = torch.where(obs_dict["all_other_agents_types"][i]>0)[0]
            agent_extent = torch.max(obs_dict["all_other_agents_history_extents"][i,agent_idx,:,:2],axis=-2)[0]
            agent_traj = torch.cat([agent_preds.positions[i,agent_idx],agent_preds.yaws[i,agent_idx]],-1)
            idx = PlanUtils.ego_sample_planning(
                ego_trajs[i].unsqueeze(0),
                agent_traj.unsqueeze(0),
                obs_dict["extent"][i:i+1, :2],
                agent_extent.unsqueeze(0),
                obs_dict["all_other_agents_types"][i:i+1,agent_idx],
                obs_dict["raster_from_world"][i].unsqueeze(0),                
                dis_map[i].unsqueeze(0),
                weights={"collision_weight": 10.0, "lane_weight": 1.0,"likelihood_weight":0.0,"progress_weight":0.0},
            )[0]
            plan.append(ego_trajs[i][idx])
        plan = torch.stack(plan,0)
        action = Action(positions=plan[...,:2],yaws=plan[...,2:])
        return action, {}


