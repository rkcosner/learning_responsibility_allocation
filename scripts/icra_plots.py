import matplotlib.pyplot as plt 
from matplotlib import cm 
import numpy as np 
import torch 
import pickle 
from tqdm import tqdm 

from tbsim.utils.trajdata_utils import parse_trajdata_batch


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

HERO_PLOT = True 
GAMMA_SURFACE = False
SAME_LANE_CL = False
INTERSECTION_CL = False
CL_STATS = True

pxls_per_meter = 2
pxl_center = [56, 112]
pxls_per_meter = 2
white = [1.0, 1.0, 1.0]
black = [0.0, 0.0, 0.0]
extent = [4.0840, 1.7300, 1.5620] # x,y,z from nuscenes
ego_extent = [4.0840, 1.7300, 1.5620]

def generate_3channel_image(batch):  
    
    colors = [[0.5, 0.0, 0.0],[0.0, 0.5, 0.0],[0.0, 0.0, 0.5],[0.0, 0.4, 0.4],[0.5, 0.3, 0.0],[0.5, 0.5, 0.0],[0.5, 0.0, 0.5]]

    semantic_channels = batch['image'][1:].cpu().detach().numpy()
    N_pxls = semantic_channels.shape[-1]
    visualizer_image = np.zeros((N_pxls, N_pxls, 3)) + 0.5
    for j, color in enumerate(colors): 
        color_np = np.array([color]) 
        s_channel = semantic_channels[j][...,None]
        visualizer_image += (s_channel @ color_np).squeeze()

    # Draw Ego Vehicle
    visualizer_image[pxl_center[1],pxl_center[0]] = white
    xs = torch.arange(-ego_extent[0]/2*pxls_per_meter, ego_extent[0]/2+1*pxls_per_meter, step = 1).int()
    ys = torch.arange(-ego_extent[1]/2*pxls_per_meter, ego_extent[1]/2+1*pxls_per_meter, step = 1).int()
    for x in xs: 
        for y in ys: 
            if x != 0 or y !=0:
                visualizer_image[pxl_center[1]+y, pxl_center[0]+x] = black 

    return visualizer_image 

def replay_sim(batch,simscene, net, device ):
    agent_idx = -1
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
            scene_action[agent.name] =  np.array(states[a, time, [0,1,3]])
        simscene.step(scene_action)
        obs = parse_trajdata_batch(simscene.get_obs())
        
        # Get network inputs
        col_states.append(states[None, -1, time,:])
        ego_states.append(states[None, 0,time,:])
        ego_imgs.append(obs["image"][None, 0,-8:,...])
        col_imgs.append(obs["image"][None, -1,-8:,...])

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
    for name in input_batch.keys(): 
        input_batch[name] = input_batch[name].to("cuda")


    gammas = net(input_batch)

    ego_gammas = gammas["gammas_A"][:,0,0,0]
    col_gammas = gammas["gammas_B"][:,0,0,0]

    data = cbf.process_batch(input_batch)
    data.requires_grad = True
    h_vals = cbf(data) 
    dhdx, = torch.autograd.grad(h_vals, inputs = data, grad_outputs = torch.ones_like(h_vals), create_graph=True)

    dhdxA = dhdx[:,:,0:4]
    dhdxB = dhdx[:,:,4:8]
    (fA, fB, gA, gB) = net.unicycle_dynamics(ego_states, col_states)
    fA = fA.to("cuda")
    fB = fB.to("cuda")
    gA = gA.to("cuda")
    gB = gB.to("cuda")


    LfhA = torch.bmm(dhdxA, fA).squeeze() 
    LfhB = torch.bmm(dhdxB, fB).squeeze()
    LghA = torch.bmm(dhdxA, gA)
    LghB = torch.bmm(dhdxB, gB) 

    natural_dynamics  = (LfhA + LfhB + cbf.alpha * h_vals[:,0])
    natural_dynamics  /= 2.0 
    
    ego_inputs = batch["inputs"][0,:,:,None].to("cuda")
    col_inputs = batch["inputs"][-1,:,:,None].to("cuda")

    LghAuA = torch.bmm(LghA, ego_inputs).squeeze()
    LghBuB = torch.bmm(LghB, col_inputs).squeeze()

    constraintA = natural_dynamics + LghAuA - ego_gammas
    constraintB = natural_dynamics + LghBuB - col_gammas

    return constraintA, constraintB

if HERO_PLOT: 
    pass

if GAMMA_SURFACE: 
    import json

    from tbsim.configs.algo_config import ResponsibilityConfig
    from tbsim.configs.base import AlgoConfig
    from tbsim.algos.factory import algo_factory
    from tbsim.safety_funcs.utils import generate_static_gamma_plots, plot_static_gammas_inline, plot_static_gammas_traj
    from matplotlib.ticker import FormatStrFormatter

    import pickle
    import matplotlib.ticker as mtick


    LABEL_CONTOUR = False

    ego_vel = 7 
    axis_font_size = 6 
    rel_vel_max = 7

    path2way_local = "/home/rkcosner/Documents/tbsim/tbsim/safety_funcs/static_scenes/batch2wayDivider.pickle"
    path4way_local = "/home/rkcosner/Documents/tbsim/tbsim/safety_funcs/static_scenes/batch4way.pickle"

    # Load Gamma Model
    file = open("/home/rkcosner/Documents/tbsim/checkpoints/idling_checkpoint/run10/run0/config.json")
    algo_cfg = AlgoConfig()
    algo_cfg.algo = ResponsibilityConfig()
    external_algo_cfg = json.load(file)
    algo_cfg.update(**external_algo_cfg)
    algo_cfg.algo.update(**external_algo_cfg["algo"])
    device = "device" 
    modality_shapes = dict()
    gamma_algo = algo_factory(algo_cfg, modality_shapes)
    checkpoint_path = "/home/rkcosner/Documents/tbsim/checkpoints/idling_checkpoint/run10/run0/checkpoints/iter9000_ep1_valLoss0.00.ckpt"
    checkpoint = torch.load(checkpoint_path)
    gamma_algo.load_state_dict(checkpoint["state_dict"])
    net = gamma_algo.nets["policy"].cuda()

    X_2way, Y_2way, gammas_A_2way, visualizer_image_2way = plot_static_gammas_inline(net, type=2, on_ngc=False, return_data = True)
    X, Y, gammas_A, visualizer_image =  plot_static_gammas_traj(net, type=4, on_ngc=False, return_data = True)
    X_line, Y_line, gammas_A_line, visualizer_image_line = plot_static_gammas_inline(net, type=4, on_ngc=False, return_data = True)

    vmax = max([gammas_A_2way.max(), gammas_A.max(),  gammas_A_line.max()]).item()
    vmin = min([gammas_A_2way.min(), gammas_A.min(),  gammas_A_line.min()]).item()

    fig, axes = plt.subplots(1,5, gridspec_kw={"width_ratios":[1,1,1,1,1]})
    fig.subplots_adjust(wspace=0.1, hspace=0)


    axes[0].imshow(np.flip(visualizer_image_2way, axis=0))
    axes[0].get_xaxis().set_ticks([])
    axes[0].get_yaxis().set_ticks([])
    axes[0].set_xlabel("Scene 1")


    out_2way = axes[1].contourf(X_2way, Y_2way,gammas_A_2way.cpu().numpy(), cmap=cm.coolwarm, vmin=vmin, vmax=vmax)#, levels = levels, vmin = vmin, vmax = vmax)
    if LABEL_CONTOUR:
        axes[1].clabel(out_2way, inline=1, fontsize=5)
    # fig.colorbar(out_cont, ax=axes[1], label="$\gamma(x_A, x_B, e_A)$")
    axes[1].set_aspect((X_2way.max()-X_2way.min())/(Y_2way.max()-Y_2way.min()))
    axes[1].yaxis.tick_right()   
    axes[1].tick_params(axis='both', which='major', labelsize=axis_font_size)
    axes[1].plot([-extent[0]/2, -extent[0]/2], [Y_2way.min(), Y_2way.max()],  color='k', linestyle="--", linewidth=0.25)
    axes[1].plot([extent[0]/2, extent[0]/2],   [Y_2way.min(), Y_2way.max()],  color='k', linestyle="--", linewidth=0.25)
    axes[1].plot([X_2way.min(), X_2way.max()], [ego_vel, ego_vel], color='k', linestyle="--", linewidth=1)
    axes[1].get_yaxis().set_visible(False)  
    axes[1].set_xlabel("$(x_j - x_i)$ [m]", fontsize=6)



    fig.set_figheight = 5
    fig.set_figwidth = 1
    axes[2].imshow(np.flip(visualizer_image, axis=0))
    axes[2].get_xaxis().set_ticks([])
    axes[2].get_yaxis().set_ticks([])
    axes[2].set_xlabel("Scene 2")


    out_line = axes[3].contourf(X_line, Y_line, gammas_A_line.cpu().numpy(), cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
    if LABEL_CONTOUR:
        axes[3].clabel(out_line, inline=1, fontsize=5)
    axes[3].set_aspect((X_line.max()-X_line.min())/(Y_line.max()-Y_line.min()))
    axes[3].plot([X_line.min(), X_line.max()], [0,0], color='k', linestyle="--", linewidth=1)
    # fig.colorbar(out_cont, ax=axes[3], label="$\gamma(x_A, x_B, e_A)$")
    axes[3].tick_params(axis='both', which='major', labelsize=axis_font_size)
    axes[3].get_yaxis().set_visible(False)  
    axes[3].plot([-extent[0]/2, -extent[0]/2], [Y_line.min(), Y_line.max()],  color='k', linestyle="--", linewidth=0.25)
    axes[3].plot([extent[0]/2, extent[0]/2],   [Y_line.min(), Y_line.max()],  color='k', linestyle="--", linewidth=0.25)
    axes[3].set_xlabel("$(x_j - x_i)$ [m]", fontsize=6)

    out_cont = axes[4].contourf(X, Y, gammas_A.cpu().numpy(), cmap=cm.coolwarm, vmin=vmin, vmax=vmax)#, levels = levels, vmin = vmin, vmax = vmax)
    if LABEL_CONTOUR:
        axes[4].clabel(out_cont, inline=1, fontsize=5)
    axes[4].set_aspect((X.max()-X.min())/(Y.max()-Y.min()))
    axes[4].tick_params(axis='both', which='major', labelsize=axis_font_size)
    axes[4].yaxis.tick_right() 
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    axes[4].xaxis.set_major_formatter(xticks)
    axes[4].plot([X.min(), X.max()], [0,0], color='k', linestyle="--", linewidth=1)
    axes[4].plot([-extent[1]/2, -extent[1]/2], [Y_line.min(), Y_line.max()],  color='k', linestyle="--", linewidth=0.25)
    axes[4].plot([extent[1]/2, extent[1]/2],   [Y_line.min(), Y_line.max()],  color='k', linestyle="--", linewidth=0.25)
    axes[4].set_ylabel("$v_j$ [m/sec]", fontsize=6)
    axes[4].yaxis.set_label_position("right")
    axes[4].set_xlabel("$(y_j - y_i)$ [m]", fontsize=6)
    axes[4].yaxis.set_major_formatter(FormatStrFormatter('%d'))
    
    plt.savefig("./paper_figures/visualize_gammas.png", dpi=300)

    # Create Colorbar
    scalar_map = cm.ScalarMappable()
    scalar_map.set_clim(vmin, vmax)
    scalar_map.set_cmap(cm.coolwarm)
    cbar = fig.colorbar(scalar_map, cax=fig.add_axes([0.90, 0, 0.01, 0.21]), label="$\gamma(\mathbf{x})$")
    cbar.ax.tick_params(axis='both', which='major', labelsize=axis_font_size)

    plt.savefig("./paper_figures/visualize_gammas_with_colorbar.svg")

    # breakpoint()
if SAME_LANE_CL: 

    torch.cuda.empty_cache()

    # Imports
    from tbsim.utils.batch_utils import set_global_batch_type
    from tbsim.safety_funcs.cbfs import BackupBarrierCBF
    from tbsim.safety_funcs.utils import scene_centric_batch_to_raw
    from tbsim.configs.eval_config import EvaluationConfig
    from tbsim.evaluation.env_builders import EnvNuscBuilder
    from tbsim.configs.algo_config import ResponsibilityConfig
    from tbsim.utils.trajdata_utils import parse_trajdata_batch
    from tbsim.algos.factory import algo_factory
    from tbsim.configs.base import AlgoConfig
    import importlib
    import h5py
    import json
    import matplotlib.patches as patches
    from matplotlib.transforms import Affine2D

    device = "cuda"
    substeps = 10

    # Load Evaluation Scene
    set_global_batch_type("trajdata")
    file = open("./results/ClosedLoopWorking/HierAgentAwareCBFQPgammas/config.json")
    eval_cfg = EvaluationConfig()
    external_cfg = json.load(file)
    eval_cfg.update(**external_cfg)
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    composer_class = getattr(policy_composers, eval_cfg.eval_class)
    composer = composer_class(eval_cfg, device, ckpt_root_dir=eval_cfg.ckpt_root_dir)


    policy, exp_config = composer.get_policy()
    # Set to scene_centric
    exp_config.algo.scene_centric = True
    env_builder = EnvNuscBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device)
    env = env_builder.get_env(split_ego=False,parse_obs=True)



    # Load Gamma Model
    file = open("/home/rkcosner/Documents/tbsim/checkpoints/idling_checkpoint/run10/run0/config.json")
    algo_cfg = AlgoConfig()
    algo_cfg.algo = ResponsibilityConfig()
    external_algo_cfg = json.load(file)
    algo_cfg.update(**external_algo_cfg)
    algo_cfg.algo.update(**external_algo_cfg["algo"])
    device = "device" 
    modality_shapes = dict()
    gamma_algo = algo_factory(algo_cfg, modality_shapes)
    checkpoint_path = "/home/rkcosner/Documents/tbsim/checkpoints/idling_checkpoint/run10/run0/checkpoints/iter9000_ep1_valLoss0.00.ckpt"
    checkpoint = torch.load(checkpoint_path)
    gamma_algo.load_state_dict(checkpoint["state_dict"])
    net = gamma_algo.nets["policy"].cuda()

    # Load CBF
    cbf = BackupBarrierCBF(T_horizon = algo_cfg.algo.cbf.T_horizon, 
                        alpha=algo_cfg.algo.cbf.alpha, 
                        veh_veh=algo_cfg.algo.cbf.veh_veh, 
                        saturate_cbf=algo_cfg.algo.cbf.saturate_cbf, 
                        backup_controller_type=algo_cfg.algo.cbf.backup_controller_type
                        )



    names = [ "gammas", "even_split","worst_case"]

    data = dict()
    h_vals = dict()
    gammas = dict()
    for name in names:
        data_path = "./results/ClosedLoopWorking/HierAgentAwareCBFQP" + name + "/run_data.pkl"
        file = open(data_path, "rb")    
        data[name] = pickle.load(file)["scene-0305"]
        h_vals[name] = np.array(data[name]["h_vals"])
        gammas[name] = np.array(data[name]["gammas"])

        h5_path = "./results/ClosedLoopWorking/HierAgentAwareCBFQP" + name + "/data.hdf5"

        with h5py.File(h5_path, "r") as file: 
            scene_name = 'scene-0305_0'
            extent             = torch.tensor(file[scene_name]["extent"])                  # [A, T, 3]
            action_pos         = torch.tensor(file[scene_name]["action_positions"])        # [A, T, 1, 2]
            action_yaws        = torch.tensor(file[scene_name]["action_yaws"])             # [A, T]
            centroid           = torch.tensor(file[scene_name]["centroid"])                # [A, T, 2]
            yaw                = torch.tensor(file[scene_name]["yaw"])                     # [A, T]
            scene_index        = torch.tensor(file[scene_name]["scene_index"])             # [A, T]
            track_id           = torch.tensor(file[scene_name]["track_id"])                # [A, T]
            world_from_agent   = torch.tensor(file[scene_name]["world_from_agent"])        # [A, T, 3, 3]
    
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

        if name == "even_split": 
            batch4forensic = scene_centric_batch_to_raw(batch, substeps=substeps)
            forensic_states = batch4forensic["states"].clone()    
            forensic_inputs = batch4forensic["inputs"].clone()    

        print("Processing Scene" + name)
        data[name]["batch"] = scene_centric_batch_to_raw(batch, BEZ_TEST=False)
        data[name]["scene_index"] = scene_index

    # Set Up Environment
    scene_indices = [data[name]["scene_index"][0,0].item()]
    env.scene_indices = scene_indices
    env.reset(scene_indices=scene_indices, start_frame_index=None)
    env.update_random_seed(1)
    simscene = env._current_scenes[0]

    obs = parse_trajdata_batch(simscene.get_obs())
    batch = {
        "image": obs["image"][0,-8:,...]
    }   
    visualizer_image = generate_3channel_image(batch)

    fig, axes = plt.subplots(1,4)
    fig.set_size_inches(10,2)

    # Plot Other Agents
    init_states = data["gammas"]["batch"]["states"][:,0,:] - data["gammas"]["batch"]["states"][0,0,:]
    ego_yaw = data["gammas"]["batch"]["states"][0,0,3].item() 
    cars = []
    for j, x0 in enumerate(init_states):
        x_agent = x0[0].item()
        y_agent = x0[1].item()
        x_pxl = int((np.cos(ego_yaw) * x_agent + np.sin(ego_yaw) * y_agent)*pxls_per_meter + pxl_center[0])
        y_pxl = int((-np.sin(ego_yaw) * x_agent + np.cos(ego_yaw) * y_agent)*pxls_per_meter + pxl_center[1])

        if y_pxl < len(visualizer_image[:,0,0]) and x_pxl < len(visualizer_image[:,0,0]) and y_pxl>=0 and x_pxl >=0: 
            visualizer_image[y_pxl, x_pxl] = black

            middle_offset = [
                extent[j,0,0].item()*pxls_per_meter/2, 
                extent[j,0,1].item()*pxls_per_meter/2
            ]
            rect = patches.Rectangle(
                (x_pxl-middle_offset[0], y_pxl-middle_offset[1]), 
                extent[j,0,0].item()*pxls_per_meter, 
                extent[j,0,1].item()*pxls_per_meter, 
                alpha = 0.7, 
                facecolor="b", 
                edgecolor="k", 
                linewidth=0.1
                )
            t = Affine2D().rotate_around(x_pxl, y_pxl, x0[3]) + axes[0].transData
            rect.set_transform(t)
            cars.append(rect)
    
    for name in names: 
        states = data[name]["batch"]["states"] - data[name]["batch"]["states"][0,0,:]
        rotated_states_x = (np.cos(ego_yaw) * states[...,0] + np.sin(ego_yaw) * states[...,1])*pxls_per_meter + pxl_center[0]
        rotated_states_y = (-np.sin(ego_yaw) * states[...,0] + np.cos(ego_yaw) * states[...,1])*pxls_per_meter + pxl_center[1]
        states[...,0] = rotated_states_x
        states[...,1] = rotated_states_y
        for j, traj in enumerate(states):
            if j==0 and name == "even_split":   
                axes[0].plot(traj[:5,0], traj[:5,1], linewidth=2, color="y")
                axes[0].plot(traj[4,0], traj[4,1], linewidth=2, color="y", marker="x")
            if j==0 and name == "gammas":   
                axes[0].plot(traj[:20,0], traj[:20,1], linewidth=2, color="k")
            if j==15 and name=="gammas": 
                axes[0].plot(traj[:20,0], traj[:20,1], linewidth=2, color="r")

    axes[0].imshow(visualizer_image, aspect="auto")
    for car in cars: axes[0].add_patch(car)
    axes[0].set_xlim(pxl_center[0]-20, pxl_center[0]+80)
    axes[0].set_ylim(pxl_center[1]-50, pxl_center[1]+50)
    axes[0].grid(False)
    axes[0].get_yaxis().set_visible(False)  
    axes[0].get_xaxis().set_visible(False)  

    # Plot h's
    also = [0,1,2]
    for i, name in enumerate(names): 
        hs = h_vals[name]
        if i == 1: 
            time_idx,_ = np.where(hs<0)
            time = np.arange(17, time_idx.item(), 1)/10
            axes[1].plot(time, hs[17:time_idx.item(),-1], linewidth=1, color="y")
            axes[1].plot(time[-1], hs[time_idx.item(),-1],"x", color="y")
        elif i==0: 
            time = np.arange(17, 35, 1)/10
            axes[1].plot(time, hs[17:35,-1], linewidth=1, color="b")
            store_val = hs[18:20,-1]
        else: 
            axes[1].plot([time[0],time[1]], store_val, marker="x", color="r", linewidth=1) 

    axes[1].grid(False)
    time = np.arange(17, 35, 1)/10
    t_min = time.min().item()
    t_max = time.max().item()
    axes[1].hlines(0,t_min, t_max, color="k", linewidth=1, linestyle="--")
    axes[1].set_ylim(-0.2, 1.2)
    axes[1].set_xlim(t_min,t_max)
    axes[1].set_title("$h(\mathbf{x})$")
    axes[1].set_xlabel("time [sec]")
    # axes[1].get_yaxis().set_visible(False)  
    axes[1].yaxis.tick_right()   


    # Plot Gammas
    gammas = gammas["gammas"]
    time = np.arange(0, len(gammas[:50]), 1)/10
    axes[2].plot(time, gammas[:50,-1], color="c")
    axes[2].yaxis.tick_right() 
    axes[2].grid(False)
    t_max = time.max().item()
    axes[2].hlines(0,-0.1, t_max, color="k", linewidth=0.5, linestyle="--")
    axes[2].set_title("$\gamma(\mathbf{x})$")
    time = np.arange(17, 35, 1)/10
    axes[2].vlines(time.min(), gammas[:50,-1].min(), gammas[:50,-1].max(),  color="k", linewidth=0.5, linestyle="--")
    axes[2].vlines(time.max(), gammas[:50,-1].min(), gammas[:50,-1].max(),  color="k", linewidth=0.5, linestyle="--")

    # plot Constraint
    batch4forensic["states"] = forensic_states
    batch4forensic["inputs"] = forensic_inputs
    constraintA, constraintB = replay_sim(batch4forensic, simscene, net, device)
    time = np.arange(0, len(constraintA[1:]), 1)/2/substeps
    indices = np.where(np.logical_and(time< 3, time>=1.8))
    axes[3].plot(time[indices], constraintA[indices].cpu().detach().numpy(), color="y")
    axes[3].plot(time[indices], constraintB[indices].cpu().detach().numpy(), color="r")
    axes[3].hlines(0,time[indices][0], time[indices][-1], color="k", linewidth=0.5, linestyle="--")
    axes[3].set_title("Constraint($\mathbf{x}$)")
    axes[3].yaxis.tick_right() 
    axes[3].grid(False)

    
    plt.savefig("test.png", dpi =300)
    

    breakpoint()

    # # Get Picture with Initial Agent States
    # states = batch["states"]
    # inputs = batch["inputs"]
    # A = states.shape[0]
    # T = states.shape[1] 
    # ego_states = []
    # ego_imgs = []
    # col_states = []
    # col_imgs = []
    # print("Replaying Experiment")
    # for time in tqdm(range(T)): 

    #     # Set up scene
    #     scene_action = dict()
    #     for a, agent in enumerate(simscene.agents):
    #         # print("agent name: ", agent, " \t agent id: ", a)
    #         scene_action[agent.name] =  np.array(states[a, time, [0,1,3]])
    #     simscene.step(scene_action)
    #     obs = parse_trajdata_batch(simscene.get_obs())


    plt.savefig("test.png", dpi =300)


    plt.savefig("./paper_figures/CL_inline.png")

if INTERSECTION_CL: 
    pass 


if CL_STATS: 
    
    path_sets = [
        ["/home/rkcosner/Documents/tbsim/paper_results/cl1/even_split/HierAgentAwareCBFQP_even_split_test_0_20_d05/run_data.pkl",
        "/home/rkcosner/Documents/tbsim/paper_results/cl1/even_split/HierAgentAwareCBFQP_even_split_test_20_40_d05/run_data.pkl",
        "/home/rkcosner/Documents/tbsim/paper_results/cl1/even_split/HierAgentAwareCBFQP_even_split_test_40_60_d05/run_data.pkl",
        "/home/rkcosner/Documents/tbsim/paper_results/cl1/even_split/HierAgentAwareCBFQP_even_split_test_60_80_d05/run_data.pkl",
        # "/home/rkcosner/Documents/tbsim/paper_results/cl1/even_split/HierAgentAwareCBFQP_even_split_test_80_100_d05/run_data.pkl",
        # "/home/rkcosner/Documents/tbsim/paper_results/cl1/even_split/HierAgentAwareCBFQP_even_split_test_100_120_d05/run_data.pkl"
        ], 
        ["/home/rkcosner/Documents/tbsim/paper_results/cl1/gammas/HierAgentAwareCBFQP_gammas_test_0_20_d05/run_data.pkl", 
        "/home/rkcosner/Documents/tbsim/paper_results/cl1/gammas/HierAgentAwareCBFQP_gammas_test_20_40_d05/run_data.pkl", 
        "/home/rkcosner/Documents/tbsim/paper_results/cl1/gammas/HierAgentAwareCBFQP_gammas_test_40_60_d05/run_data.pkl", 
        "/home/rkcosner/Documents/tbsim/paper_results/cl1/gammas/HierAgentAwareCBFQP_gammas_test_60_70_d05/run_data.pkl",
        "/home/rkcosner/Documents/tbsim/paper_results/cl1/gammas/HierAgentAwareCBFQP_gammas_test_70_80_d05/run_data.pkl",
        "/home/rkcosner/Documents/tbsim/paper_results/cl1/gammas/HierAgentAwareCBFQP_gammas_test_80_100_d05/run_data.pkl", # there's only one sample in here!!!
        # "/home/rkcosner/Documents/tbsim/paper_results/cl1/gammas/HierAgentAwareCBFQP_gammas_test_100_110_d05/run_data.pkl",
        # "/home/rkcosner/Documents/tbsim/paper_results/cl1/gammas/HierAgentAwareCBFQP_gammas_test_110_120_d05/run_data.pkl",
        ]
    ]

    if True:
        total_violations = []
        for paths in path_sets: 
            num_scenes = 0 
            safety_violations = 0 
            for path in paths: 
                file = open(path, "rb")
                data = pickle.load(file)
                file.close()
                for scene_name in data.keys(): 
                    num_scenes+=1 
                    if sum(data[scene_name]["safety_violation"])> 0: 
                        safety_violations += 1  
                print(num_scenes)
            print("\n")
            total_violations.append(safety_violations)

    total_violations = np.array(total_violations)
    breakpoint()
    # path = "/home/rkcosner/Documents/tbsim/results/HierAgentAwareCBFQPeven_split_20_thru_39/run_data.pkl"
    # file = open(path, "rb")
    # data = pickle.load(file)
    # breakpoint()