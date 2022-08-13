from platform import release
from turtle import pos
import torch 
from tbsim import dynamics as dynamics

import numpy as np 

import matplotlib.pyplot as plt
from matplotlib import cm 

from math import comb

import wandb

import pickle

from tqdm import tqdm 

def get_bezier(points):
    
    def bez_curve(t):
        n = points.shape[-2]-1
        out = 0 
        for i in range(n+1): 
            out += comb(n,i) * ((1-t)**(n-i)) * (t**i) * points[...,i,:]
        return out

    def bez_curve_deriv(t): 
        n = points.shape[-2]-1
        out = 0 
        for i in range(n+1): 
            if i == 0: 
                out += comb(n,i) * (-(1-t)**(n-i-1)*(n-i) * (t**i)) * points[...,i,:]
            elif i==n: 
                out += comb(n,i) * ((1-t)**(n-i)*i*t**(i-1)) * points[...,i,:]
            else: 
                out += comb(n,i) * (-(n-i)*(1-t)**(n-i-1) * (t**i) + (1-t)**(n-i)*i*t**(i-1)) * points[...,i,:]
        return out 

    def bez_curve_2nd_deriv(t): 
        n = points.shape[-2]-1
        out = 0 
        for i in range(n+1): 
            tmp = 0
            if i==0: 
                tmp += (n-i) * (n-i-1) * (1-t)**(n-i-2)*t**i
            elif i==1:
                tmp += (n-i) * (n-i-1) * (1-t)**(n-i-2)*t**i 
                tmp += - 2 * (n-i) * (1-t)**(n-i-1) * i * t**(i-1)
            elif i==n-1:
                tmp += - 2 * (n-i) * (1-t)**(n-i-1) * i * t**(i-1)
                tmp += (1 - t)**(n-i) * i * (i-1) * t**(i-2)
            elif i==n: 
                tmp += (1 - t)**(n-i) * i * (i-1) * t**(i-2)
            else: 
                tmp += (n-i) * (n-i-1) * (1-t)**(n-i-2)*t**i
                tmp += - 2 * (n-i) * (1-t)**(n-i-1) * i * t**(i-1)
                tmp += (1 - t)**(n-i) * i * (i-1) * t**(i-2)
            out += comb(n,i) * tmp * points[...,i,:]
        return out 

    return bez_curve, bez_curve_deriv, bez_curve_2nd_deriv

def test_bezier(): 
    y = np.array([0,1,2,3,3,5])
    x = np.linspace(0,1,len(y))
    curve = get_bezier(y)
    plt.figure()
    t = np.linspace(0,1, 100)
    plt.plot(t, curve(t))
    plt.plot(x, curve(x), 'o')
    plt.plot(x,y, '*')
    plt.savefig("bez_test.png")

def get_fits(pos_curves, yaw_curves, T = 10, Last=False): 

    pos_curve = pos_curves[0]
    pos_dot_curve = pos_curves[1]
    pos_ddot_curve = pos_curves[2]
    yaw_curve = yaw_curves[0]
    yaw_rate_curve = yaw_curves[1] 

    Taus = torch.linspace(0,1, T) 
    dt = 0.1
    dTaudt = 1.0/((T-1)*dt) # remove one for the starting point
    fit_poses = []
    fit_vels = []
    fit_yaws = []
    fit_yaw_rates = []
    fit_accels = []
    if not Last: 
        Taus = Taus[:-1]

    for tau in Taus: # include the last point to connect, but then drop it for recording
        # Fit poses, yaw rates, yaws and velocities
        yaws =  yaw_curve(tau)
        yaw_rates = yaw_rate_curve(tau) * dTaudt

        fit_yaws.append(yaws[..., None, :])
        fit_yaw_rates.append(yaw_rates[...,None, :])
        yaws = yaws.squeeze()
        yaw_rates = yaw_rates.squeeze()

        pos_dot = pos_dot_curve(tau) * dTaudt 
        fit_vel = pos_dot[...,0] * torch.cos(yaws) + pos_dot[...,1] * torch.sin(yaws)
        pos_ddot = pos_ddot_curve(tau) * dTaudt * dTaudt
        fit_accel  = torch.cos(yaws) * (pos_ddot[...,0] + pos_dot[...,1]*yaw_rates) 
        fit_accel += torch.sin(yaws) * (pos_ddot[...,1] - pos_dot[...,0]*yaw_rates)
        fit_vel = fit_vel[...,None,None]
        fit_accel = fit_accel[...,None,None]
        
        # Fitted states
        fit_poses.append(pos_curve(tau)[...,None,:] )
        fit_vels.append(fit_vel)        
        fit_accels.append(fit_accel)

    fit_pos = torch.cat(fit_poses, axis=-2)
    fit_vel = torch.cat(fit_vels, axis=-2)
    fit_accel = torch.cat(fit_accels, axis=-2)
    fit_yaw = torch.cat(fit_yaws, axis=-2)
    fit_yaw_rate = torch.cat(fit_yaw_rates, axis=-2)

    fit_states = torch.cat([fit_pos, fit_vel, fit_yaw], axis=-1)
    fit_inputs = torch.cat([fit_accel, fit_yaw_rate], axis=-1)

    return fit_states, fit_inputs
    

T_sec = 11 # Create a bezier curve for each 1 second of the trajectories
def get_bez_for_long_trajectories(batch, DeltaT=1):
    A = batch["history_positions"].shape[0]
    T_total = batch["history_positions"].shape[1]
    states = torch.empty(A, T_total, 4)
    inputs = torch.empty(A, T_total, 2)

    dt = 0.1
    for i in range(int(T_total*dt)):
        # Get Bezier Curves
        src_pos = batch["history_positions"][:,i*10:(i+1)*10+1,:]
        src_yaw = batch["history_yaws"][:,i*10:(i+1)*10+1,:]

        pos_curves = get_bezier(src_pos)
        yaw_curves = get_bezier(src_yaw) 

        fit_states, fit_inputs = get_fits(pos_curves, yaw_curves, T=T_sec)

        states[:,i*10:(i+1)*10,:] = fit_states
        inputs[:,i*10:(i+1)*10,:] = fit_inputs

    # Add the final points
    # T = batch["history_positions"].shape[1]-i*10
    # i +=1 
    # src_pos = batch["history_positions"][:,i*10:,:]
    # src_yaw = batch["history_yaws"][:,i*10:,:]
    # pos_curves = get_bezier(src_pos)
    # yaw_curves = get_bezier(src_yaw)

    # breakpoint()
    # fit_states, fit_inputs = get_fits(pos_curves, yaw_curves, T, Last=True)

    # states[:, i*10:, :] = fit_states
    # inputs[:, i*10:, :] = fit_inputs
    
    # batch["states"] = states
    # batch["inputs"] = inputs



def scene_centric_batch_to_raw(data_batch, BEZ_TEST=False):
    """
        RYAN: 
            stacks ego with other agents history
        arg: 
            - data_batch : all learning data
            - step_time 
        return: 
            Everything is ordered so that the timing index is [0 : newest], [-1 : oldest] 
            - "history_positions"
            - "history_vel"
            - "history_yaws"
            - "raw_types:
            - "history_availabilities"
            - "extents"
    """

    # Add target states (the next state in the trajectory), so that the input can be computed now 
    if ("target_positions" in data_batch) and ("history_positions" in data_batch) and ("history_availabilities" in data_batch): 
        N_future = data_batch["target_positions"].shape[-2]
        src_pos = torch.cat([data_batch["history_positions"], data_batch["target_positions"]], axis = -2) # [B, A, T, D]
        src_yaw = torch.cat([data_batch["history_yaws"], data_batch["target_yaws"]], axis=-2) # [B, A, T, D]
        src_mask = torch.cat([data_batch["history_availabilities"], data_batch["target_availabilities"]], axis=-1)# [B, A, T]
    else: 
        src_pos = data_batch["history_positions"]
        src_yaw = data_batch["history_yaws"]
        src_mask = data_batch["history_availabilities"]

    # Estimate Velocity: 3 point method when possible and 2 point method when not
    if BEZ_TEST: # 1st order approximation for debugging
        src_vel = dynamics.Unicycle.calculate_vel(src_pos, src_yaw, data_batch["dt"][0] , src_mask)
        src_vel[:, -1:,:] = data_batch["curr_speed"].unsqueeze(-1) # replace calculated velocity with true velocity
    

    """
    Get Bezier Approximations
    """
    pos_curve, pos_dot_curve, pos_ddot_curve = get_bezier(src_pos)
    yaw_curve, yaw_rate_curve, _ = get_bezier(src_yaw)    
    T = src_pos.shape[-2] 
    Taus = torch.linspace(0,1, T) 
    dTaudt = 1/((T-1)*data_batch["dt"][0]) # remove one for the starting point
    fit_poses = []
    fit_vels = []
    fit_yaws = []
    fit_yaw_rates = []
    fit_accels = []
    print("Fitting Beziers")
    for tau in Taus: 
        # Fit poses, yaw rates, yaws and velocities
        yaws =  yaw_curve(tau)
        yaw_rates = yaw_rate_curve(tau) * dTaudt

        fit_yaws.append(yaws[...,None, :])
        fit_yaw_rates.append(yaw_rates[...,None, :])
        yaws = yaws.squeeze()
        yaw_rates = yaw_rates.squeeze()

        pos_dot = pos_dot_curve(tau) * dTaudt 
        fit_vel = pos_dot[...,0] * torch.cos(yaws) + pos_dot[...,1] * torch.sin(yaws)
        pos_ddot = pos_ddot_curve(tau) * dTaudt * dTaudt
        fit_accel  = torch.cos(yaws) * (pos_ddot[...,0] + pos_dot[...,1]*yaw_rates) 
        fit_accel += torch.sin(yaws) * (pos_ddot[...,1] - pos_dot[...,0]*yaw_rates)
        fit_vel = fit_vel[...,None,None]
        fit_accel = fit_accel[...,None,None]
        
        # Fitted states
        fit_poses.append(pos_curve(tau)[...,None,:] )
        fit_vels.append(fit_vel)        
        fit_accels.append(fit_accel)

    fit_pos = torch.cat(fit_poses, axis=-2)
    fit_vel = torch.cat(fit_vels, axis=-2)
    fit_accel = torch.cat(fit_accels, axis=-2)
    fit_yaw = torch.cat(fit_yaws, axis=-2)
    fit_yaw_rate = torch.cat(fit_yaw_rates, axis=-2)

    try: 
        states = torch.cat([fit_pos, fit_vel, fit_yaw], axis=-1)
        inputs = torch.cat([fit_accel, fit_yaw_rate], axis=-1)
    except: 
        import pdb; pdb.set_trace()

    if BEZ_TEST: # 1st order approximation for debugging
        from tbsim.safety_funcs.debug_utils import view_states_and_inputs
        states_1stord = torch.cat((src_pos, src_vel, src_yaw), axis =-1) 
        inputs_1stord = (states_1stord[...,1:,2:] - states_1stord[...,:-1,2:])/data_batch["dt"][0]
        view_states_and_inputs(states, inputs, states_1stord, inputs_1stord)

    if "target_positions" in data_batch: 
        T_history = data_batch["history_positions"].shape[-2]
        data_batch["states"] = states[:,:,:-N_future,:] # Remove the future state
        data_batch["inputs"] = inputs[:,:,:-N_future,:] # Remove the future input
        data_batch["image"]  = data_batch["image"][...,T_history-1:,:,:] # Remove Image Trajectory
    else: 
        data_batch["states"] = states
        data_batch["inputs"] = inputs
    return data_batch 


def batch_to_raw_all_agents(data_batch, step_time):
    """
        RYAN: 
            stacks ego with other agents history
        arg: 
            - data_batch : all learning data
            - step_time 
        return: 
            Everything is ordered so that the timing index is [0 : newest], [-1 : oldest] 
            - "history_positions"
            - "history_vel"
            - "history_yaws"
            - "raw_types:
            - "history_availabilities"
            - "extents"
    """

    raw_type = torch.cat(
        (data_batch["type"].unsqueeze(1), data_batch["all_other_agents_types"]),
        dim=1,
    ).type(torch.int64)

    src_pos = torch.cat(
        (
            data_batch["history_positions"].unsqueeze(1),
            data_batch["all_other_agents_history_positions"],
        ),
        dim=1,
    )

    src_yaw = torch.cat(
        (
            data_batch["history_yaws"].unsqueeze(1),
            data_batch["all_other_agents_history_yaws"],
        ),
        dim=1,
    )
    
    src_mask = torch.cat(
        (
            data_batch["history_availabilities"].unsqueeze(1),
            data_batch["all_other_agents_history_availability"],
        ),
        dim=1,
    ).bool()


    """
        RYAN: extent seems to be dim=3 in nuscenes and dim=2 in L5, what does that data mean? 
    """
    extents = torch.cat(
        (
            data_batch["extent"][..., :3].unsqueeze(1),
            torch.max(data_batch["all_other_agents_history_extents"], dim=-2)[0],
        ),
        dim=1,
    )

    # estimate velocity
    src_vel = dynamics.Unicycle.calculate_vel(src_pos, src_yaw, step_time, src_mask)
    src_vel[:, 0, -1] = data_batch["curr_speed"].unsqueeze(-1)

    # Flip everything so that 0 is the current velocity and -1 is the oldest
    src_pos = torch.flip(src_pos, dims=[-2])
    src_yaw = torch.flip(src_yaw, dims=[-2])
    src_mask = torch.flip(src_mask, dims=[-1])
    src_vel = torch.flip(src_vel, dims=[-1])

    return {
        "history_positions": src_pos,
        "history_vel" : src_vel, 
        "history_yaws": src_yaw,
        "raw_types": raw_type,
        "history_availabilities": src_mask,
        "extents": extents,
    }



def plot_gammas(batch, net, relspeed=0.0, B=0, A=0):
    net.eval() 
    i = A
    with torch.no_grad():
        T = batch["states"].shape[-2]
        """
            Get Tensor Image as Numpy
                The last 7 channels contain semantic information
                - T : length of time horizon
                - 
        """

        """
            Generate Test Batch
            The test batch is going to need the following items: 
                state : [B', A', T', 4] states of the vehicles. Keep agent0 fixed at center and place agent1 everywhere else on the map
                history_availabilities : [B', A', T'] availabilities of the agents, only agent0 and agent1 are true and they're always true
                agents_from_center : [B', A', 3, 3] transformation matrix. (This doesn't actually matter since we're only going to consider the gammas from agent0's perspective) 
                image : [B', A', T'+7 ]
        """
        N_pxls = batch["image"].shape[-2] 
        pxls_per_meter = batch["raster_from_agent"][0,0].item()
        data_points_per_meter = 1.0 / 5.0
        agentFromRaster = batch["agent_from_raster"]
        pxl_center = batch["raster_from_agent"][0:2, 2]
        N_datapoints = int(N_pxls/pxls_per_meter * data_points_per_meter )
        Bprime = N_datapoints**2 # one gamma datapoint for every meter
        Aprime = 2
        Tprime = 1
        gen_states = torch.zeros((Bprime, Aprime, Tprime, 1))

        pxls_x = torch.linspace(0,N_pxls-1,N_datapoints, dtype=torch.int)
        pxls_y = pxls_x

        state_imgs = torch.zeros((Bprime,Aprime,Tprime,N_pxls,N_pxls), device=batch["image"].device)
        pxl_positions = []
        for ix, pxl_x in enumerate(pxls_x): 
            for jy, pxl_y in enumerate(pxls_y): 
                pxl_position = [[pxl_x], [pxl_y], [1.0]]
                pxl_positions.append(pxl_position)
                state_imgs[ix+jy*N_datapoints,0,0,pxl_y,pxl_x] = -1.0
                state_imgs[ix+jy*N_datapoints,0,0,int(pxl_center[1]),int(pxl_center[0])] = 1.0
        pxl_positions = torch.tensor(pxl_positions, device=agentFromRaster.device)


        state_agentA = batch["states"][B:B+1,0:1,-1:,:]
        state_agentA = torch.repeat_interleave(state_agentA, Bprime, axis=0)
        agentFromRaster = torch.repeat_interleave(agentFromRaster[None,...], Bprime, axis=0)
        state_positions = torch.bmm(agentFromRaster, pxl_positions)[:,None, :2,:] # Remove extension required for transformation 
        state_positions = torch.permute(state_positions, (0,1,3,2))
        state_vel = state_agentA[...,2:3] + relspeed # FORCE AGENTB to have the same velocity as agentA) torch.repeat_interleave(batch["states"][0:1,1,0:1,2:3], Bprime, axis=0)
        state_yaw = torch.zeros_like(state_vel)
        state_agentB = torch.cat([state_positions, state_vel, state_yaw], axis=-1)

        gen_states = torch.cat([state_agentA, state_agentB], axis=1)
        gen_availabilities = torch.ones((Bprime,Aprime, Tprime), dtype=torch.bool, device=gen_states.device)
        gen_agents_from_center = torch.eye(3)[None,None,...].repeat(Bprime, Aprime,1,1).to(gen_states.device)
        semantic_imgs = batch["image"][B:B+1,0:2,Tprime:, ...].repeat(Bprime, 1, 1, 1, 1)
        gen_image = torch.cat([state_imgs, semantic_imgs], axis=-3)

        """
            Generated Batch
        """
        test_batch ={
            "states" : gen_states, 
            "history_availabilities" : gen_availabilities, 
            "agents_from_center" : gen_agents_from_center, 
            "image" : gen_image
        }

        gammas = net(test_batch)

    net.train()

    """
    Plot Gammas Over Semantic Map
    """
    gammasA = gammas["gammas_A"].cpu().numpy()
    gammasA = gammasA.squeeze().reshape(N_datapoints, N_datapoints)
    gridX, gridY = torch.meshgrid(pxls_x, pxls_y, indexing='ij')

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1,2,1, projection="3d") #plt.axes(projection='3d')
    ax.plot_surface(gridX.numpy(), gridY.numpy(), gammasA, cmap=cm.coolwarm)

    # Set Colors 
    colors = [[0.5, 0.0, 0.0],[0.0, 0.5, 0.0],[0.0, 0.0, 0.5],[0.0, 0.4, 0.4],[0.5, 0.3, 0.0],[0.5, 0.5, 0.0],[0.5, 0.0, 0.5]]
    white = [1.0, 1.0, 1.0]
    black = [0.0, 0.0, 0.0]
    
    # Generate Semantic 3 Channel Image
    semantic_channels = test_batch['image'][0,0,Tprime:].cpu().detach().numpy()
    semantic_image = np.zeros((N_pxls, N_pxls, 3)) + 0.5
    for j, color in enumerate(colors): 
        color_np = np.array([color]) 
        s_channel = semantic_channels[j][...,None]
        semantic_image += (s_channel @ color_np).squeeze()

    # Draw Ego Vehicle
    semantic_image[int(pxl_center[1]),int(pxl_center[0])] = white
    xs = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
    ys = xs 
    for x in xs: 
        for y in ys: 
            if x != 0 or y !=0:
                semantic_image[int(pxl_center[1])+x, int(pxl_center[0])+y] = black 
    

    N_pxls = semantic_image.shape[0]
    x = np.linspace(1,N_pxls, N_pxls)
    X, Y = np.meshgrid(x, x)
    Z = (gammasA.min()-1) * np.ones(X.shape) 
    semantic_image = np.sum(semantic_image, axis=-1)
    semantic_image /= semantic_image.max()
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=plt.cm.coolwarm(semantic_image), shade=False)
    ax.set_title("gamma for agent A: [0,0,%.1f,0], agent B: [x,y,%.1f,0]" % (state_agentA[0,0,0,2].item(), relspeed + state_agentA[0,0,0,2].item()) )
    # ax.set_ylabel("y position")
    # ax.set_xlabel("x position")
    ax.set_zlabel("$\gamma(x_A, x_B, e)$")
    ax.get_xaxis().set_ticks([])#set_visible(False)
    ax.get_yaxis().set_ticks([])#set_visible(False)
    # plt.savefig("test_gamma_plotter.png")

    # Create Contour Image
    ax_cont = fig.add_subplot(1,2,2) #plt.axes(projection='3d')
    ax_cont.imshow(np.flip(semantic_image, axis=0), cmap=plt.cm.coolwarm)
    cs = ax_cont.contour(gridX.numpy(), gridY.numpy(),np.flip(gammasA, axis=1), colors='k', linewidths=3, linestyles="solid")
    ax_cont.contour(gridX.numpy(), gridY.numpy(), np.flip(gammasA, axis=1), cmap=plt.cm.coolwarm, linewidths=2)
    ax_cont.clabel(cs, cs.levels[np.where(cs.levels!=0)])
    
    if gammasA.min()<=0: 
        # Plot 0 level 
        cs = ax_cont.contour(gridX.numpy(), gridY.numpy(), np.flip(gammasA, axis=1), levels=0, colors='k', linewidths=5)
        ax_cont.contour(gridX.numpy(), gridY.numpy(), np.flip(gammasA, axis=1), levels=0, colors='w', linewidths=3, linestyles='--')
        ax_cont.clabel(cs,cs.levels)
        
    ax_cont.get_xaxis().set_ticks([])#set_visible(False)
    ax_cont.get_yaxis().set_ticks([])#set_visible(False)
    img = wandb.Image(fig)
    plt.close()

    return img


"""
    PLOT STATIC GAMMAS
"""
ego_extent = [4.0840, 1.7300, 1.5620]
pxl_center = [56, 112]
pxls_per_meter = 2
pxl_center = [56, 112]
pxls_per_meter = 2
white = [1.0, 1.0, 1.0]
black = [0.0, 0.0, 0.0]

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

def generate_static_gamma_plots(fig, visualizer_image, X, Y, gammas_A):
    fig = plt.figure()
    ax_img = fig.add_subplot(1,2,1)
    ax_img.imshow(np.flip(visualizer_image, axis=0))
    ax_img.get_xaxis().set_ticks([])#set_visible(False)
    ax_img.get_yaxis().set_ticks([])#set_visible(False)

    ax_contour = fig.add_subplot(1,2,2)
    out_cont = ax_contour.contourf(X, Y,gammas_A.cpu().numpy(), cmap=cm.coolwarm)#, levels = levels, vmin = vmin, vmax = vmax)
    fig.colorbar(out_cont, ax=ax_contour, label="$\gamma(x_A, x_B, e_A)$")
    return fig, ax_contour


def plot_static_gammas_inline(net, type): 
    torch.cuda.empty_cache()
    net.eval()
    with torch.no_grad():
        datapoints_per_meter = 0.5 
        window = 20 
        rel_vel_max = 5
        if type == 2: 
            with open("/workspace/static_scenes/batch2wayDivider.pickle", 'rb') as file: 
                batch = pickle.load(file)
                stateA = batch["states"][-1,:]
        else: 
            with open("/workspace/static_scenes/batch4way.pickle", 'rb') as file: 
                batch = pickle.load(file)
                stateA = batch["states"][0,0,-1,:]

        N_pxls = batch['image'][1:].cpu().detach().numpy().shape[-1]


        x_window = torch.arange(-window, window + 1.0/datapoints_per_meter, step=1.0/datapoints_per_meter)
        window_pxls = (x_window * pxls_per_meter).int()
        
        v_window = torch.arange(0, stateA[2].item() + rel_vel_max, step = 1)

        # Fill Batch 
        Bprime = len(x_window) * len(v_window) 
        Aprime = 2
        Tprime = 1 
        gen_states = torch.zeros(Bprime, Aprime, Tprime, 4, device=batch["image"].device)
        gen_states[:,0,0,2] = stateA[2]
        for ix, x in enumerate(x_window):
            for jv, v in enumerate(v_window): 
                gen_states[ix + len(x_window)*jv, 1, 0, :] = torch.tensor([x, 0, v, 0]) 

        gen_availabilities = torch.ones(Bprime, Aprime, Tprime, device=batch["image"].device).bool()
        gen_agents_from_center = torch.eye(3)[None,None,...].repeat(Bprime, Aprime,1,1).to(gen_states.device)

        state_imgs = torch.zeros((Bprime,Aprime,Tprime,N_pxls,N_pxls), device=batch["image"].device)
        for ix, pxl_x in enumerate(window_pxls): 
            for jy, _ in enumerate(v_window): 
                state_imgs[ix+jy*len(window_pxls),0,0,pxl_center[1],pxl_x + pxl_center[0]] = -1.0
                state_imgs[ix+jy*len(window_pxls),0,0,pxl_center[1],pxl_center[0]] = 1.0

        semantic_imgs = batch["image"][Tprime:, ...].repeat(Bprime, Aprime, Tprime, 1, 1)
        gen_image = torch.cat([state_imgs, semantic_imgs], axis=-3)

        test_batch ={
            "states" : gen_states, 
            "history_availabilities" : gen_availabilities, 
            "agents_from_center" : gen_agents_from_center, 
            "image" : gen_image
        }

        gamma_preds = net(test_batch)
    net.train()


    # Generate Semantic 3 Channel Image
    visualizer_image = generate_3channel_image(batch)
    visualizer_image[pxl_center[1], pxl_center[0] + window_pxls] = black


    X, Y = np.meshgrid(x_window, v_window)
    gammas_A = gamma_preds["gammas_A"]
    gammas_A = gammas_A.reshape(len(v_window), len(x_window))

    fig = plt.figure()
    fig, ax_contour = generate_static_gamma_plots(fig, visualizer_image, X, Y, gammas_A)
    # customize labels
    ax_contour.plot([X.min(), X.max()], [stateA[-2].item(), stateA[-2].item()], color='k', linestyle='--', linewidth=0.5)
    ax_contour.plot([ego_extent[0]/2, ego_extent[0]/2], [v_window.min(), v_window.max()], color='k', linestyle='--', linewidth=0.5)
    ax_contour.plot([-ego_extent[0]/2, -ego_extent[0]/2], [v_window.min(), v_window.max()], color='k', linestyle='--', linewidth=0.5)
    ax_contour.set_ylabel("Agent Velocity")
    ax_contour.set_xlabel("Agent Relative $x$ Position")
    img = wandb.Image(fig)

    return img

    # plt.savefig("test.png")

    # with open("./tbsim/safety_funcs/static_scenes/batchRoundabout.pickle", 'rb') as file: 
    #     batchR = pickle.load(file)



def plot_static_gammas_traj(net, type=4): 
    
    # net.eval()
    with torch.no_grad():
        datapoints_per_meter = 0.5 
        window = 20 
        rel_vel_max = 10

        if type == 4:
            with open("/workspace/static_scenes/batch4way.pickle", 'rb') as file: 
                batch = pickle.load(file)
            N_pxls = batch['image'][1:].cpu().detach().numpy().shape[-1]
            stateA = batch["states"][0,0,-1,:]
            x_traj = np.concatenate([np.ones(13)*97, 97 - np.arange(1,5, 5.0/12)]).astype(int)
            y_traj = np.linspace(112-20, 112+42, len(x_traj)).astype(int)
            theta1 = -np.pi/2
            theta2 = np.arctan2(y_traj[13:].min()- y_traj[13:].max(), x_traj.max()-x_traj.min())
            theta_traj = np.concatenate([theta1*np.ones(13), theta2*np.ones(len(x_traj[13:]))])

            x_traj = np.flip(x_traj)
            y_traj = np.flip(y_traj)
            theta_traj = np.flip(theta_traj)
        else: 
            with open("/workspace/static_scenes/batchRoundabout.pickle", 'rb') as file: 
                batch = pickle.load(file)
            N_pxls = batch['image'][1:].cpu().detach().numpy().shape[-1]
            stateA = batch["states"][-1,:]
            
            circle_center = [47, 131]
            radius = np.sqrt((56-47)**2 + (112-131)**2)
            thetas_roundabout = np.linspace(-np.pi, 0, 20)
            x_traj = (circle_center[0] + radius*np.cos(thetas_roundabout)).astype(int)
            y_traj = (circle_center[1] + radius*np.sin(thetas_roundabout)).astype(int)
            theta_traj = thetas_roundabout - np.pi/2

            theta1 = -np.pi/2
            theta2 = np.arctan2(y_traj[13:].min()- y_traj[13:].max(), x_traj.max()-x_traj.min())
            theta_traj = np.concatenate([theta1*np.ones(13), theta2*np.ones(len(x_traj[13:]))])

        pos_idx_window = torch.arange(0, len(x_traj)-1).int()
        v_window = torch.arange(0, stateA[2].item() + rel_vel_max, step = 1)

        # Fill Batch 
        Bprime = len(pos_idx_window) * len(v_window) 
        Aprime = 2
        Tprime = 1 
        gen_states = torch.zeros(Bprime, Aprime, Tprime, 4, device=batch["image"].device)
        gen_states[:,0,0,2] = stateA[2]
        for ipos in pos_idx_window:
            for jv, v in enumerate(v_window): 
                x = (x_traj[ipos]-pxl_center[0]) / pxls_per_meter
                y = (y_traj[ipos]-pxl_center[1]) / pxls_per_meter
                theta = theta_traj[ipos]
                gen_states[ipos + len(pos_idx_window)*jv, 1, 0, :] = torch.tensor([x, y, v, theta]) 

        gen_availabilities = torch.ones(Bprime, Aprime, Tprime, device=batch["image"].device).bool()
        gen_agents_from_center = torch.eye(3)[None,None,...].repeat(Bprime, Aprime,1,1).to(gen_states.device)

        state_imgs = torch.zeros((Bprime,Aprime,Tprime,N_pxls,N_pxls), device=batch["image"].device)
        for ipos in pos_idx_window: 
            for jy, _ in enumerate(v_window): 
                state_imgs[ipos+jy*len(pos_idx_window),0,0, y_traj[ipos], x_traj[ipos]] = -1.0
                state_imgs[ipos+jy*len(pos_idx_window),0,0,pxl_center[1],pxl_center[0]] = 1.0

        semantic_imgs = batch["image"][Tprime:, ...].repeat(Bprime, Aprime, Tprime, 1, 1)
        gen_image = torch.cat([state_imgs, semantic_imgs], axis=-3)

        test_batch ={
            "states" : gen_states, 
            "history_availabilities" : gen_availabilities, 
            "agents_from_center" : gen_agents_from_center, 
            "image" : gen_image
        }

        gamma_preds = net(test_batch)
    net.train()

    # Generate Road Image
    visualizer_image = generate_3channel_image(batch)
    visualizer_image[y_traj, x_traj] = black
    

    # Generate Contour Plot  
    X, Y = torch.meshgrid(pos_idx_window.float()/ (len(pos_idx_window)-1), v_window, indexing='xy')
    gammas_A = gamma_preds["gammas_A"]
    gammas_A = gammas_A.reshape(len(v_window), len(pos_idx_window))


    fig = plt.figure()
    fig, ax_contour = generate_static_gamma_plots(fig, visualizer_image, X, Y, gammas_A)
    # customize labels
    if type == 0 : 
        ax_contour.plot([X.min(), X.max()], [stateA[-2].item(), stateA[-2].item()], color='k', linestyle='--', linewidth=0.5)
    # ax_contour.plot([ego_extent[0]/2, ego_extent[0]/2], [v_window.min(), v_window.max()], color='k', linestyle='--', linewidth=0.5)
    # ax_contour.plot([-ego_extent[0]/2, -ego_extent[0]/2], [v_window.min(), v_window.max()], color='k', linestyle='--', linewidth=0.5)
    ax_contour.set_ylabel("Agent Velocity")
    ax_contour.set_xlabel("Agent Trajectory Completion")

    img = wandb.Image(fig)

    return img

if __name__ == "__main__": 
    plot_static_gammas_4way(net = 1, type = 0 )
