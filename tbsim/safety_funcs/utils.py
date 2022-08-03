from turtle import pos
import torch 
from tbsim import dynamics as dynamics

import numpy as np 

import matplotlib.pyplot as plt
from matplotlib import cm 

from math import comb

import wandb

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


def scene_centric_batch_to_raw(data_batch):
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
    src_pos = torch.cat([data_batch["history_positions"], data_batch["target_positions"]], axis = -2) # [B, A, T, D]
    src_yaw = torch.cat([data_batch["history_yaws"], data_batch["target_yaws"]], axis=-2) # [B, A, T, D]
    src_mask = torch.cat([data_batch["history_availabilities"], data_batch["target_availabilities"]], axis=-1)# [B, A, T]

    # Estimate Velocity: 3 point method when possible and 2 point method when not
    if False: # 1st order approximation for debugging
        src_vel = dynamics.Unicycle.calculate_vel(src_pos, src_yaw, data_batch["dt"][0] , src_mask)
        src_vel[:, :, -1] = data_batch["curr_speed"].unsqueeze(-1) # replace calculated velocity with true velocity
    
    

    """
    Get Bezier Approximations
    """
    pos_curve, pos_dot_curve, pos_ddot_curve = get_bezier(src_pos)
    yaw_curve, yaw_rate_curve, _ = get_bezier(src_yaw)    
    T = src_pos.shape[-2]
    Taus = torch.linspace(0,1, T) 
    dTaudt = 1.0/((T-1)*data_batch["dt"][0]) # remove one for the starting point
    fit_poses = []
    fit_vels = []
    fit_yaws = []
    fit_yaw_rates = []
    fit_accels = []
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

    if False: # 1st order approximation for debugging
        states_1stord = torch.cat((src_pos, src_vel, src_yaw), axis =-1) 
        inputs_1stord = (states_1stord[:,:,1:,2:] - states_1stord[:,:,:-1,2:])/data_batch["dt"][0]

    data_batch["states"] = states[:,:,:-1,:] # Remove the future state
    data_batch["inputs"] = inputs[:,:,:-1,:] # Remove the future input
    data_batch["image"]  = data_batch["image"][...,T-2:,:,:] # Remove Image Trajectory

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



def plot_gammas(batch, net,  B=0, A=0):
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
        state_vel = state_agentA[...,2:3] # FORCE AGENTB to have the same velocity as agentA) torch.repeat_interleave(batch["states"][0:1,1,0:1,2:3], Bprime, axis=0)
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

    fig = plt.figure()
    ax = plt.axes(projection='3d')
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
    ax.set_title("gamma for agent A: [0,0,%.1f,0], agent B: [x,y,%.1f,0]" % (state_agentA[0,0,0,2].item(),state_agentA[0,0,0,2].item()) )
    # ax.set_ylabel("y position")
    # ax.set_xlabel("x position")
    ax.set_zlabel("$\gamma(x_A, x_B, e)$")
    ax.get_xaxis().set_ticks([])#set_visible(False)
    ax.get_yaxis().set_ticks([])#set_visible(False)
    # plt.savefig("test_gamma_plotter.png")

    img = wandb.Image(fig)
    
    plt.close()

    return img  


if __name__ == "__main__": 
    test_bezier()

