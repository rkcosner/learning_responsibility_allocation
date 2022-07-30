from turtle import pos
import torch 
from tbsim import dynamics as dynamics

import numpy as np 

import matplotlib.pyplot as plt

from math import comb

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
                out += comb(n,i) * (-(1-t)**(n-i-1)*(n-i) * (t**i) + (1-t)**(n-i)*i*t**(i-1)) * points[...,i,:]
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
                tmp += (n-i) * (n-i-1) * (1-t)**(n-i-2)*t**i 
                tmp += - 2 * (n-i) * (1-t)**(n-i-1) * i * t**(i-1)
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
    src_vel = dynamics.Unicycle.calculate_vel(src_pos, src_yaw, data_batch["dt"][0] , src_mask)
    src_vel[:, :, -1] = data_batch["curr_speed"].unsqueeze(-1) # replace calculated velocity with true velocity
    
    
    pos_curve, pos_dot_curve, pos_ddot_curve = get_bezier(src_pos)
    yaw_curve, yaw_rate_curve, _ = get_bezier(src_yaw)

    
    T = src_pos.shape[-2]
    Taus = torch.linspace(0,1, T, requires_grad = True)
    dTaudt = 1.0/(T*data_batch["dt"][0])

    fit_poses = []
    fit_vels = []
    fit_yaws = []
    fit_yaw_rates = []
    fit_accels = []
    for tau in Taus: 
        # Fit poses and yaw rates
        fit_poses.append(pos_curve(tau)[...,None,:] )

        # Fit yaws and velocities
        yaws =  yaw_curve(tau)
        yaw_rates = yaw_rate_curve(tau)
        fit_yaw_rates.append(yaw_rates[...,None, :] / dTaudt)
        fit_yaws.append(yaws[...,None, :])

        yaws = yaws.squeeze()
        yaw_rates = yaw_rates.squeeze()
        pos_dot = pos_dot_curve(tau) / dTaudt 
        fit_vel = pos_dot[...,0] * torch.cos(yaws) + pos_dot[...,1] * torch.sin(yaws)
        pos_ddot = pos_ddot_curve(tau) /dTaudt / dTaudt
        fit_accel  = torch.cos(yaws) * (pos_ddot[...,0] + pos_dot[...,1]*yaw_rates) 
        fit_accel += torch.sin(yaws) * (pos_ddot[...,1] - pos_dot[...,0]*yaw_rates)

        fit_vel = fit_vel[...,None,None]
        fit_accel = fit_accel[...,None,None]
        fit_vels.append(fit_vel)
        fit_accels.append(fit_accel)


    fit_pos = torch.cat(fit_poses, axis=-2)
    fit_vel = torch.cat(fit_vels, axis=-2)
    fit_accel = torch.cat(fit_accels, axis=-2)
    fit_yaw = torch.cat(fit_yaws, axis=-2)
    fit_yaw_rate = torch.cat(fit_yaws, axis=-2)

    states = torch.cat([fit_pos, fit_vel, fit_yaw], axis=-1)
    inputs = torch.cat([fit_accel, fit_yaw_rate], axis=-1)

    # TODO: This is starting to work, but it doesn't quite work yet

    # fit_vels = []
    # N = float(src_vel.shape[-2]-1)
    # for i in range(src_vel.shape[-2]): fit_vels.append(curve(float(i)/N)[...,None])
    # fit_vel = torch.cat(fit_vels, axis = -1)
    
    # src_vels = fit_vels

    # # Estimate the input
    # states = torch.cat((src_pos, src_vel, src_yaw), axis =-1) 
    # data_batch["inputs"] = (states[:,:,1:,2:] - states[:,:,:-1,2:])/data_batch["dt"][0]

    data_batch["states"] = states[:,:,-1,:] # Remove the future state
    data_batch["inputs"] = inputs[:,:,-1,:] # Remove the future input

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

if __name__ == "__main__": 
    test_bezier()