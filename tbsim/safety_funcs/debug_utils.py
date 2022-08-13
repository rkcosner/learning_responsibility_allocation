from tkinter import N
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


from tbsim.utils.geometry_utils import transform_points_tensor, calc_distance_map
import numpy as np
import torch

# import importlib; import tbsim; importlib.reload(tbsim.safety_funcs.debug_utils); from tbsim.safety_funcs.debug_utils import * 

import sys

def get_batch_memory_size(batch): 
    batch_size = 0 
    for k in batch.keys(): 
        if torch.is_tensor(batch[k]): 
            batch_size += sys.getsizeof(batch[k].storage())
    
    print("The batch is at least: ", batch_size * 1e-9, " GB") 

def plot_gif(batch, B=0, A=0):
    i = A
    plt.figure()
    imgs = batch['image'].cpu().detach().numpy()
    center = [56, 112]
    rfc = batch['raster_from_center'][B, i, :, 2]
    i_shift = 0 * int(rfc[0].item() - center[0])
    j_shift = 0 * int(rfc[1].item() - center[1])
    N_max = imgs.shape[(-1)]
    T = imgs.shape[(-3)] - 7
    s_imgs = np.expand_dims((imgs[B, i, T:]), axis=(-1))
    s_imgs = np.expand_dims(s_imgs, axis=(-1))
    colors = [
     [
      0.5, 0.0, 0.0],
     [
      0.0, 0.5, 0.0],
     [
      0.0, 0.0, 0.5],
     [
      0.0, 0.4, 0.4],
     [
      0.5, 0.3, 0.0],
     [
      0.5, 0.5, 0.0],
     [
      0.5, 0.0, 0.5]]
    white = [1.0, 1.0, 1.0]
    black = [0.0, 0.0, 0.0]
    segmented_image = np.stack([imgs[(B, i, 0)] * 0.0, imgs[(B, i, 0)] * 0.0, imgs[(B, i, 0)] * 0.0], axis=(-1)) + 0.5
    for j, color in enumerate(colors):
        c = np.array([color])
        segmented_image += (s_imgs[j] @ c).squeeze()
    else:
        ii, jj = np.where(imgs[(B, i, 0)] == 1.0)
        s = segmented_image
        s[ii + i_shift, jj + j_shift, :] = white
        s[ii+1, jj-1] = black
        s[ii+1, jj] = black
        s[ii+1, jj+1] = black
        s[ii , jj+1] = black
        s[ii , jj-1] = black
        s[ii-1, jj+1] = black
        s[ii-1, jj] = black
        s[ii-1, jj-1] = black
        ii, jj = np.where(imgs[(B, i, 0)] == -1.0)
        s[ii + i_shift, jj + j_shift, :] = black
        patch = plt.imshow(s)
        plt.savefig('temp.png')
        plt.axis('off')

        def animate(tau):
            ii, jj = np.where(imgs[(B, i, tau)] == 1.0)
            s = segmented_image
            s[ii + i_shift, jj + j_shift, :] = white
            s[ii+1, jj-1] = black
            s[ii+1, jj] = black
            s[ii+1, jj+1] = black
            s[ii , jj+1] = black
            s[ii , jj-1] = black
            s[ii-1, jj+1] = black
            s[ii-1, jj] = black
            s[ii-1, jj-1] = black
            ii, jj = np.where(imgs[(B, i, tau)] == -1.0)
            s[ii + i_shift, jj + j_shift, :] = black
            patch.set_data(s)

        anim = animation.FuncAnimation((plt.gcf()), animate, frames=T, interval=100)
        anim.save(('imgs_B' + str(B) + 'A' + str(i) + '.gif'), writer='imagemagick', fps=10)
        plt.close()


def plot_eval_img(batch, i, collision_flag):
    B = 0
    imgs = batch['image'].cpu().detach().numpy()
    N_max = imgs.shape[(-1)]
    T = imgs.shape[(-3)] - 7
    s_imgs = np.expand_dims((imgs[i, T:]), axis=(-1))
    s_imgs = np.expand_dims(s_imgs, axis=(-1))
    colors = [
     [
      0.5, 0.0, 0.0],
     [
      0.0, 0.5, 0.0],
     [
      0.0, 0.0, 0.5],
     [
      0.0, 0.4, 0.4],
     [
      0.5, 0.3, 0.0],
     [
      0.5, 0.5, 0.0],
     [
      0.5, 0.0, 0.5]]
    white = [1.0, 1.0, 1.0]
    black = [-1.0, -1.0, -1.0]
    segmented_image = np.stack([imgs[(i, 0)] * 0.0, imgs[(i, 0)] * 0.0, imgs[(i, 0)] * 0.0], axis=(-1)) + 0.5
    for j, color in enumerate(colors):
        c = np.array([color])
        segmented_image += (s_imgs[j] @ c).squeeze()
    # else:
    #     for j in range(10):
    #         traj_agent = batch['history_positions'][0, j, :, :]
    #         traj_raster = transform_points_tensor(traj_agent, batch['raster_from_center'][(0, j)].float())
    #         traj_raster[:,] = traj_raster[:,].clip(0, N_max - 1e-05)
    #         traj_raster[:,] = traj_raster[:,].clip(0, N_max - 1e-05)
    #         for point in traj_raster:
    #             segmented_image[(int(point[1]), int(point[0]))] = 1.0
    #         else:

    final_img = segmented_image
    for t in range(T): 
        states = imgs[i, T-1,...]
        ii, jj = np.where(states==1)
        states = np.concatenate([states[...,None],states[...,None],states[...,None]], axis=-1)
        states[ii, jj, :] = white
        states[ii+1, jj-1] = black
        states[ii+1, jj] = black
        states[ii+1, jj+1] = black
        states[ii , jj+1] = black
        states[ii , jj-1] = black
        states[ii-1, jj+1] = black
        states[ii-1, jj] = black
        states[ii-1, jj-1] = black
        final_img += states

    plt.figure()
    plt.imshow(np.flip(final_img, axis=0))
    if collision_flag: 
        plt.title("Collision!")
    plt.savefig('eval' + str(i) + '.png')
    plt.close()
    return segmented_image



def plot_traj(batch, i):
    B = 2
    imgs = batch['image'].cpu().detach().numpy()
    rfc = batch['raster_from_center'][0, i, :, 2]
    i_shift = int(rfc[0].item())
    j_shift = int(rfc[1].item())
    N_max = imgs.shape[(-1)]
    T = imgs.shape[(-3)] - 7
    s_imgs = np.expand_dims((imgs[0, i, T:]), axis=(-1))
    s_imgs = np.expand_dims(s_imgs, axis=(-1))
    colors = [
     [
      0.5, 0.0, 0.0],
     [
      0.0, 0.5, 0.0],
     [
      0.0, 0.0, 0.5],
     [
      0.0, 0.4, 0.4],
     [
      0.5, 0.3, 0.0],
     [
      0.5, 0.5, 0.0],
     [
      0.5, 0.0, 0.5]]
    segmented_image = np.stack([imgs[(0, i, 0)] * 0.0, imgs[(0, i, 0)] * 0.0, imgs[(0, i, 0)] * 0.0], axis=(-1)) + 0.5
    for j, color in enumerate(colors):
        c = np.array([color])
        segmented_image += (s_imgs[j] @ c).squeeze()
    else:
        for j in range(10):
            traj_agent = batch['history_positions'][0, j, :, :]
            traj_raster = transform_points_tensor(traj_agent, batch['raster_from_center'][(0, j)].float())
            traj_raster[:,] = traj_raster[:,].clip(0, N_max - 1e-05)
            traj_raster[:,] = traj_raster[:,].clip(0, N_max - 1e-05)
            for point in traj_raster:
                segmented_image[(int(point[1]), int(point[0]))] = 1.0
            else:
                plt.imshow(segmented_image)
                plt.savefig('traj' + str(i) + '.png')
                return segmented_image


def plot_image(batch, B, A):
    imgs = batch['image'].cpu().detach().numpy()
    if len(imgs.shape) == 5: 
        plt.imshow(imgs[(B, A, -8)])
    else: 
        plt.imshow(torch.sum(imgs[A, -8:], axis=0))
    plt.axis('off')
    plt.savefig('image_' + str(B) + '_' + str(A) + '.png')


def plot_bez_test(src, fit):
    B = 0
    A = 1
    src_vel = src.cpu().detach().numpy()
    fit_vel = fit.cpu().detach().numpy()
    fig, axs = plt.subplots(nrows=3, ncols=1)
    axs[0].plot(src_vel[B, A, :], '*')
    axs[0].plot(fit_vel[B, A, :])
    axs[1].plot(src_vel[B + 1, A, :], '*')
    axs[1].plot(fit_vel[B + 1, A, :])
    axs[2].plot(src_vel[B + 2, A, :], '*')
    axs[2].plot(fit_vel[B + 2, A, :])
    plt.savefig('bez_test.png')


def view_states_and_inputs(states, inputs, gt_states, gt_inputs, A=0, B=0):
    states = states.cpu().detach().numpy()
    inputs = inputs.cpu().detach().numpy()
    gt_states = gt_states.cpu().detach().numpy()
    gt_inputs = gt_inputs.cpu().detach().numpy()
    T = states.shape[(-2)]
    dt = 0.01
    Ts = np.linspace(0, T * dt, T)
    Ts_gt = np.linspace(0, T * dt, gt_states.shape[(-2)])
    fig, axs = plt.subplots(nrows=4, ncols=1)
    labels = ['x', 'y', 'v', 'yaw']
    for i in range(4):
        if len(states.shape) == 4:  
            axs[i].plot(Ts, states[B, A, :, i])
            axs[i].plot(Ts_gt, gt_states[B, A, :, i], '*')
        else: 
            axs[i].plot(Ts, states[B, :, i])
            axs[i].plot(Ts_gt, gt_states[B, :, i], '*')
        axs[i].set_ylabel(labels[i])
    else:
        plt.savefig('fit_states_B' + str(B) + 'A' + str(A) + '.png')
        fig, axs = plt.subplots(nrows=2, ncols=1)
        labels = ['a', 'yaw rate']
        for i in range(2):
            if len(inputs.shape) == 4: 
                axs[i].plot(Ts, inputs[B, A, :, i])
                axs[i].plot(Ts_gt[:-1], gt_inputs[B, A, :, i], '*')
            else:
                axs[i].plot(Ts, inputs[B, :, i])
                axs[i].plot(Ts_gt[:-1], gt_inputs[B, :, i], '*')
            axs[i].set_ylabel(labels[i])
        else:
            plt.savefig('fit_inputs_B' + str(B) + 'A' + str(A) + '.png')
