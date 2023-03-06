from collections import OrderedDict
import numpy as np
from tbsim.utils import l5_utils

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import tbsim.utils.torch_utils as TorchUtils

from tbsim.models.rasterized_models import (
    RasterizedResponsibilityModel,
    RasterizedPlanningModel,
    RasterizedVAEModel,
    RasterizedGCModel,
    RasterizedGANModel,
    RasterizedDiscreteVAEModel,
    RasterizedECModel,
    RasterizedTreeVAEModel,
    RasterizedSceneTreeModel,
)
from tbsim.models.base_models import (
    MLPTrajectoryDecoder,
    RasterizedMapUNet,
    ResponsibilityDecoder,
)
from tbsim.models.transformer_model import TransformerModel
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.metrics as Metrics
from tbsim.utils.batch_utils import batch_utils
import tbsim.utils.loss_utils as LossUtils
import tbsim.algos.algo_utils as AlgoUtils
from tbsim.utils.geometry_utils import transform_points_tensor

from tbsim.safety_funcs.utils import (
    batch_to_raw_all_agents, 
    scene_centric_batch_to_raw, 
    plot_gammas, 
    plot_static_gammas_inline,
    plot_static_gammas_traj
)
from tbsim.safety_funcs.cbfs import (
    BackupBarrierCBF,
    ExtendedNormBallCBF,
    NormBallCBF, 
    RssCBF
)


import matplotlib.pyplot as plt

import wandb


class Responsibility(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, do_log=True):
        """
        Creates networks and places them into @self.nets.
        """
        super(Responsibility, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self._do_log = do_log
        self.on_ngc = algo_config.on_ngc
        
        # RYAN: Traj Decoder is the Decoder portion of a train VAE
        #   - feature_dim : dimension of the input feature
        #   - state_dim : dimension of output trajectory at each step
        #   - num_steps : number of future states to predict 
        #   - dynamics_type : if specified, the network predicts inputs, otherwise predict future states directly 
        #   - dynamics_kwargs : dictionary of dynamics variables
        #   - step_time : time between steps (if dynamics_model is none, this isn't used)
        #   - network_kwargs : ketword args for the decoder networks
        #   - Gaussian_var : bool flag, whether to output the variance of the predicted trajectory 
        traj_decoder = ResponsibilityDecoder(
            feature_dim = algo_config.map_feature_dim + 5, # add 5 dims for the relative positions of the 2 agents (3) plus their velocities (2)
            state_dim=algo_config.responsibility_dim,
            num_steps=algo_config.future_num_frames,
            dynamics_type=None,
            dynamics_kwargs=algo_config.responsibility_dynamics,
            step_time=algo_config.step_time,
            network_kwargs=algo_config.decoder,
            layer_dims = algo_config.decoder.layer_dims
        )

        # RYAN: 
        """
            - model_arch : (ex) resnet18, modelarchitecture
        """
        modality_shapes["image"] = (8,224,224)
        self.nets["policy"] = RasterizedResponsibilityModel(
            model_arch=algo_config.model_architecture,
            input_image_shape= modality_shapes["image"],  # [C, H, W]
            trajectory_decoder=traj_decoder,
            map_feature_dim=algo_config.map_feature_dim,
            weights_scaling=[1.0, 1.0, 1.0],
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
            constraint_loss_leaky_relu_negative_slope = algo_config.constraint_loss.leaky_relu_negative_slope,
            sum_resp_loss_leaky_relu_negative_slope   = algo_config.sum_resp_loss.leaky_relu_negative_slope, 
            max_angle=algo_config.max_angle_diff,
            normalize_constraint=algo_config.cbf.normalize_constraint
        )
        # TODO : create loss discount parameters
        if   algo_config.cbf.type == "rss_cbf": 
            self.cbf = RssCBF()
        elif algo_config.cbf.type == "norm_ball_cbf": 
            self.cbf = NormBallCBF()
        elif algo_config.cbf.type == "extended_norm_ball_cbf":
            self.cbf = ExtendedNormBallCBF()
        elif algo_config.cbf.type == "backup_barrier_cbf": 
            self.cbf = BackupBarrierCBF(T_horizon = algo_config.cbf.T_horizon, 
                                        alpha=algo_config.cbf.alpha, 
                                        veh_veh=algo_config.cbf.veh_veh, 
                                        saturate_cbf=algo_config.cbf.saturate_cbf, 
                                        backup_controller_type=algo_config.cbf.backup_controller_type
                                        )
        else: 
            raise Exception("Config Error: algo_config.cbf is not properly defined")
        

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}

    def forward(self, obs_dict):
        # RYAN: This isn't called during training??? 
        return self.nets["policy"](obs_dict)["predictions"]

    def _plot_metrics(self, batch):

        state_fig, axs = plt.subplots(4,1)
        states = batch["states"][0,...]
        inputs = batch["inputs"][0,...]
        states = states.cpu().detach().numpy()
        inputs = inputs.cpu().detach().numpy()
        for i in range(states.shape[0]): 
            for j in range(4):
                axs[j].plot(states[i,:,j], linestyle = '-')
        axs[0].set_title("States")
        axs[0].set_ylabel("x")
        axs[1].set_ylabel("y")
        axs[2].set_ylabel("v")
        axs[3].set_ylabel("yaw")
        plt.savefig("state_viewer.png")

        input_fig, input_ax = plt.subplots(2,1)
        for i in range(states.shape[0]):
            input_ax[0].plot(inputs[i,:,0], linestyle="-")
            input_ax[1].plot(inputs[i,:,1], linestyle="-")
        input_ax[0].set_title("Inputs")
        input_ax[0].set_ylabel("accel")
        input_ax[1].set_ylabel("yaw rate")
        plt.savefig("input_viewer")

        import pdb; pdb.set_trace()
        
        plotted_metrics = None
        return plotted_metrics

    def _compute_metrics(self, batch, gammas):
        _, _, percent_violations, max_violations, evensplit_percent_violations, worstcase_percent_violations = self.nets["policy"].compute_cbf_constraint_loss(self.cbf, gammas, batch)
        metrics = {
            "percent_constraint_violations" : percent_violations, 
            "max_constraint_violations" : max_violations, 
            "even_split_percent_violations" : evensplit_percent_violations, 
            "worst_case_percent_violations": worstcase_percent_violations
            }

        return metrics

    def add_inputs_vel_to_batch(self, batch): 
        # Let the "current state" be time step (k-1) and then calculate the current input as (x_k - x_{k-1})/dt 

        if self.algo_config.scene_centric == False: 
            raise Exception("Responsibility calculations are scene-centric. Please set scene_centric = True in the algo_config.py")
        if self.algo_config.dynamics.type != "Unicycle": 
            raise Exception("Using dynamics: '" +self.algo_config.dynamics +"'that have not been implemented yet. Please use unicycle or add new dynamics and state parsing")

        batch = scene_centric_batch_to_raw(batch)        
        return batch

    def training_step(self, batch, batch_idx):
        
        # A single raw batch is 0.007 GB
        batch = batch_utils().parse_batch(batch)
        # A single parsed batch is 0.025 GB
        batch = self.add_inputs_vel_to_batch(batch)
        batch["states"].requires_grad = True

        gamma_preds = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(self.cbf, gamma_preds, batch)

        total_loss = 0.0
        for lk, ell in losses.items(): 
            try: 
                loss_contribution  = ell * self.algo_config.loss_weights[lk]
                total_loss += loss_contribution
            except: 
                import pdb; pdb.set_trace()
                
        # TODO: implement metrics
        metrics = self._compute_metrics(batch, gamma_preds)  

        for lk, l in losses.items(): 
            self.log("train/losses_"+lk, l)
        self.log("train/total_loss", total_loss)
        
        for mk, m in metrics.items(): 
            self.log("train/metrics/" + mk, m)

        return {
            "loss": total_loss, 
            "all_losses": losses, 
            # "all_metrics": metrics
        }

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True) # RYAN: this slows thing down, but is required to calculate dhdx
        batch = batch_utils().parse_batch(batch)
        batch = self.add_inputs_vel_to_batch(batch)

        batch["states"].requires_grad = True

        img_m2= plot_gammas(batch,self.nets["policy"], relspeed=-2.0)
        img_0 = plot_gammas(batch, self.nets["policy"], relspeed=0)
        img_p2= plot_gammas(batch,self.nets["policy"], relspeed=2.0)

        plots = {
            "gammas_m2": img_m2, 
            "gammas_0":  img_0, 
            "gammas_p2": img_p2
            }

        gamma_preds = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(self.cbf, gamma_preds, batch))
        metrics = self._compute_metrics(batch, gamma_preds) 


        # # plotted_metrics = self._plot_metrics(batch)
        # self.batch = batch # TODO: this is probably not how this should be done... but store data for validation epoch end, for plotting metrics
        
        return {"losses": losses, "metrics": metrics, "plots" : plots}

    def validation_epoch_end(self, outputs) -> None:
        # Log Losses
        losses = []
        for j in range(len(outputs)):
            losses_batch = []
            for k in outputs[0]["losses"]:
                losses_batch.append(outputs[j]["losses"][k].item())
            losses.append(losses_batch)
        losses = np.array(losses)
        losses = np.mean(losses, axis = 0)
        for i, k in enumerate(outputs[0]["losses"]):
            self.log("val/losses_" + k, losses[i])

        self.log("val/losses_prediction_loss", 0.0)
        self.log("val/losses_goal_loss",0.0)
        self.log("val/losses_collision_loss",0.0)
        self.log("val/losses_yaw_reg_loss",0.0)

        # Log Metrics TODO: currently metrics are turned off
        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

        for j in range(len(outputs)): 
            for k in outputs[0]["plots"]:
                if wandb.run is not None: 
                    wandb.log({"val/plot_gamma_"+k: outputs[j]["plots"][k]})

        if wandb.run is not None: 
            gammas2way = plot_static_gammas_inline(self.nets["policy"], 2, self.on_ngc)
            gammas4way_samelane = plot_static_gammas_inline(self.nets["policy"], 4, self.on_ngc)
            gammas4way_cross = plot_static_gammas_traj(self.nets["policy"], 4, self.on_ngc)
            gammasRoundabout = plot_static_gammas_traj(self.nets["policy"],0, self.on_ngc)

            wandb.log({"val/plot_gammas2way":gammas2way})
            wandb.log({"val/plot_gammas4way_samelane":gammas4way_samelane})
            wandb.log({"val/plot_gammas4way_cross": gammas4way_cross})
            wandb.log({"val/plot_gammasRoundabout": gammasRoundabout})
        torch.cuda.empty_cache()
        plt.close()


    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )



    