from logging import raiseExceptions
import numpy as np
import pdb

from numpy.lib.function_base import flip
from tbsim.configs.base import AlgoConfig
import torch
import math, copy
from typing import Dict
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tbsim.dynamics import Unicycle, DoubleIntegrator
from tbsim.dynamics.base import DynType
from tbsim.models.cnn_roi_encoder import (
    CNNROIMapEncoder,
    ROI_align,
    generate_ROIs,
    Indexing_ROI_result,
    obtain_lane_flag,
)
from tbsim.utils.tensor_utils import round_2pi
from tbsim.utils.geometry_utils import (
    VEH_VEH_collision,
    VEH_PED_collision,
    PED_VEH_collision,
    PED_PED_collision,
)
from tbsim.models.Transformer import (
    make_transformer_model,
    simplelinear,
    subsequent_mask,
)


class TransformerModel(nn.Module):
    def __init__(
        self,
        algo_config,
    ):
        super(TransformerModel, self).__init__()
        self.step_time = algo_config.step_time
        self.algo_config = algo_config
        self.calc_likelihood = algo_config.calc_likelihood
        self.M = algo_config.M

        self.register_buffer(
            "weights_scaling", torch.tensor(algo_config.weights_scaling)
        )
        self.ego_weight = algo_config.ego_weight
        self.all_other_weight = algo_config.all_other_weight
        self.criterion = nn.MSELoss(reduction="none")
        self.map_enc_mode = algo_config.map_enc_mode
        "unicycle for vehicles and double integrators for pedestrians"
        self.dyn_list = {
            DynType.UNICYCLE: Unicycle(
                "vehicle", vbound=[algo_config.vmin, algo_config.vmax]
            ),
            DynType.DI: DoubleIntegrator(
                "pedestrian",
                abound=np.array([[-3.0, 3.0], [-3.0, 3.0]]),
                vbound=np.array([[-5.0, 5.0], [-5.0, 5.0]]),
            ),
        }
        if algo_config.calc_collision:
            self.col_funs = {
                "VV": VEH_VEH_collision,
                "VP": VEH_PED_collision,
                "PV": PED_VEH_collision,
                "PP": PED_PED_collision,
            }

        self.training_num = 0
        self.training_num_N = algo_config.training_num_N

        "src_dim:x,y,v,sin(yaw),cos(yaw)+16-dim type encoding"
        "tgt_dim:x,y,yaw"
        if algo_config.name == "TransformerGAN":
            self.use_GAN = True
            N_layer_enc_discr = algo_config.Discriminator.N_layer_enc
            self.GAN_static = algo_config.GAN_static
        else:
            self.use_GAN = False
            N_layer_enc_discr = None
            self.GAN_static = False

        if algo_config.vectorize_lane:
            src_dim = 21 + 3 * algo_config.points_per_lane * 4
        else:
            src_dim = 21
        self.Transformermodel, self.Discriminator = make_transformer_model(
            src_dim=src_dim,
            tgt_dim=3,
            out_dim=2,
            dyn_list=self.dyn_list.values(),
            N_t=algo_config.N_t,
            N_a=algo_config.N_a,
            d_model=algo_config.d_model,
            XY_pe_dim=algo_config.XY_pe_dim,
            temporal_pe_dim=algo_config.temporal_pe_dim,
            map_emb_dim=algo_config.map_emb_dim,
            d_ff=algo_config.d_ff,
            head=algo_config.head,
            dropout=algo_config.dropout,
            step_size=algo_config.XY_step_size,
            N_layer_enc=algo_config.N_layer_enc,
            N_layer_tgt_enc=algo_config.N_layer_tgt_enc,
            N_layer_tgt_dec=algo_config.N_layer_tgt_enc,
            M=self.M,
            use_GAN=self.use_GAN,
            GAN_static=self.GAN_static,
            N_layer_enc_discr=N_layer_enc_discr,
        )
        self.src_emb = nn.Linear(
            21,
            algo_config.d_model,
        ).cuda()

        "CNN for map encoding"
        self.CNNmodel = CNNROIMapEncoder(
            algo_config.CNN.map_channels,
            algo_config.CNN.hidden_channels,
            algo_config.CNN.ROI_outdim,
            algo_config.CNN.output_size,
            algo_config.CNN.kernel_size,
            algo_config.CNN.strides,
            algo_config.CNN.input_size,
        )

    @staticmethod
    def raw2feature(pos, vel, yaw, raw_type, mask, lanes=None, add_noise=False):
        "map raw src into features of dim 21+lane dim"

        """
        PERCEPTION_LABELS = [
        "PERCEPTION_LABEL_NOT_SET",
        "PERCEPTION_LABEL_UNKNOWN",
        "PERCEPTION_LABEL_DONTCARE",
        "PERCEPTION_LABEL_CAR",
        "PERCEPTION_LABEL_VAN",
        "PERCEPTION_LABEL_TRAM",
        "PERCEPTION_LABEL_BUS",
        "PERCEPTION_LABEL_TRUCK",
        "PERCEPTION_LABEL_EMERGENCY_VEHICLE",
        "PERCEPTION_LABEL_OTHER_VEHICLE",
        "PERCEPTION_LABEL_BICYCLE",
        "PERCEPTION_LABEL_MOTORCYCLE",
        "PERCEPTION_LABEL_CYCLIST",
        "PERCEPTION_LABEL_MOTORCYCLIST",
        "PERCEPTION_LABEL_PEDESTRIAN",
        "PERCEPTION_LABEL_ANIMAL",
        "AVRESEARCH_LABEL_DONTCARE",
        ]
        """
        dyn_type = torch.zeros_like(raw_type)
        veh_mask = (raw_type >= 3) & (raw_type <= 13)
        ped_mask = (raw_type == 14) | (raw_type == 15)
        veh_mask = veh_mask | ped_mask
        ped_mask = ped_mask * 0
        dyn_type += DynType.UNICYCLE * veh_mask
        # all vehicles, cyclists, and motorcyclists
        if add_noise:
            pos_noise = torch.randn(pos.size(0), 1, 1, 2).to(pos.device) * 0.5
            yaw_noise = torch.randn(pos.size(0), 1, 1, 1).to(pos.device) * 0.1
            if pos.ndim == 5:
                pos_noise = pos_noise.unsqueeze(1)
                yaw_noise = yaw_noise.unsqueeze(1)
            feature_veh = torch.cat(
                (
                    pos + pos_noise,
                    vel,
                    torch.cos(yaw + yaw_noise),
                    torch.sin(yaw + yaw_noise),
                ),
                dim=-1,
            )
        else:
            feature_veh = torch.cat((pos, vel, torch.cos(yaw), torch.sin(yaw)), dim=-1)

        state_veh = torch.cat((pos, vel, yaw), dim=-1)

        # pedestrians and animals
        if add_noise:
            pos_noise = torch.randn(pos.size(0), 1, 1, 2).to(pos.device) * 0.5
            yaw_noise = torch.randn(pos.size(0), 1, 1, 1).to(pos.device) * 0.1
            if pos.ndim == 5:
                pos_noise = pos_noise.unsqueeze(1)
                yaw_noise = yaw_noise.unsqueeze(1)
            ped_feature = torch.cat(
                (
                    pos + pos_noise,
                    vel,
                    vel * torch.sin(yaw + yaw_noise),
                    vel * torch.cos(yaw + yaw_noise),
                ),
                dim=-1,
            )
        else:
            ped_feature = torch.cat(
                (pos, vel, vel * torch.sin(yaw), vel * torch.cos(yaw)), dim=-1
            )
        state_ped = torch.cat((pos, vel * torch.cos(yaw), vel * torch.sin(yaw)), dim=-1)
        state = state_veh * veh_mask.view(
            [*raw_type.shape, 1, 1]
        ) + state_ped * ped_mask.view([*raw_type.shape, 1, 1])
        dyn_type += DynType.DI * ped_mask

        feature = feature_veh * veh_mask.view(
            [*raw_type.shape, 1, 1]
        ) + ped_feature * ped_mask.view([*raw_type.shape, 1, 1])

        type_embedding = F.one_hot(raw_type, 16)

        if pos.ndim == 4:
            if lanes is not None:
                feature = torch.cat(
                    (
                        feature,
                        type_embedding.unsqueeze(-2).repeat(1, 1, feature.size(2), 1),
                        lanes[:, :, None, :].repeat(1, 1, feature.size(2), 1),
                    ),
                    dim=-1,
                )
            else:
                feature = torch.cat(
                    (
                        feature,
                        type_embedding.unsqueeze(-2).repeat(1, 1, feature.size(2), 1),
                    ),
                    dim=-1,
                )

        elif pos.ndim == 5:
            if lanes is not None:
                feature = torch.cat(
                    (
                        feature,
                        type_embedding.unsqueeze(-2).repeat(
                            1, 1, 1, feature.size(-2), 1
                        ),
                        lanes[:, :, None, None, :].repeat(
                            1, feature.size(1), 1, feature.size(2), 1
                        ),
                    ),
                    dim=-1,
                )
            else:
                feature = torch.cat(
                    (
                        feature,
                        type_embedding.unsqueeze(-2).repeat(
                            1, 1, 1, feature.size(-2), 1
                        ),
                    ),
                    dim=-1,
                )
        feature = feature * mask.unsqueeze(-1)
        return feature, dyn_type, state

    @staticmethod
    def tgt_temporal_mask(p, tgt_mask):
        "use a binomial distribution with parameter p to mask out the first k steps of the tgt"
        nbatches = tgt_mask.size(0)
        T = tgt_mask.size(2)
        mask_hint = torch.ones_like(tgt_mask)
        sample = np.random.binomial(T, p, nbatches)
        for i in range(nbatches):
            mask_hint[i, :, sample[i] :] = 0
        return mask_hint

    def generate_edges(
        self,
        raw_type,
        extents,
        pos_pred,
        yaw_pred,
    ):
        veh_mask = (raw_type >= 3) & (raw_type <= 13)
        ped_mask = (raw_type == 14) | (raw_type == 15)

        agent_mask = veh_mask | ped_mask
        edge_types = ["VV", "VP", "PV", "PP"]
        edges = {et: list() for et in edge_types}
        for i in range(agent_mask.shape[0]):
            agent_idx = torch.where(agent_mask[i] != 0)[0]
            edge_idx = torch.combinations(agent_idx, r=2)
            VV_idx = torch.where(
                veh_mask[i, edge_idx[:, 0]] & veh_mask[i, edge_idx[:, 1]]
            )[0]
            VP_idx = torch.where(
                veh_mask[i, edge_idx[:, 0]] & ped_mask[i, edge_idx[:, 1]]
            )[0]
            PV_idx = torch.where(
                ped_mask[i, edge_idx[:, 0]] & veh_mask[i, edge_idx[:, 1]]
            )[0]
            PP_idx = torch.where(
                ped_mask[i, edge_idx[:, 0]] & ped_mask[i, edge_idx[:, 1]]
            )[0]
            if pos_pred.ndim == 4:
                edges_of_all_types = torch.cat(
                    (
                        pos_pred[i, edge_idx[:, 0], :],
                        yaw_pred[i, edge_idx[:, 0], :],
                        pos_pred[i, edge_idx[:, 1], :],
                        yaw_pred[i, edge_idx[:, 1], :],
                        extents[i, edge_idx[:, 0]]
                        .unsqueeze(-2)
                        .repeat(1, pos_pred.size(-2), 1),
                        extents[i, edge_idx[:, 1]]
                        .unsqueeze(-2)
                        .repeat(1, pos_pred.size(-2), 1),
                    ),
                    dim=-1,
                )
                edges["VV"].append(edges_of_all_types[VV_idx])
                edges["VP"].append(edges_of_all_types[VP_idx])
                edges["PV"].append(edges_of_all_types[PV_idx])
                edges["PP"].append(edges_of_all_types[PP_idx])
            elif pos_pred.ndim == 5:

                edges_of_all_types = torch.cat(
                    (
                        pos_pred[i, :, edge_idx[:, 0], :],
                        yaw_pred[i, :, edge_idx[:, 0], :],
                        pos_pred[i, :, edge_idx[:, 1], :],
                        yaw_pred[i, :, edge_idx[:, 1], :],
                        extents[i, None, edge_idx[:, 0], None, :].repeat(
                            pos_pred.size(1), 1, pos_pred.size(-2), 1
                        ),
                        extents[i, None, edge_idx[:, 1], None, :].repeat(
                            pos_pred.size(1), 1, pos_pred.size(-2), 1
                        ),
                    ),
                    dim=-1,
                )
                edges["VV"].append(edges_of_all_types[:, VV_idx])
                edges["VP"].append(edges_of_all_types[:, VP_idx])
                edges["PV"].append(edges_of_all_types[:, PV_idx])
                edges["PP"].append(edges_of_all_types[:, PP_idx])
        if pos_pred.ndim == 4:
            for et, v in edges.items():
                edges[et] = torch.cat(v, dim=0)
        elif pos_pred.ndim == 5:
            for et, v in edges.items():
                edges[et] = torch.cat(v, dim=1)
        return edges

    def integrate_forward(self, x0, action, dyn_type):
        """
        Integrate the state forward with initial state x0, action u
        Args:
            x0 (Torch.tensor): state tensor of size [B,Num_agent,1,4]
            action (Torch.tensor): action tensor of size [B,Num_agent,T,2]
            dyn_type (Torch.tensor(dtype=int)): [description]
        Returns:
            state tensor of size [B,Num_agent,T,4]
        """
        T = action.size(-2)
        x = [x0.squeeze(-2)] + [None] * T
        veh_mask = (dyn_type == DynType.UNICYCLE).view(*dyn_type.shape, 1)
        ped_mask = (dyn_type == DynType.DI).view(*dyn_type.shape, 1)
        if action.ndim == 5:
            veh_mask = veh_mask.unsqueeze(1)
            ped_mask = ped_mask.unsqueeze(1)
        for t in range(T):
            x[t + 1] = (
                self.dyn_list[DynType.UNICYCLE].step(
                    x[t], action[..., t, :], self.step_time
                )
                * veh_mask
                + self.dyn_list[DynType.DI].step(
                    x[t], action[..., t, :], self.step_time
                )
                * ped_mask
            )

        x = torch.stack(x[1:], dim=-2)
        pos = self.dyn_list[DynType.UNICYCLE].state2pos(x) * veh_mask.unsqueeze(
            -1
        ) + self.dyn_list[DynType.DI].state2pos(x) * ped_mask.unsqueeze(-1)
        yaw = self.dyn_list[DynType.UNICYCLE].state2yaw(x) * veh_mask.unsqueeze(
            -1
        ) + self.dyn_list[DynType.DI].state2yaw(x) * ped_mask.unsqueeze(-1)
        return x, pos, yaw

    def forward(
        self, data_batch: Dict[str, torch.Tensor], batch_idx: int = None
    ) -> Dict[str, torch.Tensor]:

        device = data_batch["history_positions"].device
        raw_type = torch.cat(
            (data_batch["type"].unsqueeze(1), data_batch["all_other_agents_types"]),
            dim=1,
        ).type(torch.int64)
        extents = torch.cat(
            (
                data_batch["extent"][..., :2].unsqueeze(1),
                torch.max(data_batch["all_other_agents_history_extents"], dim=-2)[0],
            ),
            dim=1,
        )

        src_pos = torch.cat(
            (
                data_batch["history_positions"].unsqueeze(1),
                data_batch["all_other_agents_history_positions"],
            ),
            dim=1,
        )
        "history position and yaw need to be flipped so that they go from past to recent"
        src_pos = torch.flip(src_pos, dims=[-2])
        src_yaw = torch.cat(
            (
                data_batch["history_yaws"].unsqueeze(1),
                data_batch["all_other_agents_history_yaws"],
            ),
            dim=1,
        )
        src_yaw = torch.flip(src_yaw, dims=[-2])
        src_world_yaw = src_yaw + (
            data_batch["yaw"]
            .view(-1, 1, 1, 1)
            .repeat(1, src_yaw.size(1), src_yaw.size(2), 1)
        ).type(torch.float)
        src_mask = torch.cat(
            (
                data_batch["history_availabilities"].unsqueeze(1),
                data_batch["all_other_agents_history_availability"],
            ),
            dim=1,
        ).bool()

        src_mask = torch.flip(src_mask, dims=[-1])
        # estimate velocity
        src_vel = self.dyn_list[DynType.UNICYCLE].calculate_vel(
            src_pos, src_yaw, self.step_time, src_mask
        )

        src_vel[:, 0, -1] = torch.clip(
            data_batch["speed"].unsqueeze(-1),
            min=self.algo_config.vmin,
            max=self.algo_config.vmax,
        )
        if self.algo_config.vectorize_lane:
            src_lanes = torch.cat(
                (
                    data_batch["ego_lanes"].unsqueeze(1),
                    data_batch["all_other_agents_lanes"],
                ),
                dim=1,
            ).type(torch.float)
            src_lanes = src_lanes.view(*src_lanes.shape[:2], -1)
        else:
            src_lanes = None
        src, dyn_type, src_state = self.raw2feature(
            src_pos, src_vel, src_yaw, raw_type, src_mask, src_lanes
        )

        # generate ROI based on the rasterized position
        ROI, index = generate_ROIs(
            src_pos,
            src_world_yaw,
            data_batch["centroid"],
            data_batch["raster_from_world"],
            src_mask,
            torch.tensor(self.algo_config.CNN.patch_size).to(device),
            mode=self.map_enc_mode,
        )
        image = data_batch["image"]
        CNN_out = self.CNNmodel(image, ROI)
        if self.map_enc_mode == "all":
            emb_size = (*src.shape[:-1], self.algo_config.CNN.output_size)
        elif self.map_enc_mode == "last":
            emb_size = (*src.shape[:-2], self.algo_config.CNN.output_size)

        # put the CNN output in the right location of the embedding
        map_emb = Indexing_ROI_result(CNN_out, index, emb_size)
        tgt_mask = torch.cat(
            (
                data_batch["target_availabilities"].unsqueeze(1),
                data_batch["all_other_agents_future_availability"],
            ),
            dim=1,
        ).bool()
        num = torch.arange(0, src_mask.shape[2]).view(1, 1, -1).to(src_mask.device)
        nummask = num * src_mask
        last_idx, _ = torch.max(nummask, dim=2)
        curr_state = torch.gather(
            src_state, 2, last_idx[..., None, None].repeat(1, 1, 1, 4)
        )
        curr_yaw = torch.gather(
            src_yaw, 2, last_idx[..., None, None].repeat(1, 1, 1, 1)
        )

        tgt_pos = torch.cat(
            (
                data_batch["target_positions"].unsqueeze(1),
                data_batch["all_other_agents_future_positions"],
            ),
            dim=1,
        )
        tgt_yaw = torch.cat(
            (
                data_batch["target_yaws"].unsqueeze(1),
                data_batch["all_other_agents_future_yaws"],
            ),
            dim=1,
        )

        tgt = torch.cat((tgt_pos, tgt_yaw), dim=-1)
        curr_pos_yaw = torch.cat((curr_state[..., 0:2], curr_yaw), dim=-1)

        # masking part of the target and gradually increase the masked length until the whole target is masked
        tgt_mask_hint = torch.zeros_like(tgt_mask)

        tgt = tgt - curr_pos_yaw.repeat(1, 1, tgt.size(2), 1) * tgt_mask.unsqueeze(-1)

        tgt_hint = tgt * tgt_mask_hint.unsqueeze(-1)

        tgt_mask_agent = (
            tgt_mask.any(dim=-1).unsqueeze(-1).repeat(1, 1, tgt_mask.size(-1))
        )

        seq_mask = subsequent_mask(tgt_mask.size(-1)).to(tgt.device)
        # tgt_mask = tgt_mask.unsqueeze(-1).repeat(
        #     1, 1, 1, tgt.size(-2)
        # ) * seq_mask.unsqueeze(0)
        tgt_mask_dec = tgt_mask_agent.unsqueeze(-1) * seq_mask.unsqueeze(0)

        out, prob = self.Transformermodel.forward(
            src,
            tgt_hint,
            src_mask,
            tgt_mask_dec,
            tgt_mask_agent,
            dyn_type,
            map_emb,
        )

        u_pred = self.Transformermodel.generator(out)

        if self.M > 1:
            curr_state = curr_state.unsqueeze(1).repeat(1, self.M, 1, 1, 1)

        x_pred, pos_pred, yaw_pred = self.integrate_forward(
            curr_state, u_pred, dyn_type
        )
        lane_mask = (image[:, 0] < 1.0).type(torch.float)
        if self.M == 1:
            pred_world_yaw = yaw_pred + (data_batch["yaw"].view(-1, 1, 1, 1)).type(
                torch.float
            )

        else:
            pred_world_yaw = yaw_pred + (data_batch["yaw"].view(-1, 1, 1, 1, 1)).type(
                torch.float
            )
        lane_flags = obtain_lane_flag(
            lane_mask,
            pos_pred,
            pred_world_yaw,
            data_batch["centroid"].type(torch.float),
            data_batch["raster_from_world"],
            tgt_mask_agent,
            extents.type(torch.float),
            self.algo_config.CNN.veh_ROI_outdim,
        )
        if self.M > 1:
            max_idx = torch.max(prob, dim=-1)[1]
            ego_pred_positions = pos_pred[torch.arange(0, pos_pred.size(0)), max_idx, 0]
            ego_pred_yaws = yaw_pred[torch.arange(0, pos_pred.size(0)), max_idx, 0]
        else:
            ego_pred_positions = pos_pred[:, 0]
            ego_pred_yaws = yaw_pred[:, 0]
        out_dict = {
            "predictions": {
                "positions": ego_pred_positions,
                "yaws": ego_pred_yaws,
            },
            "scene_predictions": {
                "positions": pos_pred,
                "yaws": yaw_pred,
                "prob": prob,
                "raw_outputs": x_pred,
                "lane_flags": lane_flags,
            },
            "curr_pos_yaw": curr_pos_yaw,
        }

        if self.algo_config.calc_collision:
            out_dict["scene_predictions"]["edges"] = self.generate_edges(
                raw_type, extents, pos_pred, yaw_pred
            )

        if self.calc_likelihood:

            if self.GAN_static:
                src_noisy, _, __class__ = self.raw2feature(
                    src_pos,
                    src_vel,
                    src_yaw,
                    raw_type,
                    src_mask,
                    torch.zeros_like(src_lanes) if src_lanes is not None else None,
                    add_noise=True,
                )
                src_rel = src_noisy[:, :, -1:].clone()
                src_rel[..., 0:2] -= (
                    src_noisy[:, 0:1, -1:, 0:2] * src_mask[:, :, -1:, None]
                )
                if map_emb.ndim == 4:
                    likelihood = self.Discriminator(
                        src_rel,
                        src_mask[:, :, -1:],
                        dyn_type,
                        map_emb[:, :, -1:],
                    ).view(src.shape[0], -1)
                else:
                    likelihood = self.Discriminator(
                        src_rel,
                        src_mask[:, :, -1:],
                        dyn_type,
                        map_emb.unsqueeze(-2),
                    ).view(src.shape[0], -1)
            else:
                likelihood = self.Discriminator(src, src_mask, dyn_type, map_emb).view(
                    src.shape[0], -1
                )
            if self.GAN_static:
                src_new, src_mask_new, map_emb_new = self.pred2obs_static(
                    data_batch,
                    pos_pred,
                    yaw_pred,
                    tgt_mask_agent,
                    raw_type,
                    torch.zeros_like(src_lanes) if src_lanes is not None else None,
                )
            else:
                src_new, src_mask_new, map_emb_new = self.pred2obs(
                    src_pos,
                    src_yaw,
                    src_mask,
                    data_batch,
                    pos_pred,
                    yaw_pred,
                    tgt_mask_agent,
                    raw_type,
                    self.algo_config.f_steps,
                )
            if self.M == 1:
                src_new_rel = src_new.clone()
                src_new_rel[..., 0:2] -= src_new[
                    :, 0:1, :, 0:2
                ] * src_mask_new.unsqueeze(-1)
                likelihood_new = self.Discriminator(
                    src_new_rel, src_mask_new, dyn_type, map_emb_new
                ).view(src.shape[0], -1)
            else:
                src_new_rel = src_new.clone()
                src_new_rel[..., 0:2] -= src_new[
                    :, :, 0:1, :, 0:2
                ] * src_mask_new.unsqueeze(-1)
                likelihood_new = list()
                for i in range(self.M):
                    likelihood_new.append(
                        self.Discriminator(
                            src_new_rel[:, i],
                            src_mask_new[:, i],
                            dyn_type,
                            map_emb_new[:, i],
                        )
                    )
                likelihood_new = torch.stack(likelihood_new, dim=1).view(
                    src.shape[0], self.M, -1
                )
            out_dict["scene_predictions"]["likelihood_new"] = likelihood_new
            out_dict["scene_predictions"]["likelihood"] = likelihood
        return out_dict

    def pred2obs(
        self,
        src_pos,
        src_yaw,
        src_mask,
        data_batch,
        pos_pred,
        yaw_pred,
        pred_mask,
        raw_type,
        src_lanes,
        f_steps=1,
    ):
        """generate observation for the predicted scene f_step steps into the future

        Args:
            src_pos (torch.tensor[torch.float]): xy position in src
            src_yaw (torch.tensor[torch.float]): yaw in src
            src_mask (torch.tensor[torch.bool]): mask for src
            data_batch (dict): input data dictionary
            pos_pred (torch.tensor[torch.float]): predicted xy trajectory
            yaw_pred (torch.tensor[torch.float]): predicted yaw
            pred_mask (torch.tensor[torch.bool]): mask for prediction
            raw_type (torch.tensor[torch.int]): type of agents
            src_lanes (torch.tensor[torch.float]): lane info
            f_steps (int, optional): [description]. Defaults to 1.

        Returns:
            torch.tensor[torch.float]: new src for the transformer
            torch.tensor[torch.bool]: new src mask
            torch.tensor[torch.float]: new map encoding
        """
        if pos_pred.ndim == 5:
            src_pos = src_pos.unsqueeze(1).repeat(1, self.M, 1, 1, 1)
            src_yaw = src_yaw.unsqueeze(1).repeat(1, self.M, 1, 1, 1)
            src_mask = src_mask.unsqueeze(1).repeat(1, self.M, 1, 1)
            pred_mask = pred_mask.unsqueeze(1).repeat(1, self.M, 1, 1)
            raw_type = raw_type.unsqueeze(1).repeat(1, self.M, 1)
        pos_new = torch.cat(
            (src_pos[..., f_steps:, :], pos_pred[..., :f_steps, :]), dim=-2
        )
        yaw_new = torch.cat(
            (src_yaw[..., f_steps:, :], yaw_pred[..., :f_steps, :]), dim=-2
        )
        src_mask_new = torch.cat(
            (src_mask[..., f_steps:], pred_mask[..., :f_steps]), dim=-1
        )
        vel_new = self.dyn_list[DynType.UNICYCLE].calculate_vel(
            pos_new, yaw_new, self.step_time, src_mask_new
        )
        src_new, _, _ = self.raw2feature(
            pos_new,
            vel_new,
            yaw_new,
            raw_type,
            src_mask_new,
            torch.zeros_like(src_lanes) if src_lanes is not None else None,
        )
        if yaw_new.ndim == 4:
            new_world_yaw = yaw_new + (
                data_batch["yaw"]
                .view(-1, 1, 1, 1)
                .repeat(1, yaw_new.size(1), yaw_new.size(2), 1)
            ).type(torch.float)
        elif yaw_new.ndim == 5:
            new_world_yaw = yaw_new + (
                data_batch["yaw"]
                .view(-1, 1, 1, 1, 1)
                .repeat(1, self.M, yaw_new.size(-3), yaw_new.size(-2), 1)
            ).type(torch.float)
        if self.M == 1:
            ROI, index = generate_ROIs(
                pos_new,
                new_world_yaw,
                data_batch["centroid"],
                data_batch["raster_from_world"],
                src_mask_new,
                torch.tensor(self.algo_config.CNN.patch_size).to(src_mask_new.device),
                mode="last",
            )

            CNN_out = self.CNNmodel(data_batch["image"].permute(0, 3, 1, 2), ROI)
            emb_size = (*src_new.shape[:-2], self.algo_config.CNN.output_size)
            map_emb_new = Indexing_ROI_result(CNN_out, index, emb_size)
        else:

            emb_size = (*src_new.shape[:-2], self.algo_config.CNN.output_size)
            map_emb_new = list()
            for i in range(self.M):
                ROI, index = generate_ROIs(
                    pos_new[:, i],
                    new_world_yaw[:, i],
                    data_batch["centroid"],
                    data_batch["raster_from_world"],
                    src_mask_new[:, i],
                    torch.tensor(self.algo_config.CNN.patch_size).to(
                        src_mask_new.device
                    ),
                    mode="last",
                )
                CNN_out = self.CNNmodel(data_batch["image"].permute(0, 3, 1, 2), ROI)
                map_emb_new.append(Indexing_ROI_result(CNN_out, index, emb_size))
            map_emb_new = torch.stack(map_emb_new, dim=1)
        return src_new, src_mask_new, map_emb_new

    def pred2obs_static(
        self,
        data_batch,
        pos_pred,
        yaw_pred,
        pred_mask,
        raw_type,
        src_lanes,
    ):
        """generate observation for every step of the predictions

        Args:
            data_batch (dict): input data dictionary
            pos_pred (torch.tensor[torch.float]): predicted xy trajectory
            yaw_pred (torch.tensor[torch.float]): predicted yaw
            pred_mask (torch.tensor[torch.bool]): mask for prediction
            raw_type (torch.tensor[torch.int]): type of agents
            src_lanes (torch.tensor[torch.float]): lane info
        Returns:
            torch.tensor[torch.float]: new src for the transformer
            torch.tensor[torch.bool]: new src mask
            torch.tensor[torch.float]: new map encoding
        """
        if pos_pred.ndim == 5:
            pred_mask = pred_mask.unsqueeze(1).repeat(1, self.M, 1, 1)
            raw_type = raw_type.unsqueeze(1).repeat(1, self.M, 1)

        pred_vel = self.dyn_list[DynType.UNICYCLE].calculate_vel(
            pos_pred, yaw_pred, self.step_time, pred_mask
        )
        src_new, _, _ = self.raw2feature(
            pos_pred,
            pred_vel,
            yaw_pred,
            raw_type,
            pred_mask,
            torch.zeros_like(src_lanes) if src_lanes is not None else None,
            add_noise=True,
        )
        if yaw_pred.ndim == 4:
            new_world_yaw = yaw_pred + (
                data_batch["yaw"]
                .view(-1, 1, 1, 1)
                .repeat(1, yaw_pred.size(1), yaw_pred.size(2), 1)
            ).type(torch.float)
        elif yaw_pred.ndim == 5:
            new_world_yaw = yaw_pred + (
                data_batch["yaw"]
                .view(-1, 1, 1, 1, 1)
                .repeat(1, self.M, yaw_pred.size(-3), yaw_pred.size(-2), 1)
            ).type(torch.float)
        if self.M == 1:
            ROI, index = generate_ROIs(
                pos_pred,
                new_world_yaw,
                data_batch["centroid"],
                data_batch["raster_from_world"],
                pred_mask,
                torch.tensor(self.algo_config.CNN.patch_size).to(pos_pred.device),
                mode="all",
            )
            CNN_out = self.CNNmodel(data_batch["image"].permute(0, 3, 1, 2), ROI)
            emb_size = (*src_new.shape[:-1], self.algo_config.CNN.output_size)

            map_emb_new = Indexing_ROI_result(CNN_out, index, emb_size)
        else:
            emb_size = (*src_new[:, 0].shape[:-1], self.algo_config.CNN.output_size)
            map_emb_new = list()
            for i in range(self.M):
                ROI, index = generate_ROIs(
                    pos_pred[:, i],
                    new_world_yaw[:, i],
                    data_batch["centroid"],
                    data_batch["raster_from_world"],
                    pred_mask[:, i],
                    torch.tensor(self.algo_config.CNN.patch_size).to(pos_pred.device),
                    mode="all",
                )
                CNN_out = self.CNNmodel(data_batch["image"].permute(0, 3, 1, 2), ROI)
                map_emb_new.append(Indexing_ROI_result(CNN_out, index, emb_size))
            map_emb_new = torch.stack(map_emb_new, dim=1)

        return src_new, pred_mask, map_emb_new

    def regularization_loss(self, pred_batch, data_batch):
        # velocity regularization
        vel = pred_batch["scene_predictions"]["raw_outputs"][..., 2]
        reg_loss = F.relu(vel - self.algo_config.vmax) + F.relu(
            self.algo_config.vmin - vel
        )
        return torch.sum(reg_loss) / (
            torch.sum(data_batch["target_availabilities"])
            + torch.sum(data_batch["all_other_agents_future_availability"])
        )

    def compute_losses(self, pred_batch, data_batch):
        if self.criterion is None:
            raise NotImplementedError("Loss function is undefined.")

        ego_weights = data_batch["target_availabilities"].unsqueeze(-1)

        all_other_types = data_batch["all_other_agents_types"]
        all_other_weights = (
            data_batch["all_other_agents_future_availability"].unsqueeze(-1)
            * self.all_other_weight
        )
        type_mask = ((all_other_types >= 3) & (all_other_types <= 13)).unsqueeze(-1)

        weights = torch.cat(
            (ego_weights.unsqueeze(1), all_other_weights * type_mask.unsqueeze(-1)),
            dim=1,
        )
        eta = self.algo_config.temporal_bias
        T = pred_batch["predictions"]["yaws"].shape[-2]
        temporal_weight = (
            (1 - eta + torch.arange(T) / (T - 1) * 2 * eta)
            .view(1, 1, T, 1)
            .to(weights.device)
        )
        weights = weights * temporal_weight
        mask = torch.cat(
            (
                data_batch["target_availabilities"].unsqueeze(1),
                data_batch["all_other_agents_future_availability"] * type_mask,
            ),
            dim=1,
        )
        instance_count = torch.sum(mask)

        scene_target_pos = torch.cat(
            (
                data_batch["target_positions"].unsqueeze(1),
                data_batch["all_other_agents_future_positions"],
            ),
            dim=1,
        )
        scene_target_yaw = torch.cat(
            (
                data_batch["target_yaws"].unsqueeze(1),
                data_batch["all_other_agents_future_yaws"],
            ),
            dim=1,
        )
        loss = 0
        if self.M == 1:

            loss += (
                torch.sum(
                    self.criterion(
                        scene_target_pos, pred_batch["scene_predictions"]["positions"]
                    )
                    * weights
                    * self.weights_scaling[:2]
                )
                / instance_count
            )
            ego_yaw_error = round_2pi(
                scene_target_yaw
                - pred_batch["scene_predictions"]["yaws"] * mask.unsqueeze(-1)
            )
            loss += (
                torch.sum(ego_yaw_error ** 2 * self.weights_scaling[2:] * weights)
                / instance_count
            )
        else:
            err = (
                self.criterion(
                    scene_target_pos.unsqueeze(1).repeat(1, self.M, 1, 1, 1),
                    pred_batch["scene_predictions"]["positions"],
                )
                * weights.unsqueeze(1)
                * self.weights_scaling[:2]
                * pred_batch["scene_predictions"]["prob"][:, :, None, None, None]
            )
            max_idx = torch.max(pred_batch["scene_predictions"]["prob"], dim=-1)[1]
            max_mask = torch.zeros([*err.shape[:2], 1, 1, 1], dtype=torch.bool).to(
                err.device
            )
            max_mask[torch.arange(0, err.size(0)), max_idx] = True
            nonmax_mask = ~max_mask
            loss += (
                torch.sum((err * max_mask)) + torch.sum((err * nonmax_mask).detach())
            ) / instance_count

            yaw_err = round_2pi(
                scene_target_yaw.unsqueeze(1)
                - pred_batch["scene_predictions"]["yaws"] * mask[:, None, :, :, None]
            )
            yaw_err_loss = (
                yaw_err ** 2
                * self.weights_scaling[2:]
                * weights.unsqueeze(1)
                * pred_batch["scene_predictions"]["prob"][:, :, None, None, None]
            )
            loss += (
                torch.sum((yaw_err_loss * max_mask))
                + torch.sum((yaw_err_loss * nonmax_mask).detach())
            ) / instance_count

        reg_loss = (
            self.regularization_loss(pred_batch, data_batch)
            * self.algo_config.reg_weight
        )
        if self.M == 1:
            lane_reg_loss = (
                torch.sum(
                    mask.unsqueeze(-1)
                    * (1 - pred_batch["scene_predictions"]["lane_flags"])
                )
                / instance_count
            ) * self.algo_config.lane_regulation_weight
        else:
            lane_reg_loss = (
                torch.sum(
                    mask.unsqueeze(-1).unsqueeze(1)
                    * (1 - pred_batch["scene_predictions"]["lane_flags"])
                    * pred_batch["scene_predictions"]["prob"][:, :, None, None, None]
                )
                / instance_count
            ) * self.algo_config.lane_regulation_weight
        losses = OrderedDict(
            prediction_loss=loss,
            regularization_loss=reg_loss,
            lane_reg_loss=lane_reg_loss,
        )
        if self.algo_config.calc_collision:
            coll_loss = 0
            for et, fun in self.col_funs.items():
                edges = pred_batch["scene_predictions"]["edges"][et]
                dis = fun(
                    edges[..., 0:3],
                    edges[..., 3:6],
                    edges[..., 6:8],
                    edges[..., 8:],
                ).min(dim=-1)[0]
                coll_loss += torch.sum(torch.sigmoid(-dis - 4.0)) / instance_count
            losses["coll_loss"] = coll_loss * self.algo_config.collision_weight

        if self.algo_config.calc_likelihood:
            likelihood_loss = self.algo_config.GAN_weight * (
                1 - torch.mean(pred_batch["scene_predictions"]["likelihood_new"])
            )
            losses["likelihood_loss"] = likelihood_loss

        return losses
