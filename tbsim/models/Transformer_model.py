import numpy as np
from tbsim.dynamics.base import DynType
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tbsim.dynamics import *
import tbsim.dynamics as dyn
from typing import Dict, List
import pdb
import pytorch_lightning as pl
from collections import OrderedDict
from tbsim.utils.torch_utils import get_torch_device


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class FactorizedEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, src2posfun):
        super(FactorizedEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.src2posfun = src2posfun

    def src2pos(self, src, dyn_type):

        pos = torch.zeros([*src.shape[:-1], 2]).to(src.device)
        for dt, fun in self.src2posfun.items():
            pos += fun(src) * (dyn_type == dt).view([*(dyn_type.shape), 1, 1])

        return pos

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask,
        dyn_type,
    ):
        "Take in and process masked src and target sequences."
        src_pos = self.src2pos(src, dyn_type)
        # tgt_pos = self.tgt2pos(tgt, type_index)
        return self.decode(
            self.encode(src, src_mask, src_pos),
            src_mask,
            tgt,
            tgt_mask,
            src_pos[:, :, -1:],
        )

    def encode(self, src, src_mask, src_pos):
        return self.encoder(self.src_embed(src), src_mask, src_pos)

    def decode(self, memory, src_mask, tgt, tgt_mask, pos):
        return self.decoder(
            self.tgt_embed(tgt),
            memory,
            src_mask,
            tgt_mask,
            pos,
        )


class DynamicGenerator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, dt, dyns, state2feature, feature2state):
        super(DynamicGenerator, self).__init__()
        self.dyns = dyns
        self.proj = dict()
        self.dt = dt
        self.state2feature = state2feature
        self.feature2state = feature2state
        for dyn in self.dyns:
            self.proj[dyn.type()] = nn.Linear(d_model, dyn.udim)

    def forward(self, x, tgt, type_index):
        Nagent = tgt.shape[0]
        tgt_next = [None] * Nagent
        for dyn in self.dyns:
            index = type_index[dyn.type()]
            state = self.feature2state[dyn.name](tgt[index])
            input = self.proj[dyn.type()](x)
            state_next = dyn.step(state, input, self.dt)
            x_next_raw = self.state2feature[dyn.name](state_next)
            for i in range(len(index)):
                tgt_next[index[i]] = x_next_raw[i]
        return torch.stack(tgt_next, dim=0)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FactorizedEncoder(nn.Module):
    def __init__(self, temporal_enc, agent_enc, temporal_pe, XY_pe):
        super(FactorizedEncoder, self).__init__()
        self.temporal_enc = temporal_enc
        self.agent_enc = agent_enc
        self.temporal_pe = temporal_pe
        self.XY_pe = XY_pe

    def forward(self, x, src_mask, src_pos):
        "Pass the input (and mask) through each layer in turn."
        "x:[Num_agent,T,C]"
        "pos:[Num_agent,T,2]"

        x = self.XY_pe(x, src_pos)
        x = self.agent_enc(x, src_mask)
        x = self.temporal_pe(x)
        x = self.temporal_enc(x, src_mask)
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class FactorizedDecoder(nn.Module):
    def __init__(self, temporal_dec, agent_enc, temporal_pe, XY_pe):
        super(FactorizedDecoder, self).__init__()
        self.temporal_dec = temporal_dec
        self.agent_enc = agent_enc
        self.temporal_pe = temporal_pe
        self.XY_pe = XY_pe

    def forward(self, x, memory, src_mask, tgt_mask, pos):
        "Pass the input (and mask) through each layer in turn."
        "x:[batch,Num_agent,T,C]"
        "pos:[batch,Num_agent,1,2]"
        T = x.size(-2)
        tgt_pos = pos.repeat([1, 1, T, 1])
        x = self.XY_pe(x, tgt_pos)
        # x = self.agent_enc(x, torch.diagonal(tgt_mask, dim1=-2, dim2=-1))
        x = self.agent_enc(x, tgt_mask)
        x = self.temporal_pe(x)
        x = self.temporal_dec(x, memory, src_mask, tgt_mask)

        return x


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, pooling_dim=None):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.pooling_dim = pooling_dim
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if self.pooling_dim is None:
            pooling_dim = query.ndim - 2
        else:
            pooling_dim = self.pooling_dim
        if mask is not None:
            # Same mask applied to all h heads.
            if mask.ndim == query.ndim - 1:
                mask = mask.view([*mask.shape, 1, 1]).transpose(-1, pooling_dim)
            elif mask.ndim == query.ndim:
                mask = mask.unsqueeze(-2).transpose(-2, pooling_dim)
            else:
                raise Exception("mask dimension mismatch")
        nbatches = query.size(0)
        nagent = query.size(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, nagent, -1, self.h, self.d_k)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.

        x, self.attn = attention(
            query.transpose(-2, pooling_dim),
            key.transpose(-2, pooling_dim),
            value.transpose(-2, pooling_dim),
            mask,
            dropout=self.dropout,
        )
        x = (
            x.transpose(-2, pooling_dim)
            .contiguous()
            .view(nbatches, nagent, -1, self.h * self.d_k)
        )

        # 3) "Concat" using a view and apply a final linear.
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe_shape = [1] * (x.ndim - 2) + list(x.shape[-2:])
        x = x + Variable(self.pe[:, : x.size(-2)].view(pe_shape), requires_grad=False)

        return self.dropout(x)


class PositionalEncodingNd(nn.Module):
    def __init__(self, d_model, dropout, step_size=[1]):
        super(PositionalEncodingNd, self).__init__()
        assert d_model % 2 == 0
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.step_size = step_size
        self.D = len(step_size)
        self.pe = list()

        # Compute the positional encodings once in log space.
        self.div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

    def forward(self, x, pos):
        rep_size = [1] * (x.ndim)
        rep_size[-1] = int(self.d_model / 2)
        for i in range(self.D):
            pe = torch.zeros_like(x)

            pe[..., 0::2] = torch.sin(
                pos[..., i : i + 1].repeat(*rep_size)
                / self.step_size[i]
                * self.div_term.to(x.device)
            )
            pe[..., 1::2] = torch.sin(
                pos[..., i : i + 1].repeat(*rep_size)
                / self.step_size[i]
                * self.div_term.to(x.device)
            )
            x = x + Variable(pe, requires_grad=False)
        return self.dropout(x)


def make_factorized_model(
    src_dim,
    tgt_dim,
    dyn_list,
    N_t=6,
    N_a=3,
    d_model=512,
    d_ff=2048,
    head=8,
    dropout=0.1,
    step_size=[0.1, 0.1],
):
    c = copy.deepcopy
    temporal_attn = MultiHeadedAttention(head, d_model)
    agent_attn = MultiHeadedAttention(head, d_model, pooling_dim=1)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    temporal_pe = PositionalEncoding(d_model, dropout)
    XY_pe = PositionalEncodingNd(d_model, dropout, step_size=step_size)
    temporal_enc = Encoder(EncoderLayer(d_model, c(temporal_attn), c(ff), dropout), N_t)
    agent_enc = Encoder(EncoderLayer(d_model, c(agent_attn), c(ff), dropout), N_a)
    temporal_dec = Decoder(
        DecoderLayer(d_model, c(temporal_attn), c(temporal_attn), c(ff), dropout), N_t
    )
    src2posfun = {D.type(): D.state2pos for D in dyn_list}

    Factorized_Encoder = FactorizedEncoder(
        temporal_enc, c(agent_enc), temporal_pe, XY_pe
    )
    Factorized_Decoder = FactorizedDecoder(
        temporal_dec, c(agent_enc), temporal_pe, XY_pe
    )
    model = FactorizedEncoderDecoder(
        Factorized_Encoder,
        Factorized_Decoder,
        nn.Linear(src_dim, d_model),
        nn.Linear(tgt_dim, d_model),
        nn.Linear(d_model, tgt_dim),
        src2posfun,
    )

    return model


class Transformer_model(pl.LightningModule):
    def __init__(
        self,
        algo_config,
    ):
        super(Transformer_model, self).__init__()
        self.time_step = algo_config.time_step
        self.algo_config = algo_config

        self.register_buffer(
            "weights_scaling", torch.tensor(algo_config.weights_scaling)
        )
        self.ego_weight = algo_config.ego_weight
        self.all_other_weight = algo_config.all_other_weight
        self.criterion = nn.MSELoss()

        self.dyn_list = {
            DynType.UNICYCLE: Unicycle("vehicle"),
            DynType.DI: DoubleIntegrator(
                "pedestrian",
                abound=np.array([[-3.0, 3.0], [-3.0, 3.0]]),
                vbound=np.array([[-5.0, 5.0], [-5.0, 5.0]]),
            ),
        }

        self.model = make_factorized_model(
            src_dim=21,
            tgt_dim=3,
            dyn_list=self.dyn_list.values(),
            N_t=algo_config.N_t,
            N_a=algo_config.N_a,
            d_model=algo_config.d_model,
            d_ff=algo_config.d_ff,
            head=algo_config.head,
            dropout=algo_config.dropout,
            step_size=algo_config.XY_step_size,
        )

    @staticmethod
    def raw2feature(pos, vel, yaw, raw_type, mask):
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
        dyn_type += DynType.UNICYCLE * veh_mask
        # all vehicles, cyclists, and motorcyclists
        feature_veh = torch.cat((pos, vel, torch.sin(yaw), torch.cos(yaw)), dim=-1)

        ped_mask = (raw_type == 14) | (raw_type == 15)
        # pedestrians and animals
        ped_feature = torch.cat(
            (pos, vel, vel * torch.sin(yaw), vel * torch.cos(yaw)), dim=-1
        )
        dyn_type += DynType.DI * ped_mask

        feature = feature_veh * veh_mask.view(
            [*raw_type.shape, 1, 1]
        ) + ped_feature * ped_mask.view([*raw_type.shape, 1, 1])
        try:
            type_embedding = F.one_hot(raw_type, 16)
        except:
            pdb.set_trace()

        feature = torch.cat(
            (feature, type_embedding.unsqueeze(-2).repeat(1, 1, feature.size(2), 1)),
            dim=-1,
        )
        feature = feature * mask.unsqueeze(-1)
        return feature, dyn_type

    @staticmethod
    def tgt_temporal_mask(p, tgt_mask):
        nbatches = tgt_mask.size(0)
        T = tgt_mask.size(2)
        sample = np.random.binomial(T, p, nbatches)
        for i in range(nbatches):
            tgt_mask[i, :, sample[i] :] = 0
        return tgt_mask

    def forward(
        self, data_batch: Dict[str, torch.Tensor], tgt_mask_p: float = 0.0
    ) -> Dict[str, torch.Tensor]:
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
        src_pos = torch.flip(src_pos, dims=[-2])
        src_yaw = torch.cat(
            (
                data_batch["history_yaws"].unsqueeze(1),
                data_batch["all_other_agents_history_yaws"],
            ),
            dim=1,
        )
        src_yaw = torch.flip(src_yaw, dims=[-2])
        src_yaw += (
            data_batch["yaw"]
            .view(-1, 1, 1, 1)
            .repeat(1, src_yaw.size(1), src_yaw.size(2), 1)
        )
        src_mask = torch.cat(
            (
                data_batch["history_availabilities"].unsqueeze(1),
                data_batch["all_other_agents_history_availability"],
            ),
            dim=1,
        ).bool()
        src_mask = torch.flip(src_mask, dims=[-1])
        src_vel = self.dyn_list[DynType.UNICYCLE].calculate_vel(
            src_pos, src_yaw, self.time_step, src_mask
        )
        src, dyn_type = self.raw2feature(src_pos, src_vel, src_yaw, raw_type, src_mask)

        tgt_mask = torch.cat(
            (
                data_batch["target_availabilities"].unsqueeze(1),
                data_batch["all_other_agents_future_availability"],
            ),
            dim=1,
        ).bool()
        # masking part of the target and gradually increase the masked length until the whole target is masked

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
        tgt_yaw += (
            data_batch["yaw"]
            .view(-1, 1, 1, 1)
            .repeat(1, tgt_yaw.size(1), tgt_yaw.size(2), 1)
        ) * tgt_mask.unsqueeze(-1)
        # tgt_vel = self.dyn_list[DynType.UNICYCLE].calculate_vel(
        #     tgt_pos, tgt_yaw, self.time_step, tgt_mask
        # )
        # tgt, _ = self.raw2feature(tgt_pos, tgt_vel, tgt_yaw, raw_type, tgt_mask)
        tgt = torch.cat((tgt_pos, tgt_yaw), dim=-1)
        curr_state = torch.cat((src_pos[..., -1:, :], src_yaw[..., -1:, :]), dim=-1)
        # tgt_y = tgt[...,1:,:]
        tgt = torch.cat((curr_state, tgt[..., :-1, :]), dim=2)
        tgt_mask_agent = (
            tgt_mask.any(dim=-1).unsqueeze(-1).repeat(1, 1, tgt_mask.size(-1))
        )

        tgt_mask = self.tgt_temporal_mask(tgt_mask_p, tgt_mask.clone())
        tgt_mask = torch.cat((src_mask[..., -1:], tgt_mask[..., :-1]), dim=-1)
        tgt = tgt * tgt_mask.unsqueeze(-1)

        # seq_mask = subsequent_mask(tgt_mask.size(-1)).to(self.device)
        # tgt_mask = tgt_mask.unsqueeze(-1).repeat(
        #     1, 1, 1, tgt.size(-2)
        # ) * seq_mask.unsqueeze(0)

        out = self.model.forward(
            src.to(self.device),
            tgt.to(self.device),
            src_mask.to(self.device),
            tgt_mask_agent.to(self.device),
            dyn_type.to(self.device),
        )
        tgt_y_gen = self.model.generator(out)
        ego_pred_positions = tgt_y_gen[:, 0, ..., 0:2]
        ego_pred_yaws = tgt_y_gen[:, 0, ..., 2:]
        all_other_pred_positions = tgt_y_gen[:, 1:, ..., 0:2]
        all_other_pred_yaws = tgt_y_gen[:, 1:, ..., 2:]
        out_dict = {
            "raw_outputs": tgt_y_gen,
            "predictions": {
                "positions": ego_pred_positions,
                "yaws": ego_pred_yaws,
                "all_other_positions": all_other_pred_positions,
                "all_other_yaws": all_other_pred_yaws,
            },
        }
        return out_dict

    def compute_losses(self, pred_batch, data_batch):
        if self.criterion is None:
            raise NotImplementedError("Loss function is undefined.")

        batch_size = data_batch["target_positions"].shape[0]
        # [batch_size, num_steps * 2]
        ego_targets = (
            torch.cat(
                (data_batch["target_positions"], data_batch["target_yaws"]), dim=-1
            )
        ).view(batch_size, -1) * self.ego_weight
        # [batch_size, num_steps]
        ego_weights = (
            data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling
        ).view(batch_size, -1)
        ego_pred = (
            torch.cat(
                (
                    pred_batch["predictions"]["positions"],
                    pred_batch["predictions"]["yaws"],
                ),
                dim=-1,
            )
        ).view(batch_size, -1)
        all_other_targets = (
            torch.cat(
                (
                    data_batch["all_other_agents_future_positions"],
                    data_batch["all_other_agents_future_yaws"],
                ),
                dim=-1,
            )
        ).view(batch_size, -1)
        all_other_pred = (
            torch.cat(
                (
                    pred_batch["predictions"]["all_other_positions"],
                    pred_batch["predictions"]["all_other_yaws"],
                ),
                dim=-1,
            )
        ).view(batch_size, -1)
        all_other_weights = (
            data_batch["all_other_agents_future_availability"].unsqueeze(-1)
            * self.weights_scaling
        ).view(batch_size, -1) * self.all_other_weight
        loss = torch.mean(
            self.criterion(ego_pred, ego_targets) * ego_weights
        ) + torch.mean(
            self.criterion(all_other_pred, all_other_targets) * all_other_weights
        )
        losses = OrderedDict(prediction_loss=loss)
        return losses
