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
from tbsim.utils.geometry_utils import batch_nd_transform_points
from tbsim.models.cnn_roi_encoder import CNNROIMapEncoder
from tbsim.utils.tensor_utils import round_2pi
from tbsim.utils.geometry_utils import (
    VEH_VEH_collision,
    VEH_PED_collision,
    PED_VEH_collision,
    PED_PED_collision,
)


def clones(module, n):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class FactorizedEncoderDecoder(nn.Module):
    """
    A encoder-decoder transformer model with Factorized encoder and decoder
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, src2posfun):
        """
        Args:
            encoder: FactorizedEncoder
            decoder: FactorizedDecoder
            src_embed: source embedding network
            tgt_embed: target embedding network
            generator: network used to generate output from target
            src2posfun: extract positional info from the src
        """
        super(FactorizedEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.src2posfun = src2posfun

    def src2pos(self, src, dyn_type):
        "extract positional info from src for all datatypes, e.g., for vehicles, the first two dimensions are x and y"

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
        tgt_mask_agent,
        dyn_type,
        map_emb=None,
    ):
        "Take in and process masked src and target sequences."
        src_pos = self.src2pos(src, dyn_type)
        # tgt_pos = self.tgt2pos(tgt, type_index)
        "for decoders, we only use position at the last time step of the src"
        return self.decode(
            self.encode(src, src_mask, src_pos, map_emb),
            src_mask,
            tgt,
            tgt_mask,
            tgt_mask_agent,
            src_pos[:, :, -1:],
        )

    def encode(self, src, src_mask, src_pos, map_emb):
        return self.encoder(self.src_embed(src), src_mask, src_pos, map_emb)

    def decode(self, memory, src_mask, tgt, tgt_mask, tgt_mask_agent, pos):

        return self.decoder(
            self.tgt_embed(tgt),
            memory,
            src_mask,
            tgt_mask,
            tgt_mask_agent,
            pos,
        )


class DynamicGenerator(nn.Module):
    "Incorporating dynamics to the generator to generate dynamically feasible output, not used yet"

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

    def forward(self, x, mask, mask1=None):
        "Pass the input (and mask) through each layer in turn."
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x, mask)
            else:
                if mask1 is None:
                    x = layer(x, mask)
                else:
                    x = layer(x, mask1)
        return self.norm(x)


class FactorizedEncoder(nn.Module):
    def __init__(self, temporal_enc, agent_enc, temporal_pe, XY_pe, N_layer=1):
        """
        Factorized encoder and agent axis
        Args:
            temporal_enc: encoder with attention over temporal axis
            agent_enc: encoder with attention over agent axis
            temporal_pe: positional encoding over time
            XY_pe: positional encoding over XY coordinates
        """
        super(FactorizedEncoder, self).__init__()
        self.N_layer = N_layer
        self.temporal_encs = clones(temporal_enc, N_layer)
        self.agent_encs = clones(agent_enc, N_layer)
        self.temporal_pe = temporal_pe
        self.XY_pe = XY_pe

    def forward(self, x, src_mask, src_pos, map_emb):
        """Pass the input (and mask) through each layer in turn.
        Args:
            x:[B,Num_agent,T,d_model]
            src_mask:[B,Num_agent,T]
            src_pos:[B,Num_agent,T,2]
            map_emb: [B,Num_agent,1,map_emb_dim] output of the CNN ROI map encoder
        Returns:
            embedding of size [B,Num_agent,T,d_model]
        """

        # x += (self.XY_pe(x, src_pos) + self.temporal_pe(x)) * src_mask.unsqueeze(-1)
        # if map_emb is not None:
        #     x += map_emb.unsqueeze(2)
        if map_emb.ndim == 3:
            map_emb = map_emb.unsqueeze(2).repeat(1, 1, x.size(2), 1)
        x = (
            torch.cat(
                (
                    x,
                    self.XY_pe(x, src_pos),
                    self.temporal_pe(x).repeat(x.size(0), x.size(1), 1, 1),
                    map_emb,
                ),
                dim=-1,
            )
            * src_mask.unsqueeze(-1)
        )
        for i in range(self.N_layer):
            x = self.agent_encs[i](x, src_mask)
            x = self.temporal_encs[i](x, src_mask)
        return x


class StaticEncoder(nn.Module):
    def __init__(self, agent_enc, XY_pe, N_layer=1):
        """
        Factorized encoder and agent axis
        Args:
            temporal_enc: encoder with attention over temporal axis
            agent_enc: encoder with attention over agent axis
            temporal_pe: positional encoding over time
            XY_pe: positional encoding over XY coordinates
        """
        super(StaticEncoder, self).__init__()
        self.N_layer = N_layer
        self.agent_encs = clones(agent_enc, N_layer)
        self.XY_pe = XY_pe

    def forward(self, x, src_mask, src_pos, map_emb):
        """Pass the input (and mask) through each layer in turn.
        Args:
            x:[B,Num_agent,T,d_model]
            src_mask:[B,Num_agent,T]
            src_pos:[B,Num_agent,T,2]
            map_emb: [B,Num_agent,1,map_emb_dim] output of the CNN ROI map encoder
        Returns:
            embedding of size [B,Num_agent,T,d_model]
        """

        x = (
            torch.cat(
                (
                    x,
                    self.XY_pe(x, src_pos),
                    map_emb,
                ),
                dim=-1,
            )
            * src_mask.unsqueeze(-1)
        )
        for i in range(self.N_layer):
            x = self.agent_encs[i](x, src_mask)
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module"

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
    "Encoder is made up of self-attn and feed forward"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "self attention followed by feedforward, residual and batch norm in between layers"
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        "cross attention to the embedding generated by the encoder"
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class SummaryModel(nn.Module):
    """
    map the scene information to attributes that summarizes the scene
    """

    def __init__(self, encoder, decoder, src_embed, src2posfun):
        super(SummaryModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.src2posfun = src2posfun

    def src2pos(self, src, dyn_type):
        "extract positional info from src for all datatypes, e.g., for vehicles, the first two dimensions are x and y"

        pos = torch.zeros([*src.shape[:-1], 2]).to(src.device)
        for dt, fun in self.src2posfun.items():
            pos += fun(src) * (dyn_type == dt).view([*(dyn_type.shape), 1, 1])

        return pos

    def forward(
        self,
        src,
        src_mask,
        dyn_type,
        map_emb,
    ):
        "Take in and process masked src and target sequences."
        src_pos = self.src2pos(src, dyn_type)
        return self.decode(
            self.encode(src, src_mask, src_pos, map_emb),
            src_mask,
        )

    def encode(self, src, src_mask, src_pos, map_emb):
        return self.encoder(self.src_embed(src), src_mask, src_pos, map_emb)

    def decode(self, memory, src_mask):
        return self.decoder(memory, src_mask)


class SummaryDecoder(nn.Module):
    """
    Map the encoded tensor to a description of the whole scene, e.g., the likelihood of certain modes
    """

    def __init__(
        self, temporal_attn, agent_attn, ff, emb_dim, output_dim, static=False
    ):
        super(SummaryDecoder, self).__init__()
        self.temporal_attn = temporal_attn
        self.agent_attn = agent_attn
        self.ff = ff
        self.output_dim = output_dim
        self.static = static
        self.MLP = nn.Sequential(nn.Linear(emb_dim, output_dim), nn.Sigmoid())

    def forward(self, x, mask):
        x = self.agent_attn(x, x, x, mask)
        x = self.ff(torch.max(x, dim=-3)[0]).unsqueeze(1)
        if not self.static:
            x = self.temporal_attn(x, x, x)
            x = torch.max(x, dim=-2)[0].squeeze(1)
        x = self.MLP(x)
        return x


class FactorizedDecoder(nn.Module):
    """
    Args:
        temporal_dec: decoder with attention over temporal axis
        agent_enc: decoder with attention over agent axis
        temporal_pe: positional encoding for time axis
        XY_pe: positional encoding for XY axis
    """

    def __init__(
        self,
        temporal_dec,
        agent_enc,
        temporal_enc,
        temporal_pe,
        XY_pe,
        N_layer_enc=1,
        N_layer_dec=1,
    ):
        super(FactorizedDecoder, self).__init__()
        self.temporal_dec = clones(temporal_dec, N_layer_dec)
        self.agent_enc = clones(agent_enc, N_layer_enc)
        self.temporal_enc = clones(temporal_enc, N_layer_enc)
        self.N_layer_enc = N_layer_enc
        self.N_layer_dec = N_layer_dec
        self.temporal_pe = temporal_pe
        self.XY_pe = XY_pe

    def forward(self, x, memory, src_mask, tgt_mask, tgt_mask_agent, pos):
        """
        Pass the input (and mask) through each layer in turn.
        Args:
            x (torch.tensor)): [batch,Num_agent,T_tgt,d_model]
            memory (torch.tensor): [batch,Num_agent,T_src,d_model]
            src_mask (torch.tensor): [batch,Num_agent,T_src]
            tgt_mask (torch.tensor): [batch,Num_agent,T_tgt]
            tgt_mask_agent (torch.tensor): [batch,Num_agent,T_tgt]
            pos (torch.tensor): [batch,Num_agent,1,2]

        Returns:
            torch.tensor: [batch,Num_agent,T_tgt,d_model]
        """
        T = x.size(-2)
        tgt_pos = pos.repeat([1, 1, T, 1])

        x = (
            torch.cat(
                (
                    x,
                    self.XY_pe(x, tgt_pos),
                    self.temporal_pe(x).repeat(x.size(0), x.size(1), 1, 1),
                ),
                dim=-1,
            )
            * tgt_mask_agent.unsqueeze(-1)
        )
        for i in range(self.N_layer_enc):
            x = self.agent_enc[i](x, tgt_mask_agent)
            x = self.temporal_enc[i](x, tgt_mask)
        for i in range(self.N_layer_dec):
            x = self.temporal_dec[i](x, memory, src_mask, tgt_mask)
        prob = torch.ones(x.shape[0]).to(x.device)
        return x * tgt_mask_agent.unsqueeze(-1), prob


class MultimodalFactorizedDecoder(nn.Module):
    """
    Args:
        temporal_dec: decoder with attention over temporal axis
        agent_enc: decoder with attention over agent axis
        temporal_pe: positional encoding for time axis
        XY_pe: positional encoding for XY axis
    """

    def __init__(
        self,
        temporal_dec,
        agent_enc,
        temporal_enc,
        temporal_pe,
        XY_pe,
        M,
        summary_dec,
        N_layer_enc=1,
        N_layer_dec=1,
    ):
        super(MultimodalFactorizedDecoder, self).__init__()
        self.M = M
        self.temporal_dec = clones(temporal_dec, N_layer_dec)
        self.agent_enc = clones(agent_enc, N_layer_enc)
        self.temporal_enc = clones(temporal_enc, N_layer_enc)
        self.N_layer_enc = N_layer_enc
        self.N_layer_dec = N_layer_dec
        self.temporal_pe = temporal_pe
        self.XY_pe = XY_pe
        self.summary_dec = summary_dec

    def forward(self, x, memory, src_mask, tgt_mask, tgt_mask_agent, pos):
        """
        Pass the input (and mask) through each layer in turn.
        Args:
            x (torch.tensor)): [batch,Num_agent,T_tgt,d_model]
            memory (torch.tensor): [batch,Num_agent,T_src,d_model]
            src_mask (torch.tensor): [batch,Num_agent,T_src]
            tgt_mask (torch.tensor): [batch,Num_agent,T_tgt]
            tgt_mask_agent (torch.tensor): [batch,Num_agent,T_tgt]
            pos (torch.tensor): [batch,Num_agent,1,2]

        Returns:
            torch.tensor: [batch,Num_agent,T_tgt,d_model]
        """
        T = x.size(-2)
        tgt_pos = pos.repeat([1, 1, T, 1])

        x = (
            torch.cat(
                (
                    x,
                    self.XY_pe(x, tgt_pos),
                    self.temporal_pe(x).repeat(x.size(0), x.size(1), 1, 1),
                ),
                dim=-1,
            )
            * tgt_mask_agent.unsqueeze(-1)
        )

        # adding one-hot encoding of the modes
        modes_enc = (
            F.one_hot(torch.arange(0, self.M))
            .view(1, self.M, 1, 1, self.M)
            .repeat(x.size(0), 1, x.size(1), x.size(2), 1)
        ).to(x.device)

        x = torch.cat((x.unsqueeze(1).repeat(1, self.M, 1, 1, 1), modes_enc), dim=-1)

        memory_M = memory.unsqueeze(1).repeat(1, self.M, 1, 1, 1)
        src_mask_M = src_mask.unsqueeze(1).repeat(1, self.M, 1, 1)
        tgt_mask_M = tgt_mask.unsqueeze(1).repeat(1, self.M, 1, 1, 1)
        tgt_mask_agent_M = tgt_mask_agent.unsqueeze(1).repeat(1, self.M, 1, 1)
        for i in range(self.N_layer_enc):
            x = self.agent_enc[i](x, tgt_mask_agent_M)
            x = self.temporal_enc[i](x, tgt_mask_M)
        for i in range(self.N_layer_dec):
            x = self.temporal_dec[i](
                x,
                memory_M,
                src_mask_M,
                tgt_mask_M,
            )

        prob = self.summary_dec(x, tgt_mask_agent_M).squeeze(-1)
        prob = F.softmax(prob, dim=-1)
        return x * tgt_mask_agent_M.unsqueeze(-1), prob


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
        "self attention followed by cross attention with the encoder output"

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
            pooling_dim = -2
        else:
            pooling_dim = self.pooling_dim
        if mask is not None:
            # Same mask applied to all h heads.
            if mask.ndim == query.ndim - 1:
                mask = mask.view([*mask.shape, 1, 1]).transpose(-1, pooling_dim - 1)
            elif mask.ndim == query.ndim:
                mask = mask.unsqueeze(-2).transpose(-2, pooling_dim - 1)
            else:
                raise Exception("mask dimension mismatch")

        # 1) Do all the linear projections in batch from d_model => h x d_k

        query, key, value = [
            l(x).view(*x.shape[:-1], self.h, self.d_k)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query.transpose(-2, pooling_dim - 1),
            key.transpose(-2, pooling_dim - 1),
            value.transpose(-2, pooling_dim - 1),
            mask,
            dropout=self.dropout,
        )

        x = x.transpose(-2, pooling_dim - 1).contiguous()
        x = x.view(*x.shape[:-2], self.h * self.d_k)

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

    def __init__(self, dim, dropout, max_len=5000, flipped=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        self.flipped = flipped

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        if self.flipped:
            position = -position.flip(dims=[0])
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe_shape = [1] * (x.ndim - 2) + list(x.shape[-2:-1]) + [self.dim]
        if self.flipped:
            return self.dropout(
                Variable(self.pe[:, -x.size(-2) :].view(pe_shape), requires_grad=False)
            )
        else:
            return self.dropout(
                Variable(self.pe[:, : x.size(-2)].view(pe_shape), requires_grad=False)
            )


class PositionalEncodingNd(nn.Module):
    "extension of the PE function, works for N dimensional position input"

    def __init__(self, dim, dropout, step_size=[1]):
        """
        step_size: scale of each dimension, pos/step_size = phase for the sinusoidal PE
        """
        super(PositionalEncodingNd, self).__init__()
        assert dim % 2 == 0
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        self.step_size = step_size
        self.D = len(step_size)
        self.pe = list()

        # Compute the positional encodings once in log space.
        self.div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))

    def forward(self, x, pos):
        rep_size = [1] * (x.ndim)
        rep_size[-1] = int(self.dim / 2)
        pe_shape = [*x.shape[:-1], self.dim]
        for i in range(self.D):
            pe = torch.zeros(pe_shape).to(x.device)

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
        return self.dropout(Variable(pe, requires_grad=False))


def make_factorized_model(
    src_dim,
    tgt_dim,
    out_dim,
    dyn_list,
    N_t=6,
    N_a=3,
    d_model=384,
    XY_pe_dim=64,
    temporal_pe_dim=64,
    map_emb_dim=128,
    d_ff=2048,
    head=8,
    dropout=0.1,
    step_size=[0.1, 0.1],
    N_layer_enc=1,
    N_layer_tgt_enc=1,
    N_layer_tgt_dec=1,
    M=1,
    use_GAN=False,
    GAN_static=True,
    N_layer_enc_discr=1,
):
    "first generate the building blocks, attn networks, encoders, decoders, PEs and Feedforward nets"
    c = copy.deepcopy
    temporal_attn = MultiHeadedAttention(head, d_model)
    agent_attn = MultiHeadedAttention(head, d_model, pooling_dim=-3)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    temporal_pe = PositionalEncoding(temporal_pe_dim, dropout)
    temporal_pe_flip = PositionalEncoding(temporal_pe_dim, dropout, flipped=True)
    XY_pe = PositionalEncodingNd(XY_pe_dim, dropout, step_size=step_size)
    temporal_enc = Encoder(EncoderLayer(d_model, c(temporal_attn), c(ff), dropout), N_t)
    agent_enc = Encoder(EncoderLayer(d_model, c(agent_attn), c(ff), dropout), N_a)

    src_emb = nn.Linear(src_dim, d_model - XY_pe_dim - temporal_pe_dim - map_emb_dim)
    if M == 1:
        tgt_emb = nn.Linear(tgt_dim, d_model - XY_pe_dim - temporal_pe_dim)
    else:
        tgt_emb = nn.Linear(tgt_dim, d_model - XY_pe_dim - temporal_pe_dim - M)
    generator = nn.Linear(d_model, out_dim)

    temporal_dec = Decoder(
        DecoderLayer(d_model, c(temporal_attn), c(temporal_attn), c(ff), dropout), N_t
    )
    "gather src2posfun from all agent types"
    src2posfun = {D.type(): D.state2pos for D in dyn_list}

    Factorized_Encoder = FactorizedEncoder(
        c(temporal_enc), c(agent_enc), temporal_pe_flip, XY_pe, N_layer_enc
    )
    if M == 1:
        Factorized_Decoder = FactorizedDecoder(
            c(temporal_dec),
            c(agent_enc),
            c(temporal_enc),
            temporal_pe,
            XY_pe,
            N_layer_tgt_enc,
            N_layer_tgt_dec,
        )
    else:
        mode_summary_dec = SummaryDecoder(
            c(temporal_attn), c(agent_attn), c(ff), d_model, 1
        )
        Factorized_Decoder = MultimodalFactorizedDecoder(
            temporal_dec,
            agent_enc,
            temporal_enc,
            temporal_pe,
            XY_pe,
            M,
            mode_summary_dec,
            N_layer_enc=1,
            N_layer_dec=1,
        )
    Factorized_Encoder = FactorizedEncoder(
        c(temporal_enc), c(agent_enc), temporal_pe_flip, XY_pe, N_layer_enc
    )
    if use_GAN:
        if GAN_static:
            Summary_Encoder = StaticEncoder(
                c(agent_enc),
                XY_pe,
                N_layer_enc_discr,
            )
            Summary_Decoder = SummaryDecoder(
                c(temporal_attn), c(agent_attn), c(ff), d_model, 1, static=True
            )
            static_src_emb = nn.Linear(src_dim, d_model - XY_pe_dim - map_emb_dim)
            Summary_Model = SummaryModel(
                Summary_Encoder,
                Summary_Decoder,
                c(static_src_emb),
                src2posfun,
            )
        else:
            Summary_Encoder = Summary_Encoder = FactorizedEncoder(
                c(temporal_enc),
                c(agent_enc),
                temporal_pe_flip,
                XY_pe,
                N_layer_enc_discr,
            )
            Summary_Decoder = SummaryDecoder(
                c(temporal_attn), c(agent_attn), c(ff), d_model, 1, static=True
            )
            Summary_Model = SummaryModel(
                Summary_Encoder,
                Summary_Decoder,
                c(src_emb),
                src2posfun,
            )

    else:
        Summary_Model = None
    "use a simple nn.Linear as the generator as our output is continuous"

    Transformer_Model = FactorizedEncoderDecoder(
        Factorized_Encoder,
        Factorized_Decoder,
        c(src_emb),
        c(tgt_emb),
        c(generator),
        src2posfun,
    )

    return Transformer_Model, Summary_Model


class simplelinear(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[64, 32]):
        super(simplelinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = len(hidden_dim)
        self.fhidden = nn.ModuleList()

        for i in range(1, self.hidden_layers):
            self.fhidden.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))

        self.f1 = nn.Linear(input_dim, hidden_dim[0])
        self.f2 = nn.Linear(hidden_dim[-1], output_dim)

    def forward(self, x):
        hidden = self.f1(x)
        for i in range(1, self.hidden_layers):
            hidden = self.fhidden[i - 1](F.relu(hidden))
        return self.f2(F.relu(hidden))


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
        self.Transformermodel, self.Discriminator = make_factorized_model(
            src_dim=21,
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
        self.MLP = simplelinear(
            (algo_config.history_num_frames + 1) * algo_config.d_model,
            algo_config.future_num_frames * algo_config.d_model,
            hidden_dim=[algo_config.d_model * algo_config.future_num_frames] * 3,
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
    def raw2feature(pos, vel, yaw, raw_type, mask):
        "map raw src into features of dim 21"

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
        feature_veh = torch.cat((pos, vel, torch.cos(yaw), torch.sin(yaw)), dim=-1)
        state_veh = torch.cat((pos, vel, yaw), dim=-1)

        # pedestrians and animals
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
            feature = torch.cat(
                (
                    feature,
                    type_embedding.unsqueeze(-2).repeat(1, 1, feature.size(2), 1),
                ),
                dim=-1,
            )
        elif pos.ndim == 5:
            feature = torch.cat(
                (
                    feature,
                    type_embedding.unsqueeze(-2).repeat(1, 1, 1, feature.size(-2), 1),
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
                # + self.dyn_list[DynType.DI].step(x[t], action[..., t, :], self.step_time)
                # * ped_mask
            )

        x = torch.stack(x[1:], dim=-2)
        pos = self.dyn_list[DynType.UNICYCLE].state2pos(x) * veh_mask.unsqueeze(
            -1
        )  # + self.dyn_list[DynType.DI].state2pos(x) * ped_mask.unsqueeze(-1)
        yaw = self.dyn_list[DynType.UNICYCLE].state2yaw(x) * veh_mask.unsqueeze(
            -1
        )  # + self.dyn_list[DynType.DI].state2yaw(x) * ped_mask.unsqueeze(-1)

        return x, pos, yaw

    def generate_ROIs(
        self,
        pos,
        yaw,
        centroid,
        raster_from_world,
        mask,
        patch_size,
        mode="last",
    ):
        """
        This version generates ROI for all agents only at most recent time step unless specified otherwise
        """
        if mode == "all":
            bs = pos.shape[0]
            s = torch.sin(yaw).unsqueeze(-1)
            c = torch.cos(yaw).unsqueeze(-1)
            rotM = torch.cat(
                (torch.cat((c, -s), dim=-1), torch.cat((s, c), dim=-1)), dim=-2
            )
            world_xy = ((pos.unsqueeze(-2)) @ (rotM.transpose(-1, -2))).squeeze(-2)
            world_xy += centroid.view(-1, 1, 1, 2).type(torch.float)
            Mat = raster_from_world.view(-1, 1, 1, 3, 3).type(torch.float)
            raster_xy = batch_nd_transform_points(world_xy, Mat)
            ROI = [None] * bs
            index = [None] * bs
            for i in range(bs):
                ii, jj = torch.where(mask[i])
                index[i] = (ii, jj)
                ROI[i] = torch.cat(
                    (
                        raster_xy[i, ii, jj],
                        patch_size.repeat(ii.shape[0], 1),
                        yaw[i, ii, jj],
                    ),
                    dim=-1,
                ).to(pos.device)
            return ROI, index
        elif mode == "last":
            num = torch.arange(0, mask.shape[2]).view(1, 1, -1).to(mask.device)
            nummask = num * mask
            last_idx, _ = torch.max(nummask, dim=2)

            bs = pos.shape[0]
            s = torch.sin(yaw).unsqueeze(-1)
            c = torch.cos(yaw).unsqueeze(-1)
            rotM = torch.cat(
                (torch.cat((c, -s), dim=-1), torch.cat((s, c), dim=-1)), dim=-2
            )
            world_xy = ((pos.unsqueeze(-2)) @ (rotM.transpose(-1, -2))).squeeze(-2)
            world_xy += centroid.view(-1, 1, 1, 2).type(torch.float)
            Mat = raster_from_world.view(-1, 1, 1, 3, 3).type(torch.float)
            raster_xy = batch_nd_transform_points(world_xy, Mat)
            agent_mask = mask.any(dim=2)
            ROI = [None] * bs
            index = [None] * bs
            for i in range(bs):
                ii = torch.where(agent_mask[i])[0]
                index[i] = ii
                ROI[i] = torch.cat(
                    (
                        raster_xy[i, ii, last_idx[i, ii]],
                        patch_size.repeat(ii.shape[0], 1),
                        yaw[i, ii, last_idx[i, ii]],
                    ),
                    dim=-1,
                )
            return ROI, index
        else:
            raise ValueError("mode must be 'all' or 'last'")

    @staticmethod
    def Map2Emb(CNN_out, index, emb_size):
        """put the lists of ROI align result into embedding tensor with the help of index"""
        bs = len(CNN_out)
        map_emb = torch.zeros(emb_size).to(CNN_out[0].device)
        if map_emb.ndim == 3:
            for i in range(bs):
                map_emb[i, index[i]] = CNN_out[i]
        elif map_emb.ndim == 4:
            for i in range(bs):
                ii, jj = index[i]
                map_emb[i, ii, jj] = CNN_out[i]
        else:
            raise ValueError("wrong dimension for the map embedding!")

        return map_emb

    def forward(
        self, data_batch: Dict[str, torch.Tensor], batch_idx: int = None
    ) -> Dict[str, torch.Tensor]:
        # if batch_idx is not None:
        #     self.training_num += 1
        #     tgt_mask_p = 1 - min(1.0, float(self.training_num / self.training_num_N))
        # else:
        #     tgt_mask_p = 0.0

        tgt_mask_p = 0.0

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

        src, dyn_type, src_state = self.raw2feature(
            src_pos,
            src_vel,
            src_yaw,
            raw_type,
            src_mask,
        )

        # generate ROI based on the rasterized position

        ROI, index = self.generate_ROIs(
            src_pos,
            src_world_yaw,
            data_batch["centroid"],
            data_batch["raster_from_world"],
            src_mask,
            torch.tensor(self.algo_config.CNN.patch_size).to(device),
            mode=self.map_enc_mode,
        )
        CNN_out = self.CNNmodel(data_batch["image"].permute(0, 3, 1, 2), ROI)
        if self.map_enc_mode == "all":
            emb_size = (*src.shape[:-1], self.algo_config.CNN.output_size)
        elif self.map_enc_mode == "last":
            emb_size = (*src.shape[:-2], self.algo_config.CNN.output_size)

        # put the CNN output in the right location of the embedding
        map_emb = self.Map2Emb(CNN_out, index, emb_size)
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
        tgt_mask_hint = self.tgt_temporal_mask(tgt_mask_p, tgt_mask)

        tgt = tgt - curr_pos_yaw.repeat(1, 1, tgt.size(2), 1) * tgt_mask.unsqueeze(-1)

        tgt_hint = tgt * tgt_mask_hint.unsqueeze(-1)

        tgt_mask_agent = (
            tgt_mask.any(dim=-1).unsqueeze(-1).repeat(1, 1, tgt_mask.size(-1))
        )

        seq_mask = subsequent_mask(tgt_mask.size(-1)).to(tgt.device)
        tgt_mask = tgt_mask.unsqueeze(-1).repeat(
            1, 1, 1, tgt.size(-2)
        ) * seq_mask.unsqueeze(0)

        out, prob = self.Transformermodel.forward(
            src,
            tgt_hint,
            src_mask,
            tgt_mask,
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
            },
            "curr_pos_yaw": curr_pos_yaw,
        }

        # if "all_other_agents_track_id" in data_batch.keys():
        #     out_dict["predictions"]["all_other_agents_track_id"] = data_batch[
        #         "all_other_agents_track_id"
        #     ]
        if self.algo_config.calc_collision:
            out_dict["scene_predictions"]["edges"] = self.generate_edges(
                raw_type, extents, pos_pred, yaw_pred
            )

        if self.calc_likelihood:
            if self.GAN_static:
                if map_emb.ndim == 4:
                    likelihood = self.Discriminator(
                        src[:, :, -1:],
                        src_mask[:, :, -1:],
                        dyn_type,
                        map_emb[:, :, -1:],
                    ).view(src.shape[0], -1)
                else:
                    likelihood = self.Discriminator(
                        src[:, :, -1:],
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
                likelihood_new = self.Discriminator(
                    src_new, src_mask_new, dyn_type, map_emb_new
                ).view(src.shape[0], -1)
            else:
                likelihood_new = list()
                for i in range(self.M):
                    likelihood_new.append(
                        self.Discriminator(
                            src_new[:, i],
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
        f_steps=1,
    ):
        """
        generate observation f_steps later by concatenating the predictions
        Args:
            f_steps (int, optional): number of forwarding steps. Defaults to 1.
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
            ROI, index = self.generate_ROIs(
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
            map_emb_new = self.Map2Emb(CNN_out, index, emb_size)
        else:

            emb_size = (*src_new.shape[:-2], self.algo_config.CNN.output_size)
            map_emb_new = list()
            for i in range(self.M):
                ROI, index = self.generate_ROIs(
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
                map_emb_new.append(self.Map2Emb(CNN_out, index, emb_size))
            map_emb_new = torch.stack(map_emb_new, dim=1)
        return src_new, src_mask_new, map_emb_new

    def pred2obs_static(
        self,
        data_batch,
        pos_pred,
        yaw_pred,
        pred_mask,
        raw_type,
    ):
        """
        generate observation f_steps later by concatenating the predictions
        Args:
            f_steps (int, optional): number of forwarding steps. Defaults to 1.
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
            ROI, index = self.generate_ROIs(
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

            map_emb_new = self.Map2Emb(CNN_out, index, emb_size)
        else:
            emb_size = (*src_new[:, 0].shape[:-1], self.algo_config.CNN.output_size)
            map_emb_new = list()
            for i in range(self.M):
                ROI, index = self.generate_ROIs(
                    pos_pred[:, i],
                    new_world_yaw[:, i],
                    data_batch["centroid"],
                    data_batch["raster_from_world"],
                    pred_mask[:, i],
                    torch.tensor(self.algo_config.CNN.patch_size).to(pos_pred.device),
                    mode="all",
                )
                CNN_out = self.CNNmodel(data_batch["image"].permute(0, 3, 1, 2), ROI)
                map_emb_new.append(self.Map2Emb(CNN_out, index, emb_size))
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

        # [batch_size, num_steps]
        ego_weights = data_batch["target_availabilities"].unsqueeze(-1)

        all_other_weights = (
            data_batch["all_other_agents_future_availability"].unsqueeze(-1)
            * self.all_other_weight
        )
        weights = torch.cat((ego_weights.unsqueeze(1), all_other_weights), dim=1)
        mask = torch.cat(
            (
                data_batch["target_availabilities"].unsqueeze(1),
                data_batch["all_other_agents_future_availability"],
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

        losses = OrderedDict(
            prediction_loss=loss,
            regularization_loss=reg_loss,
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
                coll_loss += torch.sum(torch.sigmoid(-dis - 4.0))
            losses["coll_loss"] = coll_loss * self.algo_config.collision_weight

        if self.algo_config.calc_likelihood:
            likelihood_loss = self.algo_config.GAN_weight * (
                1 - torch.mean(pred_batch["scene_predictions"]["likelihood_new"])
            )
            losses["likelihood_loss"] = likelihood_loss

        return losses
