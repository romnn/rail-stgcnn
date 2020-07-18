import os
import random
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torch_geometric.transforms as T
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.utils import weight_norm
from torch_geometric.data import DataLoader
from torch_geometric.nn import (
    CGConv,
    EdgeConv,
    ChebConv,
    GATConv,
    GCNConv,
    GINConv,
    MessagePassing,
    NNConv,
    SAGEConv,
    GatedGraphConv,
    BatchNorm,
    AGNNConv,
    GraphUNet,
    TopKPooling
)
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import (
    add_remaining_self_loops,
    add_self_loops,
    remove_self_loops,
    to_undirected,
)
from torch_scatter import scatter_add

from cargonet.dataset.baselinev1 import BaselineV1
from cargonet.models.normalization import MinMaxScaler, Scaler, ZScoreScaler
from cargonet.models.sociallstm import (
    ActiveRoutesModel,
    ActiveRoutesModelLSTM,
    ActiveRoutesModelLSTMGAT,
)
from cargonet.models.tempconv import TemporalConvNet
from cargonet.visualization.delays import DelayProgressPlot


class TempGCN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="mean")
        self.mlp = Seq(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(tmp)

    def update(self, aggr_out):
        return aggr_out



class MyConv(MessagePassing):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(aggr="max", **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.LeakyReLU(),
            # nn.Tanh(),
            # nn.Dropout(0.2),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_weight=None):
        return self.propagate(
            edge_index, size=(x.size(0), x.size(0)), x=x
        )

    def forward2(self, x, edge_index, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super(DynamicEdgeConv, self).forward(x, edge_index)

    def update(self, aggr_out):
        return aggr_out

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(tmp)

    def __repr__(self):
        return "{}(local_nn={}, global_nn={})".format(
            self.__class__.__name__, self.local_nn, self.global_nn
        )


class MyConv2(MyConv):
    def __init__(self, in_channels, out_channels, k=6):
        super().__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, edge_index):
        return super().forward(x, edge_index)


class Breadth(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Breadth, self).__init__()
        self.gatconv = MyConv(in_dim, out_dim)

    def forward(self, x, edge_index):
        x = torch.tanh(self.gatconv(x, edge_index))
        return x


class Depth(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super(Depth, self).__init__()
        self.lstm = torch.nn.LSTM(in_dim, hidden, 1, bias=False)

    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))
        return x, (h, c)

class TempConvLayer(torch.nn.Module):

    def __init__(self, device, k_t, c_in, c_out, dropout=0, act_func='relu'):
        super().__init__()
        self.act_func = act_func
        self.c_in = c_in
        self.c_out = c_out
        
        self.temp_conv_glu = TemporalConvNet(
            num_inputs=c_in, num_channels=[2 * c_out], dropout=dropout, kernel_size=k_t, #4,
        )
        self.temp_conv = TemporalConvNet(
            num_inputs=c_in, num_channels=[c_out], dropout=dropout, kernel_size=k_t, #4,
        )

        # self.conv = GCNConv(c_in, c_out, normalize=False)
        self.conv = MyConv(c_in, c_out)
        self.bn = nn.BatchNorm1d(c_out)

    def forward(self, x, edge_index, edge_weight=None):
        if self.c_in > self.c_out:
            raise NotImplementedError
        elif self.c_in < self.c_out:
            x_input = torch.concat([x, torch.zeros(*x.shape()[:-1], self.c_out - self.c_in)], axis=-1)
        else:
            x_input = x

        x = self.conv(x.reshape(-1, self.c_in), edge_index, edge_weight)
        x = x.view(-1, 20, self.c_out)
        x = x[:,:,:self.c_out] + x_input
        
        return F.relu(x)


class SpatioConvLayer(torch.nn.Module):

    def __init__(self, device, k_s, c_in, c_out):
        super().__init__()
        self.device = device
        self.k_s = k_s
        self.c_in = c_in
        self.c_out = c_out
        assert c_out % 4 == 0
        self.conv = MyConv(c_in, c_out)
        
    def forward(self, x, edge_index):
        if self.c_in > self.c_out:
            # bottleneck down-sampling
            raise NotImplementedError
        elif self.c_in < self.c_out:
            fill = torch.zeros(*x.shape[:-1], self.c_out - self.c_in).to(self.device)
            x_input = torch.cat([x, fill], axis=-1)
        else:
            x_input = x
        
        x = self.conv(x.reshape(-1, self.c_in).float(), edge_index)
        x = x.view(-1, 20, self.c_out)
        x = x[:,:,:self.c_out] + x_input
        return F.relu(x)


class STConvBlock(torch.nn.Module):
    
    def __init__(self, device, k_s, k_t, channels, dropout=0, act_func='GLU'):
        super().__init__()
        self.k_s = k_s
        self.k_t = k_t
        self.channels = channels
        self.dropout = dropout
        
        c_si, c_t, c_oo = channels
        self.spat_conv = SpatioConvLayer(device, k_s, c_si, c_t)
        self.temp_conv2 = TempConvLayer(device, k_t, c_t, c_oo)

    def forward(self, x, spat_edge_index, temp_edge_index, temp_edge_weight):
        x_ln = x
        for i in range(3):
            x_s = x_ln
            x_t = self.spat_conv(x_s, spat_edge_index)
            x_t = F.layer_norm(x_t, normalized_shape=x_t.size()[1:])
            x_t = F.dropout(x_t, self.dropout)
            x_o = self.temp_conv2(x_t, temp_edge_index, edge_weight=temp_edge_weight)
            x_ln = F.layer_norm(x_o, normalized_shape=x_o.size()[1:])
        x_ln = x_ln + x
        return F.dropout(x_ln, self.dropout)


class Memory(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.lstm = torch.nn.LSTM(in_dim, hidden, 1, bias=False)

    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))
        return x, (h, c)


class ActiveRoutesModelTCN(ActiveRoutesModel):
    def __init__(
        self,
        device,
        input_dim,
        output_size,
        seq_len,
        pred_seq_len,
        max_transports=1000,
        # New
        use_rnn=False,
        rnn_size=64,
        embedding_size=64,
        dropout=0.,
        gru=False,
    ):
        super().__init__(
            device,
            input_dim,
            output_size,
            seq_len,
            pred_seq_len,
            max_transports=max_transports,
        )
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.use_rnn = use_rnn

        node_fts = 8

        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(node_fts, self.embedding_size)

        self.net_node_hidden_dim = 64  # 128
        self.embed_stations = weight_norm(nn.Linear(node_fts, self.net_node_hidden_dim))
        self.embed_temp = nn.Linear(node_fts, self.net_node_hidden_dim)

        # EdgeGATConv
        self.conv1 = GATConv(self.net_node_hidden_dim, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, 64, heads=1, concat=True, dropout=0.6)

        self.conv1 = MyConv(self.net_node_hidden_dim, 64)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, 64)  # self.output_size)

        self.final_layer = nn.Linear(128 + 64, self.output_size)  # self.rnn_size

        # Just for debugging
        self.visualize = nn.Linear(64, 1)  # self.rnn_size

        # self.hidden_dim = 32 # 64  # 32
        self.hidden_dim = self.embedding_size
        self.input_dim = 8 + 3
        self.encoder = nn.Linear(self.input_dim, self.hidden_dim)

        chans = [128, 64]
        self.temp_conv = TemporalConvNet(
            num_inputs=self.hidden_dim, num_channels=chans, dropout=dropout, kernel_size=4,
        )

        chans = [self.hidden_dim, self.hidden_dim]
        self.spat_conv = TemporalConvNet(
            num_inputs=self.hidden_dim, num_channels=chans, dropout=dropout, kernel_size=4,
        )

        # self.conv_hidden = 32
        self.conv_hidden = self.hidden_dim # 64
        self.spat_k = self.seq_len + self.pred_seq_len
        self.spat_convs = nn.ModuleList()
        
        self.spat_convs.append(
            GATConv(self.net_node_hidden_dim, self.conv_hidden, heads=1, concat=True, dropout=0.2)
        )
        
        self.ground_encoder = nn.Linear(5, 16)
        self.ground_encoder2 = nn.Linear(5, self.hidden_dim + 16)

        self.ground_k = 3
        self.ground_convs = nn.ModuleList()
        self.ground_convs.append(
            GCNConv(self.conv_hidden, self.conv_hidden)
        )
        for _ in range(self.ground_k - 1):
            self.ground_convs.append(GCNConv(self.conv_hidden, self.conv_hidden))

        self.unet = GraphUNet(self.conv_hidden, 16, self.conv_hidden, depth=4, pool_ratios=0.5)
        self.pool = TopKPooling(self.conv_hidden, ratio=0.3)
        self.ground_sync = MyConv(self.hidden_dim + 16, self.conv_hidden)
        self.ground_sync2 = MyConv(self.hidden_dim + 16, self.conv_hidden)
        
        self.breadths = torch.nn.ModuleList(
            [Breadth(self.conv_hidden, self.conv_hidden) for i in range(self.spat_k)])
        
        self.depths = torch.nn.ModuleList(
            [Depth(self.conv_hidden * 2, self.conv_hidden) for i in range(self.spat_k)])
        
        
        self.pred_conv = TemporalConvNet(
            num_inputs=self.conv_hidden, num_channels=[128, 64], dropout=dropout, kernel_size=4,
        )

        tcn_hidden = self.embedding_size # 64

        self.bn = BatchNorm(2 * self.conv_hidden)
        self.final_conv = TemporalConvNet(
            # 128 -> 64 -> 32
            num_inputs=(3 if self.use_rnn else 2) * self.conv_hidden + 16,
            # num_inputs=3 * self.conv_hidden,
            num_channels=[tcn_hidden, tcn_hidden, tcn_hidden], dropout=dropout, kernel_size=8,
        )

        self.start_conv = TemporalConvNet(
            num_inputs=self.conv_hidden,
            num_channels=[tcn_hidden, tcn_hidden, tcn_hidden], dropout=dropout, kernel_size=8,
        )

        # ReLU and dropout unit
        self.lrelu = nn.LeakyReLU()  # nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.cell = nn.LSTMCell(self.hidden_dim, self.rnn_size)
        
        self.lin1 = nn.Linear(tcn_hidden, tcn_hidden)
        self.lin2 = nn.Linear(tcn_hidden, tcn_hidden)
        self.lin3 = nn.Linear(tcn_hidden, self.pred_seq_len)

        self.stgblock = STConvBlock(self.device, k_s=1, k_t=1, dropout=dropout, channels=(
            self.conv_hidden + 16, self.conv_hidden + 16, self.conv_hidden + 16
        ))

    def forward(
        self,
        data,
        net_data,
        net_state_hidden,
        net_state_cell,
        num_transports,
        transport_mask,
        current_transports,
    ):
        assert net_data.x.size(1) == 5 # has lat and lon coordinates
        total_seq_len = self.seq_len + self.pred_seq_len
        edges = data.temporal_edge_attr
        _edges = edges
        edges = torch.zeros(_edges.size(0), total_seq_len, 3).to(self.device)
        edges[:, 1:total_seq_len, :] = _edges

        full_seq = torch.cat([data.x, edges], dim=2)
        seq = full_seq[:, : self.seq_len, :]
        horizon = full_seq[:, -self.pred_seq_len :, :]

        assert not torch.isnan(seq).any()

        # Embed
        embed_seq = self.encoder(full_seq)

        # Run temp conv over it
        out = embed_seq.permute(0, 2, 1)
        out = self.start_conv(out)
        out = out.permute(0, 2, 1)
        net_embedded = self.ground_encoder(net_data.x)
        
        gei = data.ground_edge_index
        gei = gei.t().view(-1, self.seq_len + self.pred_seq_len, 2)
        gni = gei[:,:,1]
        assert gni.size(0) == data.x.size(0)
        gni = torch.clamp(gni, min=0, max=data.x.size(0))
        gn = net_embedded[gni]
        
        stout = torch.cat([out, gn], dim=-1)
        
        # stgblock
        delays = self.stgblock(stout,
            spat_edge_index=data.spatial_edge_index, temp_edge_index=data.temporal_edge_index,
            temp_edge_weight=data.temporal_edge_attr[:,:,1].view(-1)) # Planned duration

        if self.use_rnn:
            
            _net_embedded = self.ground_encoder2(net_data.x).view(-1, self.hidden_dim + 16)
            _out = delays.view(-1, self.hidden_dim  + 16)

            net_nodes = torch.cat([_net_embedded, _out])
            net_edges = net_data.edge_index
            net_weights = net_data.edge_attr[:,1].t().contiguous()
            net_bridge = to_undirected(data.ground_edge_index)

            # Pass the state to the network and run diffusion
            grounded = self.ground_sync(net_nodes, net_bridge)
            grounded = F.layer_norm(grounded, normalized_shape=grounded.size()[1:])
            
            grounded = grounded[:_net_embedded.size(0)]
            for i in range(len(self.ground_convs)):
                grounded = self.ground_convs[i](grounded, net_edges) # , edge_weight=net_weights)
                grounded = F.relu(grounded)
                grounded = F.layer_norm(grounded, normalized_shape=grounded.size()[1:])
                grounded = self.dropout(grounded)
            
            grounded = F.layer_norm(grounded, normalized_shape=grounded.size()[1:])
            
            net_state_hidden, net_state_cell = self.cell(
                grounded, (net_state_hidden, net_state_cell)
            )

            backfill = self.ground_sync2(torch.cat([net_state_hidden, _out]), net_bridge)

            backfill = backfill[_net_embedded.size(0):]
            backfill = backfill.view(-1, 20, self.hidden_dim)

        # Combine stgblock, input embedding and long term
        combined = [delays, out]
        if self.use_rnn:
            combined.append(backfill)
        out = torch.cat(combined, dim=-1)
        
        out = out

        # Run temp conv over it
        out = out.permute(0, 2, 1)
        out = self.final_conv(out)
        out = out.permute(0, 2, 1)
        out = out[:, -1, :]

        return self.lin3(out), net_state_hidden, net_state_cell