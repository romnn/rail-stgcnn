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
from torch_geometric.nn import GATConv, GCNConv, GINConv, MessagePassing, SAGEConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

from cargonet.dataset.baselinev1 import BaselineV1
from cargonet.models.baselines.model import BaselineModel
from cargonet.models.eval.losses import MAELoss
from cargonet.models.normalization import MinMaxScaler, Scaler, ZScoreScaler
from cargonet.models.sociallstm import ActiveRoutesModelLSTM, ActiveRoutesModelLSTMGAT
from cargonet.visualization.delays import DelayProgressPlot


class BaseLSTM(nn.Module):
    def __init__(
        self,
        device,
        node_input_dim,
        edge_input_dim,
        output_dim,
        max_transports,
        embedding_dim=64,
        lstm_hidden_dim=128,
        lstm_layers=3,
        batch_size=1,
        pred_seq_len=1,
        seq_len=10,
        use_gru= False,
        dropout=0,
        verbose=False,
    ):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len

        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.input_dim = self.node_input_dim + self.edge_input_dim

        self.output_dim = output_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.max_transports = max_transports
        self.lstm_layers = lstm_layers
        self.batch_size = batch_size
        self.use_gru = use_gru
        print("GRU:", self.use_gru)

        self.dropout = nn.Dropout(dropout)

        self.embedding_dim = embedding_dim
        self.encoder = nn.Linear(self.input_dim, self.embedding_dim)
        
        self.bn = nn.BatchNorm1d(self.lstm_hidden_dim)
        self.bn = nn.BatchNorm1d(20)
        self.lrelu = nn.LeakyReLU()

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.lin1 = nn.Linear(self.lstm_hidden_dim, self.lstm_hidden_dim)
        self.lin2 = nn.Linear(self.lstm_hidden_dim + (self.pred_seq_len * self.embedding_dim), self.pred_seq_len)

    def forward(self, data, x, edges, seq, net):
        past = seq[:,:self.seq_len,:]
        hist = seq[:,self.seq_len:,:]

        ehist = self.encoder(hist)
        eseq = self.encoder(past)
        ehist = F.layer_norm(ehist, normalized_shape=ehist.size()[1:])
        eseq = F.layer_norm(eseq, normalized_shape=eseq.size()[1:])
        
        if self.use_gru:
            _, h_out = self.gru(eseq)
        else:
            _, (h_out, _) = self.lstm(eseq)

        h_out = F.layer_norm(h_out, normalized_shape=h_out.size()[1:])

        h_out = h_out.view(self.lstm_layers, -1, self.lstm_hidden_dim)
        h_out = h_out[-1, :, :]
        ehist = ehist.view(h_out.size(0), -1)
        pred = torch.cat([h_out, ehist], dim=-1)
        return self.lin2(pred).float()


class BaselineLSTMModelV1(BaselineModel):

    @property
    def model_state_path(self):
        return os.path.join(self.trained_model_dir, self.name + (
            "_sim" if self.simulation else ""
        ) + (
            "_lstm" if (not self.use_gru) else ""
        ) + ".pt")

    def __init__(
        self,
        dataset,
        node_input_dim,
        edge_input_dim,
        dropout=0.1,
        max_transports=1000,
        embedding_dim=32,
        lstm_hidden_dim=32,
        lstm_layers=1,
        l1_reg=0., # 001, # 001,
        lr=0.001,
        weight_decay=0.005,
        use_gru=True,
        grad_clip=False,
        **kwargs,
    ):
        super().__init__(dataset, node_input_dim=node_input_dim, l1_reg=l1_reg, edge_input_dim=edge_input_dim, **kwargs)

        self.dropout = dropout
        self.grad_clip = grad_clip
        self.use_gru = use_gru
        self.max_transports = max_transports
        self.lstm_hidden_dim = lstm_hidden_dim

        net = self.dataset.net.to(self.device)
        self.model = BaseLSTM(
            device=self.device,
            node_input_dim=self.node_input_dim,
            edge_input_dim=self.edge_input_dim,
            output_dim=self.output_size,
            seq_len=self.seq_len,
            lstm_layers=lstm_layers,
            pred_seq_len=self.pred_seq_len,
            lstm_hidden_dim=lstm_hidden_dim,
            dropout=self.dropout,
            use_gru=self.use_gru,
            max_transports=self.max_transports,
        ).to(self.device)

        self.net = self.dataset.net.to(self.device)
        self.loss = torch.nn.MSELoss()
        # self.loss = MAELoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        decayRate = 0.99
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=decayRate
        )
        self.lr_scheduler = None