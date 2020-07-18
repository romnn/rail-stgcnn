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
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, GCNConv, GINConv, MessagePassing, SAGEConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

from cargonet.dataset.baselinev1 import BaselineV1
from cargonet.models.baselines.model import BaselineModel
from cargonet.models.eval.losses import LossCollector, MAELoss
from cargonet.models.normalization import Scaler
from cargonet.models.sociallstm import ActiveRoutesModelLSTM, ActiveRoutesModelLSTMGAT
from cargonet.visualization.delays import DelayProgressPlot


class BaseFC(nn.Module):
    def __init__(
        self,
        device,
        input_dim,
        max_transports,
        hidden_dim=128,
        lstm_layers=1,
        batch_size=1,
        pred_seq_len=1,
        seq_len=10,
        dropout=0,
        verbose=False,
    ):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_transports = max_transports
        self.lstm_layers = lstm_layers
        self.batch_size = batch_size
        self.dropout = dropout

        h2 = int(self.hidden_dim / 2)
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(h2)

        inp = self.input_dim * self.seq_len
        self.lin1 = nn.Linear(inp, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, h2)
        self.lin4 = nn.Linear(h2, self.pred_seq_len)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, ea):
        lin1 = self.lin1(x)
        x = torch.sigmoid(lin1)
        x = self.bn(x)
        x = self.dropout(x)
        lin2 = self.lin2(x)
        x = torch.sigmoid(lin2)
        x = self.dropout(x)
        x = self.lin4(x)
        return x


class FCModelV1(BaselineModel):
    def __init__(
        self,
        dataset,
        input_dim,
        dropout=0.5,
        max_transports=1000,
        hidden_dim=512,
        grad_clip=False,
        **kwargs
    ):
        super().__init__(dataset, input_dim, **kwargs)
        self.dropout = dropout
        self.grad_clip = grad_clip
        self.max_transports = max_transports
        self.hidden_dim = hidden_dim

        net = self.dataset.net.to(self.device)
        self.model = BaseFC(
            device=self.device,
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            pred_seq_len=self.pred_seq_len,
            hidden_dim=hidden_dim,
            dropout=self.dropout,
            max_transports=self.max_transports,
        ).to(self.device)

        self.net = self.dataset.net.to(self.device)
        self.loss = torch.nn.MSELoss()
        # self.loss = MAELoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=0.7
        )  # 0.001
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.0001, weight_decay=0.7)  # 0.001
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.000001, momentum=0.9)  # 0.001
        decayRate = 0.99
        # decayRate = 0.95
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=decayRate
        )

    def feed(self, x, y, ea):
        x = x[:, : self.seq_len, :].view(-1, self.seq_len * self.input_size)
        outputs = self.model(x, ea)
        expected = y[:, :].view(-1, self.pred_seq_len)
        assert expected.shape == outputs.shape

        return outputs, expected
