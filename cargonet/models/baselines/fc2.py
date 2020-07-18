import os
import random
import time
from pprint import pprint
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
from cargonet.models.eval.losses import LossCollector, MAELoss
from cargonet.models.normalization import Scaler
from cargonet.models.sociallstm import ActiveRoutesModelLSTM, ActiveRoutesModelLSTMGAT
from cargonet.visualization.delays import DelayProgressPlot


class FC2Model(nn.Module):
    def __init__(
        self, device, node_input_dim, edge_input_dim, hidden_dim, pred_seq_len, seq_len, dropout,
    ):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len
        self.hidden_dim = hidden_dim

        self.input_dim = (self.seq_len + self.pred_seq_len) * (node_input_dim + edge_input_dim)

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, int(hidden_dim/2))
        self.lin3 = weight_norm(nn.Linear(int(hidden_dim/2), self.pred_seq_len))

        self.bn = nn.BatchNorm1d(self.hidden_dim)

        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self, init="kaiming"):
        if init == "kaiming" and False:
            nn.init.kaiming_uniform_(
                self.lin11.weight, mode="fan_in", nonlinearity="leaky_relu"
            )
            nn.init.kaiming_uniform_(
                self.lin1.weight, mode="fan_in", nonlinearity="leaky_relu"
            )
            nn.init.kaiming_uniform_(
                self.lin2.weight, mode="fan_in", nonlinearity="leaky_relu"
            )
            nn.init.kaiming_uniform_(
                self.lin3.weight, mode="fan_in", nonlinearity="leaky_relu"
            )

    def forward(self, data, x, edges, seq, net):
        lin1 = self.lin1(seq.view(-1, self.input_dim)) # .half()
        x = F.relu(lin1)
        x = self.dropout(x)
        lin2 = self.lin2(x)
        x = F.relu(lin2)
        x = self.dropout(x)
        x = self.lin3(x)
        return x.float()


class FCModelV2(BaselineModel):
    def __init__(
        self, dataset, node_input_dim, edge_input_dim, dropout=0.1, 
        l1_reg=0., lr=0.001, hidden_dim=64, weight_decay=0.005, **kwargs,
    ):
        super().__init__(dataset, node_input_dim, edge_input_dim, l1_reg=l1_reg, **kwargs)
        self.model = FC2Model(
            self.device, node_input_dim=self.node_input_dim, edge_input_dim=self.edge_input_dim,
            pred_seq_len=self.pred_seq_len, seq_len=self.seq_len, hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(self.device)

        self.loss = nn.MSELoss()
        # self.loss = MAELoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.0001, weight_decay=0.7)  # 0.001
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.000001, momentum=0.9)  # 0.001
        decayRate = 0.99
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=decayRate
        )
        self.lr_scheduler = None

    @classmethod
    def hyperparameter_search(cls, epochs=20, **model_params):
        from sklearn.model_selection import ParameterGrid
        import datetime
        import json

        param_grid = dict(
            lr=[0.001], # , 0.0001],
            weight_decay=[0.01, 0.001, 0.0001],
            dropout=[0.0, 0.2, 0.4],
            hidden_dim=[32, 64, 128],
            l1_reg=[0, 0.1, 0.4],
            # batch_size=[0, 0.1, 0.4],
        )
        results = []
        for params in list(ParameterGrid(param_grid)):
            pprint(params)
            model = cls(**{**model_params, **params})
            model.train(epochs=epochs)
            _, val_losses = model.test()
            results.append((val_losses["mse"], params))
        results = sorted(results, key=lambda r: r[0])
        print("BEST")
        print(results[0])
        
        # Save as JSON
        base_path = os.path.dirname(os.path.realpath(__file__))
        models_base_path = os.path.join(base_path, "../../../trained")
        assert os.path.exists(models_base_path)

        out_file = os.path.join(models_base_path, "hps")
        out_file = os.path.join(out_file, cls.__name__ + "_" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".json")
        with open(out_file, "w+") as f:
            json.dump(results, f)
