import os
import random
import time
import math
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
from cargonet.models.tempconv import TemporalConvNet
from cargonet.visualization.delays import DelayProgressPlot


class BaseTCN(nn.Module):
    def __init__(
        self,
        device,
        node_input_dim,
        edge_input_dim,
        embedding_dim,
        conv_dim,
        pred_seq_len=1,
        seq_len=10,
        levels=3,
        dropout=0,
        kernel_size=4,
        verbose=False,
    ):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len
        self.embedding_dim = embedding_dim
        self.conv_dim = conv_dim
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.kernel_size = kernel_size
        self.levels = int(min(levels, math.log(self.seq_len + self.pred_seq_len, self.kernel_size)))
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Linear(self.node_input_dim + self.edge_input_dim + 0, self.embedding_dim)

        chans = [self.conv_dim] * self.levels
        self.temp_conv = TemporalConvNet(
            num_inputs=self.embedding_dim,
            num_channels=chans,
            dropout=dropout,
            kernel_size=self.kernel_size,
        )

        self.lrelu = nn.LeakyReLU()

        self.bn = nn.BatchNorm1d(self.embedding_dim)
        self.bn2 = nn.BatchNorm1d(self.conv_dim)

        self.lin = nn.Linear(self.conv_dim, self.pred_seq_len)
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, data, x, edges, seq, net):
        eseq = self.encoder(seq)
        eseq = eseq.permute(0, 2, 1)

        # input must be (N, C_in, L_in)
        conv_out = self.temp_conv(eseq)
        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.dropout(conv_out)
        final_hidden = conv_out[:, -1, :]

        return self.lin(final_hidden)

class BaselineTCNModelV1(BaselineModel):
    def __init__(
        self, dataset, 
        node_input_dim, edge_input_dim, dropout=0.1, lr=0.001, l1_reg=0., conv_dim=64,
        embedding_dim=64, kernel_size=8,
        weight_decay=0.001, **kwargs,
    ):
        super().__init__(dataset, node_input_dim=node_input_dim, edge_input_dim=edge_input_dim, **kwargs)

        self.dropout = dropout

        net = self.dataset.net.to(self.device)
        self.model = BaseTCN(
            device=self.device,
            node_input_dim=self.node_input_dim,
            edge_input_dim=self.edge_input_dim,
            conv_dim=conv_dim,
            kernel_size=kernel_size,
            embedding_dim=embedding_dim,
            seq_len=self.seq_len,
            pred_seq_len=self.pred_seq_len,
            dropout=self.dropout,
        ).to(self.device)

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

    @classmethod
    def hyperparameter_search(cls, epochs=20, samples=15, **model_params):
        from sklearn.model_selection import ParameterGrid
        import datetime
        import json
        import random
        from pprint import pprint

        param_grid = dict(
            lr=[0.001],
            weight_decay=[0.0, 0.001, 0.00001],
            dropout=[0.0, 0.1, 0.2],
            embedding_dim=[32, 64, 128],
            conv_dim=[32, 64, 128],
            l1_reg=[0, 0.001],
            # layers=[1, 2, 3],
            kernel_size=[2, 4, 8],
        )
        results = []
        configs = list(ParameterGrid(param_grid))
        random.shuffle(configs)
        for params in configs[:samples]:
            pprint(params)
            model = cls(**{**model_params, **params})
            model.train(epochs=epochs, val=False)
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
