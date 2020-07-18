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
from cargonet.models.eval.losses import LossCollector
from cargonet.models.normalization import Scaler
from cargonet.models.sociallstm import ActiveRoutesModelLSTM, ActiveRoutesModelLSTMGAT
from cargonet.visualization.delays import DelayProgressPlot


class MockModel:
    def eval(self):
        pass


class BaselineTimeshiftModelV1(BaselineModel):
    def __init__(
        self, dataset, node_input_dim, edge_input_dim, **kwargs,
    ):
        super().__init__(dataset=dataset, node_input_dim=node_input_dim, edge_input_dim=edge_input_dim, **kwargs)
        self.model = MockModel()

    def train(self, epochs=1, print_interval=10):
        # No need to train
        pass

    def feed(self, data, x, y, ea):
        last_delays = x[:, self.seq_len - 1, -1]
        outputs = last_delays.view(-1, 1, 1).repeat(1, self.pred_seq_len, 1)
        expected = y
        assert expected.shape == outputs.shape
        return outputs, expected
    
    def save(self, path=None):
        pass

    def load(self, path=None):
        pass