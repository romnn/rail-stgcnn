import os
import random
import time
from datetime import timedelta, datetime

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

from cargonet.preprocessing.datalake.retrieval import Retriever
from cargonet.preprocessing.graphs.tgraph import TransportGraph
from cargonet.dataset.baselinev1 import BaselineV1
from cargonet.models.baselines.model import BaselineModel
from cargonet.models.eval.losses import LossCollector
from cargonet.models.normalization import Scaler
from cargonet.models.sociallstm import ActiveRoutesModelLSTM, ActiveRoutesModelLSTMGAT
from cargonet.visualization.delays import DelayProgressPlot
from cargonet.dataset.avgdelayv1 import (
    EdgeAverageDelayDatasetV1,
    NodeAverageDelayDatasetV1,
)


class MockModel:
    def eval(self):
        pass


class BaselineAverageModelV1(BaselineModel):
    def __init__(
        self, dataset, node_input_dim, edge_input_dim, **kwargs,
    ):
        super().__init__(dataset, node_input_dim=node_input_dim, edge_input_dim=edge_input_dim, **kwargs)
        self.model = MockModel()

        base_path = os.path.dirname(os.path.realpath(__file__))
        dataset_base_path = os.path.join(base_path, "../../../datasets")
        models_base_path = os.path.join(base_path, "../../../trained")
        assert os.path.exists(dataset_base_path)
        assert os.path.exists(models_base_path)

        base_dataset_name = "average-delay-dataset-v1"
        dataset_name = "seq-average-delay-dataset-v1-10-1-1-1"
        base_dataset_path = os.path.join(dataset_base_path, base_dataset_name)
        dataset_path = os.path.join(dataset_base_path, dataset_name)

        # Initialize base dataset
        self.avg_dataset = NodeAverageDelayDatasetV1(
            root=base_dataset_path,
            name=base_dataset_name,
            limit=32,
            batch=timedelta(hours=7 * 24),
            force_reprocess=False,
            force_redownload=False,
            normalize_net=False,
        )
        
        self.avg_data, self.avg_train_data, self.avg_val_data, self.avg_train_indices, self.avg_val_indices = self.prepare_dataset(
            self.dataset, batch_size=1, shuffle=False, shuffle_after_split=False
        )

    def train(self, epochs=1, print_interval=10):
        # No need to train
        pass

    def feed(self, data, x, y, ea):
        def find_state(t):
            ti = (t - self.avg_dataset.timerange[0]) // self.avg_dataset.interval
            # print(t, ti)
            return self.avg_dataset[ti]

        def get_segment_state(t, station_id):
            state = find_state(t)
            return state.x[station_id, 0] # 0 is delay

        gei = data.ground_edge_index.t().view(-1, self.seq_len + self.pred_seq_len, 2)
        active = x.size(0)
        predictions = torch.zeros(active, self.pred_seq_len, 1).to(self.device)
        
        for tp in range(active):
            transport_id = data.transport_mask[tp].item()
            for hor in range(self.pred_seq_len):
                hor_src, hor_dest = self.seq_len + hor - 1, self.seq_len + hor
                station_idx_src, station_idx_dest = gei[tp, hor_src, 1].item(), gei[tp, hor_dest, 1].item()
                station_id_src, station_id_dest = int(self.net.x[station_idx_src, 0].item()), int(self.net.x[station_idx_dest, 0].item())
            
                first_time = datetime.fromtimestamp(data.first_time.item())

                try:
                    # map stations and fix the relative time
                    mapped_station_id = self.avg_dataset.mapping[(station_id_src, station_id_dest)]
                    arrival_src, arrival_dest = x[tp, hor_src, 1], x[tp, hor_dest, 1]
                    arrival_src = first_time + timedelta(minutes=int(arrival_src.item()))
                    arrival_dest = first_time + timedelta(minutes=int(arrival_dest.item()))
            
                    # Load the network states for the correct time steps
                    avg_state_src = get_segment_state(arrival_src, mapped_station_id).item()
                    avg_state_dest = get_segment_state(arrival_dest, mapped_station_id).item()
                    avg_state_src = 1 if avg_state_src == 0 else avg_state_src
                    avg_state_dest = 1 if avg_state_dest == 0 else avg_state_dest
                    avg_state_edge = torch.FloatTensor([avg_state_src, avg_state_dest]).mean().item()
                    planned_duration = data.temporal_edge_attr[tp, hor_dest-1, 1].item() # 1 is planned duration
                    predictions[tp, hor, 0] = x[tp, hor_src, -1] + avg_state_edge * planned_duration  # old delay plus new one planned duration
                    x[tp, hor_dest, -1] = predictions[tp,hor,0]
                except KeyError:
                    # It is very likely that those are the one without a segment excluded by the transport mask
                    continue

        outputs = predictions
        expected = y
        # print(expected.shape, outputs.shape)
        assert expected.shape == outputs.shape    
        return outputs, expected
    
    def save(self, path=None):
        pass

    def load(self, path=None):
        pass
