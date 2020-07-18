import os.path
import shutil
import statistics
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import cargonet.preprocessing.datalake.retrieval as retrieval
import cargonet.preprocessing.graphs.tgraph as tgraph
from cargonet.dataset.activeroutesv1 import ActiveRoutesV1, Encoder
from cargonet.utils.geo import GERMANY_BBOX_1
from cargonet.utils.pdf import concat_pdfs


class AccumulateEncoderV1(Encoder):

    common_node_fts = ["passed", "pause"]
    seq_route_node_fts = common_node_fts + ["delay"]
    horizon_route_node_fts = common_node_fts
    station_node_fts = ["lat", "lon"]
    predict_fts = ["delay"]

    station_edge_fts = []
    route_edge_fts = ["distance", "duration"]

    def add(self, transport, center_edge, desc, anc):
        cu, cv = center_edge
        desc_g = transport.subgraph(desc + [cv])
        anc_g = transport.subgraph(anc + [cu])
        route_g = transport.subgraph(anc + [cu, cv] + desc)

        transport_ids = list(
            set([data.get("transportId") for _, data in route_g.nodes(data=True)])
        )
        assert len(transport_ids) == 1
        transport_id = transport_ids[0]

        if not transport_id in self.route_node_mapping:
            self.route_node_mapping[transport_id] = len(self.route_node_mapping)
        t = self.route_node_mapping[transport_id]
        self.transport_mask[t] = transport_id

        pad_seq = self.seq_len - len(anc_g)

        # Common node features
        for seq, n in enumerate(nx.topological_sort(route_g)):
            stationId = transport.nodes[n].get("stationId")
            plannedArrTime = transport.nodes[n].get("plannedArrivalTime")
            plannedDepTime = transport.nodes[n].get("plannedDepartureTime")

            self.x[t, pad_seq + seq, 0] = 1  # "Node was passed"
            self.x[t, pad_seq + seq, 1] = (
                plannedDepTime - plannedArrTime
            ).total_seconds()
            
            self.current_transports[t, pad_seq + seq] = 1

        for seq, n in enumerate(nx.topological_sort(anc_g)):
            self.x[t, pad_seq + seq, 2] = transport.nodes[n].get("delay")  # Delay

        for seq, n in enumerate(nx.topological_sort(desc_g)):
            self.y_pred[t, seq, 0] = transport.nodes[n].get("delay")  # Delay

        # Temporal edges edge_attrs[:,t]
        for seq, edge in enumerate(route_g.edges):
            self.temporal_edge_attr[t, pad_seq + seq, 0] = transport.edges[edge].get(
                "distance"
            )
            self.temporal_edge_attr[t, pad_seq + seq, 1] = (
                transport.edges[edge].get("plannedDuration").total_seconds() / 60.0
            )

    def encode(self):
        num_transports = (self.current_transports == 1).sum(dim=1)
        num_transports = num_transports.numpy()
        if num_transports.size < 1:
            return Data(), self.acc
        num_transports = num_transports.max()

        data = Data(
            x=self.x,
            y=self.y_pred,
            temporal_edge_attr=self.temporal_edge_attr,
            transport_mask=self.transport_mask,
            current_transports=self.current_transports,
            num_transports=torch.IntTensor([num_transports]),
        )

        return data, self.acc


class BaselineV1(ActiveRoutesV1):
    def download(self):
        super().download()

    def process(self):
        super().process()

    @property
    def encoder(self):
        return AccumulateEncoderV1

    def aggregate(self, t, state):
        acc = nx.DiGraph()
        ustate = state.to_undirected()

        def debug(g, ref=None):
            pos = nx.spring_layout(ref if ref else g)
            if ref:
                nx.draw(ref, pos, node_size=200, node_color="green", arrows=True)
            labels = nx.get_node_attributes(g, "arrivalTime")
            nx.draw(g, pos, labels=labels, node_size=150, node_color="red", arrows=True)
            plt.show()

        # extract all subgraphs
        active_transports = [state.subgraph(c) for c in nx.connected_components(ustate)]

        def get_segment(_transport, _t):
            # Assumptions
            if not all(
                [
                    not (_transport.has_edge(u, v) and _transport.has_edge(v, u))
                    for u, v in _transport.edges
                ]
            ):
                # This can happen if a transport uses a segment twice
                pass

            # Find segment of current timestep t
            current_segment = None
            for u, v, data in _transport.edges(data=True):
                src = _transport.nodes[u].get("arrivalTime")
                dest = _transport.nodes[v].get("arrivalTime")
                if src <= _t < dest:
                    current_segment = (u, v)
            return current_segment

        # Filter only transports that fit current timeframe
        first_time = None
        active_transports = [
            tp for tp in active_transports if get_segment(tp, t) is not None
        ]
        for transport in active_transports:
            for _, data in transport.nodes(data=True):
                candidates = [
                    data.get("plannedArrivalTime"),
                    data.get("arrivalTime"),
                    data.get("plannedDepartureTime"),
                    data.get("departureTime"),
                ]
                candidate = min(candidates)
                if first_time is None:
                    first_time = candidate
                first_time = min(candidate, first_time)

        encoder = self.encoder(
            first_time=first_time,
            seq_len=self.seq_len,
            pred_seq_len=self.pred_seq_len,
            max_transports=len(active_transports),
            net=(self.net, self.net_mapping),
        )

        for transport in active_transports:
            current_segment = get_segment(transport, t)
            if current_segment is None:
                # Does not fit current timeframe
                continue
            cu, cv = current_segment
            # Include some ancestors and descendants
            desc = list(nx.nodes(nx.dfs_tree(transport, cv)))[: self.pred_seq_len]
            anc = list(nx.nodes(nx.dfs_tree(transport.reverse(copy=True), cu)))[
                : self.seq_len
            ]
            assert len(set(desc + anc)) <= self.pred_seq_len + self.seq_len + 1

            debug_sg = desc + anc + [cu, cv]
            if False:
                debug(state.subgraph(debug_sg), ref=transport)

            encoder.add(transport, center_edge=(cu, cv), anc=anc, desc=desc)

        return encoder.encode()


def build_dataset(limit, plot, rebuild=False, reprocess=False):
    dataset_name = "baseline-dataset-v1"

    base_path = os.path.dirname(os.path.realpath(__file__))
    base_dataset_path = os.path.join(base_path, "../../datasets")
    assert os.path.exists(base_dataset_path)
    dataset_path = os.path.join(base_dataset_path, dataset_name)

    try:
        dataset = BaselineV1(
            root=dataset_path,
            name=dataset_name,
            plot_processing=plot,
            limit=limit,
            batch=timedelta(hours=2),
            force_redownload=rebuild,
            force_reprocess=reprocess,
        )
        print(dataset)
        print(len(dataset))
    except Exception as e:
        raise
