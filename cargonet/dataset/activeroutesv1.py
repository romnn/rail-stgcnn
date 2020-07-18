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
from torch_geometric.utils import remove_self_loops, to_undirected

import cargonet.preprocessing.datalake.retrieval as retrieval
import cargonet.preprocessing.graphs.tgraph as tgraph
from cargonet.dataset.dataset import RailDataset
from cargonet.models.utils import encode_datetime_cyclical
from cargonet.utils.geo import GERMANY_BBOX_1
from cargonet.utils.pdf import concat_pdfs


class Encoder:
    
    def __init__(self, first_time, seq_len, pred_seq_len, max_transports, net):
        self.first_time = first_time
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len

        self.net, self.net_mapping = net
        self.static_net_size = len(self.net_mapping)
        assert (
            self.static_net_size - 1 == np.array(list(self.net_mapping.values())).max()
        )

        self.conflicting_temp = defaultdict(list)
        self.route_node_mapping = dict()
        self.transports_node_lut = defaultdict(list)
        self.transports = []
        self.transport_mask = torch.ones(max_transports, dtype=torch.long) * -1
        self.acc = nx.DiGraph()
        self.max_transports = max_transports

        n_edges_temp = (seq_len + pred_seq_len - 1) * max_transports  # unidirectional
        n_edges_spat = (seq_len + pred_seq_len) * max_transports * 2  # bidirectional


        self.ground_edge_index = torch.zeros(
            max_transports, seq_len + pred_seq_len, 2, dtype=torch.long
        )
        self.temporal_edge_index = torch.zeros(
            max_transports, seq_len + pred_seq_len - 1, 2, dtype=torch.long
        )

        self.temporal_edge_attr = torch.zeros(
            max_transports,
            seq_len + pred_seq_len - 1,
            len(self.route_edge_fts),
            dtype=torch.long,
        )

        self.y_pred = torch.zeros(max_transports, pred_seq_len, len(self.predict_fts))
        self.x = torch.zeros(
            max_transports, seq_len + pred_seq_len, len(self.seq_route_node_fts)
        )
        self.current_transports = torch.zeros(
            max_transports, seq_len + pred_seq_len, dtype=torch.bool
        )
        self.routes = torch.zeros(
            max_transports, seq_len + pred_seq_len, dtype=torch.long
        )


class RoadsAsEdgesV1(Encoder):

    common_node_fts = [
        "pause",
        "arrival_rel",
        "departure_rel",
        "month_sin",
        "month_cos",
        "arrival_sin",
        "arrival_cos",
    ]
    seq_route_node_fts = common_node_fts + ["delay"]
    horizon_route_node_fts = common_node_fts
    station_node_fts = ["lat", "lon"]
    predict_fts = ["delay"]

    station_edge_fts = []
    route_edge_fts = ["distance", "plannedDuration", "duration"]

    def add(self, t, transport, center_edge, desc, anc):
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
            
            self.x[t, pad_seq + seq, 0] = (
                plannedDepTime - plannedArrTime
            ).total_seconds() / 60.0  # Pause in minutes

            self.x[t, pad_seq + seq, 1] = (
                plannedArrTime - self.first_time
            ).total_seconds() / 60.0  # Planned Arrival in absolute delta minutes
            self.x[t, pad_seq + seq, 2] = (
                plannedDepTime - self.first_time
            ).total_seconds() / 60.0  # Planned Departure in absolute delta minutes

            # Arrival in cyclical
            cyc_p_arr_hr, cyc_p_arr_month = encode_datetime_cyclical(plannedArrTime)

            # Month
            cyc_p_arr_month_sin, cyc_p_arr_month_cos = cyc_p_arr_month
            self.x[t, pad_seq + seq, 3] = cyc_p_arr_month_sin
            self.x[t, pad_seq + seq, 4] = cyc_p_arr_month_cos

            cyc_p_arr_hr_sin, cyc_p_arr_hr_cos = cyc_p_arr_hr
            self.x[t, pad_seq + seq, 5] = cyc_p_arr_hr_sin
            self.x[t, pad_seq + seq, 6] = cyc_p_arr_hr_cos

            # Departure in cyclical (assume month is the same)
            cyc_p_arr_hr, _ = encode_datetime_cyclical(plannedArrTime)

            self.current_transports[t, pad_seq + seq] = 1
            self.routes[t, pad_seq + seq] = stationId

            this_edge_index = (self.seq_len + self.pred_seq_len) * t + (pad_seq + seq)
            station_node = self.net_mapping.get(stationId, this_edge_index)
            self.ground_edge_index[t, pad_seq + seq] = torch.LongTensor(
                [this_edge_index, station_node]
            )

            station_node = self.net_mapping.get(stationId)
            self.conflicting_temp[station_node].append(this_edge_index)

        # Sequence node features
        for seq, n in enumerate(nx.topological_sort(anc_g)):
            self.x[t, pad_seq + seq, len(self.common_node_fts)] = transport.nodes[
                n
            ].get("delay")

        # Horizon node features
        for seq, n in enumerate(nx.topological_sort(desc_g)):
            # Predictions
            self.y_pred[t, seq, 0] = transport.nodes[n].get("delay")

        for seq, edge in enumerate(route_g.edges):
            try:
                self.temporal_edge_index[t, pad_seq + seq] = torch.LongTensor(
                    [
                        (self.seq_len + self.pred_seq_len) * t + (pad_seq + seq),
                        (self.seq_len + self.pred_seq_len) * t + (pad_seq + seq + 1),
                    ]
                )

                self.temporal_edge_attr[t, pad_seq + seq, 0] = (
                    transport.edges[edge].get("distance") / 1000.0
                )
                self.temporal_edge_attr[t, pad_seq + seq, 1] = (
                    transport.edges[edge].get("plannedDuration").total_seconds() / 60.0
                )
                self.temporal_edge_attr[t, pad_seq + seq, 2] = (
                    transport.edges[edge].get("duration").total_seconds() / 60.0
                )
            except Exception as e:
                print("Error while adding temp edge attrs", str(e))

    def encode(self, t):
        self.spatial_edge_index = []
        for station, conflicting in self.conflicting_temp.items():
            # Add edge between pair of conflicting nodes in each direction
            for c1 in conflicting:
                for c2 in conflicting:
                    if c1 == c2:
                        continue
                    se = torch.LongTensor([c1, c2])
                    self.spatial_edge_index.append(se)

        if len(self.spatial_edge_index) > 0:
            self.spatial_edge_index = (
                torch.stack(self.spatial_edge_index).t().contiguous()
            )
        else:
            self.spatial_edge_index = torch.zeros(2, 0, dtype=torch.long)

        self.temporal_edge_index = self.temporal_edge_index.view(-1, 2).t().contiguous()

        self.ground_edge_index = self.ground_edge_index.view(-1, 2).t().contiguous()
        if self.ground_edge_index.size(1) > 0:
            self.ground_edge_index[0, :] += self.net.x.size(0)
            
        num_transports = (self.current_transports == 1).sum(dim=1)
        num_transports = num_transports.numpy()
        if num_transports.size < 1:
            return Data(), self.acc
            
        num_transports = num_transports.max()

        temporal_node_mask = torch.BoolTensor([1] * self.x.size(0))
        spatial_node_mask = ~temporal_node_mask

        data = Data(
            x=self.x,
            y=self.y_pred,
            t=t.timestamp(),
            first_time=self.first_time.timestamp(),
            ground_edge_index=self.ground_edge_index,
            spatial_edge_index=self.spatial_edge_index,
            temporal_edge_index=self.temporal_edge_index,
            temporal_edge_attr=self.temporal_edge_attr,
            temporal_node_mask=temporal_node_mask,
            spatial_node_mask=spatial_node_mask,
            transport_mask=self.transport_mask,
            current_transports=self.current_transports,
            routes=self.routes,
            num_transports=torch.IntTensor([num_transports]),
        )

        return data, self.acc


class ActiveRoutesV1(RailDataset):
    """
    Dataset for predicting average delay on transport edges
    """

    node_feature_mapping = ["stationId", "imId", "country", "lat", "lon"]
    edge_feature_mapping = ["distance", "popularity"]

    def __init__(
        self,
        root,
        name=None,
        transform=None,
        pre_transform=None,
        limit=1,
        plot_download=False,
        plot_processing=False,
        force_reprocess=False,
        force_redownload=False,
        normalize_net=True,
        verbose=True,
        lookbehind_steps=3,
        lookahead_steps=1,
        lookahead=timedelta(hours=1),
        lookbehind=timedelta(minutes=10),
        interval=timedelta(minutes=10),
        batch=timedelta(hours=24),
        padding=timedelta(hours=0),
        bbox=None,
        seq_interval=1,
        pred_interval=1,
        seq_len=3,
        pred_seq_len=3,
    ):
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len
        super().__init__(
            root=root,
            name=name,
            transform=transform,
            pre_transform=pre_transform,
            limit=limit,
            plot_download=plot_download,
            plot_processing=plot_processing,
            force_reprocess=force_reprocess,
            force_redownload=force_redownload,
            normalize_net=normalize_net,
            verbose=verbose,
            lookbehind_steps=lookbehind_steps,
            lookahead_steps=lookahead_steps,
            lookahead=lookahead,
            lookbehind=lookbehind,
            interval=interval,
            batch=batch,
            padding=padding,
            bbox=bbox,
            full_route=True,
            unique_index=True,
            undirected=False,
        )

    def download(self):
        super().download()

    @property
    def processed_file_names(self):
        return ["processed_%d.pt" % f for f in range(0, len(self.raw_file_names))]

    def debug_plot(
        self, t, state, prefix=None, bbox=GERMANY_BBOX_1, labels=True, size=3,
    ):
        filename = "%s/%s%s%s.pdf" % (
            self.name,
            f"{prefix}/" if prefix else "",
            prefix or "",
            t if not isinstance(t, datetime) else self.format_timestamp(t),
        )
        # TODO: need new kind of plot
        if False:
            NXTransportPlot(
                state,
                filename=filename,
                show=False,
                node_labels=False,
                edge_labels=labels,
                thickness=size,
                node_size=size,
                colorbar_range=(-1.0, 3.0),  # (-100, 300),
                bbox=bbox,
                cities=True,
                check=False,
                delay=True,
                cmap=opaque_delay_cmap if opaque else delay_cmap,
                subtitle=f"{prefix} at {str(t)}" if prefix else str(t),
            ).plot()

    @property
    def encoder(self):
        return RoadsAsEdgesV1

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
        print("%d transports" % len(active_transports))

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

            encoder.add(t, transport, center_edge=(cu, cv), anc=anc, desc=desc)

        return encoder.encode(t)

    def process(self, skip=False):
        states_count = len(self.raw_paths)
        sl, pl = (
            self.seq_len,
            self.pred_seq_len,
        )
        for i in range(states_count):
            out = os.path.join(
                self.processed_dir, self.processed_file_names[i], # - sl - pl],
            )
            
            # Check if already processed
            if skip:
                if i < 28_400:
                    print("Skipping ", i)
                    continue
                if os.path.isfile(out):
                    print("Skipping ", i)
                    continue
            
            # Read transport state at time step t and some previous steps
            state = nx.read_gpickle(self.raw_paths[i])
            t = self.timerange[i]
            t_next = t + self.interval
            self.vlog(
                "Processing t[%d] %s - %s (%d/%d, %d+%d states)" % (i, t, t_next, i, states_count, sl, pl,)
            )

            data, graph = self.aggregate(t, state)
            # print(data)

            # Plot combined graph
            if self.plot_processing:
                self.debug_plot(i, graph, prefix="combined", size=1, labels=False)

            # Apply filters and transformations
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(
                data,
                out,
            )


def build_dataset(limit, plot, rebuild=False, reprocess=False):
    dataset_name = "active-routes-dataset-v1"

    base_path = os.path.dirname(os.path.realpath(__file__))
    base_dataset_path = os.path.join(base_path, "../../datasets")
    assert os.path.exists(base_dataset_path)
    dataset_path = os.path.join(base_dataset_path, dataset_name)

    try:
        dataset = ActiveRoutesV1(
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
        print(e)

    if plot:
        print("Combining plots")
        base_fig_path = os.path.join(base_path, "../../fig")
        concat_pdfs(
            source_dir=os.path.join(base_fig_path, dataset_name),
            out_file=os.path.join(base_fig_path, "%s.pdf" % dataset_name),
        )
