import math
import os.path
import pickle
import shutil
import time
from datetime import datetime, timedelta
from pprint import pprint
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import torch_geometric.utils
from torch_geometric.data import Data, Dataset

import cargonet.preprocessing.datalake.retrieval as retrieval
import cargonet.preprocessing.graphs.tgraph as tgraph
from cargonet.models.normalization import Scaler
from cargonet.preprocessing.graphs.builders import NXTGBuilder
from cargonet.preprocessing.graphs.mergers import AverageMerger
from cargonet.utils.cli import ask
from cargonet.utils.formatting import fmt_time
from cargonet.utils.geo import GERMANY_BBOX_1
from cargonet.utils.pdf import concat_pdfs
from cargonet.visualization.colors import delay_cmap, opaque_delay_cmap
from cargonet.visualization.gmtplot import GMTTransportPlot
from cargonet.visualization.hist import HistogramPlot
from cargonet.visualization.nxplot import NXTransportPlot


class RailDataset(Dataset):

    PROCESSED_STATIC_NET_FILENAME = "processed-static-transport-net.pt"
    PROCESSED_STATIC_NET_MAPPING_FILENAME = "processed-static-transport-net.map.pickle"

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
        verbose=False,
        lookbehind_steps=3,
        lookahead_steps=1,
        lookahead=timedelta(hours=1),
        lookbehind=timedelta(minutes=10),
        interval=timedelta(minutes=10),
        batch=timedelta(hours=24),
        padding=timedelta(hours=16),
        bbox=None,
        full_route=False,
        unique_index=False,
        undirected=True,
    ):
        self.root = root
        self.name = name or self.__class__.__name__
        self.limit = limit
        self.plot_download = plot_download
        self.plot_processing = plot_processing
        self.force_reprocess = force_reprocess
        self.force_redownload = force_redownload
        self.verbose = verbose
        self.lookbehind_steps = lookbehind_steps
        self.lookahead_steps = lookahead_steps
        self.lookahead = lookahead
        self.lookbehind = lookbehind
        self.interval = interval
        self.batch = batch
        self.padding = padding
        self.bbox = bbox
        self.normalize_net = normalize_net
        self.nx_net, self.net_mapping = self.load_full_graph(bbox=bbox)
        
        self.mapping = None
        self.full_route = full_route
        self.unique_index = unique_index
        self.undirected = undirected

        # Validate configuration to avoid skew
        assert self.batch.total_seconds() % self.interval.total_seconds() == 0
        assert timedelta(hours=24).total_seconds() % self.interval.total_seconds() == 0

        # Eventually delete
        if self.force_redownload:
            if not ask("Redownload %s?" % root):
                raise AssertionError("Back off")
            try:
                shutil.rmtree(os.path.join(root, "raw"))
            except FileNotFoundError:
                pass
        if self.force_redownload or self.force_reprocess:
            if not ask("Reprocess %s?" % root):
                raise AssertionError("Back off")
            try:
                shutil.rmtree(os.path.join(root, "processed"))
            except FileNotFoundError:
                pass

        print(
            "Loaded nx net with %d nodes and %d edges"
            % (self.nx_net.number_of_nodes(), self.nx_net.number_of_edges())
        )

        # Convert the network for pyg
        self.net, self.mapping = self.process_full_graph(
            self.nx_net, self.net_mapping, reprocess=self.force_reprocess
        )

        def normalize_func(data, means, stds, **kwargs):
            data.x = Scaler.zscore(data.x, mean=means["x"], std=stds["x"])
            data.edge_attr = Scaler.zscore(
                data.edge_attr, mean=means["edge_attr"], std=stds["edge_attr"]
            )
            return data

        if self.normalize_net:
            z_score_norm = Scaler.fit(
                [self.net], normalize=normalize_func, attrs=dict(x=1, y=1, edge_attr=0,)
            )
            self.net = z_score_norm(self.net)
            print("Done with normalizing")

        print(
            "Loaded network with %d nodes and %d edges"
            % (self.net.x.size(0), self.net.edge_index.size(1))
        )

        super().__init__(root, transform, pre_transform)

    @property
    def num_node_features(self):
        return len(self.node_feature_mapping)

    @property
    def num_edge_features(self):
        return len(self.edge_feature_mapping)

    @property
    def number_of_edges(self):
        try:
            return self.net.edge_index.size(1)
        except FileNotFoundError:
            return 0

    @property
    def number_of_nodes(self):
        try:
            return self.net.x.size(0)
        except FileNotFoundError:
            return 0

    def convert_to_tg_net(self, net, mapping):
        return self.nx_to_tg(net, node_mapping=mapping), None

    @classmethod
    def load_full_graph(cls, bbox=None, min_popularity=None):
        """
        Loads the transport network from disk
        """
        try:
            mapping, n = dict(), 0
            net = nx.read_gpickle(cls.net_file_path())
            for n_id in net.nodes:
                mapping[n_id] = n
                n += 1
            total = 0
            bins = defaultdict(int)
            for u, v, data in net.edges(data=True):
                total+=1
                bins[data["popularity"]] += 1
                data["current"] = 0
            # print("1", bins[1] / total)
            # print("20+", sum([v for k, v in bins.items() if k >= 20]) / total)
            if bbox:
                pass
            return net, mapping
        except Exception as e:
            raise ValueError(
                "Failed to read net graph, did you create it?: %s" % str(e)
            )

    def process_full_graph(self, net, mapping, cache=True, reprocess=False):
        if cache and not reprocess:
            try:
                cached_net = torch.load(
                    os.path.join(self.root, self.PROCESSED_STATIC_NET_FILENAME)
                )
                cached_mapping = None
                try:
                    with open(
                        os.path.join(
                            self.root, self.PROCESSED_STATIC_NET_MAPPING_FILENAME
                        ),
                        "rb",
                    ) as cm:
                        cached_mapping = pickle.load(cm)
                except FileNotFoundError:
                    pass
                return cached_net, cached_mapping
            except FileNotFoundError:
                pass
        print("processing static net")
        print(
            "About to convert static net with %d nodes and %d edges"
            % (net.number_of_nodes(), net.number_of_edges())
        )
        processed, mapping = self.convert_to_tg_net(net, mapping=mapping)
        print("done processing static net")
        if cache:
            try:
                os.makedirs(os.path.join(self.root))
            except FileExistsError:
                pass
            torch.save(
                processed, os.path.join(self.root, self.PROCESSED_STATIC_NET_FILENAME),
            )
            if mapping:
                with open(
                    os.path.join(self.root, self.PROCESSED_STATIC_NET_MAPPING_FILENAME),
                    "wb+",
                ) as cm:
                    pickle.dump(mapping, cm)
        return processed, mapping

    @classmethod
    def build_full_graph(cls, limit=None, min_occurences=None, save=True, plot=False):
        """
        Saves the transport network as a pickeled NX graph
        """
        r = retrieval.Retriever()
        full_net = r.retrieve_full_net(limit=limit)
        print("Filtering by occurence")
        _, edges = full_net
        occurences = np.array(list(edges.values()))
        print(np.amax(occurences), np.amin(occurences))
        HistogramPlot().plot(
            occurences, bins=100, filename="station_frequencies_%d" % (limit or "")
        )
        print("Loaded full graph")
        full_net_graph = NXTGBuilder.from_nodes_edges(
            *full_net, undirected=True
        ).build()
        print("Erasing state")
        for n, data in full_net_graph.nodes(data=True):
            data["delay"] = 0
        for u, v, data in full_net_graph.edges(data=True):
            data["delay"] = 0
            data["popularity"] = edges.get(
                (u, v), 0
            )  # How many times the edge was used
        print(
            "Built NX graph with %d nodes and %d edges"
            % (full_net_graph.number_of_nodes(), full_net_graph.number_of_edges())
        )
        if save:
            nx.write_gpickle(full_net_graph, cls.net_file_path())
            print("Saved NX graph")
        if plot:
            cls.plot_full_graph(show=True)
            print("Plotted NX graph")
        return full_net_graph

    @classmethod
    def plot_full_graph(cls, backend="nx", show=False):
        net, _ = cls.load_full_graph()
        plot_backend = GMTTransportPlot if backend != "nx" else NXTransportPlot
        plot_backend(
            net,
            filename="%s.pdf" % os.path.basename(cls.net_file_path()),
            show=show,
            subtitle=os.path.basename(cls.net_file_path()),
            node_labels=False,
            edge_labels=False,
            thickness=1,
            check=False,
            delay=False,
        ).plot()
        print("Plotted %s graph" % backend)

    def vlog(self, msg):
        if self.verbose:
            print(msg)

    def nx_to_tg(self, nx_g, verbose=True, node_mapping=None):
        n_nodes = nx_g.number_of_nodes()
        nodes = torch.zeros(n_nodes, self.num_node_features, dtype=torch.float)
        
        for n, data in nx_g.nodes(data=True):
            i = n if not node_mapping else node_mapping[n]
            for j, feature in enumerate(self.node_feature_mapping):
                try:
                    pos = ["lat", "lon"]
                    if feature in pos:
                        nodes[i][j] = data["pos"][pos.index(feature)]
                    else:
                        nodes[i][j] = data[feature]
                except (TypeError, ValueError, KeyError) as e:
                    print(
                        "nx_to_tg node attr error: ",
                        e,
                        data,
                        feature,
                        data.get(feature),
                    )
                    raise

        n_edges = nx_g.number_of_edges()
        edges = torch.zeros(n_edges, 2, dtype=torch.long)
        edge_attrs = torch.zeros(n_edges, self.num_edge_features, dtype=torch.long)

        for i, edge in enumerate(nx_g.edges):
            u, v = edge
            if node_mapping:
                u, v = node_mapping[u], node_mapping[v]
            edges[i][0], edges[i][1] = u, v
            for j, feature in enumerate(self.edge_feature_mapping):
                try:
                    edge_attrs[i][j] = nx_g.edges[edge][feature]
                except (TypeError, ValueError, KeyError) as e:
                    print(
                        "nx_to_tg edge attr error: ", e, feature, nx_g.edges.get(edge)
                    )
                    raise

        if n_edges > 0:
            edges = edges.t()
            if self.undirected:
                edges = torch_geometric.utils.to_undirected(edges)

        return Data(x=nodes, edge_attr=edge_attrs, edge_index=edges.contiguous())

    @property
    def total_timerange(self):
        start, end = datetime(2019, 2, 1), datetime(2019, 10, 2)
        start = start.replace(hour=0, minute=0, second=0)
        end = end.replace(hour=0, minute=0, second=0)
        return start, end

    @classmethod
    def net_file_path(cls):
        base_path = os.path.join(cls.base_path(), "../../graphs")
        assert os.path.exists(base_path)
        return os.path.join(base_path, "full_v1.gpickle")

    @classmethod
    def base_path(cls):
        return os.path.dirname(os.path.realpath(__file__))

    @property
    def raw_file_names(self):
        # The names of the files to find in the `self.raw_dir` folder in order to skip the download
        return ["data_%s" % self.format_timestamp(t) for t in self.timerange]

    @property
    def timerange(self):
        start, end = self.total_timerange
        limit = start + self.limit * self.batch
        t = start
        timerange = []
        while t <= end and t < limit:
            timerange.append(t)
            t = t + self.interval
        return timerange

    @staticmethod
    def format_timestamp(t):
        return t.strftime("%y_%m_%d_%H_%M_%S")

    @property
    def processed_file_names(self):
        return ["processed_%d.pt" % f for f in range(len(self.raw_file_names))]

    @property
    def merger(self):
        return AverageMerger

    def debug_plot(
        self,
        t,
        state,
        prefix=None,
        bbox=GERMANY_BBOX_1,
        labels=True,
        size=3,
        opaque=False,
    ):
        filename = "%s/%s%s%s.pdf" % (
            self.name,
            f"{prefix}/" if prefix else "",
            prefix or "",
            t if not isinstance(t, datetime) else self.format_timestamp(t),
        )
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
        # print("Plotted ", filename)

    def download(self):
        print("raw_file_names = %d" % len(self.raw_file_names))
        print("processed_file_names = %d" % len(self.processed_file_names))

        r = retrieval.Retriever()
        s = r.retrieve_stations(keep_ids=True)

        start, end = self.total_timerange
        self.vlog("Start: %s End: %s (%d days)" % (start, end, (end - start).days))

        def write(_g, _t):
            raw_path = os.path.join(self.raw_dir, "data_%s" % self.format_timestamp(_t))
            nx.write_gpickle(
                _g, raw_path,
            )

        def n_of_intervals(diff):
            return diff.total_seconds() / self.interval.total_seconds()

        t, b_t = start, start - self.padding
        b_c = 0
        while b_t < end + self.batch and b_c < self.limit:
            t_next = t + self.batch
            try:
                _start = time.time()

                # Get all running transports
                b_t_running = []
                try:
                    b_t_running = r.retrieve_transport(
                        timerange=dict(
                            start=t - self.padding, end=t + self.batch + self.padding
                        )
                    )
                except Exception as e:
                    # Even when receiving fails we need to generate output for the dataset
                    print("receive transport error: ", e)

                # Create bins for each timestep
                steps = int(n_of_intervals(t_next - t))
                t_running = [[] for _ in range(steps)]

                self.vlog(
                    "received batch running transports passed=%s"
                    % fmt_time(time.time() - _start)
                )
                self.vlog("scanning %d transports" % len(b_t_running))

                # Add transport to the respective bin
                for processed, tr in enumerate(b_t_running):
                    try:
                        live = sorted(
                            tr.get("live", []), key=lambda l: l.get("eventTime")
                        )
                        assert len(live) > 1
                        t_start, t_end = (
                            live[0].get("eventTime"),
                            live[-1].get("eventTime"),
                        )
                        min_i = max(0, int(n_of_intervals(t_start - t)))
                        max_i = min(
                            steps, max(0, 1 + int(math.ceil(n_of_intervals(t_end - t))))
                        )

                        if False:
                            self.vlog(
                                "assigning transport from %s to %s into %s to %s"
                                % (
                                    t_start,
                                    t_end,
                                    t + min_i * self.interval,
                                    t + (max_i - 1) * self.interval,
                                )
                            )

                        # Build graph and validate
                        tg = tgraph.TransportGraph(
                            tr, stations=s, use_unique_index=self.unique_index
                        )
                        valid, report = tg.validate()
                        if valid:
                            # Add transport to state for steps min_i to max_i
                            for i in range(min_i, max_i):
                                t_running[i].append(tg)
                    except Exception as e:
                        if not isinstance(e, AssertionError):
                            raise
                            print("Prepare state error: %s" % str(e))
                        pass
                    if processed % 1000 == 0:
                        print("%d transports scanned" % processed)

                self.vlog(
                    "assigned %d transports to their bins passed=%s"
                    % (len(b_t_running), fmt_time(time.time() - _start))
                )

                # Have the transports now and will merge
                for i in range(steps):
                    current = t + i * self.interval
                    merger = self.merger.create(
                        nx.DiGraph(), undirected=self.undirected
                    )
                    for tg in t_running[i]:
                        t_state_update = merger.add(
                            t + i * self.interval,
                            current + self.interval,
                            tg,
                            full_route=self.full_route,
                        )
                    t_state = merger.merge()
                    write(t_state, current)
                    if self.plot_download and t_state.number_of_edges() > 0:
                        self.debug_plot(current, t_state)
                    self.vlog(
                        "t=%s t_running=%d passed=%s"
                        % (current, len(t_running[i]), fmt_time(time.time() - _start))
                    )

                t = t + self.batch
            except Exception as e:
                raise

            b_c += 1
            b_t = start + b_c * self.batch
            self.vlog("b_t=%s b_c=%d end=%s limit=%d" % (b_t, b_c, end, self.limit))

    def process(self):
        # Handled by concrete dataset implementations
        raise NotImplementedError

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        try:
            return torch.load(
                os.path.join(self.processed_dir, self.processed_file_names[idx])
            )
        except FileNotFoundError:
            raise
