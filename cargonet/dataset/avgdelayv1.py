import os.path
import shutil
import statistics
from collections import defaultdict
from datetime import datetime, timedelta

import networkx as nx
import torch
import torch_geometric
from torch_geometric.data import Dataset

import cargonet.preprocessing.datalake.retrieval as retrieval
import cargonet.preprocessing.graphs.tgraph as tgraph
from cargonet.dataset.dataset import RailDataset
from cargonet.utils.link2node import link2node
from cargonet.utils.pdf import concat_pdfs


class EdgeAverageDelayDatasetV1(RailDataset):

    node_feature_mapping = ["stationId", "imId", "country"]
    edge_feature_mapping = ["delay", "distance", "current"]

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
        running_avg_lookbehind_steps=1,
    ):
        self.running_avg_lookbehind_steps = max(1, running_avg_lookbehind_steps)
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
        )
        assert self.undirected

    def download(self):
        super().download()

    @property
    def processed_file_names(self):
        return [
            "processed_%d.pt" % f
            for f in range(self.running_avg_lookbehind_steps, len(self.raw_paths))
        ]

    @staticmethod
    def aggregate(acc, states):
        # Start with the full network and iteratively apply the considered states
        _acc = acc.copy()
        _acc = _acc.to_undirected()
        avg = defaultdict(list)

        if False:
            for _, data in acc.nodes(data=True):
                print("Acc node features:", data.keys())
                break
            for u, v, data in acc.edges(data=True):
                print("Acc edge features:", data.keys())
                break
            for _, data in states[0].nodes(data=True):
                print("State node features:", data.keys())
                break
            for u, v, data in states[0].edges(data=True):
                print("State edge features:", data.keys())
                break

        for s in states:
            s = s.to_undirected()
            for u, v, data in s.edges(data=True):
                avg[(u, v)].append(data["delay"])

        # Apply running averages
        for edge, delays in avg.items():
            delay = statistics.mean(delays)
            try:
                _acc.edges[edge]["delay"] = delay / 100
                _acc.edges[edge]["current"] = len(delays)
            except KeyError:
                pass
                # print("KEY ERROR!!")

        return _acc

    def extract_features(
        self,
        nx_g,
        edge_features=None,
        node_features=None,
        verbose=True,
        node_mapping=None,
    ):
        """
        Edges are important here
        """
        edge_features = edge_features or []
        node_features = node_features or []

        n_edges = nx_g.number_of_edges()
        edge_attrs = torch.zeros(n_edges, len(edge_features), dtype=torch.float)

        for i, edge in enumerate(nx_g.edges):
            u, v = edge
            if node_mapping:
                u, v = node_mapping[u], node_mapping[v]
            edges[i][0], edges[i][1] = u, v
            for j, feature in enumerate(edge_features):
                try:
                    edge_attrs[i][j] = nx_g.edges[u, v][feature]
                except (TypeError, ValueError, KeyError) as e:
                    print(
                        "extract_features edge attr error: ",
                        e,
                        feature,
                        nx_g.edges[edge],
                    )

        if verbose:
            delay = edge_attrs[:, self.edge_feature_mapping.index("delay")]
            print("delay: min=%d max=%d" % (delay.min().item(), delay.max().item()))

        return torch_geometric.data.Data(edge_attr=edge_attrs)

    def process(self):
        states_count = len(self.raw_paths)
        total_states = range(self.running_avg_lookbehind_steps, states_count)
        assert len(self.processed_file_names) == len(total_states)
        for i in total_states:
            # Read transport state at time step t and some previous steps
            self.vlog(
                "Processing t[%d:%d] (%d/%d, %d states)"
                % (
                    i - self.running_avg_lookbehind_steps,
                    i,
                    i,
                    states_count,
                    self.running_avg_lookbehind_steps,
                )
            )
            states = [
                nx.read_gpickle(raw_path)
                for raw_path in self.raw_paths[
                    i - self.running_avg_lookbehind_steps : i
                ]
            ]

            # Enforce undirected
            assert all([isinstance(s, nx.Graph) for s in states])

            combined = self.aggregate(self.nx_net, states)

            # Plot combined graph
            if self.plot_processing:
                self.debug_plot(
                    i, combined, prefix="combined", size=1, labels=False, opaque=True
                )

            # Extract important features and convert nx graph to a tg graph
            data = self.extract_features(
                combined,
                node_mapping=self.net_mapping,
                edge_features=self.edge_feature_mapping,
                node_features=self.node_feature_mapping,
                verbose=self.verbose,
            )

            # Apply filters and transformations
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(
                data,
                os.path.join(
                    self.processed_dir,
                    self.processed_file_names[i - self.running_avg_lookbehind_steps],
                ),
            )


class NodeAverageDelayDatasetV1(EdgeAverageDelayDatasetV1):

    node_feature_mapping = EdgeAverageDelayDatasetV1.edge_feature_mapping
    edge_feature_mapping = EdgeAverageDelayDatasetV1.node_feature_mapping

    def extract_features(
        self,
        nx_g,
        edge_features=None,
        node_features=None,
        verbose=True,
        node_mapping=None,
    ):
        """
        Nodes are important here
        """
        edge_features = edge_features or []
        node_features = node_features or []

        # Assume the data is given with delay as edge attributes
        n_nodes = nx_g.number_of_edges()
        nodes = torch.zeros(n_nodes, len(node_features), dtype=torch.float)
        for u, v, data in nx_g.edges(data=True):
            for j, feature in enumerate(node_features):
                try:
                    n = self.mapping[(u, v)]
                    nodes[n][j] = data[feature]
                except (TypeError, ValueError, KeyError) as e:
                    raise
                    print(
                        "extract_features node attr error:",
                        e,
                        data,
                        feature,
                        data[feature],
                        type(data[feature]),
                    )
                    
        if verbose:
            delay = nodes[:, node_features.index("delay")]
            print(
                "delay: mean=%d min=%d max=%d"
                % (delay.mean().item(), delay.min().item(), delay.max().item())
            )

        return torch_geometric.data.Data(x=nodes)

    def process(self):
        super().process()

    def download(self):
        super().download()

    def convert_to_tg_net(self, net):
        """
        Convert full net to tg and set the mapping
        """
        net, mapping = link2node(net, self.mapping)
        return self.nx_to_tg(net), mapping


def build_dataset(limit, plot_download, plot_processing, rebuild, reprocess, verbose):
    dataset_name = "average-delay-dataset-v1"

    base_path = os.path.dirname(os.path.realpath(__file__))
    base_dataset_path = os.path.join(base_path, "../../datasets")
    assert os.path.exists(base_dataset_path)
    dataset_path = os.path.join(base_dataset_path, dataset_name)

    try:
        print("Loading dataset")
        dataset = NodeAverageDelayDatasetV1(
            root=dataset_path,
            name=dataset_name,
            limit=limit,
            plot_download=plot_download,
            plot_processing=plot_processing,
            force_redownload=rebuild,
            force_reprocess=reprocess,
            verbose=verbose,
        )
    except Exception as e:
        raise
        print("loading dataset error: ", e)
