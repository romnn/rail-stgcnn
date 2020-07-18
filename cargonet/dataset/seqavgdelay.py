import os

import torch
from torch_geometric.data import Data

from cargonet.dataset.avgdelayv1 import NodeAverageDelayDatasetV1
from cargonet.dataset.dataset import RailDataset


class SequencedAverageDelayDatasetV1(RailDataset):

    node_feature_mapping = NodeAverageDelayDatasetV1.node_feature_mapping
    edge_feature_mapping = NodeAverageDelayDatasetV1.edge_feature_mapping

    def __init__(
        self,
        base,
        root,
        name=None,
        transform=None,
        pre_transform=None,
        force_reprocess=False,
        limit=1,
        sequence_length=10,
        sequence_interval=1,
        prediction_length=1,
        prediction_interval=1,
    ):
        self.name = name
        self.limit = limit
        self.sequence_length = sequence_length
        self.sequence_interval = sequence_interval
        self.prediction_length = prediction_length
        self.prediction_interval = prediction_interval
        self.dataset = base
        super().__init__(
            root,
            transform=transform,
            pre_transform=pre_transform,
            force_reprocess=force_reprocess,
        )

    def download(self):
        pass

    @property
    def num_node_features(self):
        return self.dataset.num_node_features

    @property
    def num_edge_features(self):
        return self.dataset.num_edge_features

    @property
    def number_of_edges(self):
        return self.dataset.number_of_edges

    @property
    def number_of_nodes(self):
        return self.dataset.number_of_nodes

    @property
    def raw_file_names(self):
        return []

    @property
    def timerange(self):
        sl, pl = (
            self.sequence_interval * self.sequence_length,
            self.prediction_interval * self.prediction_length,
        )
        return super().timerange[: len(self.dataset.processed_file_names) - sl - pl]

    @property
    def prediction_timerange(self):
        sl, pl = (
            self.sequence_interval * self.sequence_length,
            self.prediction_interval * self.prediction_length,
        )
        return [t + (sl + pl) * self.dataset.interval for t in self.timerange]

    @property
    def processed_file_names(self):
        sl, pl = (
            self.sequence_interval * self.sequence_length,
            self.prediction_interval * self.prediction_length,
        )
        return [
            "processed_%d.pt" % f
            for f in range(0, len(self.dataset.processed_file_names) - sl - pl)
        ]

    def convert_to_tg_net(self, net):
        return self.dataset.convert_to_tg_net(net)

    def process(self):
        for t, processed in enumerate(self.processed_file_names):
            # Collect sequence
            tt = t + self.sequence_length
            sequence = [
                self.dataset[t + s * self.sequence_interval]
                for s in range(self.sequence_length)
            ]

            # Collect predictions
            last_seq_index = t + self.sequence_length * self.sequence_interval
            predictions = [
                self.dataset[last_seq_index + p * self.prediction_interval]
                for p in range(self.prediction_length)
            ]

            x_seq = torch.empty(
                (
                    self.sequence_length,
                    self.dataset.number_of_nodes,
                    self.dataset.num_node_features,
                ),
                dtype=torch.float,
            )
            for si, s in enumerate(sequence):
                x_seq[si] = s.x

            y_pred = torch.empty(
                (
                    self.prediction_length,
                    self.dataset.number_of_nodes,
                    self.dataset.num_node_features,
                ),
                dtype=torch.float,
            )
            for pi, p in enumerate(predictions):
                y_pred[pi] = p.x

            # Only x is a sequence now
            data = Data(
                x=x_seq,
                y=y_pred,
            )

            # Apply filters and transformations
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(
                data, os.path.join(self.processed_dir, self.processed_file_names[t])
            )
