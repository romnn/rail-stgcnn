import tempfile
import unittest.mock as mock
from contextlib import contextmanager
from datetime import datetime
from pprint import pprint

import networkx as nx

from cargonet.dataset.avgdelayv1 import EdgeAverageDelayDatasetV1
from cargonet.preprocessing.datalake.retrieval import Retriever


@contextmanager
def average_delay_dataset_v1_harness(
    data_provider=None,
    time_range=(datetime(2019, 2, 1), datetime(2019, 2, 3)),
    node_count=10,
    ds_options=None,
):
    """
    Mocked test harness for the v1 average delay dataset
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch(
            "cargonet.dataset.avgdelayv1.EdgeAverageDelayDatasetV1.load_full_graph"
        ) as mocked_load:
            with mock.patch(
                "cargonet.dataset.avgdelayv1.EdgeAverageDelayDatasetV1.timerange",
                new_callable=mock.PropertyMock,
            ) as mocked_timerange:
                with mock.patch(
                    "cargonet.preprocessing.datalake.retrieval.Retriever.retrieve_transport"
                ) as mocked_retreive:

                    def mock_data_for_timerange(timerange, **kwargs):
                        return (
                            []
                            if not data_provider
                            else data_provider(timerange, **kwargs)
                        )

                    mocked_retreive.side_effect = mock_data_for_timerange

                    mocked_timerange.return_value = time_range

                    mock_net = nx.Graph()
                    node_count = 10
                    nodes = [
                        (i, dict(stationId=i, pos=(12 + i * 0.05, 12 + i * 0.05)))
                        for i in range(node_count)
                    ]
                    edges = [(i, i + 1) for i in range(node_count - 1)]
                    if node_count >= 10:
                        edges += [(1, 5), (5, 7), (5, 9)]  # Some skip connections
                    mock_net.add_nodes_from(nodes)
                    mock_net.add_edges_from(edges)
                    mock_mapping = dict()  # TODO
                    mocked_load.return_value = mock_net, mock_mapping

                    # for n, data in mock_net.nodes(data=True):
                    #     print(n, data)

                    # Build the dataset in a temporary directory
                    dataset = EdgeAverageDelayDatasetV1(
                        root=tmpdir, name="mock-dataset", **(ds_options or dict())
                    )

                    timerange_secs = (time_range[1] - time_range[0]).total_seconds()
                    assert (
                        len(dataset) == (timerange_secs / dataset.interval.seconds) + 1
                    )

                    yield (dataset, mock_net, mocked_timerange)
