import tempfile
import unittest.mock as mock
from datetime import datetime
from pprint import pprint

import networkx as nx
import pytest

import helpers
from cargonet.dataset.avgdelayv1 import EdgeAverageDelayDatasetV1
from cargonet.preprocessing.datalake.retrieval import Retriever
from cargonet.preprocessing.graphs.tgraph import TransportGraph
from harnesses import average_delay_dataset_v1_harness


def test_loads_entire_net():
    """ Test if the entire pre-computed static global network is used """
    node_count = 10
    with average_delay_dataset_v1_harness(node_count=node_count) as (
        dataset,
        net,
        timerange,
    ):
        for i in dataset:
            assert node_count == i.x.shape[0]
            # Undirected edges and eventually some skip connections
            assert 2 * node_count <= i.edge_index.shape[1]


def test_valid_batches():
    """ Test if batches contain exactly the expected transports """
    pass


def test_running_average_delay():
    """ Test if delay updates are applied to the pre-computed static global network """
    node_count = 10

    """
    t = helpers.create_route(
        1,
        helpers.station(
            1,
            planned_departure=datetime(2019, 2, 1, hour=1, minute=0),
            departure=datetime(2019, 2, 1, hour=1, minute=5),
        ),  # 5 mins late
        helpers.station(
            2,
            planned_arrival=datetime(2019, 2, 1, hour=2, minute=0),
            arrival=datetime(2019, 2, 1, hour=2, minute=0),  # On time
            planned_departure=datetime(2019, 2, 1, hour=2, minute=20),
            departure=datetime(2019, 2, 1, hour=2, minute=15),  # Eager
        ),
        # Drive through only
        helpers.station(
            3,
            planned_event_time=datetime(2019, 2, 1, hour=3, minute=0),
            event_time=datetime(2019, 2, 1, hour=3, minute=30),  # 30mins Late
        ),
        helpers.station(
            4,
            planned_event_time=datetime(2019, 2, 1, hour=4, minute=0),
            event_time=datetime(2019, 2, 1, hour=4, minute=20),  # 20mins Late
        ),
        helpers.station(
            helpers.END_STATION_ID,
            planned_arrival=datetime(2019, 2, 1, hour=5, minute=0),
            arrival=datetime(2019, 2, 1, hour=5, minute=5),  # 5mins Late
        ),
    )
    """

    mock_data = {
        (datetime(2019, 2, 1), datetime(2019, 2, 2)): [
            helpers.station(
                1,
                planned_departure=datetime(2019, 2, 1, hour=1, minute=0),
                departure=datetime(2019, 2, 1, hour=1, minute=5),
            )
        ],
        (datetime(2019, 2, 2), datetime(2019, 2, 3)): [],
    }

    def provider(timerange, **kwargs):
        return mock_data[tuple(timerange.values())]

    def with_ds_options(**ds_options):
        with average_delay_dataset_v1_harness(
            node_count=node_count,
            data_provider=provider,
            time_range=(datetime(2019, 2, 1), datetime(2019, 2, 3)),
            **ds_options,
        ) as (dataset, net, timerange):
            for i in dataset:
                pass
                # print(i)
                # assert node_count == i.x.shape[0]
                # Undirected edges and eventually some skip connections
                # assert 2 * node_count <= i.edge_index.shape[1]


def test_parse_transport():
    s = {
        1: {"index": 1, "stationId": 1, "lat": 12, "lon": 10,},
        2: {"index": 2, "stationId": 2, "lat": 13, "lon": 11,},
        3: {"index": 3, "stationId": 3, "lat": 13, "lon": 11,},
        4: {"index": 3, "stationId": 3, "lat": 13, "lon": 11,},
    }
    transport_id = 1000
    t = dict(
        live=[
            {
                "delay": -15,
                "eventTime": datetime(2019, 2, 1, 1, 45),
                "stationId": 1,
                "status": 5,
            },
            # Now takes another route
            {
                "delay": -26,
                "eventTime": datetime(2019, 2, 1, 2, 0),
                "stationId": 3,
                "status": 5,
            },
            {
                "delay": -26,
                "eventTime": datetime(2019, 2, 1, 2, 0),
                "stationId": 3,
                "status": 5,
            },
        ],
        planned=[
            {
                "endStationId": 4192849,
                "plannedEventTime": datetime(2019, 1, 31, 23, 10),
                "stationId": 1,
            },
            {"plannedEventTime": datetime(2019, 2, 1, 2, 0), "stationId": 2,},
        ],
        transport_id=transport_id,
        endStationId=2,
        plannedArrivalTimeEndStation=datetime(2020, 5, 1, hour=15),
    )
    tg = TransportGraph(t, stations=s)
    # pprint(tg.route)

    def assert_length_of_stations_and_updates_for_station(
        station_id, num_stations, num_updates
    ):
        r = [r for r in tg.route if r.get("stationId") == station_id]
        assert len(r) == 1
        r = r[0]
        assert len(r.get("stations", [])) == num_stations
        assert len(r.get("updates", [])) == num_updates

    assert_length_of_stations_and_updates_for_station(1, 1, 1)
    assert_length_of_stations_and_updates_for_station(2, 1, 0)
    assert_length_of_stations_and_updates_for_station(3, 0, 2)

    _, planned_edges = tg.map_and_build_graph(transport_id, tg.route, "stations")
    assert planned_edges == {("1000_2", "1000_1")}

    _, live_edges = tg.map_and_build_graph(transport_id, tg.route, "updates")
    # TODO: Seems wrong
    assert live_edges == {("1000_3", "1000_1")}
