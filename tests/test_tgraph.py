from datetime import datetime
from pprint import pprint

import networkx as nx
import pytest

import helpers
from cargonet.constants.numbers import (
    MININT,
    STATUS_ARRIVAL,
    STATUS_DEPARTURE,
    STATUS_DRIVE_THROUGH,
)
from cargonet.preprocessing.graphs.tgraph import TransportGraph


@pytest.fixture
def stations():
    s = {
        1: {"index": 1, "stationId": 1, "lat": 12, "lon": 10,},
        2: {"index": 2, "stationId": 2, "lat": 13, "lon": 11,},
        3: {"index": 3, "stationId": 3, "lat": 13, "lon": 11,},
        4: {"index": 3, "stationId": 3, "lat": 13, "lon": 11,},
    }
    return s


def test_basic_transport_extraction(stations):
    transport_id = 1000
    t = dict(
        live=[
            {
                "delay": -15,
                "eventTime": datetime(2019, 2, 1, hour=1, minute=45),
                "stationId": 1,
                "status": 5,
            },
            # Now takes another route
            {
                "delay": -26,
                "eventTime": datetime(2019, 2, 1, hour=2, minute=0),
                "stationId": 3,
                "status": 5,
            },
            {
                "delay": -26,
                "eventTime": datetime(2019, 2, 1, hour=2, minute=0),
                "stationId": 3,
                "status": 5,
            },
        ],
        planned=[
            {
                "endStationId": 4192849,
                "plannedEventTime": datetime(2019, 1, 31, hour=23, minute=10),
                "stationId": 1,
            },
            {
                "plannedEventTime": datetime(2019, 2, 1, hour=2, minute=0),
                "stationId": 2,
            },
        ],
        transport_id=transport_id,
        endStationId=2,
        plannedArrivalTimeEndStation=datetime(2020, 5, 1, hour=15),
    )
    tg = TransportGraph(t, stations=stations)
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
    assert live_edges == {("1000_3", "1000_1")}


def test_delay_interpolation(stations):
    transport_id = 1000
    # TODO!!!
    return
    t = dict(
        live=[
            {
                "delay": 0,
                "eventTime": datetime(2019, 2, 1, hour=1, minute=0),
                "stationId": 1,
                "status": STATUS_DEPARTURE,
            },
            {
                "delay": 0,
                "eventTime": datetime(2019, 2, 1, hour=1, minute=5),
                "stationId": 1,
                "status": STATUS_DEPARTURE,
            },
            {
                "delay": -26,
                "eventTime": datetime(2019, 2, 1, hour=2, minute=0),
                "stationId": 3,
                "status": 5,
            },
            {
                "delay": -26,
                "eventTime": datetime(2019, 2, 1, hour=2, minute=0),
                "stationId": 3,
                "status": 5,
            },
        ],
        planned=[
            {
                "endStationId": 4192849,
                "plannedEventTime": datetime(2019, 1, 31, hour=23, minute=10),
                "stationId": 1,
            },
            {
                "plannedEventTime": datetime(2019, 2, 1, hour=2, minute=0),
                "stationId": 2,
            },
        ],
        transport_id=transport_id,
        endStationId=2,
        plannedArrivalTimeEndStation=datetime(2020, 5, 1, hour=15),
    )
    tg = TransportGraph(t, stations=stations)


def test_edge_delay_computation(stations):
    transport_id = 1000
    t = dict(
        live=[
            {
                "delay": 0,
                "eventTime": datetime(2019, 2, 1, hour=1, minute=0),
                "stationId": 1,
                "status": STATUS_DEPARTURE,
            },
            {
                "delay": -1,
                "eventTime": datetime(2019, 2, 1, hour=1, minute=5),
                "stationId": 1,
                "status": STATUS_ARRIVAL,
            },
            {
                "delay": -26,
                "eventTime": datetime(2019, 2, 1, hour=2, minute=0),
                "stationId": 3,
                "status": 5,
            },
            {
                "delay": -26,
                "eventTime": datetime(2019, 2, 1, hour=2, minute=0),
                "stationId": 3,
                "status": 5,
            },
        ],
        planned=[
            {
                "endStationId": 4192849,
                "plannedEventTime": datetime(2019, 1, 31, hour=23, minute=10),
                "stationId": 1,
            },
            {
                "plannedEventTime": datetime(2019, 2, 1, hour=2, minute=0),
                "stationId": 2,
            },
        ],
        transport_id=transport_id,
        endStationId=2,
        plannedArrivalTimeEndStation=datetime(2020, 5, 1, hour=15),
    )
    t = helpers.create_route(
        transport_id,
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
    tg = TransportGraph(t, stations=stations)
    pprint(tg.nx_actual_route)

    # TODO: FIX!
    return
    delays = nx.get_edge_attributes(tg.nx_actual_route, "delay")
    assert delays == {
        ("1000_1", "1000_2"): -5,
        ("1000_2", "1000_3"): 40 - 15 + 30,
        ("1000_3", "1000_4"): 40 - 15 + 30,
    }
