import unittest.mock as mock
from datetime import datetime
from pprint import pprint

import mongomock
import networkx as nx
import pytest

import cargonet.preprocessing.datalake.retrieval
from cargonet.dataset.dataset import RailDataset


def insert_mock_stations(db, stations):
    for s in stations:
        s["_id"] = db.eletastations.insert_one(s).inserted_id
    return stations


def test_build_full_graph():
    db = mongomock.MongoClient().replay
    good_stations = [
        dict(stationId=1, latitude=1, longitude=1, imId=1, ruId=1, countryCode="DE"),
        dict(stationId=2, latitude=2, longitude=2, imId=2, ruId=2, countryCode="IT"),
        dict(stationId=3, latitude=3, longitude=3),
    ]
    bad_stations = [
        dict(latitude=1, longitude=1, imId=1, ruId=1, countryCode="DE"),
        dict(stationId=4, longitude=2, imId=2, ruId=2, countryCode="IT"),
    ]
    stations = insert_mock_stations(db, good_stations + bad_stations)

    with mock.patch(
        "cargonet.preprocessing.datalake.retrieval.Retriever.connect",
        new_callable=mock.PropertyMock,
    ) as mocked_connect:
        mocked_connect.return_value = None
        with mock.patch(
            "cargonet.preprocessing.datalake.retrieval.Retriever.db",
            new_callable=mock.PropertyMock,
        ) as mocked_db:
            mocked_db.return_value = db
            net = RailDataset.build_full_graph(limit=10, save=False, plot=False)
            assert net.number_of_nodes() == len(good_stations)
            for i in range(1, len(good_stations) + 1):
                assert net.nodes[i]["pos"] == (i, i)
            # Mocking example
            # for obj in objects:
            #     stored_obj = collection.find_one({"_id": obj["_id"]})
            #     stored_obj["votes"] -= 1
            #     assert (
            #         stored_obj == obj
            #     )  # by comparing all fields we make sure only votes changed
