import unittest.mock as mock
from contextlib import contextmanager
from datetime import datetime
from pprint import pprint

import mongomock
import networkx as nx
import pytest

import cargonet.preprocessing.datalake.retrieval as retrieval
from cargonet.dataset.dataset import RailDataset
from cargonet.preprocessing.datalake.cache import Cache


def insert_mock_stations(db, stations):
    for s in stations:
        s["_id"] = db.eletastations.insert_one(s).inserted_id
    return stations


def insert_mock_sections(db, sections):
    for s in sections:
        s["_id"] = db.trainsectiondata.insert_one(s).inserted_id
    return sections


def insert_mock_plans(db, plans):
    for p in plans:
        p["_id"] = db.plannedtraindata.insert_one(p).inserted_id
    return plans


def insert_mock_updates(db, updates):
    for u in updates:
        u["_id"] = db.livetraindata.insert_one(u).inserted_id
    return updates


@contextmanager
def _mocked_retrieval_database(db):
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
            yield


@contextmanager
def _mocked_cache_database(db):
    with mock.patch(
        "cargonet.preprocessing.datalake.cache.Cache.connect",
        new_callable=mock.PropertyMock,
    ) as mocked_connect:
        mocked_connect.return_value = None
        with mock.patch(
            "cargonet.preprocessing.datalake.cache.Cache.db",
            new_callable=mock.PropertyMock,
        ) as mocked_db:
            mocked_db.return_value = db
            yield


@contextmanager
def mocked_database(db):
    with _mocked_retrieval_database(db):
        with _mocked_cache_database(db):
            yield


def test_retrieve_stations():
    """
    Test if stations are correctly loaded from the datalake
    """
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

    for keep_ids in [True, False]:
        with mocked_database(db):
            r = retrieval.Retriever()
            s = r.retrieve_stations(keep_ids=keep_ids)

            # Make sure bad stations have been ignored
            assert len(s) == len(good_stations)

            for i in range(1, len(good_stations) + 1):
                # Check positions
                assert s[i]["pos"] == (i, i)

                # Check indices
                assert s[i]["index"] == (i if keep_ids else None)

            # Check country codes
            assert 0 <= s[1]["country"] < len(retrieval.Retriever.COUNTRY_CODES)
            assert 0 <= s[2]["country"] < len(retrieval.Retriever.COUNTRY_CODES)
            assert s[3]["country"] == -1


@pytest.fixture
def transport1():
    return dict(sections=[dict()], plans=[dict()], updates=[dict()],)


def test_retrieve_transport_ids(transport1):
    """
    Test if transports ids are correctly loaded from the datalake
    """
    sections = plans = updates = []
    db = mongomock.MongoClient().replay
    for t in [transport1]:
        sections += insert_mock_sections(db, t["sections"])
        plans += insert_mock_plans(db, t["plans"])
        updates += insert_mock_updates(db, t["updates"])

    with mocked_database(db):
        # Create transports cache before the query
        Cache().cache_transports_v1()
        r = retrieval.Retriever()
        ids = list(r.retrieve_transport_ids())
        assert len(ids) == 1
        pprint(ids)


@pytest.mark.skip(reason="WIP")
def test_retrieve_transports():
    """
    Test if transports are correctly loaded from the datalake
    """
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

    for keep_ids in [True, False]:
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
                r = retrieval.Retriever()
                s = r.retrieve_stations(keep_ids=keep_ids)

                # Make sure bad stations have been ignored
                assert len(s) == len(good_stations)
                pprint(s)

                for i in range(1, len(good_stations) + 1):
                    # Check positions
                    assert s[i]["pos"] == (i, i)

                    # Check indices
                    assert s[i]["index"] == (i if keep_ids else None)

                # Check country codes
                assert 0 <= s[1]["country"] < len(retrieval.Retriever.COUNTRY_CODES)
                assert 0 <= s[2]["country"] < len(retrieval.Retriever.COUNTRY_CODES)
                assert s[3]["country"] == -1
