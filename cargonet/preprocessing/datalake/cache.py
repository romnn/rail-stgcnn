import datetime
import time
from pprint import pprint

import pymongo

from cargonet.utils.formatting import fmt_time


class Cache:
    def __init__(self):
        self.db = None
        self.connect()

    def __init__(self):
        self._db = None

    def connect(self):
        client = pymongo.MongoClient("mongodb://root:example@localhost:27018")
        return client.replay

    @property
    def db(self):
        if self._db is None:
            self._db = self.connect()
        return self._db

    def create_indices(self):
        indices = [
            (
                self.db.plannedtraindata,
                "plannedDepartureTimeStartStation",
                pymongo.ASCENDING,
            ),
            (
                self.db.plannedtraindata,
                "plannedArrivalTimeEndStation",
                pymongo.ASCENDING,
            ),
            (self.db.plannedtraindata, "trainId", pymongo.ASCENDING),
            (self.db.plannedtraindata, "stationId", pymongo.ASCENDING),
            (self.db.plannedtraindata, "trainSectionId", pymongo.ASCENDING),
            # trainsectiondata
            (self.db.trainsectiondata, "euroRailRunId", pymongo.ASCENDING),
            (self.db.trainsectiondata, "trainSectionId", pymongo.ASCENDING),
            (self.db.trainsectiondata, "trainId", pymongo.ASCENDING),
            (self.db.trainsectiondata, "startStationId", pymongo.ASCENDING),
            (self.db.trainsectiondata, "endStationId", pymongo.ASCENDING),
            (
                self.db.trainsectiondata,
                "plannedArrivalTimeEndStation",
                pymongo.ASCENDING,
            ),
            (
                self.db.trainsectiondata,
                "plannedDepartureTimeStartStation",
                pymongo.ASCENDING,
            ),
            # livetraindata
            (self.db.livetraindata, "stationId", pymongo.ASCENDING),
            (self.db.livetraindata, "trainId", pymongo.ASCENDING),
            (self.db.livetraindata, "trainSectionId", pymongo.ASCENDING),
            (self.db.livetraindata, "delay", pymongo.ASCENDING),
            # eletav1
            (self.db.eletav1, "stations", pymongo.ASCENDING),
            (self.db.eletav1, "trainId", pymongo.ASCENDING),
            (self.db.eletav1, "stations.live.eventTime", pymongo.ASCENDING),
            (self.db.eletav1, "stations.plannedEventTime", pymongo.ASCENDING),
            # transportscachev1
            (
                self.db.transportscachev1,
                "plannedArrivalTimeEndStation",
                pymongo.ASCENDING,
            ),
            (
                self.db.transportscachev1,
                "plannedDepartureTimeStartStation",
                pymongo.ASCENDING,
            ),
            (self.db.transportscachev1, "euroRailRunId", pymongo.ASCENDING),
            # (db.trainsectiondata, "trainId", pymongo.ASCENDING)
        ]
        for i in indices:
            try:
                i[0].create_index([(i[1], i[2])])
            except Exception as e:
                print(e)

    def cache_transports_v1(self):
        agg = [
            {
                "$lookup": {
                    "from": "plannedtraindata",
                    "let": {"trainSectionId": "$trainSectionId"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        {
                                            "$eq": [
                                                "$trainSectionId",
                                                "$$trainSectionId",
                                            ]
                                        },
                                    ]
                                }
                            }
                        }
                    ],
                    "as": "planned",
                }
            },
            {
                "$lookup": {
                    "from": "livetraindata",
                    "let": {"trainSectionId": "$trainSectionId"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        {
                                            "$eq": [
                                                "$trainSectionId",
                                                "$$trainSectionId",
                                            ]
                                        },
                                    ]
                                }
                            }
                        }
                    ],
                    "as": "live",
                }
            },
            {"$out": "transportscachev1"},
        ]
        agg = [
            {
                "$lookup": {
                    "from": "plannedtraindata",
                    "localField": "trainSectionId",
                    "foreignField": "trainSectionId",
                    "as": "planned",
                }
            },
            {
                "$lookup": {
                    "from": "livetraindata",
                    "localField": "trainSectionId",
                    "foreignField": "trainSectionId",
                    "as": "live",
                }
            },
            {"$out": "transportscachev1"},
        ]
        self.db.trainsectiondata.aggregate(agg)


def build_caches(transports, indices):
    cache = Cache()
    start = time.time()
    c = 0

    def run(fn):
        print("Running %s" % fn.__name__)
        _start = time.time()
        fn()
        nonlocal c
        c += 1
        _end = time.time()
        print("Completed %s in %s" % (fn.__name__, fmt_time(_end - _start)))

    # Select caches to build
    if transports:
        run(cache.cache_transports_v1)
    if indices:
        run(cache.create_indices)

    end = time.time()
    print("Built %d caches in %s" % (c, fmt_time(end - start)))
