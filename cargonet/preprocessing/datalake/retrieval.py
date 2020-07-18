import datetime
from collections import defaultdict
from pprint import pprint

import click
import pymongo

import cargonet.preprocessing.graphs.tgraph as tgraph


class Retriever:

    NOT_NULL = {"$gt": -9223372036854775808}

    # list(db.eletastations.aggregate([{"$group": {"_id": "$countryCode"}}]))
    COUNTRY_CODES = [
        "RS",
        "PT",
        "SI",
        "AT",
        "FR",
        "SK",
        "RO",
        "TR",
        "BG",
        "CH",
        "DE",
        "UA",
        "BE",
        "HR",
        "IT",
        "SE",
        "CZ",
        "NL",
        "LU",
        "BY",
        "UK",
        "LT",
        "PL",
        "HU",
        "DK",
        "ES",
    ]

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

    def retrieve_transport_ids(self, limit=None):
        cursor = self.db.eletav1.find({"_id": self.NOT_NULL})
        if limit:
            cursor = cursor.limit(limit)
        for t in cursor:
            yield t.get("_id")

    def retrieve_stations(self, keep_ids=False, extent=None):
        station_id_node_mapping, c = dict(), 0

        scope = dict()
        if extent:
            scope = {
                "latitude": {"$gte": extent["lat"][0], "$lte": extent["lat"][1]},
                "longitude": {"$gte": extent["lon"][0], "$lte": extent["lon"][1]},
            }

        # Get the stations nodes
        for station in self.db.eletastations.find(
            {"stationId": self.NOT_NULL, **scope}
        ):
            try:
                s_id = station.get("stationId")
                s_lat = station.get("latitude")
                s_lon = station.get("longitude")
                try:
                    s_country = self.COUNTRY_CODES.index(station.get("countryCode"))
                except ValueError:
                    s_country = None
                s_im = station.get("imId")
                s_name = station.get("stationName")
                assert -9223372036854775808 < s_id
                assert s_lat > 0
                assert s_lon > 0
                station_id_node_mapping[s_id] = dict(
                    index=int(s_id) if keep_ids else None,
                    stationId=int(s_id),
                    stationName=s_name,
                    pos=(s_lat, s_lon),
                    # These values are considered optional
                    country=int(s_country or -1),
                    imId=int(s_im or -1),
                )
            except Exception as e:
                # Ignore bad stations
                # pprint(station)
                # print("bad station error: ", e)
                pass
        return station_id_node_mapping

    def retrieve_station(self, station_id):
        return self.db.eletastations.find_one({"stationId": int(station_id)})

    def query_stations(self, name):
        return self.db.eletastations.find({"stationName": {"$regex": name}})

    def retrieve_full_net(self, limit=None, mapping=False, isolated_nodes=False):
        net_station_id_node_mapping, net_edges = dict(), defaultdict(lambda: 0)
        s = self.retrieve_stations(keep_ids=not mapping)
        print("Retrieved %d stations" % len(s))
        all_transports = self.retrieve_transport_ids(limit=limit)
        scanned = 0
        for tid in all_transports:
            t = self.retrieve_transport(transport_id=tid)[0]
            tg = tgraph.TransportGraph(t, stations=s, keep_ids=True)
            if tg.valid:
                station_id_node_mapping, edges = tg.map_and_build_graph(
                    tid, tg.route, isolated_nodes=isolated_nodes
                )
                if mapping:
                    existing = len(net_station_id_node_mapping)
                    for i, station_mapping in enumerate(station_id_node_mapping):
                        station_id_node_mapping["index"] = existing + i

                net_station_id_node_mapping.update(station_id_node_mapping)
                for e in edges:
                    net_edges[e] = net_edges[e] + 1  # 0 if no entry
            scanned += 1
            if scanned % 1_000 == 0:
                print("Scanned %d" % scanned)
        # print("=> TOTAL: ", len(net_station_id_node_mapping), len(net_edges))
        return net_station_id_node_mapping, net_edges

    def count_transports(self):
        counts = self.db.trainsectiondata.aggregate(
            [{"$group": {"_id": "$euroRailRunId", "count": {"$sum": 1}}}]
        )
        return list(counts)[0].get("count")

    def get_transports_timerange(self):
        test = self.db.plannedtraindata.aggregate(
            [
                {
                    "$group": {
                        "_id": None,
                        "max": {"$max": "$plannedDepartureTimeStartStation"},
                        "min": {"$min": "$plannedArrivalTimeEndStation"},
                    }
                }
            ]
        )
        results = list(test)
        start, end = results[0].get("min"), results[0].get("max")
        return start, end

    def retrieve_transport(
        self, transport_id=None, timerange=None, raw_sections=False, cached=True
    ):
        if not transport_id and not timerange:
            raise ValueError("Need a transport id or a timerange")

        if transport_id and not isinstance(transport_id, list):
            transport_id = [transport_id]

        def query(_transport_id=None, _timerange=None, _cached=False):
            must_match = dict()
            if _transport_id:
                must_match["euroRailRunId"] = {
                    "$in": [int(tid) for tid in _transport_id]
                }
            if _timerange:
                must_match["$or"] = [
                    {
                        "plannedArrivalTimeEndStation": {
                            "$gte": _timerange.get("start"),
                            "$lt": _timerange.get("end"),
                        }
                    },
                    {
                        "plannedDepartureTimeStartStation": {
                            "$gte": _timerange.get("start"),
                            "$lt": _timerange.get("end"),
                        }
                    },
                ]
            if not _cached:
                agg = [
                    {"$match": must_match},
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
                ]
                cursor = self.db.trainsectiondata.aggregate(agg)
            else:
                cursor = self.db.transportscachev1.find(must_match)
            return list(cursor)

        if timerange and not transport_id:
            # Get the transport ids for this timerange first
            sections = query(_timerange=timerange, _cached=cached)
            transport_id = list(set([s.get("euroRailRunId") for s in sections]))
            timerange = None

        # Get transports by transport_id
        if len(transport_id) < 1:
            raise ValueError("No transport found")
        result = query(_transport_id=transport_id, _timerange=timerange, _cached=cached)
        if len(result) < 1:
            raise ValueError("No transport found")

        # Now group to transports with same euroRailRunId
        transports = defaultdict(list)
        for r in result:
            transports[r.get("euroRailRunId")].append(r)

        retrieved = []
        for tid, transport in transports.items():

            # Get the destination from the last section
            sections = sorted(
                [t for t in transport if len(t.get("planned", list())) > 0],
                key=lambda t: t.get("plannedArrivalTime")
                or datetime.datetime.now(),  # Do not care
            )
            if not len(sections) > 0:
                print("Failed to find a valid section with plan data")
                continue

            last_section = sections[-1]

            def sorted_plan(p):
                return sorted(p, key=lambda s: s.get("plannedEventTime"))

            def latest_ingestion(p):
                # Filter only the latest ingestion
                latest_ingestion = dict()
                for s in p:
                    ingestion, section = s.get("ingestionTime"), s.get("trainSectionId")
                    if not section in latest_ingestion:
                        latest_ingestion[section] = ingestion
                    elif latest_ingestion[section] < ingestion:
                        latest_ingestion[section] = ingestion
                return [
                    s
                    for s in p
                    if s.get("ingestionTime")
                    >= latest_ingestion[s.get("trainSectionId")]
                ]

            single_section, single_section_ingestion = None, None
            for i in range(1, len(sections)):
                sections[i]["planned"] = sorted_plan(
                    latest_ingestion(sections[i]["planned"])
                )
                plan = sections[i]["planned"]
                ingestion = sections[i].get("ingestionTime")
                if all(
                    [
                        last_section.get(k) in [p.get("stationId") for p in plan]
                        for k in ["endStationId", "startStationId"]
                    ]
                ):
                    if (
                        None in [single_section, single_section_ingestion]
                        or single_section_ingestion < ingestion
                    ):
                        single_section = sections[i]
                        single_section_ingestion = ingestion

            if single_section and False:
                sections = [single_section]
            else:
                # Have to cut to pack overlapping sections
                # Three Options:
                # - A: be restrictive and only cut following sections from the head
                # - B: be eager and cut the current station from the tail
                # - C: choose depending on prod date and default to ? when they are the same
                option = "A"
                if True:
                    for p in range(1, len(sections)):
                        sec, prev = sections[p], sections[p - 1]
                        sec_plans, prev_plans = (
                            sorted_plan(latest_ingestion(sec["planned"])),
                            sorted_plan(latest_ingestion(prev["planned"])),
                        )

                        def cut_head(target, other):
                            return [
                                s
                                for s in target
                                if s.get("plannedEventTime")
                                >= other[-1].get("plannedEventTime")
                            ]

                        def cut_tail(target, other):
                            return [
                                s
                                for s in target
                                if s.get("plannedEventTime")
                                <= other[0].get("plannedEventTime")
                            ]

                        if len(sec_plans) < 1 or len(prev_plans) < 1:
                            break

                        if option == "A":
                            sec_plans = cut_head(sec_plans, prev_plans)
                        elif option == "B":
                            prev_plans = cut_tail(prev_plans, sec_plans)
                        else:
                            if sec_plans[0].get("ingestionTime") >= prev_plans[-1].get(
                                "ingestionTime"
                            ):
                                prev_plans = cut_tail(prev_plans, sec_plans)
                            else:
                                sec_plans = cut_head(sec_plans, prev_plans)

                        sections[p]["planned"] = sec_plans
                        sections[p - 1]["planned"] = prev_plans

            if raw_sections:
                return dict(
                    sections=sections,
                    id=tid,
                )

            planned, live = [], []
            for sec in sections:
                planned = planned + sec["planned"]
                live = live + sec["live"]

            retrieved.append(
                dict(
                    live=live,
                    planned=planned,
                    transport_id=tid,
                    endStationId=last_section.get("endStationId"),
                    plannedArrivalTimeEndStation=last_section.get(
                        "plannedArrivalTimeEndStation"
                    ),
                )
            )
        return retrieved
