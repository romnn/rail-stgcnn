import datetime
import math
import os
import random
import traceback
from pprint import pprint

import click
import networkx as nx
from prettytable import PrettyTable

import cargonet.preprocessing.datalake.retrieval as retrieval
import cargonet.preprocessing.validation.constraints as constraints
from cargonet.constants.numbers import MININT
from cargonet.preprocessing.graphs.builders import NXTGBuilder
from cargonet.preprocessing.graphs.utils import calculate_delays_for_route


class TransportGraph:
    def __init__(
        self, transport, stations=None, isolated_nodes=False, use_unique_index=False
    ):
        self.transport = transport
        self.station_net = stations
        self.transport_id = transport.get("transport_id")
        self.use_unique_index = use_unique_index
        self.route = self._parse_transport(transport, isolated_nodes=isolated_nodes)

        # Cache generated graphs
        # Lazily generate graphs for different backends
        self._nx_combined_route = None
        self._nx_planned_route = None
        self._nx_actual_route = None
        self._tg_planned_route = None
        self._tg_actual_route = None

    @property
    def nx_planned_route(self):
        if self._nx_planned_route is None:
            self._nx_planned_route = NXTGBuilder.from_nodes_edges(
                *self.map_and_build_graph(
                    self.transport_id,
                    self.route,
                    use_unique_index=self.use_unique_index,
                )
            ).build()
        return self._nx_planned_route

    @property
    def nx_actual_route(self):
        if self._nx_actual_route is None:
            self._nx_actual_route = NXTGBuilder.from_nodes_edges(
                *self.map_and_build_graph(
                    self.transport_id,
                    self.route,
                    "updates",
                    use_unique_index=self.use_unique_index,
                )
            ).build()
        return self._nx_actual_route

    @property
    def actual_route(self):
        live_route = []
        for p in self.route:
            if len(p["updates"]) > 0:
                live_station = p.copy()
                live_station["stations"] = live_station["updates"]
                live_route.append(live_station)
        return live_route

    def _mapped_station(self, s_id, scale=2.0):
        """
        Maps stations based on given network of stations 
        """
        return self.station_net.get(s_id)

    def _validate(
        self,
        check_bad_edges=True,
        check_live_coverage=True,
        check_end_station_mismatch=True,
        check_missing_positions=True,
        check_delays=True,
        check_actual=False,
        max_kmph=200,
        max_delay_delta=60 * 10,  # 5 hour differences between two stations
    ):
        """
        Validates the nx transport graph based on selected constraints
        """
        try:
            bad_edges_planned = bad_edges_actual = covered = complete = None
            bad_nodes_planned = bad_nodes_actual = bad_delays = exceeding_delays = None

            if check_bad_edges:
                _, bad_edges_planned = constraints.validate_edges(
                    self.nx_planned_route, max_kmph=max_kmph
                )
                if check_actual:
                    _, bad_edges_actual = constraints.validate_edges(
                        self.nx_actual_route, max_kmph=max_kmph
                    )

            if check_live_coverage:
                covered = constraints.validate_coverage(self.route)

            if check_delays:
                bad_delays, exceeding_delays = constraints.validate_delays(
                    self, self.route, max_delay_delta=max_delay_delta
                )

            if check_end_station_mismatch:
                complete, warning = constraints.validate_completes_end_station(
                    self.transport, self.route
                )

            if check_missing_positions:
                bad_nodes_planned, bad_nodes_actual = [], []
                for n, data in self.nx_actual_route.nodes(data=True):
                    if data.get("pos") is None:
                        bad_nodes_actual.append(n)

                for n, data in self.nx_planned_route.nodes(data=True):
                    if data.get("pos") is None:
                        bad_nodes_planned.append(n)
        except Exception as e:
            raise ValueError("Unable to build:", e)

        return dict(
            bad_edges_planned=bad_edges_planned or [],
            bad_edges_actual=bad_edges_actual or [],
            bad_nodes_planned=bad_nodes_planned or [],
            bad_nodes_actual=bad_nodes_actual or [],
            bad_delays=bad_delays or [],
            exceeding_delays=exceeding_delays or [],
            covered=True if covered is None else covered,
            complete=True if complete is None else complete,
            warning=warning,
        )

    def validate(self):
        """
        Validates the transport graph
        """
        try:
            report = self._validate()
        except Exception as e:
            # raise
            # traceback.print_exc()
            return False, dict()
        valid = all(
            [
                len(report["bad_nodes_planned"]) < 1,
                len(report["bad_nodes_actual"]) < 1,
                len(report["bad_edges_planned"]) < 1,
                len(report["bad_edges_actual"]) < 1,
                len(report["bad_delays"]) < 1,
                len(report["exceeding_delays"]) < 1,
                report["covered"] is True,
                report["complete"] is True,
            ]
        )
        return valid, report

    @property
    def valid(self):
        """
        Strict validation of the transport graph
        """
        return self.validate()[0]

    def save(self, filename):
        """
        Saves the nx graph as GML
        """
        g = self.tg.copy()
        for n, data in g.nodes(data=True):
            # Decompose position because tuples are not supported
            data["lat"] = data["pos"][0]
            data["lon"] = data["pos"][1]
            # Must delete or stringify the arrival and departure datetimes for saving
            data["arrivalTime"] = str(data["arrivalTime"])
            data["departureTime"] = str(data["departureTime"])

        for n, v, data in g.edges(data=True):
            # Must delete or stringify the duration timedelta for saving
            data["duration"] = str(data["duration"])

        base_path = os.path.dirname(os.path.realpath(__file__))
        base_path = os.path.join(base_path, "../../../graphs")
        assert os.path.exists(base_path)
        nx.write_gml(g, os.path.join(base_path, "%s.gml" % filename))

    def print_route(self, interpolate_missing=True):
        """
        Prints the route of the transport
        """
        x = PrettyTable()
        head = [
            "stationId",
            "arrivalTime",
            "plannedArrivalTime",
            "departureTime",
            "plannedDepartureTime",
            "delay",
            "trainSectionId",
            "ingestionTime",
            "endStationId",
        ]
        x.field_names = head + [
            "adv. endStationId",
            "has live",
            "delay delta",
            "rel delay",
            "rel delay %",
        ]
        try:
            self.route, delays, nans = calculate_delays_for_route(
                self.route, interpolate_missing=interpolate_missing
            )
        except ValueError as e:
            # pprint(self.route)
            print("Failed to compute delays: ", str(e))
            return
        for s in range(len(self.route)):
            x.add_row(
                [self.route[s].get(h) for h in head]
                + [
                    self.transport.get("endStationId"),
                    "Yes" if len(self.route[s].get("updates", [])) > 0 else "No",
                    delays["delayDelta"][s],
                    delays["delayRel"][s],
                    delays["delayRelPercent"][s],
                ]
            )
        print(x)
        print("Total: %d stations" % len(self.route))

    def map_and_build_graph(
        self,
        transport_id,
        route,
        variant="stations",
        isolated_nodes=False,
        combine=True,
        use_unique_index=False,
    ):
        station_id_node_mapping = dict()
        n_c = 0
        last_s_id = None
        transport_edges = set()
        
        try:
            route, df, _ = calculate_delays_for_route(route)
        except ValueError as e:
            pass

        route = sorted(route, key=lambda s: s.get("arrivalTime"))
        self.route = route

        for si, s in enumerate(route):
            if variant == "stations" and len(s.get("stations", [])) < 1:
                continue
            elif variant == "updates" and len(s.get("updates", [])) < 1:
                continue

            s_id = s.get("stationId")
            m_s_id = self._mapped_station(s_id)
            if m_s_id is None:
                # No such node
                continue

            station_id_node_mapping[s_id] = m_s_id
            unique_index = "%s_%s" % (transport_id, s.get("stationId"))

            station_id_node_mapping[s_id].update(
                dict(
                    mappedIndex=station_id_node_mapping[s_id]["index"],
                    index=unique_index if use_unique_index else s.get("stationId"),
                    transportId=transport_id,
                    arrivalTime=s.get("arrivalTime"),
                    departureTime=s.get("departureTime"),
                    plannedArrivalTime=s.get("plannedArrivalTime"),
                    plannedDepartureTime=s.get("plannedDepartureTime"),
                    delay=df["delayAbs"][si],
                    delayRel=df["delayRel"][si],
                    delayRelPercent=df["delayRelPercent"][si],
                    delayDelta=df["delayDelta"][si],
                )
            )

            if isolated_nodes:
                # Add node even when there will be no edge
                station_id_node_mapping[s_id]["index"] = n_c if mapping else s_id
                n_c += 1 if mapping else 0

            if last_s_id:
                if last_s_id == s_id:
                    continue
                # Found an edge, but check if nodes exist first
                for _s_id in [s_id, last_s_id]:
                    m = self._mapped_station(_s_id)
                    if not m:
                        continue
                    station_id_node_mapping[_s_id]["index"] = m["index"]
                    if station_id_node_mapping[_s_id]["index"] is None:
                        station_id_node_mapping[_s_id]["index"] = (
                            n_c if mapping else _s_id
                        )
                        n_c += 1 if mapping else 0

                if False:  # unique_index:
                    transport_edges.add((last_s_id, s_id))
                else:
                    mi, mj = (
                        station_id_node_mapping[last_s_id],
                        station_id_node_mapping[s_id],
                    )
                    transport_edges.add((mi["index"], mj["index"]))
            last_s_id = s_id

        return station_id_node_mapping, transport_edges

    def _parse_transport(
        self, transport, isolated_nodes=False, interpolate_delay=False
    ):
        # Sort the stations of this transport by planned event time
        stations = transport.get("planned")
        stations = sorted(stations, key=lambda s: s.get("plannedEventTime"))

        updates = transport.get("live")
        updates = sorted(updates, key=lambda s: s.get("eventTime"))

        assert len(stations) > 0
        assert len(updates) > 0

        # Group by stationId
        def station(s):
            return dict(
                stations=[s],
                updates=[],
                stationId=s.get("stationId"),
                plannedArrivalTime=s.get("plannedEventTime"),
                plannedDepartureTime=s.get("plannedEventTime"),
                trainSectionId=s.get("trainSectionId"),
                ingestionTime=s.get("ingestionTime"),
                endStationId=s.get("endStationId"),
            )

        # Map planned data to route
        route = [station(stations[0])]
        for s in stations[1:]:
            if s.get("stationId") == route[-1].get("stationId"):
                # Update times
                route[-1]["stations"].append(s)
                route[-1]["endStationId"] = s.get("endStationId")
                if route[-1].get("plannedArrivalTime") > s.get("plannedEventTime"):
                    route[-1]["plannedArrivalTime"] = s.get("plannedEventTime")
                if route[-1].get("plannedDepartureTime") < s.get("plannedEventTime"):
                    route[-1]["plannedDepartureTime"] = s.get("plannedEventTime")
            else:
                route.append(station(s))

        # Map live update data to route
        for u in updates:
            inserted = False
            for i in range(len(route)):
                if route[i]["stationId"] == u.get("stationId"):
                    route[i]["updates"].append(u)
                    inserted = True
            if not inserted:
                i = 0
                eventTime = u.get("eventTime")
                plannedTime = route[i].get("plannedArrivalTime")

                # Best effort to put at the right place
                while (
                    i < len(route) - 1
                    and (None not in (eventTime, plannedTime))
                    and eventTime > plannedTime
                ):
                    i += 1
                    eventTime = u.get("eventTime")
                    plannedTime = route[i].get("plannedArrivalTime")

                s = station(
                    dict(
                        stationId=u.get("stationId"),
                        plannedEventTime=u.get("eventTime"),
                        ingestionTime=u.get("ingestionTime"),
                    )
                )
                s["stations"] = []
                s["updates"].append(u)
                route = route[:i] + [s] + route[i:]

        # Map combined data to route
        last_observed_delay = 0
        for station in route:
            planned = station["stations"]
            live = [
                u for u in station["updates"] if u.get("eventTime") and u.get("status")
            ]

            arrivals = [u.get("eventTime") for u in live if u.get("status") in [1, 3]]
            departures = [u.get("eventTime") for u in live if u.get("status") in [2, 4]]
            drivethroughs = [u.get("eventTime") for u in live if u.get("status") in [5]]

            arrival = (
                (None if len(drivethroughs) < 1 else drivethroughs[0])
                if len(arrivals) < 1
                else arrivals[0]
            )
            departure = (
                (None if len(drivethroughs) < 1 else drivethroughs[-1])
                if len(departures) < 1
                else departures[-1]
            )
            plannedArrival = station.get("plannedArrivalTime")
            plannedDeparture = station.get("plannedDepartureTime")

            delays = [u for u in station["updates"] if u.get("delay") > MININT]
            delay = None
            if len(delays) > 0:
                delay = delays[-1].get("delay")
            elif departure is not None:
                # Attempt to calculate delay
                delay = (departure - plannedDeparture).total_seconds() / 60.0
            elif interpolate_delay:
                delay = last_observed_delay

            station.update(
                dict(delay=delay, arrivalTime=arrival, departureTime=departure,)
            )
            last_observed_delay = delay
        return route
