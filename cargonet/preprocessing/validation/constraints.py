from pprint import pprint

import networkx as nx

from cargonet.preprocessing.graphs.utils import calculate_delays_for_route


def validate_edges(g, max_kmph=200):
    """
    Make sure all edges can possibly be traversed with reasonable speed
    """
    good_edges, bad_edges = [], []
    for u, v, data in g.edges(data=True):
        dur, dist_km = data.get("duration"), data.get("distance")
        # Distances < 1km are not precise enough
        if None in [dur, dist_km] or dist_km < 1.0:
            good_edges.append((u, v))
            continue
        # At least one minute
        dur_hrs = max(abs(dur.total_seconds()), 60) / 60.0 / 60.0
        # Consider bad edge if train speed would have to be faster than max_kmph to pass edge
        # print("Required speed: %f (%f km in %f hrs)" % (dist_km/dur_hrs, dist_km, dur_hrs))
        if dist_km / dur_hrs > max_kmph:
            bad_edges.append((u, v))
        else:
            good_edges.append((u, v))
    return good_edges, bad_edges


def check_for_unrelated_live_stations(route):
    unrelated = []
    all_station_ids = [s.get("stationId") for s in route]
    for u in updates:
        if u.get("stationId") not in all_station_ids:
            unrelated.append(u.get("stationId"))
    return unrelated


def validate_completes_end_station(transport, route):
    """
    Sanity check that the route is complete
    """
    expected_end_station = transport.get("endStationId")
    expected_arrival_time = transport.get("plannedArrivalTimeEndStation")
    end_station = route[-1].get("stationId")
    arrival_time = route[-1].get("arrivalTime")

    complete = False
    for s in reversed(route):
        if s.get("stationId") == expected_end_station:
            complete = True

    if arrival_time and expected_arrival_time and arrival_time >= expected_arrival_time:
        complete = True

    if not complete:
        return (
            False,
            "Expected End station %s did not match %s or any previous"
            % (expected_end_station, end_station),
        )
    return True, None


def validate_coverage(
    route,
    metric="node_coverage",
    max_gap_sec=3 * 60 * 60,
    min_node_coverage_percent=0.75,
):
    """
    Make sure there is sufficient coverage of the planned route with live data
    """
    if metric == "time":
        last_update, live = None, []
        for r in route:
            live += [l for l in r.get("updates", []) if l.get("eventTime")]
        for l in sorted(live, key=lambda l: l.get("eventTime")):
            if last_update is None:
                last_update = l.get("eventTime")
            if (l.get("eventTime") - last_update).total_seconds() > max_gap_sec:
                return False
            last_update = l.get("eventTime")
        if last_update is None:
            # Did not see any updates
            return False

    elif metric == "node_coverage":
        # Do not allow less than 75% node coverage with live data
        live_stations, planned_stations = [], []
        for r in route:
            live_stations += [
                l.get("stationId") for l in r.get("updates", []) if l.get("stationId")
            ]
            planned_stations += [
                p.get("stationId") for p in r.get("stations", []) if p.get("stationId")
            ]
        intersection = set(planned_stations).intersection(set(live_stations))
        if (
            float(len(intersection)) / float(len(set(planned_stations)))
            < min_node_coverage_percent
        ):
            return False
    return True


def validate_delays(
    tg, route, max_delta_min=60, max_delay_delta=None,
):
    bad_delays, exceeding_delays = [], []
    try:
        _, delays, nans = calculate_delays_for_route(route)
    except ValueError as e:
        return [str(e)], []
    if nans.mean() >= 5:
        return ["More than 5 NaN"], []
    for s in range(len(route)):
        delta = delays["delayDelta"][s]
        if abs(delta) > max_delta_min:
            bad_delays.append(delta)

    if max_delay_delta:
        live_edge_delays = list(
            nx.get_edge_attributes(tg.nx_actual_route, "delay").values()
        )
        live_node_delays = list(
            nx.get_node_attributes(tg.nx_actual_route, "delay").values()
        )

        planned_edge_delays = list(
            nx.get_edge_attributes(tg.nx_actual_route, "delay").values()
        )
        planned_node_delays = list(
            nx.get_node_attributes(tg.nx_actual_route, "delay").values()
        )

        delays = (
            live_edge_delays
            + live_node_delays
            + planned_edge_delays
            + planned_node_delays
        )

        for node_delays in [live_node_delays, planned_node_delays]:
            exceeding_delays += [
                abs(node_delays[ni] - node_delays[ni + 1])
                for ni in range(len(node_delays) - 1)
            ]

        # bad node delta delays
        exceeding_delays = [d for d in exceeding_delays if d > max_delay_delta]

        # bad edge delays
        exceeding_delays += [
            d for d in live_edge_delays + planned_edge_delays if d > max_delay_delta
        ]

    return bad_delays, exceeding_delays
