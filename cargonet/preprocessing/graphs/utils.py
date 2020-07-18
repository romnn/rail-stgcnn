import math
from datetime import timedelta
from pprint import pprint

import pandas as pd


def to_min(d):
    return int(math.ceil(d.total_seconds() / 60.0))


def calculate_delays_for_route(route, interpolate_missing=True):
    delays = [calculate_delay_for_station(route, s) for s in range(len(route))]
    data = dict(
        delayAbs=[d[0] for d in delays],
        delayRel=[d[1] for d in delays],
        delayRelPercent=[d[2] for d in delays],
        delayDelta=[d[3] for d in delays],
    )
    cols = ["delayAbs", "delayRel", "delayRelPercent", "delayDelta"]
    df = pd.DataFrame(data, columns=cols)
    nans = df.isna().sum(axis=0)

    if interpolate_missing:
        for c in cols:
            df[c] = df[c].interpolate()
        for c in cols:
            df[c] = df[c].bfill()
        for c in cols:
            df[c] = df[c].ffill()
        try:
            df = df.astype("int32")
        except Exception:
            raise ValueError("No delays available")
        for si in range(len(route)):
            # Fill any missing times using the interpolated delay
            abs_delay = int(df["delayAbs"][si])
            route[si]["delay"] = abs_delay

            # Infer missing arrival time
            if not route[si].get("arrivalTime"):
                route[si]["arrivalTime"] = route[si].get(
                    "plannedArrivalTime"
                ) + timedelta(minutes=abs_delay)

        for si in range(len(route)):
            # Infer missing departure time
            rel_delay = int(df["delayRel"][si])
            if not route[si].get("departureTime"):
                nsi = min(si + 1, len(route) - 1)
                dur = route[nsi].get("plannedArrivalTime") - route[si].get(
                    "plannedDepartureTime"
                )
                route[si]["departureTime"] = (
                    route[nsi].get("arrivalTime") - dur - timedelta(minutes=rel_delay)
                )

    return route, df, nans


def calculate_delay_for_station(route, station_index):
    if station_index < 0:
        raise ValueError("Bad index")
    try:
        initial_delay = to_min(
            route[0].get("departureTime") - route[0].get("plannedDepartureTime")
        )
    except (TypeError, KeyError) as e:
        initial_delay = 0

    try:
        adv_delay = int(route[station_index].get("delay"))
    except (TypeError, KeyError) as e:
        return None, None, None, None

    delta = rel_delay = rel_delay_percent = abs_delay = None
    if station_index == 0:
        rel_delay = abs_delay = initial_delay
    else:
        try:
            dur = route[station_index].get("arrivalTime") - route[
                max(0, station_index - 1)
            ].get("departureTime")
            pdur = route[station_index].get("plannedArrivalTime") - route[
                max(0, station_index - 1)
            ].get("plannedDepartureTime")
            rel_delay = to_min(dur - pdur)
            rel_delay_percent = (
                max(1, dur.total_seconds()) / max(1, pdur.total_seconds()) * 100
            )
        except Exception as e:
            pass
        try:
            abs_delay = to_min(
                route[station_index].get("departureTime")
                - route[station_index].get("plannedDepartureTime")
            )
            delta = abs_delay - adv_delay
        except Exception:
            pass
    if not abs_delay:
        abs_delay = adv_delay
        delta = None
    return abs_delay, rel_delay, rel_delay_percent, delta
