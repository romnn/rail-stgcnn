from datetime import datetime

import pytest

from cargonet.preprocessing.validation.constraints import (
    validate_coverage,
    validate_edges,
)


def test_validate_coverage():
    long_gap_route = [
        dict(
            stations=[
                dict(plannedEventTime=datetime(2020, 2, 1, hour=12), stationId=1)
            ],
            # 10 minute delay
            updates=[
                dict(eventTime=datetime(2020, 2, 1, hour=12, minute=10), stationId=1)
            ],
        ),
        dict(
            stations=[
                dict(plannedEventTime=datetime(2020, 2, 1, hour=14), stationId=2)
            ],
            updates=[],
        ),
        dict(
            stations=[
                dict(plannedEventTime=datetime(2020, 2, 1, hour=16), stationId=3)
            ],
            # 1h 10 minute delay
            updates=[
                dict(eventTime=datetime(2020, 2, 1, hour=17, minute=10), stationId=3)
            ],
        ),
    ]
    assert not validate_coverage(long_gap_route, metric="time")
    long_gap_route[1]["updates"] = [
        dict(eventTime=datetime(2020, 2, 1, hour=15), stationId=2)
    ]
    assert validate_coverage(long_gap_route, metric="time")
    assert validate_coverage(long_gap_route, metric="node_coverage")

    missing_live_data_route = [
        dict(
            stations=[
                dict(plannedEventTime=datetime(2020, 2, 1, hour=12), stationId=1)
            ],
            updates=[],
        ),
        dict(
            stations=[
                dict(plannedEventTime=datetime(2020, 2, 1, hour=14), stationId=2)
            ],
            updates=[],
        ),
    ]
    assert not validate_coverage(missing_live_data_route, metric="time")
    assert not validate_coverage(missing_live_data_route, metric="node_coverage")
