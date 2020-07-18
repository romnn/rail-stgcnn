from datetime import datetime
from pprint import pprint

import networkx as nx
import pytest

from cargonet.utils.geo import (
    center_between_points,
    center_between_stations,
    dist_m_v1,
    dist_m_v2,
)

MIN_PRECISION = 100  # 100m


@pytest.fixture
def known_locations():
    return [
        # London <-> Paris is ~340km
        (
            "LONDON<->PARIS",
            (51.4882243263235, -0.10986328125),
            (48.8502581997215, 2.3291015625),
            341220,
        ),
        # Hamburg <-> Munich is ~613km
        (
            "HAMBURG<->MUNICH",
            (53.5598889724546, 9.9755859375),
            (48.1367666796927, 11.57958984375),
            613738,
        ),
        # Wannsee <-> Griebnitzsee is ~4.6km
        (
            "WANNSEE<->GRIEBNITZSEE",
            (52.4212405457208, 13.1797313690186),
            (52.3944061012646, 13.1274604797363),
            4644,
        ),
    ]


def test_dist_m_v1(known_locations):
    for (_, p1, p2, dist_m) in known_locations:
        assert abs(dist_m_v1(p1, p2) - dist_m) < MIN_PRECISION


def test_dist_m_v2(known_locations):
    for (_, p1, p2, dist_m) in known_locations:
        assert abs(dist_m_v2(p1, p2) - dist_m) < MIN_PRECISION


def test_center_between_points(known_locations):
    for (_, p1, p2, dist_m) in known_locations:
        assert (
            abs(
                abs(dist_m_v2(center_between_points(p1, p2), p1))
                - abs(dist_m_v2(center_between_points(p1, p2), p2))
            )
            < 2_000
        )


@pytest.mark.skip(reason="Not essential")
def test_center_between_stations(known_locations):
    pass
