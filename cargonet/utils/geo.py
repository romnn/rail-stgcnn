import math

import geopy.distance
from pyrocko import model, orthodrome

GERMANY_BBOX_1 = ((47.25, 5.44), (55, 14.9))


def dist_m_v1(coord1, coord2):
    return geopy.distance.distance(coord1, coord2).m


def dist_m_v2(coord1, coord2):
    return orthodrome.distance_accurate50m(
        model.Event(lat=coord1[0], lon=coord1[1]),
        model.Event(lat=coord2[0], lon=coord2[1]),
    )


def center_between_points(coord1, coord2):
    lat = (coord1[0] + coord2[0]) / 2.0
    lon = (coord1[1] + coord2[1]) / 2.0
    return (lat, lon)


def center_between_stations(station_id1, station_id2):
    import cargonet.preprocessing.datalake.retrieval as retrieval

    r = retrieval.Retriever()
    s1, s2 = r.retrieve_station(station_id1), r.retrieve_station(station_id2)
    p1 = (s1.get("latitude"), s1.get("longitude"))
    p2 = (s2.get("latitude"), s2.get("longitude"))
    return center_between_points(p1, p2)
