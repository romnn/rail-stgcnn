import time
from itertools import combinations
from pprint import pprint

from prettytable import PrettyTable

import cargonet.preprocessing.datalake.retrieval as retrieval
from cargonet.dataset.dataset import RailDataset


def debug_station(station_id):
    r = retrieval.Retriever()
    try:
        stations = [dict(stationId=int(station_id))]
    except ValueError:
        # Query by name
        stations = list(r.query_stations(station_id))

    net, _ = RailDataset.load_full_graph()

    x = PrettyTable()
    x.field_names = [
        "stationId",
        "stationName",
        "Neighbors",
    ]
    for s in stations:
        s_id = s.get("stationId")
        try:
            node = net.nodes[s_id]
            neighbors = net.neighbors(s_id)
            x.add_row(
                [
                    s_id,
                    s.get("stationName"),
                    "\n".join([str(s) for s in list(neighbors)]),
                ]
            )
        except KeyError:
            x.add_row([s_id, "n.a.", "n.a."])
    print(x)
