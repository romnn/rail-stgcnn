import os.path
import time

import click
import networkx as nx

from cargonet.dataset.dataset import RailDataset
from cargonet.preprocessing.datalake.retrieval import Retriever
from cargonet.utils.formatting import fmt_time


def fix_countries(net, limit=None):
    new_net = nx.Graph()
    nl = el = limit
    for n, data in net.nodes(data=True):
        if not nl or 0 < nl:
            try:
                print(data)
                data["country"] = Retriever.COUNTRY_CODES.index(data["country"])
            except (KeyError, ValueError):
                data["country"] = -1
            new_net.add_node(n, **data)
            if nl:
                nl -= 1
    for u, v, data in net.edges(data=True):
        if not el or 0 < el:
            new_net.add_edge(u, v, **data)
            if el:
                el -= 1
    return new_net
