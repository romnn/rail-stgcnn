from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import pytest

from cargonet.dataset.dataset import RailDataset
from cargonet.utils.link2node import link2node

VERBOSE = False


def test_link2node():
    g = nx.Graph()
    g.add_nodes_from(range(1, 6))
    g.add_edges_from(
        [
            (1, 2, {"delay": 12}),
            (2, 3, {"delay": 11}),
            (3, 4, {"delay": 9}),
            (4, 1, {"delay": 2}),
            (1, 5, {"delay": 5}),
        ]
    )
    gg, mapping = link2node(g)

    # Assert there is a node for every source edge and features are preserved
    assert gg.number_of_nodes() == g.number_of_edges()
    assert all(
        [
            g.edges[u, v] == gg.nodes[mapping[(u, v)]]
            for u, v, data in g.edges(data=True)
        ]
    )

    if VERBOSE:
        nx.draw_spring(g)
        plt.show()

        nx.draw_spring(gg)
        plt.show()


def test_link2node_full_graph():
    g, mapping = RailDataset.load_full_graph()
    g = g.to_undirected()
    orig_edges = g.number_of_edges()
    orig_nodes = g.number_of_nodes()

    gg, l2n_mapping = link2node(g)
    gg = gg.to_undirected()
    edges = gg.number_of_edges()
    nodes = gg.number_of_nodes()

    assert orig_edges == nodes
