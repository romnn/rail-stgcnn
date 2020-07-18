from abc import ABC, abstractmethod

import networkx as nx

import cargonet.utils.geo as geo


class TransportGraphBuilder(ABC):
    def __init__(self, result):
        self.result = result

    @classmethod
    @abstractmethod
    def from_nodes_edges(
        cls, station_id_node_mapping, transport_edges, undirected=False
    ):
        pass

    def build(self):
        return self.result


class NXTGBuilder(TransportGraphBuilder):
    @classmethod
    def from_nodes_edges(
        cls, station_id_node_mapping, transport_edges, undirected=False
    ):
        """
        Builds networkX transport graph
        """
        tg = nx.Graph() if undirected else nx.DiGraph()
        for s_id, mapping in station_id_node_mapping.items():
            index = mapping["index"]
            if index is None:
                continue
            node_attrs = mapping.copy()
            node_attrs["stationId"] = s_id
            tg.add_node(index, **node_attrs)

        tg.add_edges_from(transport_edges)
        for u, v in tg.edges:
            # Add distance edge feature
            p1, p2 = tg.nodes[u].get("pos"), tg.nodes[v].get("pos")
            if None not in [p1, p2]:
                tg.edges[u, v]["distance"] = geo.dist_m_v2(p1, p2) / 1000.0

            # TODO: Add more delay metrics
            tg.edges[u, v]["delay"] = tg.nodes[v].get("delayRelPercent")  # or delayRel

            # Add planned duration edge feature
            p_d, p_a = (
                tg.nodes[u].get("plannedDepartureTime"),
                tg.nodes[v].get("plannedArrivalTime"),
            )
            if None not in [p_a, p_d]:
                tg.edges[u, v]["plannedDuration"] = p_a - p_d

            # Add real duration edge feature
            r_d, r_a = (
                tg.nodes[u].get("departureTime"),
                tg.nodes[v].get("arrivalTime"),
            )
            if None not in [r_a, r_d]:
                tg.edges[u, v]["duration"] = r_a - r_d
        return cls(tg)
