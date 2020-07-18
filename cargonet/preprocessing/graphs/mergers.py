import statistics
from abc import ABC, abstractmethod
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx


class TransportGraphMerger(ABC):
    def __init__(self, initial_state, undirected=True):
        self.undirected = undirected
        self.state = initial_state.copy()
        if undirected:
            self.state = self.state.to_undirected()

    @classmethod
    def create(cls, initial_state, undirected=True):
        return cls(initial_state, undirected=undirected)

    @abstractmethod
    def add(self, t, t_next, old, new, full_route=True):
        pass

    @abstractmethod
    def merge(self):
        pass


class OverrideMerger(TransportGraphMerger):
    def add(self, t, t_next, old, new):
        """
        Merges two NX graphs by overriding node and edge attributes of conflicting nodes
        """
        raise NotImplementedError

    def merge(self):
        raise NotImplementedError


class AverageMerger(TransportGraphMerger):
    def __init__(self, initial_state, undirected=True):
        super().__init__(initial_state, undirected)
        self.edge_delays = defaultdict(list)

    def add(self, t, t_next, new, full_route=True):
        """
        Merges two NX graphs by interpolating node and edge attributes of conflicting nodes
        """
        planned = new.nx_planned_route
        actual = new.nx_actual_route
        route = actual.to_undirected()

        def valid_edge(u, v):
            ud = route.nodes[u].get("departureTime")
            vd = route.nodes[v].get("arrivalTime")
            return ud and vd and ud <= t < vd

        for u, v, data in route.edges(data=True):
            if valid_edge(u, v) or full_route:
                if not self.state.has_edge(u, v):
                    self.state.add_node(u, **route.nodes[u])
                    self.state.add_node(v, **route.nodes[v])
                    self.state.add_edge(u, v, **data)  # Fix the delay later on
                self.edge_delays[(u, v)].append(int(data["delay"]))

    def merge(self):
        for edge_index, delays in self.edge_delays.items():
            # Safe to assume all edge already exist
            self.state.edges[edge_index]["delay"] = int(statistics.mean(delays))

        # print("edges on merge: ", self.state.number_of_edges())
        return self.state
