import os
import uuid
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import networkx as nx

import cargonet.preprocessing.validation.constraints as constraints
import cargonet.visualization.colors as colors
from cargonet.visualization.plot import Plot


class TransportPlot(Plot):
    def __init__(
        self,
        g,
        live=None,
        title=None,
        subtitle=None,
        axis=True,
        legend=True,
        arrows=True,
        cities=False,
        fontsize=15,
        thickness=1,
        node_size=2,
        node_color="black",
        node_border_color="black",
        node_border_width=0,
        filepath=None,
        filename=None,
        save=False,
        show=True,
        node_labels=False,
        edge_labels=False,
        check=False,
        delay=False,
        colorbar_range=None,
        cmap=colors.delay_cmap,
        bbox=None,
    ):
        self.g = g
        self.live = live
        self.title = title
        self.subtitle = subtitle
        self.axis = axis
        self.legend = legend
        self.arrows = arrows
        self.cities = cities
        self.thickness = thickness
        self.node_size = node_size
        self.node_color = node_color
        self.node_border_color = node_border_color
        self.node_border_width = node_border_width
        self.filepath = filepath
        self.filename = filename
        self.save = save
        self.show = show
        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.check = check
        self.delay = delay
        self.colorbar_range = colorbar_range
        self.cmap = cmap
        self.bbox = bbox

        if self.check and self.delay:
            raise ValueError("Cannot plot delays and checks at the same time!")

        # Internal
        self.vmin = None
        self.vmax = None
        self.fig = None
        self.ax = None

        super().__init__(fontsize)

    def _plot(self):
        raise NotImplementedError

    @abstractmethod
    def draw_edges(self, route, pos, **options):
        pass

    @abstractmethod
    def draw_nodes(self, route, pos, **options):
        pass

    @abstractmethod
    def draw_node_labels(self, route, pos, **options):
        pass

    @abstractmethod
    def draw_edge_labels(self, route, pos, **options):
        pass

    @abstractmethod
    def draw_cities(self, **options):
        pass

    @abstractmethod
    def plot(self, close=True):
        pass

    def write(self, filepath):
        """
        Saves the figure
        """
        plt.savefig(filepath, format="pdf", dpi=600)

    def draw_route(self, route, good_color, bad_color):
        good_edges, bad_edges = [], []
        if self.check:
            good_edges, bad_edges = constraints.validate_edges(route)
        else:
            good_edges = route.edges

        pos = nx.get_node_attributes(route, "pos")
        node_delays = nx.get_node_attributes(route, "delay")
        node_labels = station_ids = nx.get_node_attributes(route, "stationId")
        if self.delay:
            node_labels = {n: f"{l} ({node_delays[n]})" for n, l in node_labels.items()}

        edge_draw_options = dict(arrows=self.arrows, width=self.thickness, ax=self.ax)
        if self.check:
            # Draw edges
            self.draw_edges(
                route,
                pos,
                edgelist=bad_edges,
                edge_color=bad_color,
                label="Bad",
                **edge_draw_options,
            )
            self.draw_edges(
                route,
                pos,
                edgelist=good_edges,
                edge_color=good_color,
                label="Good",
                **edge_draw_options,
            )
        elif self.delay:
            delay_intensity_colors = [route.edges[u, v]["delay"] for u, v in good_edges]
            if self.colorbar_range is None:
                _vmin, _vmax = min(delay_intensity_colors), max(delay_intensity_colors)
                self.vmin = _vmin if self.vmin is None else min(self.vmin, _vmin)
                self.vmax = _vmax if self.vmax is None else max(self.vmax, _vmax)

            delays = nx.get_edge_attributes(route, "delay")
            if self.edge_labels:
                self.draw_edge_labels(
                    route, pos, label_pos=0.5, font_size=5, edge_labels=delays
                )

            self.draw_edges(
                route,
                pos,
                edge_cmap=self.colormap,
                edge_vmin=self.vmin,
                edge_vmax=self.vmax,
                edgelist=good_edges,
                edge_color=delay_intensity_colors,
                **edge_draw_options,
            )
        else:
            self.draw_edges(
                route,
                pos,
                edgelist=good_edges,
                edge_color=good_color,
                **edge_draw_options,
            )

        # Draw nodes on top
        self.draw_nodes(
            route, pos, node_size=self.node_size, ax=self.ax, node_color=self.node_color, node_border_width=self.node_border_width, node_border_color=self.node_border_color
        )
        if self.node_labels:
            self.draw_node_labels(
                route, pos, font_size=6, ax=self.ax, labels=node_labels
            )

    @property
    def colormap(self):
        return self.cmap(self.vmin, self.vmax)
