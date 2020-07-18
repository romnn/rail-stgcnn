import os
import uuid

import click
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from cargonet.preprocessing.graphs.tgraph import TransportGraph
from cargonet.visualization.tplot import TransportPlot


class NXTransportPlot(TransportPlot):
    def draw_edges(self, route, pos, **options):
        nx.draw_networkx_edges(route, pos, **options)

    def draw_nodes(self, route, pos, **options):
        nx.draw_networkx_nodes(route, pos, **options)

    def draw_node_labels(self, route, pos, **options):
        nx.draw_networkx_labels(route, pos, **options)

    def draw_edge_labels(self, route, pos, **options):
        nx.draw_networkx_edge_labels(route, pos, **options)

    def draw_cities(self, **options):
        cities = {
            "Berlin": (13.422937, 52.511991),
            "Hamburg": (9.215304, 53.712586),
            "Munich": (11.541843, 48.154955),
            "Cologne": (6.967279, 50.957886),
        }
        for c, pos in cities.items():
            self.ax.plot([pos[1]], [pos[0]], "o", color="black")  # Point
            self.ax.text(
                pos[1],
                pos[0],
                c,
                verticalalignment="bottom",
                horizontalalignment="left",
                color="black",
                fontsize=10,
            )

    def plot(self, close=True):
        self._plot()
        filepath = self.get_filepath(
            filepath=self.filepath, filename=self.filename, random=self.save
        )
        if filepath is not None:
            self.write(filepath=filepath)
        if self.show:
            plt.show()
        if close:
            plt.close()

    def _plot(self):
        size, aspect = 10, 1.5
        self.fig, self.ax = plt.subplots(figsize=(size * aspect, size))
        fig, ax = self.fig, self.ax

        if self.axis:
            ax.set_xlabel("latitude", fontsize=self.fontsize)
            ax.set_ylabel("longitude", fontsize=self.fontsize)

        if self.title:
            fig.suptitle(self.title, fontsize=1.5 * self.fontsize, fontweight="bold")

        if self.subtitle:
            ax.set_title(self.subtitle, fontsize=self.fontsize)

        self.vmin, self.vmax = None, None
        if self.colorbar_range is not None:
            self.vmin, self.vmax = self.colorbar_range

        self.draw_route(self.g, good_color="black", bad_color="red")
        if self.live is not None:
            self.draw_route(self.live, good_color="green", bad_color="blue")
        self.draw_cities()

        if self.delay:
            sm = plt.cm.ScalarMappable(
                cmap=self.colormap, norm=plt.Normalize(vmin=self.vmin, vmax=self.vmax)
            )
            sm.set_array([])

            cbar = fig.colorbar(sm, ax=ax)
            cbar.ax.set_ylabel(
                "delay in minutes", fontsize=self.fontsize
            )  # , rotation=270)

        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        if self.bbox is not None:
            # Use custom ranges as therfore scaling on the axis
            lats = self.bbox[0][0], self.bbox[1][0]
            lons = self.bbox[0][1], self.bbox[1][1]
            plt.xlim(min(lats), max(lats))
            plt.ylim(min(lons), max(lons))
        else:
            # Use equal scaling on the axis
            plt.axis("equal")

        if self.check:
            plt.legend(loc="upper right", fontsize=self.fontsize)
