import os

import click
import matplotlib.pyplot as plt
import networkx as nx
import numpy as num
from pyrocko import catalog, gmtpy, model
from pyrocko.plot.automap import Map

import cargonet.visualization.colors as colors
from cargonet.preprocessing.graphs.tgraph import TransportGraph
from cargonet.utils.geo import center_between_points, dist_m_v2
from cargonet.visualization.tplot import TransportPlot

BERLIN = (52.52, 13.405)


class GMTTransportPlot(TransportPlot):
    def __init__(
        self,
        g,
        live=None,
        title=None,
        subtitle=None,
        axis=True,
        arrows=False,
        cities=True,
        legend=True,
        fontsize=15,
        thickness=1,
        node_color="black",
        node_size=2,
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
        # GMT Specific
        center=BERLIN,
        radius=150000,
        edge_limit=10_000,
        node_limit=10_000,
        width=30,
        height=30,
        fit=True,
        grid=False,
        topo=True,
        rivers=True,
    ):
        self.center = center
        self.radius = radius
        self.edge_limit = edge_limit
        self.node_limit = node_limit
        self.width = width
        self.height = height
        self.fit = fit
        self.grid = grid
        self.topo = topo
        self.rivers = rivers
        super().__init__(
            g=g,
            live=live,
            title=title,
            subtitle=subtitle,
            axis=axis,
            legend=legend,
            arrows=arrows,
            cities=cities,
            fontsize=fontsize,
            thickness=thickness,
            node_color=node_color,
            node_size=node_size,
            node_border_color=node_border_color,
            node_border_width=node_border_width,
            filepath=filepath,
            filename=filename,
            save=save,
            show=show,
            node_labels=node_labels,
            edge_labels=edge_labels,
            check=check,
            delay=delay,
            colorbar_range=colorbar_range,
            cmap=cmap,
            bbox=bbox,
        )

    def draw_cities(self, **options):
        if self.cities:
            self.m.draw_cities(font_size=self.fontsize)

    def draw_edges(self, route, pos, **options):
        c, e_c = 0, route.number_of_edges()
        for u, v in self.g.edges:
            if self.edge_limit and c > self.edge_limit:
                break
            u_pos, v_pos = self.g.nodes[u].get("pos"), self.g.nodes[v].get("pos")
            e_lats = [u_pos[0], v_pos[0]]
            e_lons = [u_pos[1], v_pos[1]]
            
            self.m.gmt.psxy(
                in_columns=(e_lons, e_lats),
                W="%fp" % options.get("width", 2),
                G=options.get("edge_color", "black"),
                *self.m.jxyr,
            )
            c += 1
            if c % 1_000 == 0:
                print("Plotted %d of %d edges" % (c, e_c))

    def draw_nodes(self, route, pos, **options):
        c, n_lats, n_lons = 0, [], []
        n_c = route.number_of_nodes()
        for n, data in route.nodes(data=True):
            if self.node_limit and c > self.node_limit:
                break
            n_lats.append(pos[n][0])
            n_lons.append(pos[n][1])
            c += 1
            if c % 1_000 == 0:
                print("Plotted %d of %d nodes" % (c, n_c))
        self.m.gmt.psxy(
            in_columns=(n_lons, n_lats),
            S="c%fp" % options.get("node_size", 2),
            G=options.get("node_color", "black"),
            W="%d,%s,solid" % (options.get("node_border_width", 0), options.get("node_border_color", "black")),
            *self.m.jxyr,
        )

    def draw_node_labels(self, route, pos, **options):
        c, n_c = 0, route.number_of_nodes()
        for n, data in self.g.nodes(data=True):
            if self.node_limit and c > self.node_limit:
                break
            pos = data.get("pos")
            if self.node_labels:
                self.m.add_label(pos[0], pos[1], data.get("stationId"), self.fontsize)
            c += 1
            if c % 1_000 == 0:
                print("Plotted %d of %d node labels" % (c, n_c))

    def draw_edge_labels(self, route, pos, **options):
        raise NotImplementedError

    def plot(self, fit_factor=1.5, close=True):
        gmtpy.check_have_gmt()

        if self.fit:
            # Override manual settings
            lats = sorted([pos[0] for _, pos in self.g.nodes.data("pos")])
            lons = sorted([pos[1] for _, pos in self.g.nodes.data("pos")])
            min_lat, max_lat = lats[0], lats[-1]
            min_lon, max_lon = lons[0], lons[-1]
            lb, rt = (min_lat, min_lon), (max_lat, max_lon)
            self.center = center_between_points(lb, rt)
            self.radius = dist_m_v2(lb, rt) / fit_factor

        self.m = Map(
            lat=self.center[0],
            lon=self.center[1],
            radius=self.radius,
            width=self.width,
            height=self.height,
            show_grid=self.grid,
            show_topo=self.topo,
            color_dry=(238, 236, 230),
            topo_cpt_dry="light_land",
            topo_cpt_wet="light_sea",
            illuminate=True,
            illuminate_factor_ocean=0.15,
            show_rivers=self.rivers,
            show_plates=False,
            show_boundaries=True,
            show_scale=True,
            # TODO: Use german metropoles
            # custom_cities=[]
        )

        self.draw_cities()
        self.draw_route(self.g, good_color="black", bad_color="red")
        if self.live is not None:
            self.draw_route(self.live, good_color="green", bad_color="blue")

        filepath = self.get_filepath(
            filepath=self.filepath, filename=self.filename, random=self.save
        )
        if filepath is not None:
            self.m.save(filepath, psconvert=True)
