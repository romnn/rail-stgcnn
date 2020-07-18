import textwrap
from datetime import timedelta, datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from cargonet.visualization.nxplot import NXTransportPlot
from cargonet.visualization.timeseries import TimeseriesPlot


def plot_route_delay_diffusion(tid, tg):
    NXTransportPlot(
        tg.nx_actual_route,
        filename="diffusion/%s.pdf" % tid,
        show=False,
        node_labels=True,
        edge_labels=True,
        thickness=3,
        node_size=3,
        # colorbar_range=None,
        colorbar_range=(-100, 500),
        bbox=None,
        check=False,
        delay=True,
        arrows=True,
        subtitle="%s diffusion" % tid,
    ).plot()


class DelayProgressPlot(TimeseriesPlot):
    def __init__(self, fontsize=15, samples_per_minute=30, smooth=True, stations=None):
        self.stations = stations
        super().__init__(fontsize, samples_per_minute, smooth)

    def plot_route(self, tg, save=True, show_stations=True):
        self.plot_predictions(tg, save=save, show_stations=show_stations)

    def plot_predictions(
        self,
        tg,
        predictions=None,
        save=True,
        plot_type="node",
        markers=False,
        show_stations=True,
        max_stations=5,
        has_time_axis=True,
        legend=True,
        filename=None,
        fig=None,
        ax=None,
    ):
        predictions = predictions or dict()
        route = tg.nx_planned_route
        assert nx.is_directed(route)

        timestamps, delays = [], []
        if plot_type == "node":
            for n, data in route.nodes(data=True):
                timestamps.append(data.get("arrivalTime"))
                delays.append(int(data.get("delay")))

        else:
            for u, v, data in route.edges(data=True):
                timestamps.append(route.nodes[v].get("arrivalTime"))
                delays.append(int(data.get("delay")))

        delays = np.array(delays)
        max_amp = np.max(np.abs(delays)) + 10

        if None in (fig, ax): 
            size, aspect = 6, 1.5
            fig, ax = plt.subplots(figsize=(size * aspect, size))

        df = self._plot_timeseries(fig, ax, timestamps, delays, markers=False, label="ground truth", linestyle="solid", linewidth=2)
        total_duration = df.t[-1] - df.t[0]

        for p in predictions:
            times, values, label = (
                p.get("times"),
                p.get("values"),
                p.get("label")
            )
            linestyle, color, width = p.get("style"), p.get("color"), p.get("width")
            max_amp = max(max_amp, np.max(np.abs(values)))
            _ = self._plot_timeseries(
                fig, ax,
                times,
                values,
                markers=markers,
                label=label,
                linestyle=linestyle,
                color=color,
                linewidth=width,
            )

        if show_stations and self.stations:
            last_station_time = None
            for s in tg.route:
                try:
                    s_name = self.stations.get(s.get("stationId"), dict()).get(
                        "stationName"
                    )
                    t = s.get("arrivalTime")
                    tn = mdates.date2num(t)
                    delay = df[t - timedelta(minutes=1) : t + timedelta(minutes=1)][
                        "values"
                    ].mean()
                    
                    # Station marker point
                    ax.plot([tn], [delay], "o", markersize=7, markerfacecolor='w',
                        markeredgewidth=2, markeredgecolor="black")
                    if (
                        last_station_time
                        and t - last_station_time < 1 / max_stations * total_duration
                    ):
                        continue
                    ax.text(
                        tn,
                        delay,
                        s_name,
                        verticalalignment="bottom",
                        horizontalalignment="center",
                        color="black",
                        rotation=45,
                        ha='left', 
                        va='bottom',
                        fontsize=0.5 * self.fontsize,
                    )
                    last_station_time = t
                except KeyError:
                    pass

        ax.set_xlabel("time", fontsize=self.fontsize)
        ax.set_ylabel("delay [min]", fontsize=self.fontsize)

        ax.set_ybound(lower=-max_amp, upper=max_amp)

        if has_time_axis:
            day_month_time_fmt = mdates.DateFormatter("%d. %b %H:%M")
            ax.xaxis.set_major_formatter(day_month_time_fmt)
            _ = plt.xticks(rotation=45)

        if (
            legend
            and len(predictions) > 0
            and all([p.get("label") is not None for p in predictions])
        ):
            # upper right
            plt.legend(loc="lower left", fontsize=0.75 * self.fontsize)

        plt.tight_layout()
        if save:
            filename = filename or "predictions/%s.pdf" % tg.transport_id
            filepath = self.get_filepath(filename=filename)
            plt.savefig(filepath, format="pdf", dpi=600)
            print("Saved as", filename)


def plot_all_station_delay_progress(dataset, limit=1, plot_limit=500, *timeseries):
    """Print stats and debug full network
    """
    import torch
    from cargonet.dataset.avgdelayv1 import NodeAverageDelayDatasetV1
    from cargonet.preprocessing.datalake.retrieval import Retriever

    station_delays = torch.zeros(
        len(dataset), dataset.number_of_nodes, dtype=torch.float
    )
    for i, sample in enumerate(dataset):
        station_delays[i] = sample.x.view(-1)

    r = Retriever()
    s = r.retrieve_stations(keep_ids=True)

    for edge, i in dataset.mapping.items():
        if plot_limit < 1:
            break
        u, v = edge
        try:
            src, dest = s[u].get("stationName"), s[v].get("stationName")
            test = station_delays[:, i]
            if test.max() <= 0:
                continue
            TimeseriesPlot().plot_timeseries(
                [
                    dict(
                        times=dataset.timerange,
                        values=test.cpu().numpy(),
                        label="Ground truth",
                        style="solid",
                        color="black",
                    ),
                ]
                + list(timeseries),
                subtitle="%s to %s" % (src, dest),
                ylabel="avg delay",
                xlabel="time",
                filename="stationdelays/%d_to_%d.pdf" % (u, v),
            )
        except KeyError:
            print("Key Error")
        plot_limit -= 1


def plot_station_delay_progress(u, v, dataset, timeseries, limit=1, plot_limit=500):
    """Print stats and debug full network
    """
    from cargonet.preprocessing.datalake.retrieval import Retriever

    r = Retriever()
    s = r.retrieve_stations(keep_ids=True)
    try:
        src, dest = s[u].get("stationName"), s[v].get("stationName")
        TimeseriesPlot().plot_timeseries(
            list(timeseries),
            subtitle="%s to %s" % (src, dest),
            ylabel="avg delay",
            xlabel="time",
            filename="stationdelays/%d_to_%d.pdf" % (u, v),
        )
    except KeyError:
        print("Key Error")
