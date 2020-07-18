import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

from cargonet.visualization.plot import Plot
from cargonet.visualization.utils import resample_time_series


class TimeseriesPlot(Plot):
    def __init__(self, fontsize=15, samples_per_minute=30, smooth=True):
        self.samples_per_minute = samples_per_minute
        self.smooth = smooth
        super().__init__(fontsize)

    def _plot_timeseries(
        self,
        fig,
        ax,
        times,
        values,
        markers=False,
        label=None,
        capstyle="round",
        linestyle="dashed",
        linewidth=2,
        color="black",
    ):
        df = resample_time_series(
            times,
            values,
            samples_per_minute=self.samples_per_minute,
            smooth=self.smooth,
        )

        # dash_capstyle 'butt', 'round', 'projecting'
        # dash_joinstyle 'miter', 'round', 'bevel'
        ax.plot(
            df["t"],
            df["values"],
            color=color,
            label=label,
            linewidth=linewidth,
            linestyle=linestyle,
            dash_capstyle=capstyle,
            solid_capstyle=capstyle,
        )

        if markers and not self.smooth:
            # Station marker point
            ax.plot(df["t"], df["values"], "o", markersize=4)
        return df

    def plot_timeseries(
        self,
        timeseries,
        subtitle=None,
        xlabel="time",
        ylabel="delay in minutes",
        center=True,
        time_fmt="%d. %b %H:%M",
        filename=None,
        legend=True,
        has_time_axis=True,
    ):
        size, aspect = 10, 1.5
        num_plots = len(set([ts.get("index", 0) for ts in timeseries]))
        fig, axs = plt.subplots(num_plots, figsize=(size * aspect, size))

        dfs, max_amp = [], 1
        for t in timeseries:
            times, values, label, ai = (
                t.get("times"),
                t.get("values"),
                t.get("label"),
                t.get("index", 0),
            )
            linestyle, color, width = t.get("style"), t.get("color"), t.get("width")
            max_amp = max(max_amp, np.max(np.abs(values)))
            df = self._plot_timeseries(
                fig,
                axs if num_plots < 2 else axs[ai],
                times,
                values,
                label=label,
                linestyle=linestyle,
                color=color,
                linewidth=width,
            )
            dfs.append(df)

        for plot in range(num_plots):
            ax = axs if num_plots < 2 else axs[plot]
            ax.set_xlabel(xlabel, fontsize=self.fontsize)
            ax.set_ylabel(ylabel, fontsize=self.fontsize)
            if center:
                max_amp += 1
                ax.set_ybound(lower=-max_amp, upper=max_amp)

            if has_time_axis:
                ax.xaxis_date()
                ax.xaxis.set_major_formatter(mdates.DateFormatter(time_fmt))
                _ = plt.xticks(rotation=45)

            if subtitle:
                ax.set_title(subtitle, fontsize=self.fontsize)

            if (
                legend
                and len(timeseries) > 0
                and all([t.get("label") is not None for t in timeseries])
            ):
                ax.legend(loc="upper right", fontsize=self.fontsize)

        if filename:
            filepath = self.get_filepath(filename=filename)
            plt.savefig(filepath, format="pdf", dpi=600)
            print("Saved as", filename)

        plt.close()
