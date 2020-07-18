import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cargonet.visualization.colors import delay_heat_cmap
from cargonet.visualization.plot import Plot


class DelayByStationPlot(Plot):
    def __init__(self, fontsize=15, linewidth=0):
        self.linewidth = linewidth
        super().__init__(fontsize)

    def plot(
        self,
        dist,
        vmin=-100,
        vmax=100,
        center=0,
        filename=None,
        show=False,
        subtitle=None,
        x_tick_labels=None,
        y_tick_labels=None,
        xlabel="time",
        ylabel="station",
        time_fmt="%d. %b %H:%M",
        pdf=True,
    ):
        size, aspect = 10, 1.5
        fig, ax = plt.subplots(figsize=(size * aspect, size))

        sns.heatmap(
            dist.numpy(),
            linewidths=self.linewidth,
            vmin=vmin,
            vmax=vmax,
            center=center,
            cbar_kws=dict(label="Delay"),
            ax=ax,
            cmap=delay_heat_cmap(vmin=vmin, vmax=vmax, center=center),
        )

        if x_tick_labels:
            ax.set_xticklabels(x_tick_labels)
        if y_tick_labels:
            ax.set_yticklabels(y_tick_labels)

        ax.set_xlabel("time", fontsize=self.fontsize)
        ax.set_ylabel("station", fontsize=self.fontsize)

        if subtitle:
            ax.set_title(subtitle, fontsize=self.fontsize)

        # fig.subplots_adjust(bottom=0.2)
        fig.tight_layout()
        if filename:
            filepath = self.get_filepath(filename=filename)
            if pdf:
                plt.savefig(filepath + "_pdf.pdf", format="pdf", dpi=600)
            plt.savefig(filepath + "_png.png", format="png")
            print("Saved as", filename)

        if show:
            plt.show()

        plt.close()
