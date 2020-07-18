import matplotlib.pyplot as plt
import numpy as np

from cargonet.visualization.plot import Plot


class HistogramPlot(Plot):
    def __init__(self, fontsize=15):
        super().__init__(fontsize)

    def plot(self, values, bins, color="green", show=False, alpha=0.8, filename=None):
        plt.hist(values, bins, color=color, alpha=alpha)
        plt.tight_layout()
        if filename:
            filepath = self.get_filepath(filename=filename)
            plt.savefig(filepath, format="pdf", dpi=600)
            print("Saved as", filename)

        if show:
            plt.show()

        plt.close()
