import os
import shutil
import time

import cargonet.preprocessing.datalake.retrieval as retrieval
from cargonet.preprocessing.graphs.builders import NXTGBuilder
from cargonet.utils.formatting import fmt_time
from cargonet.visualization.gmtplot import GMTTransportPlot
from cargonet.visualization.nxplot import NXTransportPlot


def plot_net(limit, backend):
    r = retrieval.Retriever()
    s = r.retrieve_stations(keep_ids=True)

    plot_backend = NXTransportPlot if backend == "nx" else GMTTransportPlot

    start = time.time()
    full_net = r.retrieve_full_net(limit=limit)
    print("Loaded full graph")
    full_net_graph = NXTGBuilder.from_nodes_edges(*full_net).build()
    print("Built NX graph")
    plot_backend(
        full_net_graph,
        filename="full_net_%s_%d.pdf" % (backend, limit),
        show=False,
        labels=False,
        check=False,
    ).plot()
    end = time.time()
    print("# Plotted net in %s" % fmt_time(end - start))
