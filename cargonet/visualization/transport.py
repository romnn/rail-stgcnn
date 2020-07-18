import os
import shutil
import time

import click

import cargonet.preprocessing.datalake.retrieval as retrieval
import cargonet.preprocessing.graphs.tgraph as tgraph
from cargonet.utils.formatting import fmt_time
from cargonet.visualization.gmtplot import GMTTransportPlot
from cargonet.visualization.nxplot import NXTransportPlot


def plot_transports(
    transport_id, limit, backend, check, delay, node_labels, edge_labels, title
):
    r = retrieval.Retriever()
    s = r.retrieve_stations(keep_ids=True)

    base_path = os.path.dirname(os.path.realpath(__file__))

    if not transport_id:
        base_path = os.path.join(base_path, "../../fig/transports")
        try:
            shutil.rmtree(base_path)
        except FileNotFoundError:
            pass

    try:
        os.makedirs(base_path)
        os.makedirs(os.path.join(base_path, "bad"))
    except FileExistsError:
        pass
    assert os.path.exists(base_path)

    is_map = not backend == "nx"
    plot_backend = GMTTransportPlot if is_map else NXTransportPlot

    def plot(tid):
        print("Plotting %s" % tid)
        t = r.retrieve_transport(transport_id=tid)[0]
        tg = tgraph.TransportGraph(t, stations=s)
        _, report = tg.validate()

        plot_backend(
            tg.nx_planned_route if not delay else tg.nx_actual_route,
            filename="transports/%s%s%s.pdf"
            % (
                "bad/" if len(report.get("bad_edges_planned", [])) > 0 else "",
                tid,
                "_map" if is_map else "",
            ),
            show=False,
            subtitle=tid if title else None,
            node_labels=node_labels,
            edge_labels=edge_labels,
            thickness=3,
            check=False if delay else check,
            colorbar_range=(-100, 10_000),
            delay=delay,
            live=None if delay else tg.nx_actual_route,
        ).plot()
        time.sleep(0.2)

    start = time.time()
    total = 0
    for tid in (
        [transport_id] if transport_id else r.retrieve_transport_ids(limit=limit)
    ):
        plot(tid)
        total += 1

    end = time.time()
    print("# Plotted %d transports in %s" % (total, fmt_time(end - start)))
