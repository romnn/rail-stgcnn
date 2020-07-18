import os
import shutil
import time
import traceback
from datetime import datetime

import cargonet.preprocessing.datalake.retrieval as retrieval
import cargonet.preprocessing.graphs.tgraph as tgraph
import cargonet.visualization.nxplot as nxplot
from cargonet.visualization.transport import plot_transports


def check_transports(transport_id, limit, plot_valid=False):
    r = retrieval.Retriever()
    s = r.retrieve_stations(keep_ids=True)

    stats, invalid = dict(), 0

    def incr_counter(stat_key, value=1):
        nonlocal stats
        stats[stat_key] = 1 if not stat_key in stats else stats[stat_key] + value

    def check(tid):
        print("Checking %s" % tid)
        t = r.retrieve_transport(transport_id=tid)[0]
        tg = tgraph.TransportGraph(t, stations=s)
        valid, report = tg.validate()

        if not valid:
            if len(report.get("bad_edges_planned", [])) > 0:
                print("=> Found %d bad edges" % len(report.get("bad_edges_planned")))
                incr_counter("bad edges")
            if len(report.get("bad_delays", [])) > 0:
                print("=> Found %d bad delays" % len(report.get("bad_delays")))
                incr_counter("bad delays")
            if len(report.get("exceeding_delays", [])) > 0:
                print(
                    "=> Found %d exceeding delays" % len(report.get("exceeding_delays"))
                )
                incr_counter("exceeding delays")
            if not report.get("covered", True):
                print("=> Not sufficiently covered")
                incr_counter("bad coverage")
            if not report.get("complete", True):
                print("=> Not complete")
                incr_counter("incomplete")
            nonlocal invalid
            invalid += 1
        else:
            if plot_valid:
                plot_transports(
                    transport_id=tid,
                    limit=None,
                    backend="nx",
                    check=True,
                    delay=True,
                    node_labels=True,
                    edge_labels=True,
                    title="%d (considered valid)" % tid,
                )
            for n, data in tg.nx_actual_route.nodes(data=True):
                a = data.get("arrivalTime")
                if datetime(2019, 2, 7) <= a <= datetime(2019, 2, 14):
                    raise ValueError("%s is a valid transport with an arrival time of %s" % (tid, a))

    total, failed = 0, 0
    for tid in (
        [transport_id] if transport_id else r.retrieve_transport_ids(limit=limit)
    ):
        try:
            check(tid)
        except Exception as e:
            raise
            traceback.print_exc()
            failed += 1
            invalid += 1
        total += 1

    print("# Total: %d (%d failed)" % (total, failed))
    for stat, count in stats.items():
        print("# %s: %d" % (stat, count))
    print(
        "# => {} of {} ({}%) did not pass checks".format(
            invalid, total, round(float(invalid) / float(total) * 100.0)
        )
    )
