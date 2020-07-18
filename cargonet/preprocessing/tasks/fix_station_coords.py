import time

import cargonet.preprocessing.datalake.retrieval as retrieval
import cargonet.preprocessing.graphs.tgraph as tgraph
import cargonet.utils.geo as geo
import cargonet.visualization.nxplot as nxplot


def fix_station_coordinates(show):
    r = retrieval.Retriever()

    fixes = []

    def update_coordinates(station_id, coords):
        r.db.eletastations.update_one(
            {"stationId": int(station_id)},
            {"$set": {"latitude": float(coords[0]), "longitude": float(coords[1])}},
            upsert=False,
        )

    # Update
    for fix in fixes:
        update_coordinates(fix.get("stationId"), fix.get("coords"))

    # Then display
    fixed = 0
    total_bad_edges = []
    for fix in fixes:
        tid = fix.get("transportId")
        t = r.retrieve_transport(transport_id=tid)[0]
        s = r.retrieve_stations(keep_ids=True)
        tg = tgraph.TransportGraph(t, stations=s)
        _, bad_edges, _ = tg.validate()
        total_bad_edges += bad_edges

        nxplot.plot_nx_graph(
            tg.graph,
            filename="transports/%s%s.pdf" % ("bad/" if bad_edges else "", tid),
            show=show or fix.get("show") is True,
            labels=True,
            check=True,
        )
        time.sleep(0.2)
        fixed += 1

    print(
        "Corrected %d stations. Have %d bad edges now" % (fixed, len(total_bad_edges))
    )
