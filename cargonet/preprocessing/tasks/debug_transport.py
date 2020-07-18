import time
from itertools import combinations
from pprint import pprint

import networkx as nx
from prettytable import PrettyTable

import cargonet.preprocessing.datalake.retrieval as retrieval
import cargonet.preprocessing.graphs.tgraph as tgraph
import cargonet.visualization.colors as colors


def debug_combinations(sections, stations, all_live):
    results = []
    for i in range(1, len(sections) + 1):
        for sec in list(combinations(sections, i)):
            x = dict()
            x["planned"] = []
            x["live"] = []
            for _s in sec:
                x["planned"] += _s.get("planned")
                x["live"] += _s.get("live")
            x["endStationId"] = sec[-1].get("endStationId")
            x["plannedEventTime"] = sec[-1].get("plannedArrivalTime")
            x["eventTime"] = sec[-1].get("plannedArrivalTime")
            x["plannedArrivalTimeEndStation"] = sec[-1].get(
                "plannedArrivalTimeEndStation"
            )

            error, bad_edges, problems = None, [], dict()
            try:
                tg = tgraph.TransportGraph(x, stations=stations)
                _, bad_edges, problems = tg.validate()

                nxplot.plot_nx_graph(
                    tg.graph, show=True, labels=True, check=True, live=tg.live
                )
                time.sleep(0.2)
            except Exception as e:
                # raise
                error = str(e)

            section_ids = [s.get("trainSectionId") for s in sec]
            results.append(
                {
                    "SectionId": ",".join([str(s) for s in section_ids]),
                    "Stations": len(x["planned"]),
                    "Bad edges": len(bad_edges) > 0 or error is not None,
                    "Live Cov": "%d/%d"
                    % (
                        len(
                            [
                                xx
                                for xx in all_live
                                if xx.get("trainSectionId") in section_ids
                            ]
                        ),
                        len(all_live),
                    ),
                    "Problems": "\n".join(
                        list(problems.values()) + ([] if error is None else [error])
                    ),
                }
            )

    x = PrettyTable()
    x.field_names = ["SectionId", "Stations", "Bad edges", "Live Cov", "Problems"]
    for r in results:
        x.add_row([r.get(h) for h in x.field_names])
    print(x)


def debug_live_sections(sections, all_live):
    # Sort all live data
    section_ids = [s.get("trainSectionId") for s in sections]

    def get_section(sid):
        f = [s for s in sections if s.get("trainSectionId") == sid]
        if len(f) > 0:
            return f[0]
        return dict()

    all_live = sorted(all_live, key=lambda l: l.get("eventTime"))
    all_live_stations = []
    for l in all_live:
        if l.get("stationId") not in all_live_stations:
            all_live_stations.append(l.get("stationId"))

    x = PrettyTable()
    x.field_names = ["Live station"] + section_ids
    for i in range(len(all_live_stations)):
        observed = all_live_stations[i]
        strict = [
            "x"
            if observed
            in [p.get("stationId") for p in get_section(s).get("planned", [])]
            else ""
            for s in section_ids
        ]
        x.add_row([observed] + strict)
    print(x)
