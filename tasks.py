"""
Tasks for maintaining the project.
Execute 'invoke --list' for guidance on using Invoke
"""


from pathlib import Path
from invoke import task
from pprint import pprint
from collections import defaultdict
import os
import numpy as np

Path().expanduser()

ROOT_DIR = Path(__file__).parent
TEST_DIR = ROOT_DIR.joinpath("tests")
SOURCE_DIR = ROOT_DIR.joinpath("cargonet")
PYTHON_DIRS = [str(d) for d in [SOURCE_DIR, TEST_DIR]]


@task(help={"check": "Checks if source is formatted without applying changes"})
def format(c, check=False):
    """Format code
    """
    python_dirs_string = " ".join(PYTHON_DIRS)
    black_options = "--diff" if check else ""
    c.run("black {} {}".format(black_options, python_dirs_string))
    isort_options = "--recursive {}".format("--check-only" if check else "")
    c.run("isort {} {}".format(isort_options, python_dirs_string))


@task
def lint(c):
    """Lint code
    """
    c.run("flake8 {}".format(SOURCE_DIR))


@task
def test(c):
    """Run tests
    """
    c.run("python -m pytest -s")


@task
def type_check(c):
    """Check types
    """
    c.run("mypy")


@task
def plot_full_graph(c):
    """Plots the full graph
    """
    RailDataset.plot_full_graph(backend="gmt", show=False)


@task
def plot_transports(
    c,
    transport_id=None,
    limit=3_000,
    backend="nx",
    check=True,
    delay=False,
    node_labels=True,
    edge_labels=False,
    title=True,
):
    """Plots all or selected transports
    """
    import cargonet.visualization.transport as transports

    transports.plot_transports(
        transport_id=transport_id, limit=limit, backend=backend, check=check, delay=delay, node_labels=node_labels, edge_labels=edge_labels, title=title
    )


@task
def plot_net(c, limit=3_000, backend="nx"):
    """Plots the entire net
    """
    import cargonet.preprocessing.tasks.plot_net as pn

    pn.plot_net(
        transport_id, limit, backend, check, delay, node_labels, edge_labels, title
    )


@task
def fix_station_coordinates(c, show=False):
    """Fixes invalid station coordinates
    """
    import cargonet.preprocessing.tasks.fix_station_coordinates as fsc

    fsc.fix_station_coordinates(
        transport_id, limit, backend, check, delay, node_labels, edge_labels, title
    )


@task
def debug_transports_combos(c, transport_id):
    """Debug transport section combinations
    """
    import cargonet.preprocessing.tasks.debug_transport as dt
    from cargonet.preprocessing.datalake.retrieval import Retriever

    r = Retriever()
    s = r.retrieve_stations(keep_ids=True)
    t_raw = r.retrieve_transport(transport_id=transport_id, raw_sections=True)
    all_live = []
    for sec in t_raw.get("sections"):
        for l in sec.get("live"):
            all_live.append(l)
    dt.debug_combinations(t_raw.get("sections"), s, all_live)


@task
def debug_transports_delays(c, transport_id, smooth=True):
    """Debug transport delays
    """
    import cargonet.preprocessing.tasks.debug_transport as dt
    from cargonet.preprocessing.datalake.retrieval import Retriever
    from cargonet.preprocessing.graphs.tgraph import TransportGraph
    from cargonet.visualization.delays import DelayProgressPlot

    r = Retriever()
    s = r.retrieve_stations(keep_ids=True)
    t = r.retrieve_transport(transport_id=transport_id)[0]
    tg = TransportGraph(t, stations=s)
    DelayProgressPlot(stations=s, smooth=smooth).plot_route(
        tg, save=True, show_stations=True
    )


@task
def plot_predicted_delay(c, transport_id):
    """Debug transport delays
    """
    import cargonet.visualization.delays as d
    from cargonet.preprocessing.graphs.tgraph import TransportGraph
    from cargonet.preprocessing.datalake.retrieval import Retriever

    r = Retriever()
    s = r.retrieve_stations(keep_ids=True)
    t = r.retrieve_transport(transport_id=transport_id)[0]
    tg = TransportGraph(t, stations=s)
    d.DelayProgressPlot(stations=s).plot_predictions(tg, save=True, show_stations=True)


@task
def debug_transport_sections(c, transport_id):
    """Debug transport sections
    """
    import cargonet.preprocessing.tasks.debug_transport as dt
    from cargonet.preprocessing.datalake.retrieval import Retriever

    r = Retriever()
    s = r.retrieve_stations(keep_ids=True)
    t_raw = r.retrieve_transport(transport_id=transport_id, raw_sections=True)
    all_live = []
    for sec in t_raw.get("sections"):
        for l in sec.get("live"):
            all_live.append(l)
    dt.debug_live_sections(t_raw.get("sections"), all_live)


@task
def debug_station(c, station_id):
    """Debug station
    """
    import cargonet.preprocessing.tasks.debug_station as ds

    ds.debug_station(station_id)


@task
def check_transports(c, transport_id=None, limit=3_000, plot_valid=False):
    """Debug station
    """
    import cargonet.preprocessing.tasks.check_transports as ct

    ct.check_transports(transport_id, limit, plot_valid=plot_valid)


def _receive_tg(transport_id):
    from cargonet.preprocessing.graphs.tgraph import TransportGraph
    from cargonet.preprocessing.datalake.retrieval import Retriever

    r = Retriever()
    t = r.retrieve_transport(transport_id=transport_id)[0]
    s = r.retrieve_stations(keep_ids=True)
    return TransportGraph(t, stations=s)


@task
def export_transports(c, transport_id):
    """Export transport to GML
    """
    tg = _receive_tg(transport_id)
    tg.save("transport_%s" % transport_id)


@task
def print_route(c, transport_id, interpolate_delays=True):
    """Print transport route
    """
    tg = _receive_tg(transport_id)
    tg.print_route(interpolate_missing=interpolate_delays)


@task
def build_caches(c, transports=False, indices=False):
    """Build datalake caches
    """
    import cargonet.preprocessing.datalake.cache as cache

    cache.build_caches(transports, indices)


@task
def build_avgdelayv1(
    c,
    limit=1,
    plot_download=False,
    plot_processing=False,
    rebuild=False,
    reprocess=False,
    verbose=True,
):
    """Build avgdelayv1 dataset
    """
    from cargonet.dataset.avgdelayv1 import build_dataset

    build_dataset(limit, plot_download, plot_processing, rebuild, reprocess, verbose)


@task
def train_avgdelayv1(
    c,
    plot=True,
    limit=32,
    epochs=1,
    train=False,
    evaluate=True,
    reprocess=False,
    redownload=False,
    device=None,
):
    """Train avgdelayv1 model
    """
    from cargonet.models.avgdelayv1 import train_model

    train_model(
        plot,
        limit=limit,
        epochs=epochs,
        train=train,
        evaluate=evaluate,
        reprocess=reprocess,
        redownload=redownload,
        device=device,
    )


@task
def train_activeroutesv1(
    c,
    plot=True,
    limit=32,
    epochs=100,
    train=False,
    evaluate=True,
    reprocess=False,
    redownload=False,
    bptt=False,
    device=None,
):
    """Train activeroutesv1 model
    """
    from cargonet.models.activeroutesv1 import train_model

    train_model(
        plot,
        limit=limit,
        epochs=epochs,
        train=train,
        evaluate=evaluate,
        reprocess=reprocess,
        redownload=redownload,
        device=device,
        bptt=bptt
    )

@task
def train_baselines(
    c,
    plot=False,
    limit=32,
    epochs=100,
    train=False,
    all=False,
    search=False,
    fc=False,
    fc2=False,
    ts=False,
    tcn=False,
    rf=False,
    lstm=False,
    avg=False,
    svm=False,
    evaluate=True,
    reprocess=False,
    redownload=False,
    device=None,
):
    """Train baseline models
    """
    from cargonet.models.baselines.train import train_model

    train_model(
        plot=plot,
        limit=limit,
        epochs=epochs,
        all=all,
        search=search,
        rf=rf,
        ts=ts,
        lstm=lstm,
        fc=fc,
        tcn=tcn,
        fc2=fc2,
        svm=svm,
        avg=avg,
        train=train,
        evaluate=evaluate,
        reprocess=reprocess,
        redownload=redownload,
        device=device,
    )

@task
def make_tn(
    c,
    transport_id,
):
    """ Test """
    from cargonet.visualization.gmtplot import GMTTransportPlot
    from cargonet.preprocessing.datalake.retrieval import Retriever
    from cargonet.preprocessing.graphs.tgraph import TransportGraph
    
    r = Retriever()
    stations = r.retrieve_stations(keep_ids=True)

    # Load the transport
    tg = TransportGraph(r.retrieve_transport(transport_id=transport_id)[0], stations=stations)
    
    GMTTransportPlot(
        tg.nx_actual_route,
        check=False,
        filename="predictions/compare/%s.pdf" % transport_id,
        node_size=17,
        thickness=4,
        fontsize=25,
        node_border_color="black",
        node_border_width=3,
        node_color="white",
    ).plot(fit_factor=1.9)


@task
def test_models(
    c,
    plot=True,
    limit=32, # 8 months
    device=None,
    pred_seq_len=10,
    horizons=None,
    linear=False,
):
    """Train activeroutesv1 model
    """
    import torch
    import matplotlib.pyplot as plt
    from datetime import timedelta
    from cargonet.models.model import MLModel
    from cargonet.models.activeroutesv1 import ActiveRoutesModelV1
    from cargonet.models.baselines.fc2 import FCModelV2
    from cargonet.models.baselines.lstm import BaselineLSTMModelV1
    from cargonet.models.baselines.timeshift import BaselineTimeshiftModelV1
    from cargonet.dataset.activeroutesv1 import ActiveRoutesV1
    from cargonet.dataset.simulator import Simulation
    from cargonet.models.normalization import Scaler
    from cargonet.visualization.delays import DelayProgressPlot
    from cargonet.visualization.gmtplot import GMTTransportPlot
    from cargonet.preprocessing.datalake.retrieval import Retriever
    from cargonet.preprocessing.graphs.tgraph import TransportGraph
    from cargonet.models.utils import rec_dd
    import networkx as nx

    base_path = os.path.dirname(os.path.realpath(__file__))
    dataset_base_path = os.path.join(base_path, "datasets")
    models_base_path = os.path.join(base_path, "trained")
    assert os.path.exists(dataset_base_path)
    assert os.path.exists(models_base_path)

    dataset_name = "active-routes-v1"
    dataset_path = os.path.join(dataset_base_path, dataset_name)

    simulation_dataset_name = "simulation-v1"
    simulation_dataset_path = os.path.join(dataset_base_path, simulation_dataset_name)

    pred_seq_len=10
    ds_options = dict(seq_len=10, pred_seq_len=pred_seq_len)
    batch_hours = 7 * 24
    horizons = [0, 3, 6, 9]

    use_simulation = False

    if use_simulation:
        dataset = Simulation(
            root=simulation_dataset_path,
            name=simulation_dataset_name,
            limit=32 * 10 * 2,
            **ds_options
        )
    else:
        dataset = ActiveRoutesV1(
            root=dataset_path,
            name=dataset_name,
            limit=limit,
            batch=timedelta(hours=batch_hours),
            **ds_options
        )

    model_options = dict(
        node_input_dim=len(dataset.encoder.seq_route_node_fts),
        edge_input_dim=len(dataset.encoder.route_edge_fts),
        shuffle=False,
        shuffle_after_split=False,
    )

    delay_stddev = None

    def normalize_func(data, means, stds, **kwargs):
        data.x = Scaler.zscore(data.x, mean=means["x"], std=stds["x"])
        nonlocal delay_stddev
        delay_stddev = stds["x"][-1]

        data.temporal_edge_attr = Scaler.zscore(
            data.temporal_edge_attr,
            mean=means["temporal_edge_attr"],
            std=stds["temporal_edge_attr"],
        )
        
        assert not torch.isnan(data.temporal_edge_attr).any()
        assert not torch.isnan(data.x).any()
        return data

    class MockModel:
        name = "Missing"

    print("Creating models")
    ar_model = sf_ar_model = fc2_model = lstm_model = ts_model = MockModel()
    if True: # YES
        ar_model = ActiveRoutesModelV1(
            dataset, device=device, use_rnn=False, **ds_options, **model_options
        )
        ar_model.load()
    if False:
        sf_ar_model = ActiveRoutesModelV1(
            dataset, device=device, use_rnn=True, **ds_options, **model_options
        )
        sf_ar_model.load()
    if True: # YES
        fc2_model = FCModelV2(
            dataset, device=device, **ds_options, **model_options
        )
        fc2_model.load()
    if False:
        lstm_model = BaselineLSTMModelV1(
            dataset, device=device, **ds_options, **model_options
        )
        lstm_model.load()
    if True: # YES
        ts_model = BaselineTimeshiftModelV1(
            dataset, device=device, **ds_options, **model_options
        )
        ts_model.load()
    
    models = [
        ar_model,
        sf_ar_model,
        ts_model,
        fc2_model,
        lstm_model,
    ]
    models = [m for m in models if not isinstance(m, MockModel)]
    print("Evaluating %d models for horizons %s" % (len(models), horizons))

    trained_limit = 32
    cache = "%s_norm_%d_%d" % (dataset.name, batch_hours, trained_limit)
    print("fitting normalization", cache)
    z_score_norm = Scaler.fit(
        models[0].train_data,
        normalize=normalize_func,
        attrs=dict(temporal_edge_attr=1, x=1, y=1,),
        cache=cache
    )
    for model in models:
        if isinstance(model, BaselineTimeshiftModelV1):
            continue
        model.dataset.transform = z_score_norm
        model.init_loaders()
    print("done fitting normalization")

    # DEBUG
    if False:
        for data in models[0].val_data:
            print(data.x)
            break
        return

    if False:
        distr = []
        for j, data in enumerate(ts_model.data):
            if data.x is None or torch.isnan(data.x).any():
                distr.append(0)
                continue
            distr.append(data.x.size(0))
        fig, ax = plt.subplots(tight_layout=True)
        ax.fill_between(range(len(distr)), 0, distr)
        # ax.plot(range(len(distr)), distr) # , bins=int(len(distr) * 0.5))
        plt.show()
        return
    

    long_val = [34877359, 34904458]
    wrong = [34813294, 34834374]
    plot_limit = 100_000

    COLORS = {
        # a2de96 light green
        # 01a9b4 blue
        ts_model.name: "#a2de96", # light green
        getattr(fc2_model, "name", "FC2"): "#fc7e2f", # orange
        getattr(lstm_model, "name", "LSTM"): "#fbd46d", # yellow
        ar_model.name: "#f40552", # red
        sf_ar_model.name: "#c3edea", # light blue
    }

    RENAME = {
        ts_model.name: "Timeshift",
        getattr(fc2_model, "name", "FC2"): "FCNN",
        getattr(lstm_model, "name", "LSTM"): "LSTM",
        ar_model.name: "RailSTGCNN",
        sf_ar_model.name: "Stateful RailSTGCNN",
    }

    r = Retriever()
    stations = r.retrieve_stations(keep_ids=True)

    summary = MLModel.test_models(models, pred_seq_len=pred_seq_len, debug=long_val)
    lengths = []
    # print(summary)
    for transport, results in summary.items():

        # if not transport in long_val:
        #     continue

        # if not transport in []:
        #     continue

        if plot_limit < 1:
            continue

        # Load the transport
        tg = TransportGraph(r.retrieve_transport(transport_id=transport)[0], stations=stations)

        # Sort the results by time first first
        ts = sorted(results.keys())
        
        plot_trajs = True
        if plot_trajs and not use_simulation:
            GMTTransportPlot(
                tg.nx_actual_route,
                check=False,
                filename="predictions/compare/%s.pdf" % transport,
                node_size=17,
                thickness=4,
                fontsize=25,
                node_border_color="black",
                node_border_width=3,
                node_color="white",
            ).plot(fit_factor=1.9)

        plot_limit -= 1
        
        for hor in (horizons if horizons is not None else range(pred_seq_len)):
            try:
                route = list(nx.topological_sort(tg.nx_actual_route))
                transport_preds = defaultdict(lambda: [None] * len(route))
                raw_transport_preds = rec_dd()

                for i, n in enumerate(route):
                    for t in ts:
                        for mdl, predictions in results[t].items():
                            # predictions: p_s_i+1, p_s_i+2, ..., p_s_i+n
                            for j, pred in enumerate(reversed(predictions)):
                                # pred: p_s_i+n
                                s, p = pred
                                if n == s and len(predictions) - j > hor:
                                    # print("Found %d/%d" % (-i-1, len(route)))
                                    # print("Found %d/%d at %d from %s" % (i+1, len(route), len(predictions) - j - 1, t))
                                    transport_preds[mdl][i] = (s, p)
                for t in ts:
                    for mdl, predictions in results[t].items():
                        # Here the prediction values are added
                        if len(predictions) <= hor:
                            continue
                        s, p = predictions[hor]
                        raw_transport_preds[mdl][s] = p

                assert all([len(route) == len(preds) for preds in transport_preds.values()])

                timeseries = []
                for mdl, preds in raw_transport_preds.items():
                    if mdl == "labeled":
                        continue
                    if len(preds) < 20:
                        continue
                    lengths.append(len(preds))

                    # times = [pt for pt, pp in preds]
                    # values = [pp for pt, pp in preds]
                    # print(times)
                    # print(values)
                    
                    times, values = [], []
                    for s, p in preds.items():
                        for n in nx.topological_sort(tg.nx_actual_route):
                            if n == s:
                                position = tg.nx_actual_route.nodes[n].get("arrivalTime")
                                times.append(position)
                                values.append(p)

                    assert len(times) == len(values)

                    # Sort items and times
                    pls = zip(times, values)
                    pls = sorted(pls, key=lambda x: x[0])
                    
                    times = [x[0] for x in pls]
                    values = [x[1] for x in pls]

                    timeseries.append(
                        dict(
                            label=RENAME.get(mdl, mdl),
                            # times = np.linspace(0, len(preds), len(preds))
                            times=times, # if not linear else np.linspace(0, len(preds), len(preds)),
                            values=values,
                            index=0,
                            style="solid", # "dashed",
                            color=COLORS.get(mdl, "black"),
                            width=2,
                        ),
                    )
                
                if len(timeseries) < 1:
                    continue
                
                size, aspect = 10, 1.5
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(size * aspect, size))

                ds_name = "ELETA" if not use_simulation else "SIM"
                DelayProgressPlot(smooth=False, stations=stations, fontsize=25).plot_predictions(
                    tg=tg,
                    fig=fig,
                    ax=ax,
                    predictions=timeseries,
                    save=True,
                    markers=False,
                    show_stations=True,
                    has_time_axis=not linear,
                    filename="predictions/compare/%s_%s_prediction_%s.pdf" % (ds_name, transport, str(hor)),
                )
                plt.close()
            except Exception as e:
                print(e)

    if len(lengths) > 0:
        lens = torch.FloatTensor(lengths)
        print("mean: %f min: %f max: %f" % (lens.mean(), lens.min(), lens.max()))
    print("Delay stddev is", delay_stddev)

@task
def build_station_net(c, limit=1_000_000, min_occurences=None, plot=False):
    """Build station network graph
    """
    from cargonet.dataset.avgdelayv1 import EdgeAverageDelayDatasetV1

    print("Rebuilding station net")
    EdgeAverageDelayDatasetV1.build_full_graph(
        limit=limit, min_occurences=min_occurences, plot=plot
    )


@task
def build_activeroutesv1(c, limit=1, plot=True, rebuild=False, reprocess=False):
    """Build avgdelayv1 dataset
    """
    from cargonet.dataset.activeroutesv1 import build_dataset

    build_dataset(limit, plot, rebuild=rebuild, reprocess=reprocess)


@task
def build_baselinev1(c, limit=1, plot=True, rebuild=False, reprocess=False):
    """Build baseline dataset
    """
    from cargonet.dataset.baselinev1 import build_dataset

    build_dataset(limit, plot, rebuild=rebuild, reprocess=reprocess)


@task
def debug_net(c, limit=1_00):
    """Print stats and debug full network
    """
    from cargonet.dataset.dataset import RailDataset

    net, _ = RailDataset.load_full_graph()
    c = 0
    for u, v, data in net.edges(data=True):
        if c > limit:
            break
        print(data.get("delay"))
        c += 1


def get_figure_path(filename):
    base_path = os.path.dirname(os.path.realpath(__file__))
    base_fig_path = os.path.join(base_path, "fig")
    assert os.path.exists(base_fig_path)
    return os.path.join(base_fig_path, filename)

def get_dataset_path(dataset_name):
    base_path = os.path.dirname(os.path.realpath(__file__))
    base_dataset_path = os.path.join(base_path, "datasets")
    assert os.path.exists(base_dataset_path)
    return os.path.join(base_dataset_path, dataset_name)


@task
def plot_delay_distribution(
    c,
    limit=1,
    vertical=1000,
    current=False,
    pdf=False,
    station_labels=False,
    reprocess=False,
):
    """Print stats and debug full network
    """
    import torch
    from cargonet.dataset.avgdelayv1 import NodeAverageDelayDatasetV1
    from cargonet.visualization.heat import DelayByStationPlot
    from cargonet.preprocessing.datalake.retrieval import Retriever

    dataset_name = "average-delay-dataset-v1"
    dataset_path = get_dataset_path(dataset_name)

    print("Loading dataset")
    dataset = NodeAverageDelayDatasetV1(
        root=dataset_path, name=dataset_name, limit=limit, force_reprocess=reprocess
    )

    r = Retriever()
    s = r.retrieve_stations(keep_ids=True)

    reverse_mapping = {value: key for key, value in dataset.mapping.items()}

    i_current = dataset.node_feature_mapping.index("current")
    i_delay = dataset.node_feature_mapping.index("delay")

    station_delays = torch.zeros(
        len(dataset),
        dataset.number_of_nodes,
        len(dataset.node_feature_mapping),
        dtype=torch.float,
    )
    for i, sample in enumerate(dataset):
        station_delays[i] = sample.x

    print("Average station delay:", station_delays[:, :, i_delay].float().mean().item())
    print("Average current:", station_delays[:, :, i_current].float().mean().item())

    def get_y_tick_labels(interval):
        edges = [reverse_mapping[si] for si in interval]
        return [
            "%s to %s"
            % (
                s.get(e[0], dict()).get("stationName"),
                s.get(e[1], dict()).get("stationName"),
            )
            for e in edges
        ]

    for i in range(0, station_delays.size(1), vertical):
        values = station_delays[
            :, i : i + vertical, i_current if current else i_delay
        ].T
        print(values.mean(), values.max(), values.min())
        DelayByStationPlot().plot(
            values * (50 if current else 1 / 100),
            y_tick_labels=get_y_tick_labels(range(i, i + vertical))
            if station_labels
            else None,
            # x_label=[s[reverse_station_mapping[si]] for si in range(i : i + max_vertical)],
            filename="station-delay-distribution/%d_%s"
            % (i, "current" if current else "delay"),
            pdf=pdf,
            vmin=0,
            vmax=5 if current else 3,
            center=0 if current else 1,
        )


@task
def plot_station_delay_progress(c, limit=1, plot_limit=100, *timeseries):
    """Print stats and debug full network
    """
    from cargonet.dataset.avgdelayv1 import NodeAverageDelayDatasetV1
    from cargonet.visualization.delays import plot_station_delay_progress

    dataset_name = "average-delay-dataset-v1"
    dataset_path = get_dataset_path(dataset_name)

    print("Loading dataset")
    dataset = NodeAverageDelayDatasetV1(
        root=dataset_path, name=dataset_name, limit=limit,
    )

    plot_station_delay_progress(
        dataset, limit=limit, plot_limit=plot_limit, *timeseries
    )
