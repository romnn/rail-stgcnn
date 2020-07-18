import os.path
from datetime import timedelta

import matplotlib.pyplot as plt
import torch

from cargonet.dataset.activeroutesv1 import ActiveRoutesV1
from cargonet.dataset.simulator import Simulation
from cargonet.models.baselines.fc import FCModelV1
from cargonet.models.baselines.fc2 import FCModelV2
from cargonet.models.baselines.lstm import BaselineLSTMModelV1
from cargonet.models.baselines.rf import BaselineRandomForestModelV1
from cargonet.models.baselines.svm import BaselineSVMModelV1
from cargonet.models.baselines.tcn import BaselineTCNModelV1
from cargonet.models.baselines.average import BaselineAverageModelV1
from cargonet.models.baselines.timeshift import BaselineTimeshiftModelV1
from cargonet.models.eval.losses import LossCollector
from cargonet.models.normalization import Scaler


def train_model(
    plot=True,
    limit=15,
    epochs=100,
    search=False,
    all=False,
    lstm=False,
    ts=False,
    rf=False,
    tcn=False,
    svm=False,
    fc=False,
    fc2=False,
    avg=False,
    train=False,
    evaluate=True,
    reprocess=False,
    redownload=False,
    device=None,
):
    torch.cuda.empty_cache()
    if device:
        print("Using", device)

    base_path = os.path.dirname(os.path.realpath(__file__))
    dataset_base_path = os.path.join(base_path, "../../../datasets")
    models_base_path = os.path.join(base_path, "../../../trained")
    assert os.path.exists(dataset_base_path)
    assert os.path.exists(models_base_path)

    dataset_name = "active-routes-v1"
    dataset_path = os.path.join(dataset_base_path, dataset_name)

    simulation_dataset_name = "simulation-v2"
    simulation_dataset_path = os.path.join(dataset_base_path, simulation_dataset_name)

    ds_options = dict(seq_len=10, pred_seq_len=10,)
    batch_hours = 7 * 24

    use_simulation = False

    if use_simulation:
        dataset = Simulation(
            root=simulation_dataset_path,
            name=simulation_dataset_name,
            limit=32 * 10 * 2,
            force_reprocess=reprocess,
            **ds_options
        )
    else:
        dataset = ActiveRoutesV1(
            root=dataset_path,
            name=dataset_name,
            limit=32,
            batch=timedelta(hours=batch_hours),
            force_reprocess=reprocess,
            force_redownload=redownload,
            normalize_net=not (ts or avg), 
            **ds_options
        )

    def normalize_func(data, means, stds, **kwargs):
        data.x = Scaler.zscore(
            data.x, mean=means["x"], std=stds["x"]
        )
        if (svm or rf):
            delay_index = -1
            data.y = Scaler.zscore(
                data.y, mean=means["x"][delay_index], std=stds["x"][delay_index]
            )
        data.temporal_edge_attr = Scaler.zscore(
            data.temporal_edge_attr,
            mean=means["temporal_edge_attr"],
            std=stds["temporal_edge_attr"],
        )
        return data

    model_options = dict(
        node_input_dim=len(dataset.encoder.seq_route_node_fts),
        edge_input_dim=len(dataset.encoder.route_edge_fts),
        simulation=use_simulation,
        shuffle=False # True
    )

    batch_size = 2048

    def process_model(model_cls, params):
        model = model_cls(**params)
        if train:
            cache = "%s_norm_%d_%d" % (model.dataset.name, batch_hours, limit)
            if not (ts or avg):
                print("fitting normalization")
                z_score_norm = Scaler.fit(
                    model.train_data,
                    normalize=normalize_func,
                    attrs=dict(temporal_edge_attr=1, x=1, y=1,),
                    cache=cache
                )
                model.dataset.transform = z_score_norm
                model.init_loaders()
                print("done fitting normalization")
            train_losses = model.train(epochs=epochs)
            model.save()
            if train_losses and False:
                # Plot loss curve
                plt.plot(train_losses)
                plt.savefig(
                    os.path.join(models_base_path, model.name + "_loss.pdf"),
                    format="pdf",
                    dpi=600,
                )
        elif search:
            model_cls.hyperparameter_search(**params)
        else:
            # Load the model
            try:
                model.load()
            except FileNotFoundError:
                print("No trained model to load. Train one first using --train")

        if evaluate:
            print("Testing the model")
            if not (ts or avg):
                print("fitting normalization")
                cache = "%s_norm_%d_%d" % (model.dataset.name, batch_hours, limit)
                z_score_norm = Scaler.fit(
                    model.train_data,
                    normalize=normalize_func,
                    attrs=dict(temporal_edge_attr=1, x=1, y=1,),
                    cache=cache
                )
                model.dataset.transform = z_score_norm
                model.init_loaders()
                print("done fitting normalization")
            val_accs, val_losses = model.test()
            print(LossCollector.format(val_losses))
            plot_len = 200
            model.plot_primitive_prediction(
                "val", val_losses["ys"][:plot_len], val_losses["xs"][:plot_len]
            )

    # Initialize model
    init_params = dict(dataset=dataset, device=device, plot=plot, loader_batch_size=1, batch_size=batch_size, **ds_options, **model_options)
    if lstm:
        process_model(BaselineLSTMModelV1, init_params)
    elif svm:
        process_model(BaselineSVMModelV1, init_params)
    elif fc:
        process_model(FCModelV1, init_params)
    elif fc2:
        process_model(FCModelV2, init_params)
    elif rf:
        process_model(BaselineRandomForestModelV1, init_params)
    elif ts:
        process_model(BaselineTimeshiftModelV1, init_params)
    elif tcn:
        process_model(BaselineTCNModelV1, init_params)
    elif avg:
        process_model(BaselineAverageModelV1, init_params)
    else:
        print("No baseline model specified")

    return

    if evaluate:
        model.eval()
        print("Evaluating model...")
        val_acc, val_loss = model.test(plot=plot)
        print("Validation acc:", val_acc.view(-1))
        print("Validation MSE loss: {:.4f}".format(val_loss))
        print("Mean validation acc: {:.4f}".format(val_acc.mean().item()))
        return

        from cargonet.models.predictor import AvgDelayV1Predictor
        from cargonet.visualization.delays import plot_station_delay_progress

        p = AvgDelayV1Predictor(model=model, dataset=dataset)

        for d, sample in enumerate(dataset[:1]):
            pred = model.predict(sample)
            for s in range(0, 3):
                plt.plot(
                    range(0, 10),
                    sample.x[:, s, 0].cpu().detach().numpy(),
                    color="black",
                )
                plt.plot(
                    range(10, 12),
                    sample.y[:, s, 0].repeat(2, 1).cpu().detach().numpy(),
                    color="blue",
                    linestyle="solid",
                )
                plt.plot(
                    range(10, 12),
                    pred.T[s, 0].repeat(2, 1).cpu().detach().numpy(),
                    color="red",
                    linestyle="dashed",
                )
                plt.show()

        return

        node_max = 1000
        node_batch = 1000
        time_batch = 1000
        for b in range(0, dataset.number_of_nodes, node_batch):
            for t in range(0, len(dataset), time_batch):
                if node_max <= b:
                    return

                ds = dataset[t : t + time_batch]
                # Ground truth
                station_delays = torch.zeros(time_batch, node_batch, dtype=torch.float)
                print(station_delays.shape)
                for d, sample in enumerate(ds):
                    station_delays[d] = sample.x.view(-1)[b : b + node_batch].detach()

                plt.plot(station_delays[:, i].cpu().detach().numpy(), color="red")
                plt.show()

                # Predict
                preds = p.compare_predictions(
                    dataset=ds, b=b, time_batch=time_batch, node_batch=node_batch
                )

                for edge, i in dataset.mapping.items():
                    u, v = edge

                    i -= node_batch
                    if not i in range(node_batch):
                        continue

                    test = station_delays[:, i]
                    print(test.mean(), test.min(), test.max())
                    if station_delays[:, i].max() <= 0:
                        continue

                    plot_station_delay_progress(
                        u,
                        v,
                        dataset,
                        timeseries=[
                            dict(
                                times=dataset.timerange[t : t + time_batch],
                                values=station_delays[:, i].cpu().detach().numpy(),
                                label="Ground truth",
                                style="solid",
                                color="black",
                            ),
                            dict(
                                times=dataset.prediction_timerange[t : t + time_batch],
                                values=preds[:, i].cpu().detach().numpy(),
                                label="Prediction [1]",
                                style="dashed",
                                color="blue",
                            ),
                        ],
                    )
