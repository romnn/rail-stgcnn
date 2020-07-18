import os
import random
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torch_geometric.transforms as T
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, GCNConv, GINConv, MessagePassing, SAGEConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

from cargonet.dataset.activeroutesv1 import ActiveRoutesV1
from cargonet.dataset.simulator import Simulation
from cargonet.models.eval.losses import LossCollector, MAELoss
from cargonet.models.gtcn import ActiveRoutesModelTCN
from cargonet.models.model import MLModel
from cargonet.models.normalization import MinMaxScaler, Scaler, ZScoreScaler
from cargonet.models.sociallstm import ActiveRoutesModelLSTM, ActiveRoutesModelLSTMGAT
from cargonet.models.stgcnv1 import ActiveRoutesModelSTGCNV1
from cargonet.visualization.delays import DelayProgressPlot

class ActiveRoutesModelV1(MLModel):
    
    @property
    def model_state_path(self):
        # # "_stateful" if self.use_rnn else ""
        return os.path.join(self.trained_model_dir, self.name + (
            "_sim" if self.simulation else ""
        ) + ".pt")

    def __init__(
        self,
        dataset,
        node_input_dim,
        edge_input_dim,
        output_size=1,
        seq_len=3,
        pred_seq_len=3,
        # rnn_size=64 + 16,
        rnn_size=64 + 16,
        use_rnn=False,
        dropout=0.1, # 4,  # .3,
        # dropout=0.6 , # 0.3
        # embedding_size=32,
        # embedding_size=64,
        embedding_size=64,
        lr=0.001,
        l1_reg=0.00, # 0001, #0001, #.01, # 0001,
        weight_decay=0.001,
        max_transports=1000,
        grad_clip=False,
        shuffle_after_split=None,
        **kwargs
    ):
        shuffle_after_split = (not use_rnn) if (shuffle_after_split is None) else shuffle_after_split
        super().__init__(dataset, 
            l1_reg=l1_reg, shuffle_after_split=shuffle_after_split, **kwargs) # chunks=2,
        
        self.use_rnn = use_rnn
        self.name = self.name + ("_stateful" if self.use_rnn else "")
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.grad_clip = grad_clip
        self.embedding_size = embedding_size
        self.max_transports = max_transports
        

        self.output_size = output_size
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len

        net = self.dataset.net.to(self.device)
        self.model = ActiveRoutesModelTCN(
            device=self.device,
            input_dim=self.node_input_dim,
            output_size=self.output_size,
            seq_len=self.seq_len,
            pred_seq_len=self.pred_seq_len,
            rnn_size=self.rnn_size,
            use_rnn=self.use_rnn,
            dropout=self.dropout,
            embedding_size=self.embedding_size,
            max_transports=self.max_transports,
        ).to(self.device)

        self.net = self.dataset.net.to(self.device)
        self.loss = torch.nn.MSELoss()
        # self.loss = MAELoss()

        """
        self.optimizer = torch.optim.AdamW([
            dict(params=self.model.encoder.parameters(), lr=lr, weight_decay=weight_decay),
            dict(params=self.model.stgblock.parameters(), lr=lr, weight_decay=weight_decay),
            dict(params=self.model.start_conv.parameters(), lr=lr, weight_decay=weight_decay),
            dict(params=self.model.final_conv.parameters(), lr=lr, weight_decay=weight_decay),
            dict(params=self.model.ground_encoder.parameters(), lr=lr, weight_decay=weight_decay),
            dict(params=self.model.ground_sync.parameters(), lr=lr, weight_decay=weight_decay),
            dict(params=self.model.ground_sync2.parameters(), lr=lr, weight_decay=weight_decay),
            dict(params=self.model.cell.parameters(), lr=0.00001, weight_decay=0.1),
            # self.model.parameters(), lr=lr, weight_decay=weight_decay
        ])  # lr=0.0001
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001) # , weight_decay=0)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.000000001, weight_decay=1.0)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)
        # optimizer = torch.optim.Adam(net.parameters(), weight_decay=args.lambda_param)

        self.lr_scheduler = None
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        
    def init_rnn_state(self):
        net_size = self.dataset.net.x.size(0)
        net_state_hidden = torch.autograd.Variable(torch.zeros(net_size, self.rnn_size), requires_grad=False).to(
            self.device
        )
        net_cell_states = torch.autograd.Variable(torch.zeros(net_size, self.rnn_size), requires_grad=False).to(self.device)
        return net_state_hidden, net_cell_states

    def feed(self, data, net_state_hidden=None, net_cell_states=None):
        x = data.x
        y = data.y
        
        net_size = self.net.x.size(0)
        if net_state_hidden is None:
            net_state_hidden = Variable(torch.zeros(net_size, self.rnn_size))
        if net_cell_states is None:
            net_cell_states = Variable(torch.zeros(net_size, self.rnn_size))

        net_state_hidden = net_state_hidden.to(self.device)
        net_cell_states = net_cell_states.to(self.device)

        outputs, net_state_hidden, net_cell_states = self.model(
            data,
            self.net,
            net_state_hidden,
            net_cell_states,
            data.num_transports,
            data.transport_mask,
            data.current_transports,
        )
        outputs = outputs
        expected = y.view(-1, self.pred_seq_len)
        if self.denormalize:
            delay_index = -1
            outputs = self.tf.inverse_zscore(
                outputs,
                mean=self.tf.means["x"][delay_index],
                std=self.tf.stds["x"][delay_index]
            )
            expected = self.tf.inverse_zscore(
                expected,
                mean=self.tf.means["x"][delay_index],
                std=self.tf.stds["x"][delay_index]
            )
        
        # print(expected.shape, outputs.shape)
        assert expected.shape == outputs.shape
        
        return outputs, expected, net_state_hidden, net_cell_states


def train_model(
    plot,
    limit=1,
    epochs=100,
    reprocess=False,
    redownload=False,
    device=None,
    train=False,
    bptt=False,
    evaluate=True,
):

    torch.cuda.empty_cache()
    if device:
        print("Using", device)
    print("bptt", bptt)

    base_path = os.path.dirname(os.path.realpath(__file__))
    dataset_base_path = os.path.join(base_path, "../../datasets")
    models_base_path = os.path.join(base_path, "../../trained")
    assert os.path.exists(dataset_base_path)
    assert os.path.exists(models_base_path)

    dataset_name = "active-routes-v1"
    dataset_path = os.path.join(dataset_base_path, dataset_name)

    simulation_dataset_name = "simulation-v1"
    simulation_dataset_path = os.path.join(dataset_base_path, simulation_dataset_name)

    ds_options = dict(seq_len=10, pred_seq_len=10,)
    batch_hours = 7 * 24 # 1 Week

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
            limit=limit,
            batch=timedelta(hours=batch_hours),
            force_reprocess=reprocess,
            force_redownload=redownload,
            **ds_options
        )

    denormalize = False
    model_options = dict(
        node_input_dim=len(dataset.encoder.seq_route_node_fts),
        edge_input_dim=len(dataset.encoder.route_edge_fts),
        simulation=use_simulation,
        denormalize=denormalize,
    )

    def normalize_func(data, means, stds, **kwargs):
        data.x = Scaler.zscore(data.x, mean=means["x"], std=stds["x"])
        if denormalize:
            delay_index = -1
            data.y = Scaler.zscore(data.y, mean=means["x"][delay_index], std=stds["x"][delay_index])
        data.temporal_edge_attr = Scaler.zscore(
            data.temporal_edge_attr,
            mean=means["temporal_edge_attr"],
            std=stds["temporal_edge_attr"],
        )
        assert not torch.isnan(data.temporal_edge_attr).any()
        assert not torch.isnan(data.x).any()
        return data

    # Initialize model
    model = ActiveRoutesModelV1(
        dataset, device=device, shuffle=False, loader_batch_size=1, use_rnn=bptt,
        shuffle_after_split=None if not use_simulation else (not bptt),
        **ds_options, **model_options
    )

    print("fitting normalization")
    cache = "%s_norm_%d_%d" % (dataset.name, batch_hours, limit)
    z_score_norm = Scaler.fit(
        model.train_data,
        normalize=normalize_func,
        attrs=dict(temporal_edge_attr=1, x=1, y=1,),
        cache=cache
    )
    model.dataset.transform = z_score_norm
    model.init_loaders()
    print("done fitting normalization")

    # Train
    if train:
        if bptt:
            train_losses = model.bptt_train(epochs=epochs)
        else:
            train_losses = model.train(epochs=epochs)
        model.save()
        if train_losses:
            # Plot loss curve
            plt.plot(train_losses)
            plt.savefig(
                os.path.join(models_base_path, model.name + "_loss.pdf"),
                format="pdf",
                dpi=600,
            )
    else:
        # Load the model
        try:
            model.load()
        except FileNotFoundError:
            print("No trained model to load. Train one first using --train")

    if evaluate:
        print("Avaluating ")
        val_accs, val_losses = model.test()
        print(LossCollector.format(val_losses))
        plot_len = 400
        model.plot_primitive_prediction(
            "val", val_losses["ys"][-plot_len:], val_losses["xs"][-plot_len:]
        )

    return
    if evaluate:
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
                # continue
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
