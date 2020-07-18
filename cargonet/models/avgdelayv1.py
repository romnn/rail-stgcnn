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

from cargonet.dataset.avgdelayv1 import (
    EdgeAverageDelayDatasetV1,
    NodeAverageDelayDatasetV1,
)
from cargonet.dataset.seqavgdelay import SequencedAverageDelayDatasetV1
from cargonet.models.gcnlstm import GCNLSTM
from cargonet.models.model import MLModel
from cargonet.models.normalization import MockFeatures, NormalizeFeatures


class AverageDelayModelV1(MLModel):
    @staticmethod
    def prepare_dataset(
        dataset,
        device,
        batch_size=1,
        seq_len=10,
        pred_len=1,
        train_val_ratio=0.7,
        shuffle=True,
    ):
        """
        Prepare the dataset for training and validation
        """
        # Shuffle start indices
        time_len = len(dataset.timerange)
        start_indices = list(range(0, time_len - seq_len))
        if shuffle:
            random.shuffle(start_indices)

        split_index = int(time_len * train_val_ratio)
        train_indices = start_indices[:split_index]
        val_indices = start_indices[split_index:]
        print(
            "Split: train=[%d:%d] val=[%d:%d]"
            % (0, split_index, split_index, len(start_indices))
        )

        train_loader = DataLoader(
            dataset[train_indices], batch_size=batch_size, shuffle=shuffle
        )
        val_loader = DataLoader(
            dataset[val_indices], batch_size=batch_size, shuffle=shuffle
        )
        return train_loader, val_loader

        def generator(_indices, _seq_len, _batch_size):
            current = 0
            for t in range(0, len(_indices), _batch_size):
                xs = torch.empty((_batch_size, _seq_len, dataset.number_of_nodes)).to(
                    device
                )
                ys = torch.empty((_batch_size, dataset.number_of_nodes)).to(device)
                for index in _indices:
                    xs[t] = dataset[index : index + seq_len]
                    ys[t] = dataset[index + seq_len]
                yield xs, ys

        return (
            generator(train_indices, seq_len, batch_size),
            generator(val_indices, seq_len, batch_size),
        )

    def __init__(self, dataset, device=None, verbose=False):
        super().__init__(verbose=verbose, device=device)
        # Load the dataset
        self.dataset = dataset
        self.train_data, self.val_data = self.prepare_dataset(self.dataset, self.device)
        net = self.dataset.net.to(self.device)
        self.model = GCNLSTM(
            net=net,
            k=3,
            num_nodes=self.dataset.number_of_nodes,
            feature_size=512,
            lstm_hidden_dim=512,
            dropout=0.5,
            verbose=self.verbose,
        ).to(self.device)
        self.loss = torch.nn.MSELoss(reduction="sum")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.0001,
        )

    def train_step(self, data):
        data = data.to(self.device)
        self.model.hidden = self.model.init_hidden()
        outputs = self.model(data)
        expected = data.y.detach()[:, :, 0].view(1, self.dataset.number_of_nodes)
        loss = self.loss(outputs, expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def test(self):
        self.model.eval()
        losses = []
        for data in self.val_data:
            data = data.to(self.device)
            pred = self.model(data)
            # print("pred mean=%f min=%f max=%f" % (pred.mean().item(), pred.min().item(), pred.max().item()))
            expected = data.y[:, :, 0]
            # print("expected mean=%f min=%f max=%f" % (expected.mean().item(), expected.min().item(), expected.max().item()))
            losses.append(self.loss_MSE(pred, expected))
        return losses, torch.tensor(losses).mean().item()

        logits, accs = self.model(), []
        for _, mask in self.data("train_mask", "val_mask", "test_mask"):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    def train(self, epochs=1, print_interval=1):
        self.model.train()
        self.model.reset_parameters()
        cur_time = prev_time = time.time()
        best_val_acc = test_acc = 0
        losses = []
        loss = 0
        for epoch in range(1, epochs + 1):
            # Train
            epoch_loss = []
            for data in self.train_data:
                loss = self.train_step(data)
                epoch_loss.append(loss)
            # Validate
            losses += epoch_loss
            cur_time = time.time()
            log = "Epoch: {:03d}, Time: {:.1f}s, Train: {:.4f}"
            if epoch % print_interval == 0:
                print(log.format(epoch, cur_time - prev_time, loss))
            prev_time = cur_time
        return losses


def train_model(
    plot,
    limit=1,
    epochs=1,
    reprocess=False,
    redownload=False,
    device=None,
    train=False,
    evaluate=True,
    normalize=True,
):
    # Load the base dataset
    torch.cuda.empty_cache()

    if device:
        print("Using", device)

    base_path = os.path.dirname(os.path.realpath(__file__))
    dataset_base_path = os.path.join(base_path, "../../datasets")
    models_base_path = os.path.join(base_path, "../../trained")
    assert os.path.exists(dataset_base_path)
    assert os.path.exists(models_base_path)

    base_dataset_name = "average-delay-dataset-v1"
    dataset_name = "seq-average-delay-dataset-v1-10-1-1-1"
    base_dataset_path = os.path.join(dataset_base_path, base_dataset_name)
    dataset_path = os.path.join(dataset_base_path, dataset_name)

    # Initialize base dataset
    base_dataset = NodeAverageDelayDatasetV1(
        root=base_dataset_path,
        name=base_dataset_name,
        limit=limit,
        batch=timedelta(hours=7 * 24),
        force_reprocess=reprocess,
        force_redownload=redownload,
    )

    class Scaler(object):
        def __init__(self, xmean, xmax, xmin, xstd):
            self.xmean = xmean
            self.xmax = xmax
            self.xmin = xmin
            self.xstd = xstd

        @classmethod
        def fit(cls, dataset, clamp=None):
            # Fit node features
            xmin, xmax, xmean, xstd = [], [], [], []
            for i, d in enumerate(dataset):
                if clamp:
                    # TODO
                    d.x = torch.clamp(d.x, min=-1000, max=1000)
                imax, _ = d.x.max(dim=0)
                imin, _ = d.x.min(dim=0)
                xmax.append(imax)
                xmin.append(imin)
                xmean.append(d.x.mean(dim=0))
                xstd.append(d.x.std(dim=0))
            xmin, _ = torch.stack(xmin).min(dim=0)
            xmax, _ = torch.stack(xmax).max(dim=0)
            xmean = torch.stack(xmean).mean(dim=0)
            xstd = torch.stack(xstd).mean(dim=0)

            xmin = xmin.view(1, -1).repeat(d.x.size(0), 1)
            xmax = xmax.view(1, -1).repeat(d.x.size(0), 1)
            xmean = xmean.view(1, -1).repeat(d.x.size(0), 1)
            xstd = xstd.view(1, -1).repeat(d.x.size(0), 1)

            return cls(xmean=xmean, xmax=xmax, xmin=xmin, xstd=xstd)

        def __repr__(self):
            return "{}()".format(self.__class__.__name__)

    class MinMaxScaler(Scaler):
        def __call__(self, data):
            data.x -= self.xmin  # bring the lower range to 0
            data.x /= self.xmax  # bring the upper range to 1
            return data

    class ZScoreScaler(Scaler):
        def __call__(self, data):
            data.x = (data.x - self.xmean) / self.xstd
            return data

    minmax = ZScoreScaler.fit(base_dataset)

    # Find min and max delays and distances
    for i, d in enumerate(base_dataset):
        # Clamp first to avoid huge values
        pass

    # Initialize sequenced dataset
    dataset = SequencedAverageDelayDatasetV1(
        base_dataset,
        root=dataset_path,
        name=dataset_name,
        limit=limit,
        force_reprocess=reprocess,
        transform=minmax,
    )

    # Initialize model
    model = AverageDelayModelV1(dataset, device=device)

    print("Done")

    # Train
    if train:
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
        model.load()

    if evaluate:
        model.eval()
        print("Evaluating model...")
        
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
                # plt.plot(range(10, 12), data.y[:,0].repeat(2, 1), color="blue")
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
