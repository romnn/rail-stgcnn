import os.path
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch
import torch.autograd
import torch.nn as nn
from prettytable import PrettyTable
from datetime import datetime
from torch_geometric.data import DataLoader

from cargonet.models.eval.losses import LossCollector
from cargonet.models.utils import register_hooks
from cargonet.visualization.delays import DelayProgressPlot

if True:
    torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


val_interval = 3

class MLModel(ABC):

    FORCE_GPU = True

    def __init__(
        self,
        dataset,
        simulation=False,
        name=None,
        verbose=False,
        shuffle=False,
        shuffle_after_split=True,
        device=None,
        plot=False,
        l1_reg=0,
        weight_decay=0,
        denormalize=False,
        batch_size=1,
        loader_batch_size=1,
    ):
        self.name = name or self.__class__.__name__
        self.verbose = verbose
        self.shuffle = shuffle
        self.shuffle_after_split = shuffle_after_split
        self.loader_batch_size = loader_batch_size
        self.simulation = simulation
        self.batch_size = batch_size
        self.l1_reg = l1_reg
        self.weight_decay = weight_decay
        self.loader_batch_size = loader_batch_size
        self.plot = plot
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() or self.FORCE_GPU else "cpu")
        )

        self.train_metric_collector = LossCollector()
        self.val_metric_collector = LossCollector()
        
        self.denormalize = denormalize
        self.dataset = dataset
        self.init_loaders()

    def init_loaders(self):
        self.tf = self.dataset.transform
        self.data, self.train_data, self.val_data, self.train_indices, self.val_indices = self.prepare_dataset(
            self.dataset, batch_size=self.loader_batch_size, shuffle=self.shuffle, shuffle_after_split=self.shuffle_after_split,
        )

    @property
    def trained_model_dir(self):
        base_path = os.path.dirname(os.path.realpath(__file__))
        models_base_path = os.path.join(base_path, "../../trained")
        assert os.path.exists(models_base_path)
        return models_base_path

    @property
    def model_state_path(self):
        return os.path.join(self.trained_model_dir, self.name + ("_sim" if self.simulation else "") + ".pt")

    def collect_train_metrics(self, metrics):
        return self.train_metric_collector.collect_metrics(metrics)

    def collect_val_metrics(self, metrics):
        return self.val_metric_collector.collect_metrics(metrics)

    def print_eval_summary(self, nd=2):
        val_acc, val_loss = self.test()
        x = PrettyTable()
        x.field_names = ["metric", "train", "val"]
        x.add_row(
            [
                "MSE",
                round(self.train_metric_collector.mses[-1], nd),
                round(val_loss["mse"], nd),
            ]
        )
        x.add_row(
            [
                "ACC",
                round(self.train_metric_collector.accs[-1], nd),
                round(val_loss["acc"], nd),
            ]
        )
        x.add_row(
            [
                "MAE",
                round(self.train_metric_collector.maes[-1], nd),
                round(val_loss["mae"], nd),
            ]
        )
        x.add_row(
            [
                "RMSE",
                round(self.train_metric_collector.rmses[-1], nd),
                round(val_loss["rmse"], nd),
            ]
        )
        print(x)

    def plot_primitive_prediction(self, i, outputs, expected, smooth=False):
        outputs = outputs.view(-1).cpu().numpy()
        expected = expected.reshape(-1).cpu().numpy()

        DelayProgressPlot(smooth=smooth).plot_timeseries(
            timeseries=[
                dict(
                    label="prediction",
                    times=np.linspace(0, len(outputs), len(outputs)),
                    values=outputs,
                    index=0,
                ),
                dict(
                    label="ground truth",
                    times=np.linspace(0, len(expected), len(expected)),
                    values=expected,
                    index=0,
                ),
                dict(
                    label="diff",
                    times=np.linspace(0, len(expected), len(expected)),
                    values=outputs - expected,
                    index=1,
                ),
            ],
            has_time_axis=False,
            filename="predictions/%s/prediction-%s" % (self.name, str(i)),
        )

    def plot_prediction(self):
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

    @staticmethod
    def prepare_dataset(
        dataset, batch_size=1, train_val_ratio=0.5, shuffle=False, chunks=2*4, shuffle_after_split=True
    ):
        random.seed(123456)

        # Shuffle start indices
        time_len = len(dataset.timerange)
        start_indices = list(range(0, time_len))
        if shuffle:
            random.shuffle(start_indices)

        split_index = int(time_len * train_val_ratio)

        train_indices, val_indices = [], []
        if shuffle_after_split:
            assert train_val_ratio == 0.5
            # Minimize skew in the dataset while avoiding any overlaps
            
            chunk_size = int(time_len/chunks)
            for chunk in range(chunks):
                if chunk%2 == 0:
                    train_indices += start_indices[chunk * chunk_size:(chunk + 1) * chunk_size]
                else:
                    val_indices += start_indices[chunk * chunk_size:(chunk + 1) * chunk_size]

        else:
            # RNNs require absolute ordering
            train_indices = start_indices[:int(split_index)]
            val_indices = start_indices[int(split_index):]
            # Swap
            train_indices, val_indices = val_indices, train_indices

        print(
            "Split: train=[%d:%d] val=[%d:%d] ratio=%f shuffle=%s shuffle_after_split=%s"
            % (0, split_index, split_index, len(start_indices), train_val_ratio, shuffle, shuffle_after_split)
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
        )
        train_loader = DataLoader(
            dataset[train_indices],
            batch_size=batch_size,
            num_workers=0,
            shuffle=shuffle_after_split,
        )
        val_loader = DataLoader(
            dataset[val_indices],
            batch_size=batch_size,
            num_workers=0,
            shuffle=shuffle_after_split,
        )
        return loader, train_loader, val_loader, train_indices, val_indices

    def predict(self, *args, **kwargs):
        """ Wrap feed by default """
        return self.feed(*args, **kwargs)

    def save(self, path=None):
        print("Saving to", path or self.model_state_path)
        torch.save(self.model.state_dict(), path or self.model_state_path)

    def load(self, path=None):
        print("Loading from", path or self.model_state_path)
        self.model.load_state_dict(torch.load(path or self.model_state_path))

    def init_rnn_state(self):
        return None, None

    def train(self, epochs=1, print_interval=1, val=True, debug=False):
        cur_time = prev_time = time.time()
        torch.set_anomaly_enabled(debug)

        net_state_hidden, net_cell_states = self.init_rnn_state()

        for epoch in range(1, epochs + 1):
            self.model.train()

            epoch_loss_collector = LossCollector()

            for i, data in enumerate(self.train_data):
                # print("%d of %d" % (i, len(self.train_data)))

                if data.x is None or torch.isnan(data.x).any():
                    # print("Skipping", i, data.x, torch.isnan(data.x).any(), data.x is None)
                    continue

                data = data.to(self.device)
                self.optimizer.zero_grad()

                # Compute gradients
                outputs, expected, net_state_hidden, net_cell_states = self.feed(data, net_state_hidden, net_cell_states)
                mask = data.current_transports[:, -self.pred_seq_len :]
                # print(outputs.shape, expected.shape)
                outputs[~mask] = 0
                expected[~mask] = 0
                
                l1_regularization = 0.
                if True:
                    for param in self.model.parameters():
                        l1_regularization = l1_regularization + param.abs().sum()
                
                assert not torch.isnan(expected).any()
                loss = self.loss(outputs, expected)
                
                loss = loss + self.l1_reg * l1_regularization

                loss.backward()

                epoch_loss_collector.collect(outputs, expected)

                total_norm = 0
                for p in self.model.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm = total_norm + param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                assert total_norm == total_norm
                # print("Param norm:", total_norm)

                # Update parameters
                # nn.utils.clip_grad_norm_(self.model.parameters(), 10e0)
                self.optimizer.step()

            if self.lr_scheduler:
                self.lr_scheduler.step()
            loss = epoch_loss_collector.reduce()
            self.collect_train_metrics(loss)

            cur_time = time.time()
            log = "Epoch: {:03d}, Train: {:.4f} ({:.4f}) Val: {:.4f} ({:.4f}) Acc: {:.4f}, Time: {:.1f}s"
            if epoch % print_interval == 0:
                val_loss = dict(mse=-1, mae=-1, acc=-1)
                if val and epoch % val_interval == 0:
                    # Validate
                    val_acc, val_loss = self.test()
                print(
                    log.format(
                        epoch,
                        loss["mse"], loss["mae"],
                        val_loss["mse"], val_loss["mae"],
                        val_loss["acc"],
                        cur_time - prev_time,
                    )
                )
            prev_time = cur_time

        self.print_eval_summary()

    def bptt_train(self, epochs=1, print_interval=1, k1=1, k2=3, seq=5, skip=None, debug=False):
        cur_time = prev_time = time.time()
        torch.set_anomaly_enabled(debug)
        retain_graph = k1 < k2

        for epoch in range(1, epochs + 1):
            self.model.train()

            net_state_hidden, net_cell_states = self.init_rnn_state()
            epoch_loss_collector = LossCollector()
            states = [(None, self.init_rnn_state())]

            total_loss = 0
            for i, data in enumerate(self.train_data):
                if skip and i % skip != 0:
                    continue

                if data.x is None or torch.isnan(data.x).any():
                    # print("Skipping", i, data.x, torch.isnan(data.x).any(), data.x is None)
                    continue

                data = data.to(self.device)
                self.optimizer.zero_grad()

                # Compute gradients
                outputs, expected, net_state_hidden, net_cell_states = self.feed(data, net_state_hidden, net_cell_states)
                
                def repackage_hidden(h):
                    """Wraps hidden states in new Tensors, to detach them from their history."""

                    if isinstance(h, torch.Tensor):
                        return h.detach()
                    else:
                        return tuple(repackage_hidden(v) for v in h)

                mask = data.current_transports[:, -self.pred_seq_len:]
                outputs[~mask] = 0
                expected[~mask] = 0

                l1_regularization = 0.
                for param in self.model.parameters():
                    l1_regularization = l1_regularization + param.abs().sum()
                
                loss = self.loss(outputs, expected)

                loss = loss + self.l1_reg * l1_regularization

                total_loss = total_loss + loss

                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    param_norm = p.grad.data.norm(2)
                    total_norm = total_norm + param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                # print("Paramm norm:", total_norm)

                if i % seq == 0:
                    total_loss.backward()
                    epoch_loss_collector.collect(outputs, expected)
                    # Update parameters
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.optimizer.step()

                    # Cut the gradient graph
                    net_state_hidden = repackage_hidden(net_state_hidden)
                    net_cell_states = repackage_hidden(net_cell_states)
                    total_loss = 0

            loss = epoch_loss_collector.reduce()
            self.collect_train_metrics(loss)

            cur_time = time.time()
            log = "Epoch: {:03d}, Train: {:.4f} ({:.4f}) Val: {:.4f} ({:.4f}) Acc: {:.4f}, Time: {:.1f}s"
            if epoch % print_interval == 0:
                val_loss = dict(mse=-1, mae=-1, acc=-1)
                val_acc = [-1]
                if epoch % val_interval == 0:
                    # Validate, this uses another hidden state for the model
                    val_acc, val_loss = self.test()
                print(
                    log.format(
                        epoch,
                        loss["mse"], loss["mae"],
                        val_loss["mse"], val_loss["mae"],
                        val_loss["acc"],
                        cur_time - prev_time,
                    )
                )
            prev_time = cur_time

        self.print_eval_summary()

    @torch.no_grad()
    def bptt_test(self, plot=False):
        raise NotImplementedError

    @torch.no_grad()
    def test(self, plot=False):
        self.model.eval()
        net_state_hidden, net_cell_states = self.init_rnn_state()
        accs, val_loss_collector = [], LossCollector()
        for j, data in enumerate(self.val_data):
            if data.x is None or torch.isnan(data.x).any():
                # print("Skipping validation of", j, data)
                continue
            data = data.to(self.device)
            outputs, expected, net_state_hidden, net_cell_states = self.feed(data, net_state_hidden, net_cell_states)
            mask = data.current_transports[:, -self.pred_seq_len :]
            outputs[~mask] = 0
            expected[~mask] = 0

            # print(outputs, expected)
            val_loss_collector.collect(outputs, expected)
        return accs, val_loss_collector.reduce()

    @classmethod
    @torch.no_grad()
    def test_models(cls, models, pred_seq_len, debug=None, limit=None):
        from cargonet.models.baselines.model import BaselineModel
        from cargonet.models.utils import rec_dd
        debug = debug or []
        
        results = rec_dd()

        for model in models:
            model_name = model.name
            start = model.dataset.timerange[len(model.val_data)]
            end = model.dataset.timerange[-1]
            print("Evaluating %s (%s to %s)" % (model_name, start, end))
            model.model.eval()
            net_state_hidden, net_cell_states = model.init_rnn_state()
            accs, val_loss_collector = [], LossCollector()
            for j, data in enumerate(model.val_data):
                
                if data.x is None or torch.isnan(data.x).any():
                    continue

                if limit and limit  < 1:
                    break

                if limit:
                    limit -= 1
                
                data = data.to(model.device)

                t = model.dataset.timerange[j]

                if isinstance(model, BaselineModel):
                    outputs, expected = model.predict(data, data.x, data.y, data.temporal_edge_attr)
                else:
                    outputs, expected, net_state_hidden, net_cell_states = model.predict(data, net_state_hidden, net_cell_states)

                mask = data.current_transports[:, -pred_seq_len:]
                outputs[~mask] = 0
                expected[~mask] = 0

                assert data.transport_mask.size(0) == outputs.size(0) == expected.size(0)

                for i, tm in enumerate(data.transport_mask):
                    i_mask = data.current_transports[i, -pred_seq_len:]
                    i_stations = data.routes[i, -pred_seq_len:][i_mask]
                    i_outputs = outputs[i][i_mask].view(-1)
                    i_expected = expected[i][i_mask].view(-1)
                    
                    assert i_stations.shape == i_outputs.shape
                    
                    i_stations = i_stations.tolist()
                    i_outputs = i_outputs.tolist()
                    i_expected = i_expected.tolist()

                    results[tm.item()][t]["labeled"] = list(zip(i_stations, i_expected))
                    results[tm.item()][t][model_name] = list(zip(i_stations, i_outputs))

        return results
