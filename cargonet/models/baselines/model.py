import time

import pickle
import torch
import os.path
from prettytable import PrettyTable
from torch.autograd import Variable

from cargonet.models.eval.losses import LossCollector
from cargonet.models.model import MLModel


class BaselineModel(MLModel):
    
    def __init__(
        self,
        dataset,
        node_input_dim,
        edge_input_dim,
        plot=False,
        verbose=False,
        device=None,
        batch_size=1,
        loader_batch_size=1,
        output_size=1,
        seq_len=3,
        pred_seq_len=3,
        **kwargs,
    ):
        super().__init__(
            dataset,
            verbose=verbose,
            device=device,
            plot=plot,
            batch_size=batch_size,
            loader_batch_size=loader_batch_size,
            **kwargs
        )

        self.net = self.dataset.net.to(self.device)
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.output_size = output_size
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len

    def train(self, epochs=1, print_interval=1, val=True, validation_interval=3):
        cur_time = prev_time = time.time()
        for epoch in range(1, epochs + 1):
            self.model.train()

            epoch_loss_collector = LossCollector()

            self.dataset.encoder.horizon_route_node_fts

            # Prepare batches
            if False:
                total_seq_len = self.seq_len + self.pred_seq_len
                restx = torch.zeros(0, total_seq_len, len(self.dataset.encoder.seq_route_node_fts))
                resty = torch.zeros(0, self.pred_seq_len, len(self.dataset.encoder.predict_fts))
                restea = torch.zeros(0, total_seq_len - 1, len(self.dataset.encoder.route_edge_fts))
                resttei = torch.zeros(0, total_seq_len, 2)

                for data in self.train_data:
                    if data.x is None:
                        continue
                    x = torch.cat([restx, data.x], dim=0)
                    y = torch.cat([resty, data.y], dim=0)
                    ea = torch.cat([restea, data.temporal_edge_attr], dim=0)
                    
                    for b_s in range(0, x.size(0), self.batch_size):
                        batchx = x[b_s : b_s + self.batch_size, :, :]
                        batchy = y[b_s : b_s + self.batch_size, :, :]
                        batchea = ea[b_s : b_s + self.batch_size, :, :]
                        batchx = batchx.to(self.device)
                        batchy = batchy.to(self.device)
                        batchea = batchea.to(self.device)

                        self.optimizer.zero_grad()
                        outputs, expected = self.feed(None, batchx, batchy, batchea)

                        
                        mask = data.current_transports[:outputs.size(0), -self.pred_seq_len :]
                        outputs[~mask] = 0
                        expected[~mask] = 0

                        l1_regularization = 0.
                        for param in self.model.parameters():
                            l1_regularization += param.abs().sum()
                        
                        loss = self.loss(outputs, expected)

                        loss = loss + self.l1_reg * l1_regularization

                        epoch_loss_collector.collect(outputs, expected)

                        # Compute gradients
                        loss.backward()

                        # Update parameters
                        self.optimizer.step()

                    restx = x[x.size(0) % self.batch_size :, :, :]
                    resty = y[y.size(0) % self.batch_size :, :, :]
                    restea = ea[ea.size(0) % self.batch_size :, :, :]
            else:
                for data in self.train_data:
                    if data.x is None:
                        continue

                    data = data.to(self.device)
                
                    self.optimizer.zero_grad()
                    outputs, expected = self.feed(data, data.x, data.y, data.temporal_edge_attr)

                    mask = data.current_transports[:outputs.size(0), -self.pred_seq_len :]
                    outputs[~mask] = 0
                    expected[~mask] = 0
                    
                    l1_regularization = 0.
                    for param in self.model.parameters():
                        l1_regularization += param.abs().sum()
                    
                    loss = self.loss(outputs, expected)
                    loss = loss + self.l1_reg * l1_regularization

                    epoch_loss_collector.collect(outputs, expected)

                    # Compute gradients
                    loss.backward()

                    # Update parameters
                    self.optimizer.step()

            if self.lr_scheduler:
                self.lr_scheduler.step()
            loss = epoch_loss_collector.reduce()
            self.collect_train_metrics(loss)

            cur_time = time.time()
            log = "Epoch: {:03d}, Train: {:.4f} ({:.4f}) Val: {:.4f} ({:.4f}) Acc: {:.4f} Time: {:.1f}s"
            if epoch % print_interval == 0:
                val_loss = dict(mse=-1, mae=-1, acc=-1)
                if val and epoch % validation_interval == 0:
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

    @torch.no_grad()
    def test(self, plot=False):
        self.model.eval()
        accs, val_loss_collector = [], LossCollector()
        use_train = False
        if use_train:
            print("WARNING: Using train data for testing")
        for j, data in enumerate(self.train_data if use_train else self.val_data):
            if data.x is None:
                continue
            data = data.to(self.device)

            outputs, expected = self.feed(data, data.x, data.y, data.temporal_edge_attr)
            
            mask = data.current_transports[:, -self.pred_seq_len :]
            outputs[~mask] = 0
            expected[~mask] = 0
            val_loss_collector.collect(outputs, expected)
        return accs, val_loss_collector.reduce()

    def get_full_seq(self, x, edges):
        total_seq_len = self.seq_len + self.pred_seq_len
        _edges = edges.view(-1, total_seq_len - 1, edges.size(-1))
        edges = torch.zeros(_edges.size(0), total_seq_len, edges.size(-1)).to(self.device)
        edges[:, 1:total_seq_len, :] = _edges

        return torch.cat([x, edges], dim=2)

    def feed(self, data, x, y, ea):
        assert not torch.isnan(x).any()
        assert not torch.isnan(ea).any()

        ea = ea
        x = x
        full_seq = self.get_full_seq(x, ea)

        outputs = self.model(data, x, ea, full_seq, net=self.net)
        
        if self.denormalize:
            delay_index = -1
            outputs = self.tf.inverse_zscore(
                outputs,
                mean=self.tf.means["x"][delay_index],
                std=self.tf.stds["x"][delay_index]
            )

        expected = y[:, :].view(-1, self.pred_seq_len)
        # print(expected.shape, outputs.shape)
        assert expected.shape == outputs.shape

        return outputs, expected


class BaselineSklearnModel(BaselineModel):
    
    def plot_pred(self, pred, gt, lookahead=1, linewidth=2):
        pred_range = range(0, len(gt))
        gt_range = pred_range
        plt.plot(gt_range, gt, linewidth=linewidth, color="green")
        plt.plot(pred_range, pred, linewidth=linewidth, color="red")
        plt.show()

    @classmethod
    def collect_dataset(cls, dataset, pred_lookahead=0, limit=None):
        x, y = [], []
        for data in dataset:
            if data.x is None:
                continue
            if limit is not None and limit < 0:
                break
            mask = data.current_transports[:, 10 + pred_lookahead]
            x.append(data.x[mask])
            y.append(data.y[mask])
            if limit is not None:
                limit -= 1

        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        
        y = y[:, pred_lookahead, 0]
        assert x.size(0) == y.size(0)
        return x, y

    def fit(self, idx, train_x, train_y):
        self.models[idx].fit(train_x.cpu().numpy(), train_y.cpu().numpy().ravel())

    def preprocess(self, train_x, train_y):
        train_x = train_x.view(-1, (self.seq_len + self.pred_seq_len) * self.node_input_dim)
        return train_x, train_y

    def train(self, epochs=1, print_interval=1):
        _train_loss_collector, _val_loss_collector = LossCollector(), LossCollector()
        for pred_lookahead in range(1, self.pred_seq_len + 1):

            print("Collecting dataset for lookahead", pred_lookahead)
            train_x, train_y = self.collect_dataset(
                self.train_data, pred_lookahead=pred_lookahead - 1
            )
            val_x, val_y = self.collect_dataset(
                self.val_data, pred_lookahead=pred_lookahead - 1
            )

            train_x, train_y = self.preprocess(train_x, train_y)
            val_x, val_y = self.preprocess(val_x, val_y)

            print("Training lookahead", pred_lookahead)
            self.fit(pred_lookahead-1, train_x, train_y)

            train_predictions = self.models[pred_lookahead-1].predict(train_x)
            val_predictions = self.models[pred_lookahead-1].predict(val_x)

            train_err = _train_loss_collector.collect(
                torch.from_numpy(train_predictions).float(), train_y
            )
            val_err = _val_loss_collector.collect(
                torch.from_numpy(val_predictions).float(), val_y
            )

            print(
                "lookahead", pred_lookahead, "train:", LossCollector.format(train_err)
            )
            print("lookahead", pred_lookahead, "val:", LossCollector.format(val_err))

            if self.plot:
                self.plot_pred(
                    val_predictions[:200], val_y[:200], lookahead=pred_lookahead
                )

        self.collect_train_metrics(_train_loss_collector.reduce())
        self.collect_val_metrics(_val_loss_collector.reduce())
        self.print_eval_summary()
        return _val_loss_collector

    def print_eval_summary(self, nd=2):
        x = PrettyTable()
        reduced_train = self.train_metric_collector.reduce()
        reduced_val = self.val_metric_collector.reduce()
        x.field_names = ["metric", "train", "val"]
        x.add_row(
            [
                "MSE",
                round(reduced_train["mse"], nd),
                round(reduced_val["mse"], nd),
            ]
        )
        x.add_row(
            [
                "ACC",
                round(reduced_train["acc"], nd),
                round(reduced_val["acc"], nd),
            ]
        )
        x.add_row(
            [
                "MAE",
                round(reduced_train["mae"], nd),
                round(reduced_val["mae"], nd),
            ]
        )
        x.add_row(
            [
                "RMSE",
                round(reduced_train["rmse"], nd),
                round(reduced_val["rmse"], nd),
            ]
        )
        print(x)

    @torch.no_grad()
    def test(self, plot=False):
        accs, val_loss_collector = [], LossCollector()
        for pred_lookahead in range(1, self.pred_seq_len + 1):
            print("Collecting", pred_lookahead)
            val_x, val_y = self.collect_dataset(
                self.train_data, pred_lookahead=pred_lookahead - 1, limit=None #100
            )
            print("Done collecting", pred_lookahead)

            val_x, val_y = self.preprocess(val_x, val_y)
            val_predictions = self.models[pred_lookahead-1].predict(val_x)
            val_predictions = torch.from_numpy(val_predictions).float()
            if self.denormalize:
                delay_index = -1
                val_predictions = self.tf.inverse_zscore(
                    val_predictions,
                    mean=self.tf.means["x"][delay_index],
                    std=self.tf.stds["x"][delay_index]
                )
                val_y = self.tf.inverse_zscore(
                    val_y,
                    mean=self.tf.means["x"][delay_index],
                    std=self.tf.stds["x"][delay_index]
                )
            print(val_predictions)
            print(val_y)
            val_loss_collector.collect(val_predictions, val_y)

        return accs, val_loss_collector.reduce()

    def predict(self, data, x, y, ea, **kwargs):
        """ Combine the individual models """
        predictions = torch.zeros(x.size(0), self.pred_seq_len)
        for pred_lookahead in range(self.pred_seq_len):
            out = torch.from_numpy(self.models[pred_lookahead].predict(x)).float()
            predictions[:,pred_lookahead] = out
        
        expected = y[:, :].view(-1, self.pred_seq_len)
        if self.denormalize:
            delay_index = -1
            predictions = self.tf.inverse_zscore(
                predictions,
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

        return predictions, expected

    def spec_model_filename(self, idx):
        _base = os.path.basename(self.model_state_path)
        _dir = os.path.dirname(self.model_state_path)
        return os.path.join(_dir, "%d_%s" % (idx, _base))

    def save(self, path=None):
        for i, model in enumerate(self.models):
            _path = path or self.spec_model_filename(i)
            print("Saving to", _path, i)
            with open(_path, mode="wb") as f:
                pickle.dump(model, f)

    def load(self, path=None):
        for i, _ in enumerate(self.models):
            _path = path or self.spec_model_filename(i)
            print("Loading from", _path, i)
            with open(_path, mode="rb") as f:
                self.models[i] = pickle.load(f)