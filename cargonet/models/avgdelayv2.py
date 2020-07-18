"""
This implementation makes use of a predictor and a GCN for predicting edge features
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
from torch.utils.data import DataLoader
from torch_geometric.data import ClusterData, ClusterLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor

from logger import Logger


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GCNInference(torch.nn.Module):
    def __init__(self, weights):
        super(GCNInference, self).__init__()
        self.weights = weights

    def forward(self, x, adj):
        for i, (weight, bias) in enumerate(self.weights):
            x = adj @ x @ weight + bias
            x = np.clip(x, 0, None) if i < len(self.weights) - 1 else x
        return x


class DelayPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(DelayPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, loader, optimizer, device):
    model.train()

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        h = model(data.x, data.edge_index)

        src, dst = data.edge_index
        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(
            0, data.x.size(0), src.size(), dtype=torch.long, device=device
        )
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = src.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size, device):
    predictor.eval()
    print("Evaluating full-batch GNN on CPU...")

    weights = [
        (conv.weight.cpu().detach().numpy(), conv.bias.cpu().detach().numpy())
        for conv in model.convs
    ]
    model = GCNInference(weights)

    x = data.x.numpy()
    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1])
    adj = adj.set_diag()
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    adj = adj.to_scipy(layout="csr")

    h = torch.from_numpy(model(x, adj)).to(device)

    def test_split(split):
        source = split_edge[split]["source_node"].to(device)
        target = split_edge[split]["target_node"].to(device)
        target_neg = split_edge[split]["target_node_neg"].to(device)

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return (
            evaluator.eval({"y_pred_pos": pos_pred, "y_pred_neg": neg_pred,})[
                "mrr_list"
            ]
            .mean()
            .item()
        )

    train_mrr = test_split("eval_train")
    valid_mrr = test_split("valid")
    test_mrr = test_split("test")

    return train_mrr, valid_mrr, test_mrr


def run():

    cluster_data = ClusterData(
        data,
        num_parts=args.num_partitions,
        recursive=False,
        save_dir=dataset.processed_dir,
    )

    loader = ClusterLoader(
        cluster_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = GCN(
        data.x.size(-1),
        args.hidden_channels,
        args.hidden_channels,
        args.num_layers,
        args.dropout,
    ).to(device)
    predictor = LinkPredictor(
        args.hidden_channels, args.hidden_channels, 1, args.num_layers, args.dropout
    ).to(device)

    evaluator = Evaluator(name="ogbl-citation")
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()), lr=args.lr
        )
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, loader, optimizer, device)
            print(f"Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}")

            if epoch > 49 and epoch % args.eval_steps == 0:
                result = test(
                    model,
                    predictor,
                    data,
                    split_edge,
                    evaluator,
                    batch_size=64 * 1024,
                    device=device,
                )
                logger.add_result(run, result)

                train_mrr, valid_mrr, test_mrr = result
                print(
                    f"Run: {run + 1:02d}, "
                    f"Epoch: {epoch:02d}, "
                    f"Loss: {loss:.4f}, "
                    f"Train: {train_mrr:.4f}, "
                    f"Valid: {valid_mrr:.4f}, "
                    f"Test: {test_mrr:.4f}"
                )

        logger.print_statistics(run)
    logger.print_statistics()
