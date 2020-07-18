import os
import random
import time
from datetime import timedelta
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torch_geometric.transforms as T
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, GCNConv, GINConv, MessagePassing, SAGEConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

from cargonet.dataset.baselinev1 import BaselineV1
from cargonet.models.baselines.model import BaselineSklearnModel
from cargonet.models.eval.losses import LossCollector
from cargonet.models.normalization import MinMaxScaler, Scaler, ZScoreScaler
from cargonet.models.sociallstm import ActiveRoutesModelLSTM, ActiveRoutesModelLSTMGAT
from cargonet.visualization.delays import DelayProgressPlot


class BaselineRandomForestModelV1(BaselineSklearnModel):
    def __init__(
        self,
        dataset,
        node_input_dim,
        edge_input_dim,
        plot=False,
        verbose=False,
        device=None,
        batch_size=1,
        output_size=1,
        seq_len=3,
        pred_seq_len=3,
        **kwargs
    ):
        super().__init__(
            dataset=dataset,
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            plot=plot,
            verbose=verbose,
            device=device,
            batch_size=batch_size,
            output_size=output_size,
            seq_len=seq_len,
            pred_seq_len=pred_seq_len,
            **kwargs
        )
        self.denormalize = True
        self.models = [
            RandomForestRegressor(
                n_estimators=400,
                # n_estimators=10,
                min_samples_split=10,
                min_samples_leaf=1,
                bootstrap=True,
                max_features="sqrt",
                max_depth=60,
                # max_depth=5,
                n_jobs=-1,
            )
        ] * self.pred_seq_len

    def preprocess(self, train_x, train_y):
        train_x = train_x[:,:,-1].view(-1, self.seq_len + self.pred_seq_len)
        return train_x, train_y

    @classmethod
    def hyperparameter_search(cls, **model_params):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ["auto", "sqrt"]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap,
        }
        pprint(random_grid)

        results = dict()
        pred_lookahead = 10
        model = cls(**model_params)

        print("Collecting dataset for lookahead", pred_lookahead)
        train_x, train_y = model.collect_dataset(
            model.train_data, pred_lookahead=pred_lookahead - 1
        )
        val_x, val_y = model.collect_dataset(
            model.val_data, pred_lookahead=pred_lookahead - 1
        )

        train_x, train_y = model.preprocess(train_x, train_y)
        val_x, val_y = model.preprocess(val_x, val_y)

        rf = RandomForestRegressor()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=random_grid,
            n_iter=15,
            cv=2,
            verbose=2,
            random_state=42,
            n_jobs=-1,
        )
        rf_random.fit(train_x.cpu().numpy(), train_y.cpu().numpy().ravel())
        results[pred_lookahead] = rf_random.best_params_

        pprint(results)
