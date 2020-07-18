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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR, LinearSVR
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, GCNConv, GINConv, MessagePassing, SAGEConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

from cargonet.dataset.baselinev1 import BaselineV1
from cargonet.models.baselines.model import BaselineSklearnModel
from cargonet.models.normalization import MinMaxScaler, Scaler, ZScoreScaler
from cargonet.models.sociallstm import ActiveRoutesModelLSTM, ActiveRoutesModelLSTMGAT
from cargonet.visualization.delays import DelayProgressPlot


class BaselineSVMModelV1(BaselineSklearnModel):
    
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
        **kwargs,
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
            LinearSVR(C=0.01, max_iter=10e5)
        ] * self.pred_seq_len
        
    def fit(self, idx, train_x, train_y):
        self.models[idx].fit(train_x.cpu().numpy(), train_y.cpu().numpy().ravel())

    def preprocess(self, train_x, train_y):
        train_x = train_x[:,:,-1].view(-1, self.seq_len + self.pred_seq_len)
        return train_x, train_y

    @classmethod
    def hyperparameter_search_old(cls, **model_params):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {"C": Cs, "gamma": gammas}
        pprint(param_grid)

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

        sve = SVR(kernel="rbf")
        sve = LinearSVR(C=0.1, max_iter=10e2)
        
        sve_random = RandomizedSearchCV(
            estimator=sve,
            param_distributions=param_grid,
            n_iter=10,
            cv=2,
            verbose=2,
            random_state=42,
            n_jobs=8,
        )

        # Fit the random search model
        sve_random.fit(train_x.cpu().numpy(), train_y.cpu().numpy().ravel())
        results[pred_lookahead] = sve_random.best_params_

        pprint(results)

    @classmethod
    def hyperparameter_search(cls, **model_params):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        param_grid = {"C": Cs}
        pprint(param_grid)

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

        sve = LinearSVR(C=10, max_iter=10e2)
        
        sve_random = RandomizedSearchCV(
            estimator=sve,
            param_distributions=param_grid,
            n_iter=10,
            cv=2,
            verbose=2,
            random_state=42,
            n_jobs=8,
        )

        # Fit the random search model
        sve_random.fit(train_x.cpu().numpy(), train_y.cpu().numpy().ravel())
        results[pred_lookahead] = sve_random.best_params_

        pprint(results)
