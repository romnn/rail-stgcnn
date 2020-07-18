import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        return torch.sqrt(self.mse(x, y))


class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        abs_err = torch.abs(x - y)
        return abs_err.mean()


class ACCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, tolerance=30):
        diff = torch.abs(y - x)
        valid_total = diff[diff == diff].sum()
        correct = diff[diff < tolerance].sum()
        return correct / valid_total


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = MAELoss()
        self.acc = ACCLoss()
        self.rmse = RMSELoss()

    def forward(self, x, y):
        return dict(
            mse=self.mse(x, y).item(),
            mae=self.mae(x, y).item(),
            acc=self.acc(x, y).item(),
            rmse=self.rmse(x, y).item(),
        )


class LossCollector:
    def __init__(self):
        self.loss = CombinedLoss()
        self.mses, self.accs, self.maes, self.rmses = [], [], [], []
        self._xs, self._ys = [], []

    def collect(self, x, y):
        self._xs.append(x.reshape(-1))
        self._ys.append(y.reshape(-1))
        metrics = self.loss(x, y)
        return self.collect_metrics(metrics)

    def collect_metrics(self, metrics):
        self.mses.append(metrics["mse"])
        self.accs.append(metrics["acc"])
        self.maes.append(metrics["mae"])
        self.rmses.append(metrics["rmse"])
        return metrics

    def reduce(self):
        self.xs = None if len(self._xs) < 1 else torch.cat(self._xs, dim=0)
        self.ys = None if len(self._ys) < 1 else torch.cat(self._ys, dim=0)
        mse=torch.FloatTensor(self.mses)
        mae=torch.FloatTensor(self.maes)
        acc=torch.FloatTensor(self.accs)
        rmse=torch.FloatTensor(self.rmses)
        return dict(
            mse=mse[mse == mse].mean().item(),
            mae=mae[mae == mae].mean().item(),
            acc=acc[acc == acc].mean().item(),
            rmse=rmse[rmse == rmse].mean().item(),
            xs=self.xs,
            ys=self.ys,
        )

    @staticmethod
    def format(metrics):
        return "ACC={:.4f} MSE={:.4f} MAE={:.4f} RMSE={:.4f}".format(
            metrics["acc"], metrics["mse"], metrics["mae"], metrics["rmse"]
        )

    def summary(self):
        return self.format(self.reduce())


def loss(have, want):
    print("MAE  {:.4f}".format(mae(have, want).item()))
    print("ACC {:.4f}".format(acc(have, want).item()))
    print("RMSE {:.4f}".format(rmse(have, want).item()))
