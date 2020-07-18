from collections import defaultdict

import numpy as np
import torch
import os.path
import pickle


class NormalizeFeatures(object):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, data):
        # linear rescale to range [0, 1]
        data.x = data.x.float()
        data.x -= self.min_val  # bring the lower range to 0
        data.x /= self.max_val  # bring the upper range to 1
        data.y = data.y.float()
        data.y -= self.min_val  # bring the lower range to 0
        data.y /= self.max_val  # bring the upper range to 1
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class MockFeatures(object):
    def sample_sine(self, seq, l, ft):
        x = torch.FloatTensor(np.linspace(0, 3, seq))
        x = x.view(seq, 1, 1).repeat(1, l, ft)
        x = x + torch.randn(1, l, 2).repeat(seq, 1, 1)
        s = (torch.sin(x) + 1.0) / 2.0
        return s

    def sine(self, data):
        x_seq, y_seq = data.x.size(0), data.y.size(0)
        l = data.x.size(1)  # Same for x and y
        ft = 2
        s = self.sample_sine(x_seq + y_seq, l, ft)
        data.x = s[:x_seq, :]
        data.y = s[x_seq : x_seq + y_seq, :]
        return data

    def __init__(self, mock_func=None):
        self.mock_func = mock_func or self.sine

    def __call__(self, data):
        return self.mock_func(data)

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class Scaler(object):
    def __init__(
        self, callback, mins, maxes, means, stds
    ):
        self.callback = callback
        self.mins = mins
        self.maxes = maxes
        self.means = means
        self.stds = stds

    @classmethod
    def _min_max_mean_std(cls, m, dim=0):
        # m[m != m] = 0
        _max, _ = m.max(dim=dim)
        _min, _ = m.min(dim=dim)
        mean = m.float().mean(dim=dim)
        std = m.float().std(dim=dim)
        return _min, _max, mean, std

    @classmethod
    def fit(cls, dataset, normalize, clamp=None, attrs=None, cache=None):

        base_path = os.path.dirname(os.path.realpath(__file__))
        models_base_path = os.path.join(base_path, "../../trained/cache")
        assert os.path.exists(models_base_path)

        # Check cache first
        if cache is not None:
            cache = os.path.join(models_base_path, cache + ".pickle")
            # if the file exists, load it
            try:
                with open(cache, "rb") as f:
                    cached = pickle.load(f)
                    return cls(callback=normalize, **cached)
            except Exception:
                pass

        attrs = attrs or dict(x=0, y=0, edge_attr=0,)
        # Fit node features
        _mins, _maxs, _means, _stds = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )

        for i, d in enumerate(dataset):
            for attr, dim in attrs.items():
                if hasattr(d, attr) and getattr(d, attr) is not None:
                    if getattr(d, attr).size(0) < 1:
                        continue
                    imin, imax, imean, istd = cls._min_max_mean_std(
                        getattr(d, attr), dim=dim
                    )
                    _mins[attr].append(imin)
                    _maxs[attr].append(imax)
                    _means[attr].append(imean)
                    _stds[attr].append(istd)

        mins, maxs, means, stds = dict(), dict(), dict(), dict()
        for attr, values in _mins.items():
            __mins, _ = torch.cat(values, dim=0).min(dim=0)
            __min, _ = __mins.min(dim=0)
            mins[attr] = __min

        for attr, values in _maxs.items():
            __maxs, _ = torch.cat(values, dim=0).max(dim=0)
            __max, _ = __maxs.max(dim=0)
            maxs[attr] = __max

        for attr, values in _means.items():
            means[attr] = torch.cat(values, dim=0).mean(dim=0)

        for attr, values in _stds.items():
            stds[attr] = torch.cat(values, dim=0).mean(dim=0)
            
        # print(stds)
        if cache is not None:
            # Save it to the cache
            with open(cache, "wb+") as f:
                pickle.dump(dict(mins=mins, maxes=maxs, means=means, stds=stds), f)

        return cls(callback=normalize, mins=mins, maxes=maxs, means=means, stds=stds)

    @classmethod
    def minmax(self, m, _min, _max):
        m -= _min  # bring the lower range to 0
        m /= _max  # bring the upper range to 1
        return m

    @classmethod
    def zscore(self, m, mean, std):
        std[std == 0] += 1e-9
        m = (m - mean) / std
        return m

    @classmethod
    def inverse_zscore(self, m, mean, std):
        m = m * std + mean
        return m

    def __call__(self, data):
        if data.x is None:
            return data
        return self.callback(
            data, mins=self.mins, maxes=self.maxes, means=self.means, stds=self.stds,
        )

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
