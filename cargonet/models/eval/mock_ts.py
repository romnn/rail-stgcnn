import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def generate_random_timeseries(seed=123):
    np.random.seed(123)
    vol = 0.30
    lag = 30
    n = 10000
    df = pd.DataFrame(np.random.randn(n) * np.sqrt(vol) * np.sqrt(1 / 252.0)).cumsum()
    df = df.to_numpy()

    look_back = 3
    predict = 1
    total = len(df) - look_back - predict
    train_val_ratio = 0.3
    train_val_split = int(len(df) * train_val_ratio)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df)
    train = df[:train_val_split]
    val = df[train_val_split:]

    def create_dataset(dataset, look_back=1, predict=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - predict):
            a = dataset[i : (i + look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back + predict - 1])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back, predict)
    trainX = trainX.reshape(-1, look_back)

    valX, valY = create_dataset(val, look_back, predict)
    valX = valX.reshape(-1, look_back)

    return trainX, trainY, valX, valY
