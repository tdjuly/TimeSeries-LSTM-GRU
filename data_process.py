# -*- coding:utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True     # for GUP cuda only


def load_data():
    path = os.path.dirname(os.path.realpath(__file__)) + '/data/port_data2.csv'
    df = pd.read_csv(path, encoding='gbk')
    columns = df.columns
    df.fillna(df.mean(), inplace=True)
    _max1 = np.max(df[columns[1]])        # df[columns[1]] is the target time series
    _min1 = np.min(df[columns[1]])
    for i in range(len(df.columns)):
        _max = np.max(df[columns[i]])  # df[columns[1]] is the target time series
        _min = np.min(df[columns[i]])
        df[columns[i]] = (df[columns[i]] - _min) / (_max - _min)   # min-max normalisation

    return df, _max1, _min1


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def nn_seq_us(batch_size):
    print('data processing...')
    data, max_value, min_value = load_data()
    load = data[data.columns[1]]
    load = load.tolist()
    load = torch.FloatTensor(load).view(-1)     # .view is to reshape the tensor, -1 means only one row and many columns
    data = data.values.tolist()
    seq = []
    for i in range(len(data) - 1):
        df_x = []       # train input
        df_y = []       # train target

        for j in range(i, i + 1):
            df_x.append(load[j])
        df_y.append(load[i + 1])

        df_x = torch.FloatTensor(df_x).view(-1)
        df_y = torch.FloatTensor(df_y).view(-1)

        seq.append((df_x, df_y))

    train_set = seq[0:int(len(seq) * 0.7)]
    # test_set = seq[int(len(seq) * 0.8):len(seq)]
    test_set = seq

    train_len = int(len(train_set) / batch_size) * batch_size
    test_len = int(len(test_set) / batch_size) * batch_size
    # train_set, test_set = train_set[:train_len], test_set[:test_len]
    train_set, test_set = train_set[:train_len], test_set

    train = MyDataset(train_set)
    test = MyDataset(test_set)

    train_set = DataLoader(dataset=train, batch_size=batch_size, shuffle=False, num_workers=0)
    test_set = DataLoader(dataset=test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_set, test_set, max_value, min_value


def nn_seq_ms(batch_size):
    print('data processing...')
    data, max_value, min_value = load_data()
    load = data[data.columns[1]]
    load = load.tolist()
    data = data.values.tolist()
    seq = []
    for i in range(len(data) - 1):
        df_x = []
        df_y = []
        for j in range(i, i + 1):
            x = [load[j], data[j][21]]
            for c in range(2, 21):
                x.append(data[j][c])
            df_x.append(x)
        df_y.append(load[i + 1])
        df_x = torch.FloatTensor(df_x)
        df_y = torch.FloatTensor(df_y).view(-1)
        seq.append((df_x, df_y))

    train_set = seq[0:int(len(seq) * 0.7)]
    test_set = seq

    train_len = int(len(train_set) / batch_size) * batch_size
    test_len = int(len(test_set) / batch_size) * batch_size
    train_set, test_set = train_set[:train_len], test_set

    train = MyDataset(train_set)
    test = MyDataset(test_set)

    train_set = DataLoader(dataset=train, batch_size=batch_size, shuffle=False, num_workers=0)
    test_set = DataLoader(dataset=test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_set, test_set, max_value, min_value


def nn_seq_mm(batch_size, num):
    print('data processing...')
    data, max_value, min_value = load_data()
    load = data[data.columns[1]]
    load = load.tolist()
    data = data.values.tolist()
    seq = []

    for i in range(0, len(data) - 24 - num, num):
        df_x = []
        df_y = []

        for j in range(i, i + 24):
            x = [load[j]]
            for c in range(2, 8):
                x.append(data[j][c])
            df_x.append(x)

        for j in range(i + 24, i + 24 + num):
            df_y.append(load[j])

        df_x = torch.FloatTensor(df_x)
        df_y = torch.FloatTensor(df_y).view(-1)
        seq.append((df_x, df_y))

    # print(seq[-1])
    train_set = seq[0:int(len(seq) * 0.7)]
    test_set = seq[int(len(seq) * 0.7):len(seq)]

    train_len = int(len(train_set) / batch_size) * batch_size
    test_len = int(len(test_set) / batch_size) * batch_size
    train_set, test_set = train_set[:train_len], test_set[:test_len]

    train = MyDataset(train_set)
    test = MyDataset(test_set)
    train_set = DataLoader(dataset=train, batch_size=batch_size, shuffle=False, num_workers=0)
    test_set = DataLoader(dataset=test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_set, test_set, max_value, min_value


def get_mape(x, y):
    """
    :param x:true
    :param y:pred
    :return:MAPE
    """
    return np.mean(np.abs(100*(x-y)/x))


def get_rmse(x, y):
    return np.sqrt(np.mean(np.square(x - y)))
