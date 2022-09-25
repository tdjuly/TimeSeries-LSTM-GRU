# -*- coding:utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import decomposition
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def load_data():
    path = os.path.dirname(os.path.realpath(__file__)) + '/data/port_data.csv'
    df = pd.read_csv(path, encoding='gbk')

    columns = df.columns
    df.fillna(df.mean(), inplace=True)
    _max1 = np.max(df[columns[1]])        # df[columns[1]] is the target time series
    _min1 = np.min(df[columns[1]])

    # min-max normalisation for all data
    for i in range(len(df.columns)):
        _max = np.max(df[columns[i]])  # df[columns[1]] is the target time series
        _min = np.min(df[columns[i]])
        df[columns[i]] = (df[columns[i]] - _min) / (_max - _min)   # min-max normalisation

    return df, _max1, _min1


def nn_seq_ms(batch_size):
    print('pca data processing...')

    data, max_value, min_value = load_data()

    # PCA start
    data_before_pca = data.iloc[:, 2:21]
    pca = decomposition.PCA(n_components=4)
    pca.fit(data_before_pca)
    data_after_pca = pca.fit_transform(data_before_pca)
    data_after_pca = pd.DataFrame(data_after_pca)
    data_after_pca_columns = data_after_pca.columns

    for i in range(len(data_after_pca.columns)):
        _max = np.max(data_after_pca[data_after_pca_columns[i]])  # df[columns[1]] is the target time series
        _min = np.min(data_after_pca[data_after_pca_columns[i]])
        data_after_pca[data_after_pca_columns[i]] = (data_after_pca[data_after_pca_columns[i]] - _min) / (_max - _min)  # min-max normalisation
    # PCA end

    load = data[data.columns[1]]
    load_df = pd.DataFrame(load)
    load = load.tolist()

    data = pd.concat([load_df, data_after_pca], axis=1)
    data = np.array(data)
    data.tolist()

    seq = []
    for i in range(len(data) - 1):
        df_x = []
        df_y = []
        for j in range(i, i + 1):
            x = [load[j]]
            # for c in range(1, 1):
            #     x.append(data[j][c])
            df_x.append(x)
        df_y.append(load[i + 1])
        df_x = torch.FloatTensor(df_x)
        df_y = torch.FloatTensor(df_y).view(-1)
        seq.append((df_x, df_y))

    train_set = seq[0:int(len(seq) * 0.8)]
    test_set = seq

    train_len = int(len(train_set) / batch_size) * batch_size
    test_len = int(len(test_set) / batch_size) * batch_size
    train_set, test_set = train_set[:train_len], test_set

    train = MyDataset(train_set)
    test = MyDataset(test_set)

    train_set = DataLoader(dataset=train, batch_size=batch_size, shuffle=False, num_workers=0)
    test_set = DataLoader(dataset=test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_set, test_set, max_value, min_value
