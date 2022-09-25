# -*- coding: utf-8 -*-
import os
import random
import argparse
from itertools import chain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from models import LSTM, BiLSTM, GRU
from data_process import get_mape, get_rmse


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


seed = 20
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    _hidden_size = [8, 16, 32, 64, 100, 128, 256, 512]
    port_name = []
    output = []

    for ii in range(len(_hidden_size)):
        print(_hidden_size[ii])
        # -----------------set parameters--------------------------------
        parser = argparse.ArgumentParser()

        parser.add_argument('--epochs', type=int, default=5000, help='input dimension')
        parser.add_argument('--input_size', type=int, default=1, help='input dimension')
        parser.add_argument('--output_size', type=int, default=1, help='output dimension')
        parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
        parser.add_argument('--num_layers', type=int, default=1, help='num layers')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer, adam or SDG')
        parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
        parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction, False=One direction')
        parser.add_argument('--divide_ratio', type=float, default=0.8, help='training set ratio')

        args = parser.parse_args()

        LSTM_PATH = './model/Univariate-SingleStep-LSTM.pkl'

        # train(args, LSTM_PATH, 'us')
        input_size, hidden_size, num_layers, output_size, batch_size, ratio = \
            args.input_size, _hidden_size[ii], args.num_layers, args.output_size, args.batch_size, args.divide_ratio

        # -----------------------------------data processing-----------------------------------
        # train_set, test_set, m, n = nn_seq_us(batch_size=batch_size)
        path = os.path.dirname(os.path.realpath(__file__)) + '/data/port_data.csv'
        df = pd.read_csv(path, encoding='gbk')
        df.fillna(df.mean(), inplace=True)
        columns = df.columns

        for jj in range(len(df.columns)-1):
            print('\t', columns[jj+1])

            _max = np.max(df[columns[jj+1]])
            _min = np.min(df[columns[jj+1]])
            df[columns[jj+1]] = (df[columns[jj+1]] - _min) / (_max - _min)  # min-max normalisation
            data, m, n = df, _max, _min

            load = data[data.columns[1]]
            load = load.tolist()
            load = torch.FloatTensor(load).view(-1)  # .view is to reshape the tensor, -1 means only one row and many columns
            data = data.values.tolist()
            seq = []
            for i in range(len(data) - 1):
                df_x = []  # train input
                df_y = []  # train target

                for j in range(i, i + 1):
                    df_x.append(load[j])
                df_y.append(load[i + 1])

                df_x = torch.FloatTensor(df_x).view(-1)
                df_y = torch.FloatTensor(df_y).view(-1)

                seq.append((df_x, df_y))

            train_set = seq[0:int(len(seq) * ratio)]
            test_set = seq

            train_len = int(len(train_set))
            test_len = len(seq) - train_len

            train = MyDataset(train_set)
            test = MyDataset(test_set)

            train_set = DataLoader(dataset=train, batch_size=batch_size, shuffle=False, num_workers=0)
            test_set = DataLoader(dataset=test, batch_size=batch_size, shuffle=False, num_workers=0)

            # -----------------------------------define training model-----------------------------------
            # define model structure
            if args.bidirectional:
                model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)
            else:
                model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)      # LSTM
                # model = GRU(input_size, hidden_size, output_size).to(device)  # GRU

            # define loss function
            loss_function = nn.MSELoss().to(device)

            # define optimizer
            if args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

            # -----------------------------------training---------------------------------------------------
            loss = 0
            for i in range(args.epochs):
                for (seq, label) in train_set:
                    seq = seq.to(device)
                    label = label.to(device)
                    y_pred = model(seq)  # 等价于model.forward(seq)
                    loss = loss_function(y_pred.reshape(1, 1), label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # print('epoch', i, ':', loss.item())

            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, LSTM_PATH)

            # -------------------------------------------load fitted model------------------------------------
            if args.bidirectional:
                model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)
            else:
                model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)
                # model = GRU(input_size, hidden_size, output_size).to(device)
            model.load_state_dict(torch.load(LSTM_PATH)['model'])
            model.eval()

            # ------------------------------------------ testing----------------------------------------------
            pred = []
            y = []
            for (seq, target) in test_set:
                target = list(chain.from_iterable(target.data.tolist()))
                y.extend(target)
                seq = seq.to(device)
                with torch.no_grad():
                    y_pred = model(seq)
                    y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                    pred.extend(y_pred)

            y, pred = np.array([y]), np.array([pred])
            y = (m - n) * y + n
            y = y.flatten()
            pred = (m - n) * pred + n
            pred = pred.flatten()

            # train fitted
            train_fitted = pred[:train_len]
            train_real = y[:train_len]
            train_MAPE = get_mape(train_real, train_fitted)
            train_RMSE = get_rmse(train_real, train_fitted)
            # print("\ntrain fitted:", train_fitted)
            # print('train real:', train_real)
            # print('train_MAPE:', train_MAPE)
            # print('train_RMSE:', train_RMSE)

            # test forecast results
            test_results = pred[train_len:]
            test_real = y[train_len:]
            test_MAPE = get_mape(test_real, test_results)
            test_RMSE = get_rmse(test_real, test_results)
            # print("\ntest forecasted: ", test_results)
            # print('test real: ', test_real)
            # print('test_MAPE:', test_MAPE)
            # print('test_RMSE:', test_RMSE)

            # --------------------------------plot------------------------------------------
            # x = [i for i in range(1, len(y.T) + 1)]
            #
            # plt.plot(x, y.reshape(-1), c='green', marker='x', ms=5, alpha=0.75, label='true')
            # plt.plot(x, pred.reshape(-1), c='red', marker='o', ms=5, alpha=0.75, label='pred')
            #
            # plt.grid(axis='y')
            # plt.legend()
            # plt.show()
            _output = [hidden_size, columns[jj+1], train_fitted, train_real, train_MAPE, train_RMSE, test_results, test_real, test_MAPE, test_RMSE]
            output.append(_output)

    # --------------------------------save results--------------------------------
    import csv

    f = open('output.csv', 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    header = ('hidden_size', 'port', 'train_fitted', 'train_real', 'train_MAPE', 'train_RMSE', 'test_results', 'test_real', 'test_MAPE', 'test_RMSE')
    csv_writer.writerow(header)
    for data in output:
        csv_writer.writerow(data)
    f.close()
