# -*- coding:utf-8 -*-
from itertools import chain
import torch
from torch import nn
from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt

from models import LSTM, BiLSTM, GRU
from data_process import nn_seq_us, nn_seq_ms, nn_seq_mm, device, get_mape, get_rmse, setup_seed
# from data_precess_PCA import nn_seq_ms

setup_seed(20)


def train(args, path, flag):
    """
    args: model inforamtion
    path: location to save the best model
    flag: model type (single/multi-variate+single/multi-step)
    """

    input_size, hidden_size, num_layers, output_size, batch_size = \
        args.input_size, args.hidden_size, args.num_layers, args.output_size, args.batch_size

    # select different data preprocessing method
    # Dtr = training set, Dte = testing set
    if flag == 'us':
        train_set, test_set, m, n = nn_seq_us(batch_size=batch_size)
    elif flag == 'ms':
        train_set, test_set, m, n = nn_seq_ms(batch_size=batch_size)
    else:
        train_set, test_set, m, n = nn_seq_mm(batch_size=batch_size, num=args.output_size)

    # define model structure (LSTM / BiLSTM)
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)
    else:
        # model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)      # LSTM
        model = GRU(input_size, hidden_size, output_size).to(device)        # GRU

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
            y_pred = model(seq)     # 等价于model.forward(seq)
            loss = loss_function(y_pred.reshape(1, 1), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch', i, ':', loss.item())

    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, path)


def test(args, path, flag):

    input_size, hidden_size, num_layers, output_size, batch_size = \
        args.input_size, args.hidden_size, args.num_layers, args.output_size, args.batch_size

    if flag == 'us':
        train_set, test_set, m, n = nn_seq_us(batch_size=batch_size)
    elif flag == 'ms':
        train_set, test_set, m, n = nn_seq_ms(batch_size=batch_size)
    else:
        train_set, test_set, m, n = nn_seq_mm(batch_size=batch_size, num=args.output_size)

    # ----------------------load fitted model-------------------------------------
    print('loading model...')
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)
    else:
        # model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)
        model = GRU(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(torch.load(path)['model'])
    model.eval()

    # ------------------------predict / testing----------------------------------------------
    print('\n\npredicting...')
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
    pred = (m - n) * pred + n

    y = y.flatten()
    pred = pred.flatten()
    print('fitted: ', pred[0:12])
    print('MAPE:', get_mape(y[0:12], pred[0:12]))
    print('RMSE:', get_rmse(y[0:12], pred[0:12]))

    print("pred:", pred[-5:])
    print('MAPE:', get_mape(y[-5:], pred[-5:]))
    print('RMSE:', get_rmse(y[-5:], pred[-5:]))

    # --------------------------------plot------------------------------------------
    print('plotting...')
    x = [i for i in range(1, len(y.T)+1)]

    plt.plot(x, y.reshape(-1), c='green', marker='x', ms=5, alpha=0.75, label='true')
    plt.plot(x, pred.reshape(-1), c='red', marker='o', ms=5, alpha=0.75, label='pred')

    plt.grid(axis='y')
    plt.legend()
    plt.show()

    # ------------save model info & result--------------------

