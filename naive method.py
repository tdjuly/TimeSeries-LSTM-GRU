# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from data_process import get_mape, get_rmse


if __name__ == '__main__':
    port_name = []
    output = []
    ratio = 0.8

    # -----------------------------------data processing-----------------------------------
    # train_set, test_set, m, n = nn_seq_us(batch_size=batch_size)
    path = os.path.dirname(os.path.realpath(__file__)) + '/data/port_data.csv'
    df = pd.read_csv(path, encoding='gbk')
    df.fillna(df.mean(), inplace=True)
    columns = df.columns

    train_len = int((len(df) - 1) * ratio)
    test_len = (len(df) - 1) - train_len

    for jj in range(len(df.columns)-1):
        print('\t', columns[jj+1])

        # train set
        train_real = np.asarray(df.iloc[1:train_len+1, jj+1])
        train_fitted = np.asarray(df.iloc[0:train_len, jj+1])
        train_MAPE = get_mape(train_real, train_fitted)
        train_RMSE = get_rmse(train_real, train_fitted)

        # test set
        test_real = np.asarray(df.iloc[train_len+1:, jj+1])
        test_results = np.asarray(df.iloc[train_len:train_len + test_len, jj+1])
        test_MAPE = get_mape(test_real, test_results)
        test_RMSE = get_rmse(test_real, test_results)

        _output = [columns[jj+1], train_fitted, train_real, train_MAPE, train_RMSE, test_results, test_real, test_MAPE, test_RMSE]
        output.append(_output)

    # --------------------------------save results--------------------------------
    import csv

    f = open('output.csv', 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    header = ('port', 'train_fitted', 'train_real', 'train_MAPE', 'train_RMSE', 'test_results', 'test_real', 'test_MAPE', 'test_RMSE')
    csv_writer.writerow(header)
    for data in output:
        csv_writer.writerow(data)
    f.close()
