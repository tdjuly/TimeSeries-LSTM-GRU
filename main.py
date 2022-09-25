# -*- coding: utf-8 -*-
from args import us_args_parser, ms_args_parser, mm_args_parser
from util import train, test

# us = Univariate-SingleStep, ms = Multivariate-SingleStep, mm = Multivariate-MultiStep
model_type = 'us'

if __name__ == '__main__':
    if model_type == 'us':
        args = us_args_parser()
        LSTM_PATH = './model/Univariate-SingleStep-LSTM.pkl'

        train(args, LSTM_PATH, model_type)
        test(args, LSTM_PATH, model_type)

    elif model_type == 'ms':
        args = ms_args_parser()
        LSTM_PATH = './model/Multivariate-SingleStep-LSTM.pkl'

        train(args, LSTM_PATH, model_type)
        test(args, LSTM_PATH, model_type)

    elif model_type == 'mm':
        args = mm_args_parser()
        LSTM_PATH = './model/Multivariate-MultiStep_LSTM.pkl'

        train(args, LSTM_PATH, model_type)
        test(args, LSTM_PATH, model_type)
    else:
        print('Wrong model type')
