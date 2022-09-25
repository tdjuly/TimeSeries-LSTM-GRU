# -*- coding:utf-8 -*-
import argparse
import torch


# Univariate-SingleStep-LSTM
def us_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=5, help='input dimension')
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

    args = parser.parse_args()

    return args


# Multivariate-SingleStep-LSTM
def ms_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=5000, help='input dimension')
    parser.add_argument('--input_size', type=int, default=2, help='input dimension')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')

    args = parser.parse_args()

    return args


# Multivariate-MultiStep-LSTM
def mm_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100, help='input dimension')
    parser.add_argument('--input_size', type=int, default=7, help='input dimension')
    parser.add_argument('--output_size', type=int, default=4, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=True, help='LSTM direction')

    args = parser.parse_args()

    return args
