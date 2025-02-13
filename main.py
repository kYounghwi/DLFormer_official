# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:33:37 2025

@author: Younghwi Kim
"""
import torch
from torch import nn
from torch import optim

import DLFormer.exp as EXP
import src.data_factory as data_factory

from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train DLFormer model')
    parser.add_argument('--model_name', type=str, default='DLFormer')
    parser.add_argument('--data', type=str, default='Exchange')
    parser.add_argument('--root_path', type=str, default='data/')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--pred', type=int, default=3, help='Forecasting term')
    parser.add_argument('--seq', type=int, default=12, help='Window size')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=256, help='Attention dimension')
    parser.add_argument('--d_ff', type=int, default=512, help='Feedforward dimension')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--dropout', type=float, default=0.01)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', default=True, action='store_true')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--plot_interval', type=int, default=30)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--loss', type=str, default='mse')
    return parser.parse_args()

def main():
    args = get_args()
    
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'{args.device} is available')
    
    now = datetime.now()
    df, args.freq, args.embed = data_factory.data_select(args.data, args.root_path)
    args.n_features = df.shape[1]
    args.cols = df.columns
    args.label_len = min(args.pred, args.seq)
    
    train_data, train_loader = data_factory.data_provider(args.root_path, args.data, args.features, args.batch_size, args.seq, args.label_len, args.pred, 'train')
    val_data, val_loader = data_factory.data_provider(args.root_path, args.data, args.features, args.batch_size, args.seq, args.label_len, args.pred, 'val')
    test_data, test_loader = data_factory.data_provider(args.root_path, args.data, args.features, args.batch_size, args.seq, args.label_len, args.pred, 'test')
    
    model = EXP.build_model(args)
    
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.MAELoss()
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr) 
    
    EXP.train(args, model, criterion, optimizer, train_loader, val_loader, now)
    EXP.test(args, criterion, optimizer, test_loader, now)

if __name__ == "__main__":
    main()

