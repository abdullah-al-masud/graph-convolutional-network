"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

import argparse

def parse_arguments(show=False, jupyter=False):

    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default='kaggle-molecules', help='path to the directory of the data')
    parser.add_argument('--val-ratio', type=float, default=.15, help='validation ratio for training')
    parser.add_argument('--test-ratio', type=float, default=.15, help='test data ratio for evaluation')
    parser.add_argument('--problem-type', type=str, default='energy-prediction', help='what problem are we trying to run. Two options available. 1) energy prediction, 2) bond prediction')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')
    parser.add_argument('--learning-rate', type=float, default=.0001, help='learning rate for training')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--save-path', type=str, default='output', help='path to the directory to save outputs')
    parser.add_argument('--lr-reduce', type=float, default=.995, help='learning rate reducing factor')
    parser.add_argument('--save-interval', type=int, default=50, help='interval in epochs to save intermediate model weights')
    
    args = parser.parse_args() if not jupyter else parser.parse_args([])
    if show:
        print(args)
    
    return args