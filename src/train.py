"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

import model
import pickle
import os
import sys
from msdlib.mlutils import modeling
import pdb
sys.path.append(os.getcwd())
from config import parse_arguments


def train_for_energy(data_path, args):
    if isinstance(data_path, str):
        m, train_loader, val_loader, test_loader = pickle.load(open(data_path, 'rb'))
    else:
        m, train_loader, val_loader, test_loader = data_path
    
    node_features = train_loader.dataset[0].x.shape[1]
    edge_features = train_loader.dataset[0].edge_attr.shape[1]

    gnn_model = model.Regressor(node_features, edge_features)

    os.makedirs(os.path.join(args.save_path, 'runs'), exist_ok=True)
    tmodel = model.modified_Tmodel(
        model=gnn_model,
        model_type='regressor',
        # tensorboard_path=os.path.join(args.save_path, 'runs'),
        interval=args.save_interval,
        savepath=args.save_path,
        epoch=args.epoch,
        learning_rate=args.learning_rate,
        lr_reduce=args.lr_reduce)
    
    tmodel.fit(train_loader=train_loader, val_loader=val_loader, evaluate=False)

    result, all_results = tmodel.evaluate(
        data_sets=[train_loader, val_loader, test_loader],
        set_names=['Train', 'Validation', 'Test'],
        savepath=args.save_path)
    print('regression result:\n', result)


def train_for_bondtype(data_path, args):
    if isinstance(data_path, str):
        m, train_loader, val_loader, test_loader, index2bond = pickle.load(open(data_path, 'rb'))
    else:
        m, train_loader, val_loader, test_loader, index2bond = data_path
    
    node_features = train_loader.dataset[0].x.shape[1]
    edge_features = train_loader.dataset[0].edge_attr.shape[1]

    gnn_model = model.BondClassifier(node_features, edge_features, len(index2bond))

    os.makedirs(os.path.join(args.save_path, 'runs'), exist_ok=True)
    tmodel = model.modified_Tmodel(
        model=gnn_model,
        model_type='multi-classifier',
        interval=args.save_interval,
        savepath=args.save_path,
        epoch=args.epoch,
        learning_rate=args.learning_rate,
        lr_reduce=args.lr_reduce,
        class_xrot=30)
    
    tmodel.fit(train_loader=train_loader, val_loader=val_loader, evaluate=False)

    result, all_results = tmodel.evaluate(
        data_sets=[train_loader, val_loader, test_loader],
        set_names=['Train', 'Validation', 'Test'],
        savepath=args.save_path,
        index2label=index2bond)
    print('classification result:\n', result)


if __name__ == "__main__":

    args = parse_arguments()
    if args.problem_type == 'energy-prediction':
        data_path = os.path.join(args.save_path, 'dataloaders_energy.pickle')
        train_for_energy(data_path, args)
    elif args.problem_type == 'bond-prediction':
        data_path = os.path.join(args.save_path, 'dataloaders_bondtype.pickle')
        train_for_bondtype(data_path, args)
    else:
        print('Problem-type not supported!')