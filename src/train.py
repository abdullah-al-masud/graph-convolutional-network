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


def train(data_path, args):
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


if __name__ == "__main__":

    args = parse_arguments()
    data_path = os.path.join(os.getcwd(), 'dataloaders.pickle')
    train(data_path, args)
