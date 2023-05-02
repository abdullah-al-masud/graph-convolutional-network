"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pdb
import pandas as pd
import numpy as np
from tqdm import tqdm
from msdlib.msd.processing import one_hot_encoding
import os
import sys
import pickle

sys.path.append(os.getcwd())
from config import parse_arguments


def load_molecules_for_energy(data_dir, dtype=torch.float32, val_ratio=.15, test_ratio=.15, batch_size=32):
    struct = pd.read_csv(os.path.join(data_dir, 'structures.csv'))
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    # test = pd.read_csv('kaggle-molecules/test.csv')
    energy = pd.read_csv(os.path.join(data_dir, 'potential_energy.csv'))

    train = extract_distance(struct, train)
    molecules = create_graph_for_energy(struct, train, energy, dtype=dtype)
    tr_mol, val_mol, test_mol = split_dataset(molecules=molecules, val_ratio=val_ratio, test_ratio=test_ratio)
    train_loader = create_dataloader(tr_mol, batch_size=batch_size)
    val_loader = create_dataloader(val_mol, batch_size=batch_size)
    test_loader = create_dataloader(test_mol, batch_size=batch_size)

    with open('dataloaders.pickle', 'wb') as f:
        pickle.dump([molecules, train_loader, val_loader, test_loader], f, protocol=pickle.HIGHEST_PROTOCOL)

    return train_loader, val_loader, test_loader


def split_dataset(molecules, val_ratio=.15, test_ratio=.15):

    names = list(molecules.keys())
    total = len(names)
    np.random.shuffle(names)

    test_names = names[:int(test_ratio * total)]
    val_names = names[int(test_ratio * total): int(test_ratio * total) + int(val_ratio * total)]
    train_names = names[int(test_ratio * total) + int(test_ratio * total) + int(val_ratio * total):]

    train_molecules = {n: molecules[n] for n in train_names}
    val_molecules = {n: molecules[n] for n in val_names}
    test_molecules = {n: molecules[n] for n in test_names}

    return train_molecules, val_molecules, test_molecules


def create_dataloader(data_dict, batch_size=32, shuffle=True):

    loader = DataLoader(list(data_dict.values()), batch_size=batch_size, shuffle=shuffle)
    return loader


def create_graph_for_energy(struct, df, energy, dtype=torch.float32):

    # feat_names = ['proton', 'neutron', 'mass', 'outer_electron', 'outer_electron_def', 'total_shell']
    atom_features = {'C': [6, 6, 12, 4, 4, 2], 'H': [1, 0, 1, 1, 1, 1], 'N': [7, 7, 14, 5, 3, 1], 'O': [8, 8, 16, 6, 2, 1], 'F': [9, 10, 19, 7, 1, 1]}
    bond_types = ['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC', '3JHN']
    
    # one hot encoding of bond types
    _df = df.set_index('molecule_name').copy()
    # types = pd.DataFrame(OneHotEncoder().fit_transform(_df[['type']]).toarray(), columns=['type_%s'%c for c in bond_types], index=_df.index, dtype=float)
    types = pd.DataFrame(one_hot_encoding(_df['type'], class_label=bond_types), columns=['type_%s'%c for c in bond_types], index=_df.index, dtype=float)
    _df = pd.concat([_df, types], axis=1, sort=False)
    _df['std_scalar_coupling_constant'] = (_df['scalar_coupling_constant'] - _df['scalar_coupling_constant'].mean()) / _df['scalar_coupling_constant'].std()
    
    # reindexing of energy DF
    energy.set_index('molecule_name', inplace=True)
    
    st = struct.sort_values(['molecule_name', 'atom_index']).set_index('molecule_name').copy()

    # creating molecules
    molecules = {}
    for m in _df.index.unique():
        mol = _df.loc[[m]]
        edge_index = torch.tensor(mol[['atom_index_0', 'atom_index_1']].values.tolist() + mol[['atom_index_1', 'atom_index_0']].values.tolist()).t().contiguous()
        edge_features = torch.tensor(mol[['type_%s'%c for c in bond_types] + ['distance', 'std_scalar_coupling_constant']].values, dtype=dtype).repeat((2, 1))
        y = torch.tensor([energy['potential_energy'].loc[m]], dtype=dtype)
        features = torch.tensor([atom_features[a] for a in st.loc[m]['atom']], dtype=dtype)
        molecules[m] = Data(x=features, edge_index=edge_index, y=y, edge_attr=edge_features)
    
    return molecules


def extract_distance(struct, df):
    dist = []
    _struct = struct.copy()
    _struct.set_index(['molecule_name', 'atom_index'], inplace=True)
    print('calculating distance...')
    for i in tqdm(range(df.shape[0])):
        xyz1 = _struct.loc[(df['molecule_name'].iloc[i], df['atom_index_0'].iloc[i])][['x', 'y', 'z']]
        xyz2 = _struct.loc[(df['molecule_name'].iloc[i], df['atom_index_1'].iloc[i])][['x', 'y', 'z']]
        d = np.sqrt(np.square(xyz1 - xyz2).sum())
        dist.append(d)
    df['distance'] = dist

    return df


if __name__ == "__main__":
    args = parse_arguments(show=True)
    dtype = torch.float32
    train_loader, val_loader, test_loader = load_molecules_for_energy(args.data_dir, dtype=dtype, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    
