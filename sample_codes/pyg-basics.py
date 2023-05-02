import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset



def build_methane():
    # definition of methane
    # CH4
    edge_index = torch.tensor(
        [[0, 1], [1, 0],
        [0, 2], [2, 0],
        [0, 3], [3, 0],
        [0, 4], [4, 0]], dtype=torch.long)
    
    # calling continguous is necessary here
    edge_index = edge_index.t().contiguous()

    # description of features
    # #proton, #neutron, atomic mass, charge, outer shell electrons
    x = torch.tensor(
        [[6, 6, 12, 0, 4],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1]], dtype=torch.float)

    # lets assume edge feature is distance between two atoms
    # for 4 edges, we had to define 8 edge_index entries. So, similarly, we need to define 8 feature values with shape (8, 1).
    edge_feature = torch.tensor([[1.5, 2.4, 1.6, 1.3, 1.5, 2.4, 1.6, 1.5]]).T

    # label: lets imagine label is binary whether the molecule is organic or not. Methane is organic so 1
    y = torch.tensor(1)

    # data definition
    from torch_geometric.data import Data
    molecule = Data(x=x, edge_index=edge_index, edge_attr=edge_feature, y=y)
    
    print(molecule)
    print(molecule.x)
    print(molecule.edge_index)


def load_enzymes():

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    print(dataset[0])


if __name__ == "__main__":
    build_methane()