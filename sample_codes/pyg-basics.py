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

    # description of features
    # #proton, #neutron, atomic mass, charge, outer shell electrons
    x = torch.tensor(
        [[6, 6, 12, 0, 4],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.t().contiguous())
    print(data.x)
    print(data.edge_index)
    print(data)


def load_enzymes():

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    print(dataset[0])


if __name__ == "__main__":
    build_methane()