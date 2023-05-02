from torch_geometric.datasets import TUDataset


def main():

    data = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

    dataset_info(data)

    sample = data[0]
    print(sample)

    data_properties(data)


def main_mutag():

    data = TUDataset(root='/tmp/MUTAG', name='MUTAG')

    dataset_info(data)

    sample = data[0]
    print(sample)

    data_properties(data)


def dataset_info(data):
    print('dataset length:', len(data))
    print(data.num_features)
    print(data.num_classes)

    print(data.get_summary())
    

def data_properties(data):

    print(dir(data))



if __name__ == "__main__":

    main()