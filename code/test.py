from data_loader import LRGBDataset, TUDataset
from config import config

for dataset_name in ['peptides-func']:
    val_dataset = LRGBDataset(root="./data", name=dataset_name, split='val')
    for data in val_dataset:
        print(dataset_name, data)
        print(data.x)
        print(data.y)
        print(data.edge_index)
        print(data.edge_attr)
        break 