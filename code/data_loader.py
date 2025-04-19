import os
from torch_geometric.loader import DataLoader
from graphgps.loader.master_loader import (
    preformat_PCQM4Mv2Contact,
    preformat_Peptides
)


def load_lrgb_dataset(name, split='train', batch_size=32, data_root='../data'):
    """
    Load an LRGB dataset using graphgps loader interface.

    Args:
        name (str): One of ['pcqm-contact', 'peptides-functional', 'peptides-structural']
        split (str): One of ['train', 'val', 'test']
        batch_size (int): PyG DataLoader batch size
        data_root (str): Path where dataset will be stored

    Returns:
        DataLoader for the requested split
    """
    name = name.lower()
    dataset_dir = os.path.join(data_root, name.replace('-', '_'))

    if name.startswith('pcqm-contact'):
        dataset = preformat_PCQM4Mv2Contact(dataset_dir, name)
    elif name.startswith('peptides-'):
        dataset = preformat_Peptides(dataset_dir, name)
    else:
        raise ValueError(f"Unsupported dataset '{name}'.")

    # Extract appropriate split
    split_map = {'train': 0, 'val': 1, 'test': 2}
    split_idx = split_map[split]
    indices = dataset.split_idxs[split_idx]
    dataset_split = dataset[indices]

    return DataLoader(dataset_split, batch_size=batch_size, shuffle=(split == 'train'))