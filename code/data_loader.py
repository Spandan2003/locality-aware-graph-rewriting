import os
import os.path as osp
import pickle
import shutil
from typing import Callable, List, Optional

import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import read_tu_data

class LRGBDataset(InMemoryDataset):
    r"""The `"Long Range Graph Benchmark (LRGB)"`
    <https://arxiv.org/abs/2206.08164>`_
    datasets which is a collection of 5 graph learning datasets with tasks
    that are based on long-range dependencies in graphs. See the original
    `source code <https://github.com/vijaydwivedi75/lrgb>`_ for more details
    on the individual datasets.
    """

    names = [
        'pascalvoc-sp', 'coco-sp', 'pcqm-contact', 'peptides-func',
        'peptides-struct'
    ]

    urls = {
        'pascalvoc-sp':
        'https://www.dropbox.com/s/8x722ai272wqwl4/pascalvocsp.zip?dl=1',
        'coco-sp':
        'https://www.dropbox.com/s/r6ihg1f4pmyjjy0/cocosp.zip?dl=1',
        'pcqm-contact':
        'https://www.dropbox.com/s/qdag867u6h6i60y/pcqmcontact.zip?dl=1',
        'peptides-func':
        'https://www.dropbox.com/s/ycsq37q8sxs1ou8/peptidesfunc.zip?dl=1',
        'peptides-struct':
        'https://www.dropbox.com/s/zgv4z8fcpmknhs8/peptidesstruct.zip?dl=1'
    }

    dwnld_file_name = {
        'pascalvoc-sp': 'voc_superpixels_edge_wt_region_boundary',
        'coco-sp': 'coco_superpixels_edge_wt_region_boundary',
        'pcqm-contact': 'pcqmcontact',
        'peptides-func': 'peptidesfunc',
        'peptides-struct': 'peptidesstruct'
    }

    def __init__(
            self,
            root: str,
            name: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
        ):
        self.name = name.lower()
        assert self.name in self.names
        assert split in ['train', 'val', 'test']

        super().__init__(root, transform, pre_transform, pre_filter)

        if pre_transform is not None:
            # Include metadata based on pre_transform
            transform_name = getattr(pre_transform, '__name__', pre_transform.__class__.__name__)
            self._processed_dir = f"{self.processed_dir}_{transform_name}"
        else:
            self._processed_dir = self.processed_dir

        # Path to the processed data
        path = osp.join(self.processed_dir, f'{split}.pt')

        # Load the dataset
        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        # Dynamically return the processed_dir path based on pre_transform
        name = "processed"
        if self.pre_transform is not None:
            transform_name = getattr(self.pre_transform, '__name__', self.pre_transform.__class__.__name__)
            name += f"_{transform_name}"
        return osp.join(self.root, self.name, name)

    @property
    def raw_file_names(self) -> List[str]:
        if self.name.split('-')[1] == 'sp':
            return ['train.pickle', 'val.pickle', 'test.pickle']
        else:
            return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.urls[self.name], self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, self.dwnld_file_name[self.name]),
                  self.raw_dir)
        os.unlink(path)

    def process(self):
        if self.name == 'pcqm-contact':
            # PCQM-Contact
            self.process_pcqm_contact()
        else:
            if self.name == 'coco-sp':
                # Label remapping for coco-sp.
                # See self.label_remap_coco() func
                label_map = self.label_remap_coco()

            for split in ['train', 'val', 'test']:
                if self.name.split('-')[1] == 'sp':
                    # PascalVOC-SP and COCO-SP
                    with open(osp.join(self.raw_dir, f'{split}.pickle'),
                              'rb') as f:
                        graphs = pickle.load(f)
                elif self.name.split('-')[0] == 'peptides':
                    # Peptides-func and Peptides-struct
                    with open(osp.join(self.raw_dir, f'{split}.pt'),
                              'rb') as f:
                        graphs = torch.load(f)

                data_list = []
                for graph in tqdm(graphs, desc=f'Processing {split} dataset'):
                    if self.name.split('-')[1] == 'sp':
                        x = graph[0].to(torch.float)
                        edge_attr = graph[1].to(torch.float)
                        edge_index = graph[2]
                        y = torch.LongTensor(graph[3])
                    elif self.name.split('-')[0] == 'peptides':
                        x = graph[0]
                        edge_attr = graph[1]
                        edge_index = graph[2]
                        y = graph[3]

                    if self.name == 'coco-sp':
                        for i, label in enumerate(y):
                            y[i] = label_map[label.item()]

                    data = Data(x=x, edge_index=edge_index,
                                edge_attr=edge_attr, y=y)

                    if self.pre_filter is not None and not self.pre_filter(
                            data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)

                torch.save(self.collate(data_list),
                           osp.join(self.processed_dir, f'{split}.pt'))

    def label_remap_coco(self):
        # Util function for name 'COCO-SP'
        original_label_idx = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78,
            79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]

        label_map = {}
        for i, key in enumerate(original_label_idx):
            label_map[key] = i

        return label_map

    def process_pcqm_contact(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pt'), 'rb') as f:
                graphs = torch.load(f)

            data_list = []
            for graph in tqdm(graphs, desc=f'Processing {split} dataset'):
                x = graph[0]
                edge_attr = graph[1]
                edge_index = graph[2]
                edge_label_index = graph[3]
                edge_label = graph[4]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            edge_label_index=edge_label_index,
                            edge_label=edge_label)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))

class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    """

    url = "https://www.chrsmrrs.com/graphkerneldatasets"
    cleaned_url = (
        "https://raw.githubusercontent.com/nd7141/" "graph_datasets/master/datasets"
    )

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        use_node_attr: bool = False,
        use_edge_attr: bool = False,
        cleaned: bool = False,
    ):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)

        out = torch.load(self.processed_paths[0])
        if not isinstance(out, tuple) or len(out) != 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again."
            )
        self.data, self.slices, self.sizes = out

        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

        # Add an internal property for the modified directory
    @property
    def modified_processed_dir(self):
        return self._processed_dir if hasattr(self, '_processed_dir') else self.processed_dir
    @property
    def raw_file_names(self) -> List[str]:
        return [
            f"{self.name}_train.graph",
            f"{self.name}_test.graph",
            f"{self.name}_val.graph",
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return [
            f"{self.name}_train.graph",
            f"{self.name}_test.graph",
            f"{self.name}_val.graph",
        ]

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)

    def process(self):
        """Process the dataset into torch_geometric format"""
        for split in ['train', 'val', 'test']:
            path = osp.join(self.raw_dir, f"{self.name}_{split}.graph")
            graphs = read_tu_data(path)

            data_list = []
            for graph in graphs:
                edge_index = graph[0]
                x = graph[1]
                y = graph[2]
                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)

            torch.save(self.collate(data_list), osp.join(self.processed_dir, f"{self.name}_{split}.pt"))


    def label_remap_coco(self):
        # Util function for name 'COCO-SP'
        # to remap the labels as the original label idxs are not contiguous
        original_label_idx = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78,
            79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]

        label_map = {}
        for i, key in enumerate(original_label_idx):
            label_map[key] = i

        return label_map

    def process_pcqm_contact(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pt'), 'rb') as f:
                graphs = torch.load(f)

            data_list = []
            for graph in tqdm(graphs, desc=f'Processing {split} dataset'):
                """
                PCQM-Contact
                Each `graph` is a tuple (x, edge_attr, edge_index,
                                        edge_label_index, edge_label)
                    Shape of x : [num_nodes, 9]
                    Shape of edge_attr : [num_edges, 3]
                    Shape of edge_index : [2, num_edges]
                    Shape of edge_label_index: [2, num_labeled_edges]
                    Shape of edge_label : [num_labeled_edges]

                    where,
                    num_labeled_edges are negative edges and link pred labels,
                    https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/loader/dataset/pcqm4mv2_contact.py#L192
                """
                x = graph[0]
                edge_attr = graph[1]
                edge_index = graph[2]
                edge_label_index = graph[3]
                edge_label = graph[4]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            edge_label_index=edge_label_index,
                            edge_label=edge_label)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)

    Stats:
        .. list-table::
            :widths: 20 10 10 10 10 10
            :header-rows: 1

            * - Name
              - #graphs
              - #nodes
              - #edges
              - #features
              - #classes
            * - MUTAG
              - 188
              - ~17.9
              - ~39.6
              - 7
              - 2
            * - ENZYMES
              - 600
              - ~32.6
              - ~124.3
              - 3
              - 6
            * - PROTEINS
              - 1,113
              - ~39.1
              - ~145.6
              - 3
              - 2
            * - COLLAB
              - 5,000
              - ~74.5
              - ~4914.4
              - 0
              - 3
            * - IMDB-BINARY
              - 1,000
              - ~19.8
              - ~193.1
              - 0
              - 2
            * - REDDIT-BINARY
              - 2,000
              - ~429.6
              - ~995.5
              - 0
              - 2
            * - ...
              -
              -
              -
              -
              -
    """

    url = "https://www.chrsmrrs.com/graphkerneldatasets"
    cleaned_url = (
        "https://raw.githubusercontent.com/nd7141/" "graph_datasets/master/datasets"
    )

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        use_node_attr: bool = False,
        use_edge_attr: bool = False,
        cleaned: bool = False,
    ):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)

        out = torch.load(self.processed_paths[0])
        if not isinstance(out, tuple) or len(out) != 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again."
            )
        self.data, self.slices, self.sizes = out

        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f"processed"
        metadata = "_cleaned" if self.cleaned else ""
        metadata += (
            f"_{self.pre_transform.__self__}" if self.pre_transform is not None else ""
        )
        name += metadata
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        return self.sizes["num_node_labels"]

    @property
    def num_node_attributes(self) -> int:
        return self.sizes["num_node_attributes"]

    @property
    def num_edge_labels(self) -> int:
        return self.sizes["num_edge_labels"]

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes["num_edge_attributes"]

    @property
    def raw_file_names(self) -> List[str]:
        names = ["A", "graph_indicator"]
        return [f"{self.name}_{name}.txt" for name in names]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url(f"{url}/{self.name}.zip", folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [
                    self.pre_transform(d)
                    for d in tqdm(data_list, desc="Applying pre_transform...")
                ]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        torch.save((self.data, self.slices, sizes), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name}({len(self)})"