import os
import kmbio  # fork of biopython PDB with some changes in how the structure, chain, etc. classes are defined.
import proteinsolver
import torch
import torch_geometric
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from .dataset_utils import*

import sys

class LSTMDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_dir, 
        annotations_path,
        device,
        transform=None, 
        target_transform=None
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.annotations = torch.Tensor(torch.load(annotations_path))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        x = torch.load(f"{self.data_dir}/data_{idx}.pt", map_location=self.device)
        y = self.annotations[idx]
        return x, y


class LSTMEnergyDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        paths, 
        targets, 
        transform=None, 
        target_transform=None
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.annotations = torch.Tensor(targets).unsqueeze(1)
        self.paths = paths
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])
        time_idx = [92, 116, 140]  # time variables to be removed
        for i in time_idx:
            x[:,i] = 0.0
        x = torch.from_numpy(x).float()
        y = self.annotations[idx]
        return x, y


class ProteinDataset(torch_geometric.data.Dataset):
    """
    Variation of torch_geometric.data.Dataset.

    Provides a framework for data handling and pre-processing pipeline. Pre-processing
    is mainly performed using kmbio.PDB for PDB file handling and proteinsolver utility-
    code for preparing input for GNN use.

    Args:
    root: path to directory containing raw/processed file directories.
    file_names: list of files with raw data to use.
    targets: list of ground truth values for each raw file.
    overwrite: Bool indicating if the processed files should be overwritten.
    transform: see torch_geometric.data.Dataset docs
    pre_transform: see torch_geometric.data.Dataset docs
    """

    def __init__(
        self,
        processed_dir,
        file_names,
        targets,
        overwrite=False,
        root=None,
        cores=1,
        transform=None,
        pre_transform=None,
    ):
        self._raw_file_names = file_names
        self.targets = targets
        self._overwrite = overwrite
        self.cores = cores
        self._processed_dir = processed_dir
        self.n_files = len(self._raw_file_names)
        if self._overwrite:
            self._processed_file_names = ["dummy"] 
        else:
            self._processed_file_names = self.generate_file_names()
    
        super().__init__(root, transform, pre_transform)

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def raw_file_names(self):
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return self._processed_file_names

    def generate_file_names(self):
        return [Path(f"{self._processed_dir}/data_{i}.pt") for i in range(self.n_files)]

    def process(self):
        def _sub_process(j, raw_file, target, processed_path):
            # Read data from raw_path and process
            structure_all = kmbio.PDB.load(raw_file)
            structure_all = merge_chains(structure_all)
            structure = kmbio.PDB.Structure(j, structure_all[0].extract("A"))
            pdata = proteinsolver.utils.extract_seq_and_adj(
                structure, "A", remove_hetatms=True
            )
            data = proteinsolver.datasets.row_to_data(pdata)
            data = proteinsolver.datasets.transform_edge_attr(data)
            data.y = torch.Tensor([target])
            data.chain_map = np.array([res.chain for res in list(structure.residues)])

            #if self.pre_transform is not None:
            #    data = self.pre_transform(data)

            torch.save(data, processed_path)
        
        os.makedirs(self._processed_dir, exist_ok=True)
        processed_file_set = set(self._processed_dir.glob("data_*"))
        i = 0
        self._processed_file_names = self.generate_file_names()
        args = list()
        for raw_file, target, processed_path in zip(self._raw_file_names, self.targets, self._processed_file_names):
            if processed_path not in processed_file_set or self._overwrite:
                args.append([i, raw_file, target, processed_path])
            i += 1
        Parallel(n_jobs=self.cores)(delayed(_sub_process)(*arg) for arg in args)

    def len(self):
        return len(self._processed_file_names)

    def get(self, idx):
        return torch.load(Path(f"{self._processed_dir}/data_{idx}.pt"))
