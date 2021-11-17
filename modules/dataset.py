import os
import torch
import numpy as np
import pandas as pd
import kmbio  # fork of biopython PDB with some changes in how the structure, chain, etc. classes are defined.
import proteinsolver

from torch_geometric.data import Dataset
from pathlib import Path
from joblib import Parallel, delayed


class ProteinDataset(Dataset):
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
            data = proteinsolver.datasets.transform_edge_attr(data)  # ?
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


def merge_chains(structure, merged_chain_name="A"):
    """merges a structure with multiple chains into a single chain"""
    # generate empty structure
    new_structure = kmbio.PDB.Structure(structure.id)
    new_model = kmbio.PDB.Model(0)
    new_structure.add(new_model)
    new_chain = kmbio.PDB.Chain(merged_chain_name)
    new_model.add(new_chain)

    # sort chains according to index of first residue
    chains = list(structure.chains)
    start_positions = [
        list(chain.residues)[0].id[1] for chain in chains
    ]  # idx 1 is residue position
    sorted_chains = [chain for _, chain in sorted(zip(start_positions, chains))]

    chain_pos_offset = 0  # constant to offset positions of residues in other chains
    for i, chain in enumerate(sorted_chains):
        res_list = list(chain.residues)
        for j, res in list(enumerate(res_list))[::-1]:  # reverse to prevent duplicate idxs
            if i > 0:  # skip first chain
                res.id = (res.id[0], j + chain_pos_offset + 1, res.id[2])
            res.chain = chain.id
        chain_pos_offset += res_list[-1].id[1]
        new_chain.add(chain.residues)
    return new_structure


def get_metadata(path):
    """get targets, etc. from dataset"""
    metadata = dict()
    infile = open(path)
    for line in infile:
        if line[0] != "#":
            line = line.strip().split(",")
            data_id = line[0]
            data_bind = line[5]
            metadata[data_id] = int(data_bind)
    return metadata


def get_data(model_dir, metadata):
    """
    Loads a set of protein models for a given dir.
    """
    metadata = get_metadata(metadata)

    target_list = list()
    path_list = list()
    for model in model_dir.glob("*"):  # iterate over and collect each model
        path_list.append(model)
        model_id = model.name.split("_")[0]
        target = metadata[model_id]
        target_list.append(target)
    return np.array(path_list), np.array(target_list)
