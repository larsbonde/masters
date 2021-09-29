import os
import torch
import numpy as np
import kmbio  # fork of biopython PDB with some changes in how the structure, chain, etc. classes are defined.
import proteinsolver

from torch_geometric.data import Dataset
from pathlib import Path


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
        transform=None,
        pre_transform=None,
    ):
        self._raw_file_names = file_names
        self.targets = targets
        self._overwrite = overwrite
        self._processed_dir = processed_dir

        if self._overwrite:
            self._processed_file_names = ["dummy"]
        else:
            n_files = len(self._raw_file_names)
            self._processed_file_names = [
                Path(f"{self._processed_dir}/data_{i}.pt") for i in range(n_files)
            ]
    
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

    def process(self):
        os.makedirs(self._processed_dir, exist_ok=True)
        processed_file_set =  set(self._processed_dir.glob("data_*"))
        print(len(processed_file_set))
        i = 0
        for raw_file, target, processed_path in zip(self._raw_file_names, self.targets, self._processed_file_names):
            if processed_path not in processed_file_set or self._overwrite:
                print(processed_path)
                # Read data from raw_path and process
                structure_all = kmbio.PDB.load(raw_file)
                structure_all = merge_chains(structure_all)
                structure = kmbio.PDB.Structure(i, structure_all[0].extract("A"))

                pdata = proteinsolver.utils.extract_seq_and_adj(
                    structure, "A", remove_hetatms=True
                )
                data = proteinsolver.datasets.row_to_data(pdata)
                data = proteinsolver.datasets.transform_edge_attr(data)  # ?
                data.y = torch.Tensor(target)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, processed_path)
            else:
                print("skipping", i)
            i += 1

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

    chain_len = 0  # constant to offset positions of residues in other chains
    for i, chain in enumerate(sorted_chains):
        res_list = list(chain.residues)
        if i > 0:  # skip first chain
            for j, res in list(enumerate(res_list))[
                ::-1
            ]:  # iterate in reverse to prevent duplicate idxs
                res.id = (res.id[0], j + chain_len + 1, res.id[2])
        chain_len += res_list[-1].id[1]
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
