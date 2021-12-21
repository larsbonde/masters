import kmbio
import numpy as np
import torch
import torch_geometric
from pathlib import Path


def filter_peptides(partition_1, partition_2, unique_peptides, filtered_peptides, metadata):  # TODO modularize to take a single partition instead of two
    filtered_indices = list()
    filtered_partitions = list()

    for pep in filtered_peptides:
        filtered_indices.extend(list(metadata[metadata["peptide"] == pep].index))
        filtered_partitions.extend(np.where(unique_peptides == pep)[0])

    partition_1 = [part for i, part in enumerate(partition_1) if i not in filtered_partitions]
    partition_2 = [part for i, part in enumerate(partition_2) if i not in filtered_partitions]

    filtered_indices = set(filtered_indices)

    for i in range(len(partition_1)):
        train_part, valid_part = partition_1[i], partition_2[i]
        train_part = [i for i in train_part if i not in filtered_indices]
        valid_part = [i for i in valid_part if i not in filtered_indices]
        partition_1[i], partition_2[i] = train_part, valid_part

    unique_peptides = np.delete(unique_peptides, filtered_partitions)

    return partition_1, partition_2, unique_peptides


def generate_2_loo_partitions(metadata):
    """
    Generates leave-one-out partitions given a df with metadata
    """
    unique_peptides = metadata["peptide"].unique()

    metadata["merged_chains"] = metadata["CDR3a"] + metadata["CDR3b"]
    loo_train_partitions = list()
    loo_valid_partitions = list()
    for pep in unique_peptides:
        valid_df = metadata[metadata["peptide"] == pep]
        valid_unique_cdr = valid_df["merged_chains"].unique()

        # get training rows and drop swapped data
        train_df = metadata[metadata["peptide"] != pep]
        train_df = train_df[~train_df["merged_chains"].str.contains('|'.join(valid_unique_cdr))]

        loo_train_partitions.append(list(train_df.index))
        loo_valid_partitions.append(list(valid_df.index))

   # hacky dataset fix
    loo_train_partitions, loo_valid_partitions, unique_peptides = filter_peptides(
        loo_train_partitions, 
        loo_valid_partitions,
        ["CLGGLLTMV", "ILKEPVHGV"],
        metadata,
    )
    return loo_train_partitions, loo_valid_partitions, unique_peptides


def generate_3_loo_partitions(metadata, valid_pep="KTWGQYWQV"):
    """
    Generates leave-one-out partitions given a df with metadata. NOTE: Drops swapped data.
    """
    metadata = metadata[metadata["origin"] != "swapped"]
    unique_peptides = metadata["peptide"].unique()
    unique_peptides = np.delete(unique_peptides, np.where(unique_peptides == valid_pep))
    
    valid_df = metadata[metadata["peptide"] == valid_pep]
    
    loo_train_partitions = list()
    loo_test_partitions = list()
    
    for pep in unique_peptides:
        test_df = metadata[metadata["peptide"] == pep]
        train_df = metadata[(metadata["peptide"] != pep) & (metadata["peptide"] != valid_pep)]

        loo_train_partitions.append(list(train_df.index))
        loo_test_partitions.append(list(test_df.index))
        
    valid_partition = list(valid_df.index)

    # hacky dataset fix
    loo_train_partitions, loo_test_partitions, unique_peptides = filter_peptides(
        loo_train_partitions, 
        loo_test_partitions,
        unique_peptides,
        ["CLGGLLTMV", "ILKEPVHGV"],
        metadata,
    )
    return loo_train_partitions, loo_test_partitions, valid_partition, unique_peptides


def create_gnn_embeddings(
    dataset, 
    out_dir, 
    device, 
    gnn_func, 
    cores=1, 
    overwrite=False, 
    chain_keys=np.array(["P", "M", "A", "B"])
):  
    def _sub_process(out_path, data, chain_keys=chain_keys, gnn_func=gnn_func):
        #data = data.to("cpu")
        with torch.no_grad():
            out = gnn_func(data.x, data.edge_index, data.edge_attr)

        # add positional encoding of chains
        positional_encoding = np.zeros((len(data.x), len(chain_keys)))
        for j, p in enumerate(data.chain_map[0]):
            positional_encoding[j][np.where(chain_keys == p)] = 1
        positional_encoding = torch.Tensor(positional_encoding)
        out = torch.cat((out, positional_encoding), dim=1)

        torch.save(out, out_path)

    out_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
    
    data_loader = torch_geometric.loader.DataLoader(dataset, shuffle=False, batch_size=1)
    out_files = list()
    targets = list()
    for i, data in enumerate(data_loader):
        out_path = out_dir / f"data_{i}.pt"
        if not out_path.is_file() or overwrite:
            out_files.append(out_path)
        targets.append([data.y])
    torch.save(targets, out_dir / f"targets.pt")
    
    for arg in zip(out_files, data_loader):  # use instead of failed attempt at parallelizing pytorch stuff
        _sub_process(*arg)


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
