import sys
sys.path.append('/home/projects/ht3_aim/people/sebdel/masters/') # add my repo to python path
import os
import torch
import torch_geometric
import kmbio  # fork of biopython PDB with some changes in how the structure, chain, etc. classes are defined.
import numpy as np
import pandas as pd
import proteinsolver
import modules

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from Bio import SeqIO

from modules.dataset import *
from modules.dataset_utils import *
from modules.utils import *
from modules.models import *
from modules.lstm_utils import *

np.random.seed(0)


def find_subset_idx(seq, subset_seq):
    subset_indices = list()
    subset_len = len(subset_seq)
    for i in range(len(seq) - subset_len + 1):
        if np.all(seq[i : i + subset_len] == subset_seq):
            return [i, i + subset_len]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Only root needs to be changed
root = Path("/home/projects/ht3_aim/people/sebdel/masters/data/")
data_root = root / "neat_data"
metadata_path = data_root / "metadata.csv"
processed_dir = data_root / "processed"
state_file = root / "state_files" / "e53-s1952148-d93703104.state"
full_seq_path = data_root / "full_seqs.fsa"

model_dir = data_root / "raw" / "tcrpmhc"
energy_model_dir = data_root / "raw" / "energy_terms_mock"

# Get metadata
paths = list(model_dir.glob("*"))
join_key = [int(x.name.split("_")[0]) for x in paths]
path_df = pd.DataFrame({'#ID': join_key, 'path': paths})

metadata = pd.read_csv(metadata_path)
metadata = metadata.join(path_df.set_index("#ID"), on="#ID", how="inner")  # filter to non-missing data
metadata = metadata.reset_index(drop=True)

raw_files = np.array(metadata["path"])
ps_targets = np.array(metadata["binder"])

dataset_pre = ProteinDataset(
    processed_dir / "proteinsolver_preprocess", 
    raw_files, 
    ps_targets, 
    cores=20, 
    overwrite=False
)

energy_paths = list(energy_model_dir.glob("*"))
join_key = [int(x.name.split("_")[0]) for x in energy_paths]
path_df = pd.DataFrame({'#ID': join_key})
metadata = metadata.join(path_df.set_index("#ID"), on="#ID", how="inner")  # filter to energy/raw overlap

dataset_emb = LSTMDataset(
    data_dir=processed_dir / "energy_terms_pos", 
    annotations_path=processed_dir / "energy_terms_pos" / "targets.pt",
    device=device,
)

# Create GNN embeddings (gnn.forward_without_last_layer=128 dim, gnn.forward=20 dim)
out_dir = processed_dir / "energy_terms_cdr_pep_only"
out_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

chain_keys = np.array(["P", "M", "A", "B"])

j = 0
for i, record in enumerate(SeqIO.parse(full_seq_path, "fasta")):
    print("-------")
    seq = np.array(list(record.seq))
    seq_id = int(record.id)
    if seq_id in metadata["#ID"]:
        print(j)
        print(i)
        print(metadata.iloc[j]["#ID"])
        data = dataset_pre[i]
        x = dataset_emb[j][0]
        metadata_row = metadata.iloc[j]
        
        new_data = list()
        for cdr3, tcr in [["CDR3a", "A"], ["CDR3b", "B"]]:
            # get TCR and CDR3 seq
            tcr_seq = seq[np.where(data.chain_map == tcr)]
            cdr3_seq = np.array(list(metadata_row[cdr3]))
            
            # find indices of CDR3 in TCR
            start_idx, end_idx = find_subset_idx(tcr_seq, cdr3_seq)
            
            # slice embedded data to correct chain using pos encoding and subset CDR indices
            pos_encoding_key = np.where(chain_keys == tcr)[0][0] - 4 # subtract 4 to get same key as when data was generated
            cdr3_embed = x[x[:, pos_encoding_key] == 1]  
            cdr3_embed = cdr3_embed[start_idx:end_idx]
            new_data.append(cdr3_embed)
        
        pos_encoding_key = np.where(chain_keys == "P")[0][0] - 4
        peptide_embed = x[x[:, pos_encoding_key] == 1]
        new_data.append(peptide_embed)
        new_data = torch.vstack(new_data)
        torch.save(new_data, out_dir / f"data_{j}.pt")

        j += 1
