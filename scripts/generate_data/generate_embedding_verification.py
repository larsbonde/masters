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

from modules.dataset import *
from modules.dataset_utils import *
from modules.utils import *
from modules.models import *
from modules.lstm_utils import *

np.random.seed(0)
torch.manual_seed(0)

class ChainFilter(kmbio.PDB.Select):
    def __init__(self, subset):
        self.subset = subset

    def accept_chain(self, chain):
        if chain.id in self.subset:
            return 1
        else:
            return 0


# Only root needs to be changed
root = Path("/home/projects/ht3_aim/people/sebdel/masters/data/")
data_root = root / "neat_data"
metadata_path = data_root / "metadata.csv"
processed_dir = data_root / "processed"
state_file = root / "state_files" / "e53-s1952148-d93703104.state"
model_dir = data_root / "raw" / "tcrpmhc"

# Get metadata
paths = list(model_dir.glob("*"))
join_key = [int(x.name.split("_")[0]) for x in paths]
path_df = pd.DataFrame({'#ID': join_key, 'path': paths})

metadata = pd.read_csv(metadata_path)
metadata = metadata.join(path_df.set_index("#ID"), on="#ID", how="inner")  # filter to non-missing data
metadata = metadata.reset_index(drop=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

raw_files = metadata[metadata["binder"] == 1]["path"]

pmhc_chain_subset = ["M", "P"]
p_chain_subset = ["P"]
annotated_paths = list()

outdir_1 = data_root / "raw" / "pmhc"
outdir_2 = data_root / "raw" / "p"

outdir_1.mkdir(parents=True, exist_ok=True)
outdir_2.mkdir(parents=True, exist_ok=True)

overwrite = False
for raw_file in raw_files:

    model_id = raw_file.name.split("_")[0]
    pmhc_file_name = outdir_1 / f"{model_id}_pmhc.pdb"
    p_file_name =  outdir_2/ f"{model_id}_p.pdb"
    
    if overwrite or (not pmhc_file_name.is_file() or not p_file_name.is_file()):
        structure  = kmbio.PDB.load(raw_file)
    
        io = kmbio.PDB.io.PDBIO()
        io.set_structure(structure)
        io.save(pmhc_file_name, ChainFilter(subset=pmhc_chain_subset))
    
        io = kmbio.PDB.io.PDBIO()
        io.set_structure(structure)
        io.save(p_file_name, ChainFilter(subset=p_chain_subset))

    annotated_paths.append([raw_file, "0"])  # add indices of peptide
    annotated_paths.append([pmhc_file_name, "1"])  # add indices of peptide
    annotated_paths.append([p_file_name, "2"])  # add indices of peptide

embedding_verification_metadata_path = data_root / "embedding_verification_metadata.csv"
with open(embedding_verification_metadata_path, "w") as metadata_outfile:
    for data in annotated_paths:
        print(data[0], data[1], sep=",", file=metadata_outfile)

# init proteinsolver gnn and preprocess data for GNN embedding
num_features = 20
adj_input_size = 2
hidden_size = 128

gnn = Net(
    x_input_size=num_features + 1, 
    adj_input_size=adj_input_size, 
    hidden_size=hidden_size, 
    output_size=num_features
)
gnn.load_state_dict(torch.load(state_file, map_location=device))
gnn.eval()
gnn = gnn.to(device)

# load dataset
raw_files = list()
targets = list()
with open(embedding_verification_metadata_path, "r") as infile:
    for line in infile:
        line = line.strip().split(",")
        raw_files.append(line[0])
        targets.append(int(line[1]))

dataset = ProteinDataset(
    processed_dir / "proteinsolver_preprocess_embedding_verification", 
    raw_files, 
    targets, 
    cores=12, 
    overwrite=True
)

# Create GNN embeddings (gnn.forward_without_last_layer=128 dim, gnn.forward=20 dim)
gnn_func = gnn.forward_without_last_layer
out_dir = processed_dir / "proteinsolver_embedding_verification"
out_dir.mkdir(parents=True, exist_ok=True)

targets = list()
for i in range(len(dataset)):
    out_path = out_dir / f"data_{i}.pt"
    data = dataset[i]
    if not out_path.is_file() or overwrite:
        with torch.no_grad():
            out = gnn_func(data.x, data.edge_index, data.edge_attr)
        peptide_emb = out[data.chain_map[0] == "P"]  # idx 0 as we only have 1 batch
        torch.save(peptide_emb, out_path)
    targets.append([data.y])
torch.save(targets, out_dir / f"targets.pt")
