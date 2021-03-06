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

# Only root needs to be changed
root = Path("/home/projects/ht3_aim/people/sebdel/masters/data/")
data_root = root / "neat_data"
metadata_path = data_root / "metadata.csv"
processed_dir = data_root / "processed"
state_file = root / "state_files" / "e53-s1952148-d93703104.state"
out_dir = root / "state_files" / "tcr_binding"
model_dir = data_root / "raw" / "foldx_repair"

# Get metadata
paths = list(model_dir.glob("*"))
join_key = [int(x.name.split("_")[0]) for x in paths]
path_df = pd.DataFrame({'#ID': join_key, 'path': paths})

metadata = pd.read_csv(metadata_path)
metadata = metadata.join(path_df.set_index("#ID"), on="#ID", how="inner")  # filter to non-missing data
metadata = metadata.reset_index(drop=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

raw_files = np.array(metadata["path"])
targets = np.array(metadata["binder"])

dataset = ProteinDataset(
    processed_dir / "proteinsolver_preprocess_foldx_repair", 
    raw_files, 
    targets, 
    cores=12, 
    overwrite=True
)

# Create GNN embeddings (gnn.forward_without_last_layer=128 dim, gnn.forward=20 dim)
gnn_func = gnn.forward_without_last_layer
out_dir = processed_dir / "proteinsolver_embeddings_pos_foldx_repair"
create_gnn_embeddings(dataset, out_dir, device, gnn_func, cores=1, overwrite=True)
