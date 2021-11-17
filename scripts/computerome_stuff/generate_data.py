import sys
sys.path.append('/home/projects/ht3_aim/people/sebdel/masters/') # add my repo to python path
import os
import torch
import torch.nn.functional as F
import torch_geometric
import kmbio  # fork of biopython PDB with some changes in how the structure, chain, etc. classes are defined.
import numpy as np
import pandas as pd
import proteinsolver
import modules

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, BatchSampler
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import *
from torch import nn, optim
from pathlib import Path

from modules.dataset import *
from modules.utils import *
from modules.model import *
from modules.lstm_utils import *

np.random.seed(1)

def create_gnn_embeddings(
    dataset, 
    out_dir, 
    device, 
    gnn_func, 
    cores=1, 
    overwrite=False, 
    chain_keys=np.array(["P", "M", "A", "B"])
):  
    def _sub_process(out_path, data, chain_keys, gnn_func):
            data = data.to("cpu")
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

    sub_process = lambda out_path, data: _sub_process(
        out_path,
        data,
        gnn_func=gnn_func,
        chain_keys=chain_keys,
    )
    Parallel(n_jobs=cores)(delayed(sub_process)(*arg) for arg in zip(out_files, data_loader))


# Only root needs to be changed
root = Path("/home/projects/ht3_aim/people/sebdel/masters/data/")
data_root = root / "neat_data"
metadata_path = data_root / "metadata.csv"
processed_dir = data_root / "processed" / "tcr_binding"
state_file = root / "state_files" / "e53-s1952148-d93703104.state"
out_dir = root / "state_files" / "tcr_binding"
model_dir = data_root / "raw" / "tcrpmhc"

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

dataset = ProteinDataset(processed_dir, raw_files, targets, cores=30, overwrite=False)

# Create GNN embeddings (gnn.forward_without_last_layer=128 dim, gnn.forward=20 dim)
gnn_func = gnn.forward_without_last_layer
out_dir = processed_dir / "gnn_out_pos_128_test"
create_gnn_embeddings(dataset, out_dir, device, gnn_func, cores=30, overwrite=False)