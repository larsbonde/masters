#! /usr/bin/env python3

import sys
sys.path.append('/home/projects/ht3_aim/people/sebdel/masters/') # add my repo to python path
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import modules

from torch.utils.data import SubsetRandomSampler, BatchSampler
from torch import nn, optim
from pathlib import Path

from modules.dataset_utils import *
from modules.dataset import *
from modules.utils import *
from modules.models import *
from modules.gnn_utils import *

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = Path("/home/projects/ht3_aim/people/sebdel/masters/data/")
data_root = root / "neat_data"
metadata_path = data_root / "metadata.csv"
processed_dir = data_root / "processed" / "tcr_binding"
state_file = root / "state_files" / "e53-s1952148-d93703104.state"
out_dir = root / "state_files" / "tcr_binding" / "proteinsolver_finetuning"
model_dir = data_root / "raw" / "tcrpmhc"

paths = list(model_dir.glob("*"))
join_key = [int(x.name.split("_")[0]) for x in paths]
path_df = pd.DataFrame({'#ID': join_key, 'path': paths})

metadata = pd.read_csv(metadata_path)
metadata = metadata.join(path_df.set_index("#ID"), on="#ID", how="inner")  # filter to non-missing data
metadata = metadata.reset_index(drop=True)

raw_files = np.array(metadata["path"])
targets = np.array(metadata["binder"])
dataset = ProteinDataset(processed_dir, raw_files, targets, overwrite=False)

loo_train_partitions, loo_valid_partitions, unique_peptides = generate_loo_partitions(metadata)

# GNN params
num_features = 20
adj_input_size = 2
hidden_size = 128

# general params
batch_size = 4
epochs = 4
learning_rate = 1e-2
lr_decay = 1 #0.98
w_decay = 0

# touch files to ensure output
n_splits = len(unique_peptides)
save_dir = get_non_dupe_dir(out_dir)
loss_paths = touch_output_files(save_dir, "loss", n_splits)
state_paths = touch_output_files(save_dir, "state", n_splits)
pred_paths = touch_output_files(save_dir, "pred", n_splits)

i = 0
for train_idx, valid_idx in zip(loo_train_partitions, loo_valid_partitions):
    
    net = MyGNN(
        x_input_size=num_features + 1, 
        adj_input_size=adj_input_size, 
        hidden_size=hidden_size, 
        output_size=num_features
    )
    net.load_state_dict(torch.load(state_file, map_location=device))
    net.linear_out = nn.Linear(hidden_size, 1)  # rewrite final layer to scalar outptu
    net = net.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        net.parameters(), 
        lr=learning_rate, 
        weight_decay=w_decay,
    ) 
    scheduler = optim.lr_scheduler.MultiplicativeLR(
        optimizer, 
        lr_lambda=lambda epoch: lr_decay
    )
    
    net, train_losses, valid_losses = gnn_train(
        net,
        epochs,
        criterion,
        optimizer,
        scheduler,
        dataset,
        train_idx,
        valid_idx,
        batch_size,
        device,
    )
    torch.save(net.state_dict(), state_paths[i])
    torch.save({"train": train_losses, "valid": valid_losses}, loss_paths[i])
    
    pred, true = gnn_predict(net, dataset, valid_idx, device)     
    torch.save({"y_pred": pred, "y_true": true}, pred_paths[i])
    
    i += 1