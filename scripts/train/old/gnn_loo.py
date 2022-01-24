#! /usr/bin/env python3

import sys
sys.path.append('/home/projects/ht3_aim/people/sebdel/masters/') # add my repo to python path
import torch
import numpy as np
import pandas as pd
import modules

from torch.utils.data import SubsetRandomSampler, BatchSampler
from torch import nn, optim
from pathlib import Path
from sklearn.model_selection import KFold

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
processed_dir = data_root / "processed"
state_file = root / "state_files" / "e53-s1952148-d93703104.state"
out_dir = root / "state_files" / "tcr_binding" / "proteinsolver_finetune_80_cv"
model_dir = data_root / "raw" / "tcrpmhc"
cluster_path = data_root / "clusterRes_cluster.tsv"

paths = list(model_dir.glob("*"))
join_key = [int(x.name.split("_")[0]) for x in paths]
path_df = pd.DataFrame({'#ID': join_key, 'path': paths})

metadata = pd.read_csv(metadata_path)
metadata = metadata.join(path_df.set_index("#ID"), on="#ID", how="inner")  # filter to non-missing data
metadata = metadata.reset_index(drop=True)

raw_files = np.array(metadata["path"])
targets = np.array(metadata["binder"])
dataset = ProteinDataset(
    processed_dir / "proteinsolver_preprocess", 
    raw_files, 
    targets, 
    overwrite=False
)

train_partitions, test_partitions = K_fold_CV_from_clusters(cluster_path, n_splits)

# GNN params
num_features = 20
adj_input_size = 2
hidden_size = 128

# general params
batch_size = 8
n_splits = 5
epochs = 600
learning_rate = 1e-5
lr_decay = 0.999
w_decay = 1e-3

# touch files to ensure output
save_dir = get_non_dupe_dir(out_dir)
loss_paths = touch_output_files(save_dir, "loss", n_splits)
state_paths = touch_output_files(save_dir, "state", n_splits)
pred_paths = touch_output_files(save_dir, "pred", n_splits)

i = 0
for outer_train_idx, test_idx in zip(train_partitions, test_partitions):
    best_inner_fold_valid_losses = list()
    inner_train_losses = list()
    inner_valid_losses = list()
    best_inner_fold_models = list()
    CV = KFold(n_splits=n_splits, shuffle=True)
    for train_idx, valid_idx in CV.split(outer_train_idx):
    
    net = MyGNN(
        x_input_size=num_features + 1, 
        adj_input_size=adj_input_size, 
        hidden_size=hidden_size, 
        output_size=num_features
    )
    net.load_state_dict(torch.load(state_file, map_location=device))
    net.linear_out = nn.Linear(hidden_size, 1)  # rewrite final layer to scalar output
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
        early_stopping=True,
    )
    torch.save(net.state_dict(), state_paths[i])
    torch.save({"train": train_losses, "valid": valid_losses}, loss_paths[i])
    
    pred, true = gnn_predict(net, dataset, test_idx, device)     
    torch.save({"y_pred": pred, "y_true": true}, pred_paths[i])
    
    i += 1
