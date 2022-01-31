#! /usr/bin/env python3

import sys
sys.path.append('/home/projects/ht3_aim/people/sebdel/masters/') # add my repo to python path
import torch
import numpy as np
import pandas as pd
import modules
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", default="default")
parser.add_argument("-s", "--drop_swapped", action="store_true", default=False)
parser.add_argument("-c", "--cluster", default="cdr3ab")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = Path("/home/projects/ht3_aim/people/sebdel/masters/data/")
data_root = root / "neat_data"
metadata_path = data_root / "metadata.csv"
processed_dir = data_root / "processed"
state_file = root / "state_files" / "e53-s1952148-d93703104.state"

if args.mode == "default":
    model_dir = data_root / "raw" / "tcrpmhc"
    proc_dir = processed_dir / "proteinsolver_preprocess"
    out_dir = root / "state_files" / "tcr_binding" / "proteinsolver_finetune_80_cv"

if args.cluster == "cdr3ab":
    cluster_path = data_root / "clusterRes_cluster.tsv"
    out_dir.parent / str(out_dir.name + "_cluster_cdr3ab")
if args.cluster == "cdr3b":
    cluster_path = data_root / "clusterRes_cdr3b_cluster.tsv"
    out_dir.parent / str(out_dir.name + "_cluster_cdr3b")

if args.drop_swapped:
    out_dir = out_dir.parent / str(out_dir.name + "_no_swapped")

paths = list(model_dir.glob("*"))
join_key = [int(x.name.split("_")[0]) for x in paths]
path_df = pd.DataFrame({'#ID': join_key, 'path': paths})

metadata = pd.read_csv(metadata_path)
metadata = metadata.join(path_df.set_index("#ID"), on="#ID", how="inner")  # filter to non-missing data
metadata = metadata.reset_index(drop=True)

raw_files = np.array(metadata["path"])
targets = np.array(metadata["binder"])
dataset = ProteinDataset(
    proc_dir, 
    raw_files, 
    targets, 
    overwrite=False
)

# GNN params
num_features = 20
adj_input_size = 2
hidden_size = 128

# general params
batch_size = 8
epochs = 200
learning_rate = 1e-5
lr_decay = 0.999
w_decay = 1e-3

# touch files to ensure output
n_splits = 5
save_dir = get_non_dupe_dir(out_dir)
loss_paths = touch_output_files(save_dir, "loss", n_splits)
state_paths = touch_output_files(save_dir, "state", n_splits)
pred_paths = touch_output_files(save_dir, "pred", n_splits)

partitions = partition_clusters(cluster_path, n_splits)

if args.drop_swapped:
    filtered_indices = list(metadata[metadata["origin"] == "swapped"].index)
    for i in range(n_splits):
        part = [j for j in partitions[i] if j not in filtered_indices]
        partitions[i] = part

extra_print_str = "\nSaving to {}\nFold: {}\nPeptide: {}"

for i in range(n_splits):
    best_inner_fold_valid_losses = list()
    inner_train_losses = list()
    inner_valid_losses = list()
    best_inner_fold_models = list()

    test_idx = partitions[i]
    outer_train_folds = [partitions[j] for j in range(n_splits) if j != i]
    inner_train_partitions, inner_valid_partitions = join_partitions(outer_train_folds)
    for train_idx, valid_idx in zip(inner_train_partitions, inner_valid_partitions):
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

        best_inner_fold_valid_losses.append(min(valid_losses))
        inner_train_losses.append(train_losses)
        inner_valid_losses.append(valid_losses)
        best_inner_fold_models.append(copy.deepcopy(net.state_dict()))

    best_inner_idx = best_inner_fold_valid_losses.index(min(best_inner_fold_valid_losses))
    net.load_state_dict(best_inner_fold_models[best_inner_idx])
    train_losses = inner_train_losses[best_inner_idx]
    valid_losses = inner_valid_losses[best_inner_idx]

    torch.save(net.state_dict(), state_paths[i])
    torch.save({"train": train_losses, "valid": valid_losses}, loss_paths[i])
    
    pred, true = gnn_predict(net, dataset, test_idx, device)     
    torch.save({"y_pred": pred, "y_true": true}, pred_paths[i])
    
    i += 1
