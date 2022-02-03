#! /usr/bin/env python3

import sys
sys.path.append('/home/projects/ht3_aim/people/sebdel/masters/') # add my repo to python path
import torch
import numpy as np
import pandas as pd
import modules
import copy
import argparse

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, BatchSampler
from torch import nn, optim
from pathlib import Path
from sklearn.model_selection import KFold

from modules.dataset_utils import *
from modules.dataset import *
from modules.utils import *
from modules.models import *
from modules.lstm_utils import *

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode")
parser.add_argument("-s", "--drop_swapped", action="store_true", default=False)
parser.add_argument("-c", "--cluster", default="cdr3ab")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = Path("/home/projects/ht3_aim/people/sebdel/masters/data/")
data_root = root / "neat_data"
metadata_path = data_root / "metadata.csv"
processed_dir = data_root / "processed" 
state_file = root / "state_files" / "e53-s1952148-d93703104.state"
model_dir = data_root / "raw" / "tcrpmhc"

if args.mode == "ps":
    data = processed_dir / "proteinsolver_embeddings_pos"
    targets = processed_dir / "proteinsolver_embeddings_pos" / "targets.pt"
    out_dir = root / "state_files" / "tcr_binding" / "lstm_ps_80_cv"
    batch_size = 8
    embedding_dim = 128
    hidden_dim = 128
    num_layers = 2 

if args.mode == "esm":
    data = processed_dir / "esm_embeddings_pos"
    targets = processed_dir / "proteinsolver_embeddings_pos" / "targets.pt"
    out_dir = root / "state_files" / "tcr_binding" / "lstm_esm_80_cv"
    batch_size = 8
    embedding_dim = 1280
    hidden_dim = 128 
    num_layers = 2

if args.mode == "esm_ps":
    data = processed_dir / "proteinsolver_esm_embeddings_pos"
    targets = processed_dir / "proteinsolver_embeddings_pos" / "targets.pt"
    out_dir = root / "state_files" / "tcr_binding" / "lstm_esm_ps_80_cv"
    batch_size = 8
    embedding_dim = 1280 + 128
    hidden_dim = 128 
    num_layers = 2

if args.mode == "ps_foldx":
    model_dir = data_root / "raw" / "foldx_repair"
    data=processed_dir / "proteinsolver_embeddings_pos_foldx_repair"
    targets=processed_dir / "proteinsolver_embeddings_pos_foldx_repair" / "targets.pt"
    out_dir = root / "state_files" / "tcr_binding" / "lstm_ps_foldx_80_cv"
    batch_size = 8
    embedding_dim = 128
    hidden_dim = 128
    num_layers = 2 

if args.mode == "esm_ps_foldx":
    model_dir = data_root / "raw" / "foldx_repair"
    data = processed_dir / "proteinsolver_esm_embeddings_pos_foldx_repair"
    targets = processed_dir / "proteinsolver_embeddings_pos_foldx_repair" / "targets.pt"
    out_dir = root / "state_files" / "tcr_binding" / "lstm_esm_ps_foldx_80_cv"
    batch_size = 8
    embedding_dim = 1280 + 128
    hidden_dim = 128 
    num_layers = 2

if args.mode == "ps_rosetta":
    model_dir = data_root / "raw" / "rosetta_repair"
    data = processed_dir / "proteinsolver_embeddings_pos_rosetta_repair"
    targets = processed_dir / "proteinsolver_embeddings_pos_rosetta_repair" / "targets.pt"
    out_dir = root / "state_files" / "tcr_binding" / "lstm_ps_rosetta_80_cv"
    batch_size = 8
    embedding_dim = 128
    hidden_dim = 128 
    num_layers = 2

if args.mode == "esm_ps_rosetta":
    model_dir = data_root / "raw" / "rosetta_repair"
    data = processed_dir / "proteinsolver_esm_embeddings_pos_rosetta_repair"
    targets = processed_dir / "proteinsolver_embeddings_pos_rosetta_repair" / "targets.pt"
    out_dir = root / "state_files" / "tcr_binding" / "lstm_esm_ps_rosetta_80_cv"
    batch_size = 8
    embedding_dim = 1280 + 128
    hidden_dim = 128 
    num_layers = 2

if args.mode == "blosum":
    data = processed_dir / "blosum_embeddings_pos"
    targets = processed_dir / "proteinsolver_embeddings_pos" / "targets.pt"
    out_dir = root / "state_files" / "tcr_binding" / "lstm_blosum_80_cv"
    batch_size = 8
    embedding_dim = 21
    hidden_dim = 128 
    num_layers = 2

if args.cluster == "cdr3ab":
    cluster_path = data_root / "clusterRes_cluster.tsv"
    out_dir = out_dir.parent / str(out_dir.name + "_cluster_cdr3ab")
if args.cluster == "cdr3b":
    cluster_path = data_root / "clusterRes_cdr3b_cluster.tsv"
    out_dir = out_dir.parent / str(out_dir.name + "_cluster_cdr3b")
if args.cluster == "cdr3b_low_cov":
    cluster_path = data_root / "clusterRes_cdr3b_test_cov_25_cluster.tsv"
    out_dir = out_dir.parent / str(out_dir.name + "_cluster_cdr3b_low_cov")

if args.drop_swapped:
    out_dir = out_dir.parent / str(out_dir.name + "_no_swapped")

paths = list(model_dir.glob("*"))
join_key = [int(x.name.split("_")[0]) for x in paths]
path_df = pd.DataFrame({'#ID': join_key, 'path': paths})

metadata = pd.read_csv(metadata_path)
metadata = metadata.join(path_df.set_index("#ID"), on="#ID", how="inner")  # filter to non-missing data
metadata = metadata.reset_index(drop=True)
metadata["merged_chains"] = metadata["CDR3a"] + metadata["CDR3b"]
unique_peptides = metadata["peptide"].unique()

dataset = LSTMDataset(
    data_dir=data, 
    annotations_path=targets
)

# general params
epochs = 150
learning_rate = 5e-5
lr_decay = 0.995
w_decay = 1e-3
dropout = 0.6  # test scheduled dropout. Can set droput using net.layer.dropout = 0.x https://arxiv.org/pdf/1703.06229.pdf

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

available_indices = list(metadata.index)
for i in range(n_splits):
    part = [j for j in partitions[i] if j in available_indices]
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
        net = QuadLSTM(
            embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout,
        )
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
        
        net, train_losses, valid_losses = lstm_train(
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
            extra_print=extra_print_str.format(save_dir, i, unique_peptides[i]),
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
    
    pred, true = lstm_predict(net, dataset, test_idx, device)     
    torch.save({"y_pred": pred, "y_true": true}, pred_paths[i])
