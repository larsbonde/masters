#! /usr/bin/env python3

import sys
sys.path.append('/home/projects/ht3_aim/people/sebdel/masters/') # add my repo to python path
import torch
import numpy as np
import pandas as pd
import modules
import argparse

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, BatchSampler
from torch import nn, optim
from pathlib import Path

from modules.dataset_utils import *
from modules.dataset import *
from modules.utils import *
from modules.models import *
from modules.lstm_utils import *

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", default="default")
parser.add_argument("-s", "--drop_swapped", action="store_true", default=False)
parser.add_argument("-c", "--cluster", default="medium")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = Path("/home/projects/ht3_aim/people/sebdel/masters/data/")
data_root = root / "neat_data"
metadata_path = data_root / "metadata.csv"
processed_dir = data_root / "processed" 
state_file = root / "state_files" / "e53-s1952148-d93703104.state"


if args.mode == "ps":
    model_dir = data_root / "raw" / "tcrpmhc"
    data = processed_dir / "proteinsolver_embeddings_pos"
    targets = processed_dir / "proteinsolver_embeddings_pos" / "targets.pt"
    out_dir = root / "state_files" / "tcr_binding" / "lstm_ps_nested_loo_cv"
    batch_size = 8
    embedding_dim = 128
    hidden_dim = 128
    num_layers = 2 

if args.mode == "esm":
    model_dir = data_root / "raw" / "tcrpmhc"
    data = processed_dir / "esm_embeddings_pos"
    targets = processed_dir / "proteinsolver_embeddings_pos" / "targets.pt"
    out_dir = root / "state_files" / "tcr_binding" / "lstm_esm_nested_loo_cv"
    batch_size = 8
    embedding_dim = 1280
    hidden_dim = 128 
    num_layers = 2

if args.mode == "esm_ps":
    model_dir = data_root / "raw" / "tcrpmhc"
    data = processed_dir / "proteinsolver_esm_embeddings_pos"
    targets = processed_dir / "proteinsolver_embeddings_pos" / "targets.pt"
    out_dir = root / "state_files" / "tcr_binding" / "lstm_esm_ps_nested_loo_cv"
    batch_size = 8
    embedding_dim = 1280 + 128
    hidden_dim = 128 
    num_layers = 2

if args.mode == "ps_foldx":
    model_dir = data_root / "raw" / "foldx_repair"
    data=processed_dir / "proteinsolver_embeddings_pos_foldx_repair"
    targets=processed_dir / "proteinsolver_embeddings_pos_foldx_repair" / "targets.pt"
    out_dir = root / "state_files" / "tcr_binding" / "lstm_ps_foldx_nested_loo_cv"
    batch_size = 8
    embedding_dim = 128
    hidden_dim = 128
    num_layers = 2 

if args.mode == "esm_ps_foldx":
    model_dir = data_root / "raw" / "foldx_repair"
    data = processed_dir / "proteinsolver_esm_embeddings_pos_foldx_repair"
    targets = processed_dir / "proteinsolver_embeddings_pos_foldx_repair" / "targets.pt"
    out_dir = root / "state_files" / "tcr_binding" / "lstm_esm_ps_foldx_nested_loo_cv"
    batch_size = 8
    embedding_dim = 1280 + 128
    hidden_dim = 128 
    num_layers = 2

if args.mode == "ps_rosetta":
    model_dir = data_root / "raw" / "rosetta_repair"
    data = processed_dir / "proteinsolver_embeddings_pos_rosetta_repair"
    targets = processed_dir / "proteinsolver_embeddings_pos_rosetta_repair" / "targets.pt"
    out_dir = root / "state_files" / "tcr_binding" / "lstm_ps_rosetta_nested_loo_cv"
    batch_size = 8
    embedding_dim = 128
    hidden_dim = 128 
    num_layers = 2

if args.mode == "esm_ps_rosetta":
    model_dir = data_root / "raw" / "rosetta_repair"
    data = processed_dir / "proteinsolver_esm_embeddings_pos_rosetta_repair"
    targets = processed_dir / "proteinsolver_embeddings_pos_rosetta_repair" / "targets.pt"
    out_dir = root / "state_files" / "tcr_binding" / "lstm_esm_ps_rosetta_nested_loo_cv"
    batch_size = 8
    embedding_dim = 1280 + 128
    hidden_dim = 128 
    num_layers = 2

if args.cluster == "medium":
    cluster_path = data_root / "clusterRes_cdr3b_50_raw_idx_cluster.tsv"
if args.cluster == "low":
    cluster_path = data_root / "clusterRes_cdr3b_25_raw_idx_cluster.tsv"
    out_dir = out_dir.parent / str(out_dir.name + "_cluster_25")

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
    annotations_path=targets,
    device=device
)

# general params
epochs = 100
learning_rate = 5e-5
lr_decay = 0.995
w_decay = 1e-3
dropout = 0.6 # test scheduled dropout. Can set droput using net.layer.dropout = 0.x https://arxiv.org/pdf/1703.06229.pdf

outer_train_partitions, test_partitions, unique_peptides = generate_loo_partitions(metadata, drop_swapped=args.drop_swapped)

# touch files to ensure output
n_splits = len(unique_peptides)
save_dir = get_non_dupe_dir(out_dir)
loss_paths = touch_output_files(save_dir, "loss", n_splits)
state_paths = touch_output_files(save_dir, "state", n_splits)
pred_paths = touch_output_files(save_dir, "pred", n_splits)

extra_print_str = "\nSaving to {}\nFold: {}\nPeptide: {}"

i = 0
for outer_train_idx, test_idx in zip(outer_train_partitions, test_partitions):
    best_inner_fold_valid_losses = list()
    inner_train_losses = list()
    inner_valid_losses = list()
    best_inner_fold_models = list()

    inner_metadata = metadata.iloc[outer_train_idx].copy(deep=True)
    inner_metadata = inner_metadata.sample(frac=1)  # shuffle rows to get even partitions
    inner_partitions = partition_clusters(inner_metadata, cluster_path)
    inner_train_partitions, inner_valid_partitions = join_partitions(inner_partitions)

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
    
    i += 1

