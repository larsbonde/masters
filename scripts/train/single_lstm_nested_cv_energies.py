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
parser.add_argument("-m", "--mode", default="default")
parser.add_argument("-s", "--swapped", action="store_true", default=False)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = Path("/home/projects/ht3_aim/people/sebdel/masters/data/")
data_root = root / "neat_data"
metadata_path = data_root / "metadata.csv"
processed_dir = data_root / "processed" 
state_file = root / "state_files" / "e53-s1952148-d93703104.state"
out_dir = root / "state_files" / "tcr_binding" / "lstm_single_energy_80_cv"
cluster_path = data_root / "clusterRes_cluster.tsv"

n_splits = 5
partitions = [list() for _ in range(n_splits)]
targets = list()

model_energies_dir = Path("/home/projects/ht3_aim/people/idamei/data/train_data")

paths = list(model_energies_dir.glob("*"))
for i, path in enumerate(paths):
    split = str(path).split("_")
    
    bind_str = split[-2]
    if bind_str == "pos":
        bind = 1
    else:
        bind = 0
    targets.append(bind)
    part = int(split[-3][0]) - 1
    partitions[part].append(i)

dataset = LSTMEnergyDataset(
    paths=paths,
    targets=targets
)

# LSTM params
batch_size = 32
embedding_dim = 142
hidden_dim = 256 #128 #32
num_layers = 2  # from 2

# general params
epochs = 150
learning_rate = 1e-4
lr_decay = 0.995
w_decay = 1e-3
dropout = 0.6  # test scheduled dropout. Can set droput using net.layer.dropout = 0.x https://arxiv.org/pdf/1703.06229.pdf

# touch files to ensure output
save_dir = get_non_dupe_dir(out_dir)
loss_paths = touch_output_files(save_dir, "loss", n_splits)
state_paths = touch_output_files(save_dir, "state", n_splits)
pred_paths = touch_output_files(save_dir, "pred", n_splits)

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
        net = MyLSTM(
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
            collate_fn=pad_collate,
            #extra_print=extra_print_str.format(save_dir, i, unique_peptides[i]),
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
    
    pred, true = lstm_predict(net, dataset, test_idx, device, collate_fn=pad_collate)     
    torch.save({"y_pred": pred, "y_true": true}, pred_paths[i])
