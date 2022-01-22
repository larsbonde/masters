#! /usr/bin/env python3

import sys
sys.path.append('/home/projects/ht3_aim/people/sebdel/masters/') # add my repo to python path
import torch
import numpy as np
import pandas as pd
import modules
import copy

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = Path("/home/projects/ht3_aim/people/sebdel/masters/data/")
data_root = root / "neat_data"
metadata_path = data_root / "metadata.csv"
processed_dir = data_root / "processed" 
state_file = root / "state_files" / "e53-s1952148-d93703104.state"
out_dir = root / "state_files" / "tcr_binding" / "lstm_ps_80_cv"
model_dir = data_root / "raw" / "tcrpmhc"
cluster_path = data_root / "clusterRes_cluster.tsv"

paths = list(model_dir.glob("*"))
join_key = [int(x.name.split("_")[0]) for x in paths]
path_df = pd.DataFrame({'#ID': join_key, 'path': paths})

metadata = pd.read_csv(metadata_path)
metadata = metadata.join(path_df.set_index("#ID"), on="#ID", how="inner")  # filter to non-missing data
metadata = metadata.reset_index(drop=True)
metadata["merged_chains"] = metadata["CDR3a"] + metadata["CDR3b"]
unique_peptides = metadata["peptide"].unique()

dataset = LSTMDataset(
    data_dir=processed_dir / "proteinsolver_embeddings_pos", 
    annotations_path=processed_dir / "proteinsolver_embeddings_pos" / "targets.pt"
)

# LSTM params
batch_size = 8
embedding_dim = 128
hidden_dim = 128 #128 #32
num_layers = 2  # from 2

# general params
epochs = 15  # TODO set to 150 again!!!!!!!!!!
learning_rate = 1e-4
lr_decay = 0.99
w_decay = 1e-3
dropout = 0.6  # test scheduled dropout. Can set droput using net.layer.dropout = 0.x https://arxiv.org/pdf/1703.06229.pdf

# touch files to ensure output
n_splits = 5
save_dir = get_non_dupe_dir(out_dir)
loss_paths = touch_output_files(save_dir, "loss", n_splits)
state_paths = touch_output_files(save_dir, "state", n_splits)
pred_paths = touch_output_files(save_dir, "pred", n_splits)

train_partitions, test_partitions = K_fold_CV_from_clusters(cluster_path, n_splits)

extra_print_str = "\nSaving to {}\nFold: {}\nPeptide: {}"

i = 0
for outer_train_idx, test_idx in zip(train_partitions, test_partitions):
    best_inner_fold_valid_losses = list()
    inner_train_losses = list()
    inner_valid_losses = list()
    best_inner_fold_models = list()
    CV = KFold(n_splits=n_splits, shuffle=True)
    for train_idx, valid_idx in CV.split(outer_train_idx):
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
    print(best_inner_fold_models[best_inner_idx])
    print(net.state_dict())
    net.load_state_dict(best_inner_fold_models[best_inner_idx])
    train_losses = inner_train_losses[best_inner_idx]
    valid_losses = inner_valid_losses[best_inner_idx]

    torch.save(net.state_dict(), state_paths[i])
    torch.save({"train": train_losses, "valid": valid_losses}, loss_paths[i])
    
    pred, true = lstm_predict(net, dataset, test_idx, device)     
    torch.save({"y_pred": pred, "y_true": true}, pred_paths[i])
    
    i += 1
