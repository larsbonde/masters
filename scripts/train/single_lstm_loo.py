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
parser.add_argument("-s", "--swapped", action="store_true", default=False)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = Path("/home/projects/ht3_aim/people/sebdel/masters/data/")
data_root = root / "neat_data"
metadata_path = data_root / "metadata.csv"
processed_dir = data_root / "processed" 
state_file = root / "state_files" / "e53-s1952148-d93703104.state"
model_dir = data_root / "raw" / "tcrpmhc"
cluster_path = data_root / "clusterRes_cdr3b_50_cluster.tsv"

if args.mode == "ps":
    model_dir = data_root / "raw" / "tcrpmhc"
    data = processed_dir / "proteinsolver_embeddings_pos"
    targets = processed_dir / "proteinsolver_embeddings_pos" / "targets.pt"
    out_dir = root / "state_files" / "tcr_binding" / "lstm_ps_single"
    batch_size = 8
    embedding_dim = 128 + 4
    hidden_dim = 256
    num_layers = 2

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

loo_train_partitions, loo_test_partitions, loo_valid_partitions, unique_peptides = generate_3_loo_partitions(
    metadata,
    cluster_path,
    drop_swapped=args.drop_swapped,
    )

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
n_splits = len(unique_peptides)
save_dir = get_non_dupe_dir(out_dir)
loss_paths = touch_output_files(save_dir, "loss", n_splits)
state_paths = touch_output_files(save_dir, "state", n_splits)
pred_paths = touch_output_files(save_dir, "pred", n_splits)

extra_print_str = "\nSaving to {}\nFold: {}\nPeptide: {}"

i = 0
for train_idx, test_idx, valid_idx in zip(loo_train_partitions, loo_test_partitions, loo_valid_partitions):
    
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
        extra_print=extra_print_str.format(save_dir, i, unique_peptides[i]),
        early_stopping=True,
    )
    torch.save(net.state_dict(), state_paths[i])
    torch.save({"train": train_losses, "valid": valid_losses}, loss_paths[i])
    
    pred, true = lstm_predict(net, dataset, test_idx, device, collate_fn=pad_collate)     
    torch.save({"y_pred": pred, "y_true": true}, pred_paths[i])
    
    i += 1
