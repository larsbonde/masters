#! /usr/bin/env python3

import sys
sys.path.append('/home/projects/ht3_aim/people/sebdel/masters/') # add my repo to python path
import torch
import numpy as np
import pandas as pd
import modules

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
out_dir = root / "state_files" / "tcr_binding" / "lstm_embedding_verification"
model_dir = data_root / "raw" / "tcrpmhc"

dataset = LSTMDataset(
    data_dir=processed_dir / "proteinsolver_embedding_verification", 
    annotations_path=processed_dir / "proteinsolver_embedding_verification" / "targets.pt"
)

# LSTM params
batch_size = 8
embedding_dim = 128
hidden_dim = 128 #128 #32
num_layers = 2  # from 2

# general params
n_splits = 5
epochs = 150
learning_rate = 1e-4
lr_decay = 0.99
w_decay = 1e-3
dropout = 0.6  # test scheduled dropout. Can set droput using net.layer.dropout = 0.x https://arxiv.org/pdf/1703.06229.pdf

# touch files to ensure output
save_dir = get_non_dupe_dir(out_dir)
loss_paths = touch_output_files(save_dir, "loss", n_splits)
state_paths = touch_output_files(save_dir, "state", n_splits)
pred_paths = touch_output_files(save_dir, "pred", n_splits)

extra_print_str = "\nSaving to {}\nFold: {}\nPeptide: {}"

CV = KFold(n_splits=n_splits, shuffle=True)
i = 0
for train_idx, test_idx in CV.split(dataset):
    
    net = MyLSTM(
        embedding_dim=embedding_dim, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        dropout=dropout,
        output_dim=3,
    )
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        net.parameters(), 
        lr=learning_rate, 
        weight_decay=w_decay,
    )
    scheduler = optim.lr_scheduler.MultiplicativeLR(
        optimizer, 
        lr_lambda=lambda epoch: lr_decay
    )
    
    net, train_losses, valid_losses = lstm_embedding_test_train(
        net,
        epochs,
        criterion,
        optimizer,
        scheduler,
        dataset,
        train_idx,
        test_idx,
        batch_size,
        device,
        collate_fn=pad_collate,
        #extra_print=extra_print_str.format(save_dir, i, unique_peptides[i]),
        early_stopping=True,
    )
    torch.save(net.state_dict(), state_paths[i])
    torch.save({"train": train_losses, "valid": valid_losses}, loss_paths[i])
    
    pred, true = lstm_predict(
        net, 
        dataset,
        test_idx, 
        device, 
        collate_fn=pad_collate, 
        out_fn=torch.nn.Softmax(dim=0)
    )     
    torch.save({"y_pred": pred, "y_true": true}, pred_paths[i])
    
    i += 1
