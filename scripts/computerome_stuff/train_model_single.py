#! /usr/bin/env python3

import sys
sys.path.append('/home/projects/ht3_aim/people/sebdel/masters/') # add my repo to python path
import os
import torch
import torch.nn.functional as F
import torch_geometric
import kmbio  # fork of biopython PDB with some changes in how the structure, chain, etc. classes are defined.
import numpy as np
import pandas as pd
#import proteinsolver
import modules

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, BatchSampler
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import *
from torch import nn, optim
from pathlib import Path

#from modules.dataset import *
from modules.utils import *
#from modules.model import *
from modules.lstm_utils import *

#np.random.seed(1)


def pad_collate(batch, pad_val=0):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=pad_val)
    yy_pad = nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=pad_val)

    return xx_pad, yy_pad, x_lens, y_lens


def lstm_quad_train(
    model,
    epochs,
    criterion,
    optimizer,
    scheduler,
    dataset,
    train_idx,
    valid_idx,
    batch_size,
    device,
    extra_print=None,
):
    train_losses = list()
    valid_losses = list()
    
    for e in range(epochs):
        
        train_sampler = BatchSampler(SubsetRandomSampler(train_idx), batch_size=batch_size, drop_last=False)
        valid_sampler = BatchSampler(SubsetRandomSampler(valid_idx), batch_size=1, drop_last=False)
        
        train_loader = DataLoader(dataset=dataset, batch_sampler=train_sampler, pin_memory=True, collate_fn=pad_collate)
        valid_loader = DataLoader(dataset=dataset, batch_sampler=valid_sampler, pin_memory=True, collate_fn=pad_collate)

        train_len = len(train_loader)
        valid_len = len(valid_loader)
        
        #if e > 5:
        #    net.linear_dropout.dropout = 0.6
        #    net.lstm_1.dropout = 0.6
        #    net.lstm_2.dropout = 0.6
        #    net.lstm_3.dropout = 0.6
        #    net.lstm_4.dropout = 0.6
            
        
        train_loss = 0
        model.train()
        j = 0
        print(">train")
        for xx, y in train_loader:    
            y = y.to(device)
            xx = (x.to(device) for x in xx)
            y_pred = model(*xx)
            optimizer.zero_grad()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

#            display_func(j, train_len, e, train_losses, valid_losses, extra_print)
            j += 1
        print(">valid")
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for xx, y in valid_loader:    
                y = y.to(device)
                xx = (x.to(device) for x in xx)
                y_pred = model(*xx)
                loss = criterion(y_pred, y)
                valid_loss += loss.item()
        
        scheduler.step()
        train_losses.append(train_loss / train_len)
        valid_losses.append(valid_loss / valid_len)

    return model, train_losses, valid_losses


def lstm_quad_predict(model, dataset, idx, device):
    data_loader = DataLoader(dataset=dataset, sampler=idx, batch_size=1, collate_fn=pad_collate)
    pred = list()
    true = list()
    with torch.no_grad():
        for xx, y in data_loader:    
            xx = (x.to(device) for x in xx)
            y_pred = model(*xx)
            pred.append(torch.sigmoid(y_pred))
            true.append(y)
    return torch.Tensor(pred), torch.Tensor(true)


root = Path("/home/projects/ht3_aim/people/sebdel/masters/data/")
data_root = root / "neat_data"
metadata_path = data_root / "metadata.csv"
processed_dir = data_root / "processed" / "tcr_binding"
state_file = root / "state_files" / "e53-s1952148-d93703104.state"
out_dir = root / "state_files" / "tcr_binding" / "single_lstm"
model_dir = data_root / "raw" / "tcrpmhc"

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

paths = list(model_dir.glob("*"))
join_key = [int(x.name.split("_")[0]) for x in paths]
path_df = pd.DataFrame({'#ID': join_key, 'path': paths})

metadata = pd.read_csv(metadata_path)
metadata = metadata.join(path_df.set_index("#ID"), on="#ID", how="inner")  # filter to non-missing data
metadata = metadata.reset_index(drop=True)
unique_peptides = metadata["peptide"].unique()

metadata["merged_chains"] = metadata["CDR3a"] + metadata["CDR3b"]
loo_train_partitions = list()
loo_valid_partitions = list()
for pep in unique_peptides:
    valid_df = metadata[metadata["peptide"] == pep]
    valid_unique_cdr = valid_df["merged_chains"].unique()

    #valid_unique_cdra = valid_df["CDR3a"].unique()
    #valid_unique_cdrb = valid_df["CDR3b"].unique()
    
    # get training rows and drop swapped data
    train_df = metadata[metadata["peptide"] != pep]
    train_df = train_df[~train_df["merged_chains"].str.contains('|'.join(valid_unique_cdr))]
    
    #train_df = train_df[~train_df["CDR3a"].str.contains('|'.join(valid_unique_cdra))]
    #train_df = train_df[~train_df["CDR3b"].str.contains('|'.join(valid_unique_cdrb))]
    
    loo_train_partitions.append(list(train_df.index))
    loo_valid_partitions.append(list(valid_df.index))

dataset = LSTMDataset(
    data_dir=processed_dir / "gnn_out_pos_128", 
    annotations_path=processed_dir / "gnn_out_pos_128" / "targets.pt"
)

# hacky dataset fix
# hacky dataset fix
# hacky dataset fix
filtered_peptides = ["CLGGLLTMV", "ILKEPVHGV"]
filtered_indices = list()
filtered_partitions = list()

for pep in filtered_peptides:
    filtered_indices.extend(list(metadata[metadata["peptide"] == pep].index))
    filtered_partitions.extend(np.where(unique_peptides == pep)[0])

loo_train_partitions = [part for i, part in enumerate(loo_train_partitions) if i not in filtered_partitions]
loo_valid_partitions = [part for i, part in enumerate(loo_valid_partitions) if i not in filtered_partitions]

filtered_indices = set(filtered_indices)

for i in range(len(loo_train_partitions)):
    train_part, valid_part = loo_train_partitions[i], loo_valid_partitions[i]
    train_part = [i for i in train_part if i not in filtered_indices]
    valid_part = [i for i in valid_part if i not in filtered_indices]
    loo_train_partitions[i], loo_valid_partitions[i] = train_part, valid_part
    
unique_peptides = np.delete(unique_peptides, filtered_partitions)

# LSTM params
batch_size = 64
embedding_dim = 128
hidden_dim = 128 #128 #32
num_layers = 3  # from 2
epochs = 60
learning_rate = 1e-3
lr_decay = 0.98 #TODO set some val, 1=no effect
w_decay = 1e-4
dropout = 0.8  # test scheduled dropout. Can set droput using net.layer.dropout = 0.x https://arxiv.org/pdf/1703.06229.pdf

# touch files to ensure output
n_splits = len(unique_peptides)
save_dir = get_non_dupe_dir(out_dir)
loss_paths = touch_output_files(save_dir, "loss", n_splits)
state_paths = touch_output_files(save_dir, "state", n_splits)
pred_paths = touch_output_files(save_dir, "pred", n_splits)

extra_print_str = "\nSaving to {}\nFold: {}\nPeptide: {}"

i = 0
for train_idx, valid_idx in zip(loo_train_partitions, loo_valid_partitions):
    print(f">{i}")   
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
        #weight_decay=w_decay,
    )  # test learning rate scheduler to reduce validation volatility
    scheduler = optim.lr_scheduler.MultiplicativeLR(
        optimizer, 
        lr_lambda=lambda epoch: lr_decay
    )
 
    net, train_losses, valid_losses = lstm_quad_train(
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
        extra_print_str.format(save_dir, i, unique_peptides[i]),
    )
    torch.save(net.state_dict(), state_paths[i])
    torch.save({"train": train_losses, "valid": valid_losses}, loss_paths[i])
    print(">pred")
    pred, true = lstm_quad_predict(net, dataset, valid_idx, device)     
    torch.save({"y_pred": pred, "y_true": true}, pred_paths[i])
    
    i += 1


#n_splits = len(unique_peptides)
#threshold = 0.2

# compute metrics
#perf_data = dict()
#for i in range(n_splits):
#    data = torch.load(pred_paths[i])
#    pred = data["y_pred"]
#    true = data["y_true"]

    # auc
#    auc = roc_auc_score(true, pred)
#    fpr, tpr, thr = roc_curve(true, pred, pos_label=1)
    
#    thresh_pred = torch.zeros(len(pred))
#    thresh_pred[pred >= threshold] = 1
#    mcc = matthews_corrcoef(true, thresh_pred)
    
#    pep = unique_peptides[i]
#    perf_data[pep] = [fpr, tpr, auc, mcc]

#    print(auc, mcc)

#performance_file = save_dir / "performance_data.pt"
#torch.save(perf_data, performance_file)
