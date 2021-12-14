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
from sklearn.ensemble import RandomForestClassifier

#from modules.dataset import *
from modules.utils import *
#from modules.model import *
from modules.lstm_utils import *

root = Path("/home/projects/ht3_aim/people/sebdel/masters/data/")
data_root = root / "neat_data"
metadata_path = data_root / "metadata.csv"
processed_dir = data_root / "processed" / "tcr_binding"
out_dir = root / "state_files" / "tcr_binding" / "randomforest_baseline"/"250"
model_dir = data_root / "raw" / "tcrpmhc"

model_dir = data_root / "raw" / "tcrpmhc"

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
    
    # get training rows and drop swapped data
    train_df = metadata[metadata["peptide"] != pep]
    train_df = train_df[~train_df["merged_chains"].str.contains('|'.join(valid_unique_cdr))]

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

n_splits = len(unique_peptides)
save_dir = get_non_dupe_dir(out_dir)
pred_paths = touch_output_files(save_dir, "pred", n_splits)
n_estimators = 250

pad_dummy_tensor = torch.zeros((500, 128))  # some number greater than longest protein in dataset

i = 0
for train_idx, valid_idx in zip(loo_train_partitions, loo_valid_partitions):

    # get train input
    x_train = [dataset[i][0][:,:-4] for i in train_idx]

    # pad and reshape data to n_sample x n_feature dim
    x_train.append(pad_dummy_tensor)
    x_train = nn.utils.rnn.pad_sequence(x_train, batch_first=True, padding_value=0)
    x_train = x_train[:-1]  # remove padding tensor
    batch_dim, len_dim, embed_dim = x_train.shape
    x_train = x_train.reshape((batch_dim, len_dim * embed_dim))

    # get train labels
    y_train = torch.Tensor([dataset[i][1].item() for i in train_idx])

    # get valid input
    x_valid = [dataset[i][0][:,:-4] for i in valid_idx]

    # pad and reshape data to n_sample x n_feature dim
    x_valid.append(pad_dummy_tensor)
    x_valid = nn.utils.rnn.pad_sequence(x_valid, batch_first=True, padding_value=0)
    x_valid = x_valid[:-1]  # remove padding tensor
    batch_dim, len_dim, embed_dim = x_valid.shape
    x_valid = x_valid.reshape((batch_dim, len_dim * embed_dim))

    # get train labels
    y_valid = torch.Tensor([dataset[i][1].item() for i in valid_idx])

    # fit model and get valid preds
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(x_train.numpy(), y_train.numpy())

    pred = clf.predict_proba(x_valid.numpy())[:,1]


    torch.save({"y_pred": pred, "y_true": y_valid}, pred_paths[i])

    i += 1
