import sys
sys.path.append('/home/sebastian/masters/') # add my repo to python path
import os
import torch
import torch_geometric
import kmbio  # fork of biopython PDB with some changes in how the structure, chain, etc. classes are defined.
import numpy as np
import proteinsolver
import modules

from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import *
from torch import nn, optim
from modules.dataset import *
from modules.utils import *
from modules.model import *
from modules.my_model import *


root = Path("/home/sebastian/masters/data/")
data_root = root / "neat_data"
metadata_path = data_root / "embedding_dataset.csv"
processed_dir = data_root / "processed" / "embedding_verification"
state_file = root / "state_files" / "e53-s1952148-d93703104.state"
out_dir = root / "state_files" / "embedding_verification" 

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# load dataset
raw_files = list()
targets = list()
with open(metadata_path, "r") as infile:
    for line in infile:
        line = line.strip().split(",")
        raw_files.append(line[0])
        targets.append(int(line[1]))

raw_files = np.array(raw_files)
targets = np.array(targets)

dataset = ProteinDataset(processed_dir, raw_files, targets, overwrite=False)

# init proteinsolver gnn
num_features = 20
adj_input_size = 2
hidden_size = 128

gnn = Net(
    x_input_size=num_features + 1, 
    adj_input_size=adj_input_size, 
    hidden_size=hidden_size, 
    output_size=num_features
)
gnn.load_state_dict(torch.load(state_file, map_location=device))
gnn.eval()
gnn = gnn.to(device)

# init LSTM
num_classes = 3
num_layers = 2
hidden_size = 26

net = MyLSTM(num_classes, num_features, num_layers, hidden_size)
net = net.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001) 

# training params
epochs = 1
n_splits = 5
batch_size = 5

# touch files to ensure output
save_dir = get_non_dupe_dir(out_dir)
loss_paths = touch_output_files(save_dir, "loss", n_splits)
state_paths = touch_output_files(save_dir, "state", n_splits)
pred_paths = touch_output_files(save_dir, "pred", n_splits)

CV = KFold(n_splits=n_splits, shuffle=True)
i = 0
for train_idx, valid_idx in CV.split(dataset):
    
    train_subset = dataset[torch.LongTensor(train_idx)][0:10]
    valid_subset = dataset[torch.LongTensor(valid_idx)][0:10]
    
    net = MyLSTM(num_classes, num_features, num_layers, hidden_size)
    net = net.to(device)
    
    # partial function - gnn arg is static, x is given later
    gnn_transform = lambda x: gnn_to_lstm_batch(
        x, 
        gnn_instance=gnn, 
        device=device,
        num_classes=num_classes
)
    
    net, train_subset_losses, valid_subset_losses = train_model(
        model=net,
        epochs=epochs, 
        criterion=criterion,
        optimizer=optimizer,
        train_data=train_subset, 
        valid_data=valid_subset,
        batch_size=batch_size,
        device=device,
        transform=gnn_transform,
)

    torch.save({"train": train_subset_losses, "valid": valid_subset_losses}, loss_paths[i])
    torch.save(net.state_dict(), state_paths[i])
    
    # perform test preds
    y_pred, y_true = predict(
        model=net, 
        data=train_subset, 
        batch_size=batch_size,
        device=device,
        transform=gnn_transform,
)

    torch.save({"y_pred": y_pred, "y_true": y_true,}, pred_paths[i])
    
    i += 1