import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
from torch import nn, optim
from .utils import *


class MyLSTM(nn.Module):
    def __init__(self,  num_classes, num_features, num_layers, hidden_size):
        super(MyLSTM, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers, 
            dropout=0.5, 
            bidirectional=True
        )
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(hidden_size * 2, num_classes)
        
        torch.nn.init.xavier_uniform_(self.linear.weight)
    
    def forward(self, x):
        x = nn.utils.rnn.pack_sequence(x)
        x, (h, c) = self.lstm(x)
        h_cat = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        out = self.dropout(h_cat)
        out = self.linear(out)
        return out


def gnn_to_lstm_batch(data, gnn_instance, device, num_classes):
    """function for bridging gnn output to lstm"""
    data = data.to(device)
    y = data.y
    with torch.no_grad():
        out = gnn_instance(data.x, data.edge_index, data.edge_attr)
    
    batches = torch.unique(data.batch)
    sliced_embeddings = list()
    encoded_y = list()
    for batch_idx in batches:
        # split sub graphs into batches
        batch_slice = torch.nonzero(data.batch == batch_idx)
        chain_map = data.chain_map[batch_idx]
        one_batch_peptide_emb = out[batch_slice][chain_map == "P"]  # get peptide only
        sliced_embeddings.append(one_batch_peptide_emb.squeeze(1))
        
        # one hot encode targets
        sliced_y = int(y[batch_idx].item())
        one_hot_y = np.zeros(num_classes)
        one_hot_y[sliced_y] = 1
        encoded_y.append(one_hot_y)
        
    sliced_embeddings.sort(key=lambda x: len(x))
    encoded_y = torch.Tensor(encoded_y)
    
    return sliced_embeddings, encoded_y


def train_model(
    model,
    epochs,
    criterion,
    optimizer,
    train_data, 
    valid_data,
    batch_size,
    device,
    transform=None,
):
    """general function for training a model"""
    train_losses = list()
    valid_losses = list()
    for i in range(epochs):
        train_loader = iter(torch_geometric.loader.DataLoader(train_data, batch_size=batch_size))
        valid_loader = iter(torch_geometric.loader.DataLoader(valid_data, batch_size=batch_size))
        
        train_len = len(train_loader)
        valid_len = len(valid_loader)

        train_loss = 0
        model.train()
        for j, x in enumerate(train_loader):
            if transform:
                x, y = transform(x)  # y needs to be changed to multiclass y i.e. [0,0,1] instead of [2].
            else:
                x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            display_func(j, train_len, i, train_losses, valid_losses)

        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for x in valid_loader:
                if transform:
                    x, y = transform(x)
                else:
                    x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                valid_loss += loss.item()

        train_losses.append(train_loss / train_len)
        valid_losses.append(valid_loss / valid_len)
    return model, train_losses, valid_losses


def predict(model, data, batch_size, device, transform=None):
    data_loader = iter(torch_geometric.loader.DataLoader(data, batch_size=batch_size))
    pred = list()
    true = list()
    with torch.no_grad():
        for x in data_loader:
            if transform:
                x, y = transform(x)
            else:
                x, y = x.to(device), y.to(device)
            out = model(x)
            pred.append(F.softmax(out, dim=1))
            true.append(y)
    return torch.cat(pred), torch.cat(true)
