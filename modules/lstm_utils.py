import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from .utils import *


class MyLSTM(nn.Module):
    """placeholder model"""
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


class LSTMDataset(Dataset):
    def __init__(
        self, 
        data_dir, 
        annotations_path, 
        transform=None, 
        target_transform=None
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.annotations = torch.Tensor(torch.load(annotations_path))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        x = torch.load(f"{self.data_dir}/data_{idx}.pt")
        y = self.annotations[idx]
        return x, y


def pad_collate(batch, pad_val=21):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=pad_val)
    yy_pad = nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=pad_val)

    return xx_pad, yy_pad, x_lens, y_lens


def lstm_train(
    model,
    epochs,
    criterion,
    optimizer,
    train_data, 
    valid_data,
    batch_size,
    device,
):
    train_losses = list()
    valid_losses = list()
    
    for e in range(epochs):
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

        train_len = len(train_loader)
        valid_len = len(valid_loader)

        train_loss = 0
        model.train()
        j = 0
        for x, y, _, _ in train_loader:    
            y = y.long().squeeze(1).to(device)
            x = x.float().to(device)
            y_pred = model(x)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            display_func(j, train_len, e, train_losses, valid_losses)
            j += 1
            
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y, _, _ in valid_loader:    
                y = y.long().squeeze(1).to(device)
                x = x.float().to(device)
                y_pred = model(x)
                loss = F.cross_entropy(y_pred, y)
                valid_loss += loss.item()

        train_losses.append(train_loss / train_len)
        valid_losses.append(valid_loss / valid_len)

    return model, train_losses, valid_losses


def lstm_predict(model, data, device):
    data_loader = DataLoader(dataset=data, batch_size=1, shuffle=False, collate_fn=pad_collate)
    pred = list()
    true = list()
    with torch.no_grad():
        for x, y, _, _ in valid_loader:    
            y = y.long().squeeze(1).to(device)
            x = x.float().to(device)
            y_pred = model(x)
            pred.append(F.softmax(y_pred, dim=1))
            true.append(y)
    return pred, true
