import torch
import torch_geometric
import numpy as np
import copy
from .utils import *
from torch.utils.data import SubsetRandomSampler, BatchSampler


def gnn_train(
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
    early_stopping=False,
    verbose=False,
    extra_print=None,
):
    train_losses = list()
    valid_losses = list()
    epochs_since_last_improv = 0
    best_valid_loss = float("inf")
    best_model = model.state_dict()

    for e in range(epochs):
        
        train_sampler = BatchSampler(SubsetRandomSampler(train_idx), batch_size=batch_size, drop_last=False)
        valid_sampler = BatchSampler(SubsetRandomSampler(valid_idx), batch_size=1, drop_last=False)
        
        train_loader = torch_geometric.loader.DataLoader(dataset=dataset, batch_sampler=train_sampler, pin_memory=True)
        valid_loader = torch_geometric.loader.DataLoader(dataset=dataset, batch_sampler=valid_sampler, pin_memory=True)

        train_len = len(train_loader)
        valid_len = len(valid_loader)
        
        train_loss = 0
        model.train()
        j = 0
        for data in train_loader:    
            data = data.to(device)
            y_pred = model(data.x, data.edge_index, data.edge_attr, data.batch).squeeze(1)
            optimizer.zero_grad()
            loss = criterion(y_pred, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if verbose:
                display_func(j, train_len, e, train_losses, valid_losses, extra_print)
            j += 1
        
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for data in valid_loader:    
                data = data.to(device)
                y_pred = model(data.x, data.edge_index, data.edge_attr, data.batch).squeeze(1)
                loss = criterion(y_pred, data.y)
                valid_loss += loss.item()
        
        scheduler.step()
        train_losses.append(train_loss / train_len)
        valid_losses.append(valid_loss / valid_len)

        if valid_losses[-1] < best_valid_loss:
            best_model = copy.deepcopy(model.state_dict())
            best_valid_loss = valid_losses[-1]
            epochs_since_last_improv = 0
        else:
            epochs_since_last_improv += 1

        if epochs_since_last_improv > 30 and early_stopping:
            model.load_state_dict(best_model)
            break

    if early_stopping:
        model.load_state_dict(best_model)

    return model, train_losses, valid_losses


def gnn_predict(model, dataset, idx, device):
    data_loader = torch_geometric.loader.DataLoader(dataset=dataset, sampler=idx, batch_size=1)
    pred = list()
    true = list()
    model.eval()
    with torch.no_grad():
        for data in data_loader:    
            data = data.to(device)
            y_pred = model(data.x, data.edge_index, data.edge_attr, data.batch).squeeze(1)
            pred.append(torch.sigmoid(y_pred))
            true.append(data.y)
    return torch.Tensor(pred), torch.Tensor(true)
