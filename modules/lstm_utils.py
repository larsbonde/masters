import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from .utils import *
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, BatchSampler


def pad_collate(batch, pad_val=0):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=pad_val)
    yy_pad = nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=pad_val)

    return xx_pad, yy_pad


def pad_collate_chain_split(batch, pad_val=0, n_split=4, n_encode=4):
    (xx, yy) = zip(*batch)
    x_split_batch = [list() for _ in range(n_split)]
    for x in xx:
        for i in range(n_encode):
            x_split_batch[i].append(x[x[:,-i - 1] == 1][:,:-n_encode])  # slice based on positional encoding and remove encoding part

    for i in range(n_split):
        x_split_batch[i] = nn.utils.rnn.pad_sequence(
            x_split_batch[i], 
            batch_first=True, 
            padding_value=pad_val
        )
    yy_pad = nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=pad_val)
    return x_split_batch, yy_pad


def lstm_train(
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
    collate_fn=pad_collate_chain_split,
    early_stopping=False,
    verbose=False,
    extra_print=None,
):
    train_losses = list()
    valid_losses = list()
    #epochs_since_last_improv = 0
    best_valid_loss = float("inf") 
    best_model = model.state_dict()

    for e in range(epochs):
        
        train_sampler = BatchSampler(SubsetRandomSampler(train_idx), batch_size=batch_size, drop_last=True)
        valid_sampler = BatchSampler(SubsetRandomSampler(valid_idx), batch_size=1, drop_last=False)
        
        train_loader = DataLoader(dataset=dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
        valid_loader = DataLoader(dataset=dataset, batch_sampler=valid_sampler, collate_fn=collate_fn)

        train_len = len(train_loader)
        valid_len = len(valid_loader)
        
        train_loss = 0
        model.train()
        j = 0
        for xx, y in train_loader:    
            y = y.to(device)
            if type(xx) == list:
                xx = (x.to(device) for x in xx)
                y_pred = model(*xx)
            else:
                xx = xx.to(device)
                y_pred = model(xx)
            optimizer.zero_grad()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if verbose:
                display_func(j, train_len, e, train_losses, valid_losses, extra_print)
            j += 1
        
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for xx, y in valid_loader:
                y = y.to(device)    
                if type(xx) == list:
                    xx = (x.to(device) for x in xx)
                    y_pred = model(*xx)
                else:
                    xx = xx.to(device)
                    y_pred = model(xx)
                loss = criterion(y_pred, y)
                valid_loss += loss.item()
        
        scheduler.step()
        train_losses.append(train_loss / train_len)
        valid_losses.append(valid_loss / valid_len)

        if valid_losses[-1] < best_valid_loss:
            best_model = model.state_dict()
            best_valid_loss = valid_losses[-1]
        #    epochs_since_last_improv = 0
        #else:
        #    epochs_since_last_improv += 1

        #if epochs_since_last_improv > 20 and early_stopping:
        #    model.load_state_dict(best_model)
        #    break

    if early_stopping:
        model.load_state_dict(best_model)

    return model, train_losses, valid_losses


def lstm_predict(model, dataset, idx, device, collate_fn=pad_collate_chain_split):
    data_loader = DataLoader(dataset=dataset, sampler=idx, batch_size=1, collate_fn=collate_fn)
    pred = list()
    true = list()
    model.eval()
    with torch.no_grad():
        for xx, y in data_loader:
            y = y.to(device)    
            if type(xx) == list:
                xx = (x.to(device) for x in xx)
                y_pred = model(*xx)
            else:
                xx = xx.to(device)
                y_pred = model(xx)
            pred.append(torch.sigmoid(y_pred))
            true.append(y)
    return torch.Tensor(pred), torch.Tensor(true)
