import atexit
import copy
import csv
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
from .lstm_utils import*
from torch.nn.modules.container import ModuleList
from torch_geometric.nn.inits import reset
from torch_geometric.utils import scatter_, to_dense_adj
from torch_geometric.nn import global_mean_pool

# From https://gitlab.com/elaspic/elaspic2/-/tree/master/src/elaspic2/plugins/proteinsolver/data/ps_191f05de


class MyLSTM(nn.Module):
    def __init__(self,  embedding_dim, hidden_dim, num_layers, output_dim=1,  dropout=0.0):
        super(MyLSTM, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers, 
            dropout=dropout, 
            bidirectional=True,
            batch_first=True,
        )
        self.linear_dropout = nn.Dropout(p=dropout)
        self.batch_norm = nn.BatchNorm1d(num_features=hidden_dim)
        self.linear_1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, output_dim)
        
        torch.nn.init.xavier_uniform_(self.linear_1.weight) 
        torch.nn.init.xavier_uniform_(self.linear_2.weight)    

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        h_cat = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        print(h.shape)
        out = self.linear_1(h_cat)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.linear_dropout(out)
        out = self.linear_2(out)
        return out


class QuadLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout=0.0):
        super(QuadLSTM, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.lstm_1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )       
        self.lstm_2 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )        
        self.lstm_3 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout, 
            batch_first=True,
        )     
        self.lstm_4 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout, 
            batch_first=True,
        )
        
        self.linear_dropout = nn.Dropout(p=dropout)
        self.batch_norm = nn.BatchNorm1d(num_features=hidden_dim)
        self.linear_1 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, 1)
        
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
    
    def forward(self, x_1, x_2, x_3, x_4):
        _, (h_1, _) = self.lstm_1(x_1)
        _, (h_2, _) = self.lstm_2(x_2)
        _, (h_3, _) = self.lstm_3(x_3)
        _, (h_4, _) = self.lstm_4(x_4)
        h_cat = torch.cat((h_1[-1], h_2[-1], h_3[-1], h_4[-1]), dim=1)
        out = self.linear_1(h_cat)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.linear_dropout(out)
        out = self.linear_2(out)
        return out


class TripleLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout=0.0):
        super(TripleLSTM, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.lstm_1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )       
        self.lstm_2 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )        
        self.lstm_3 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout, 
            batch_first=True,
        )     

        self.linear_dropout = nn.Dropout(p=dropout)
        self.batch_norm = nn.BatchNorm1d(num_features=hidden_dim)
        self.linear_1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, 1)
        
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
    
    def forward(self, x_1, x_2, x_3):
        _, (h_1, _) = self.lstm_1(x_1)
        _, (h_2, _) = self.lstm_2(x_2)
        _, (h_3, _) = self.lstm_3(x_3)
        h_cat = torch.cat((h_1[-1], h_2[-1], h_3[-1]), dim=1)
        out = self.linear_1(h_cat)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.linear_dropout(out)
        out = self.linear_2(out)
        return out


# ProteinSolver stuff
class EdgeConvMod(torch.nn.Module):
    def __init__(self, nn, aggr="max"):
        super().__init__()
        self.nn = nn
        self.aggr = aggr
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x, edge_index, edge_attr=None):
        """ """
        row, col = edge_index
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        # TODO: Try -x[col] instead of x[col] - x[row]
        if edge_attr is None:
            out = torch.cat([x[row], x[col]], dim=-1)
        else:
            out = torch.cat([x[row], x[col], edge_attr], dim=-1)
        out = self.nn(out)
        x = scatter_(self.aggr, out, row, dim_size=x.size(0))

        return x, out

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


class EdgeConvBatch(nn.Module):
    def __init__(self, gnn, hidden_size, batch_norm=True, dropout=0.2):
        super().__init__()

        self.gnn = gnn

        x_post_modules = []
        edge_attr_post_modules = []

        if batch_norm is not None:
            x_post_modules.append(nn.LayerNorm(hidden_size))
            edge_attr_post_modules.append(nn.LayerNorm(hidden_size))

        if dropout:
            x_post_modules.append(nn.Dropout(dropout))
            edge_attr_post_modules.append(nn.Dropout(dropout))

        self.x_postprocess = nn.Sequential(*x_post_modules)
        self.edge_attr_postprocess = nn.Sequential(*edge_attr_post_modules)

    def forward(self, x, edge_index, edge_attr=None):
        x, edge_attr = self.gnn(x, edge_index, edge_attr)
        x = self.x_postprocess(x)
        edge_attr = self.edge_attr_postprocess(edge_attr)
        return x, edge_attr


def get_graph_conv_layer(input_size, hidden_size, output_size):
    mlp = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
    gnn = EdgeConvMod(nn=mlp, aggr="add")
    graph_conv = EdgeConvBatch(gnn, output_size, batch_norm=True, dropout=0.2)
    return graph_conv


class MyEdgeConv(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x, edge_index, edge_attr=None):
        """ """
        row, col = edge_index
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        # TODO: Try -x[col] instead of x[col] - x[row]
        if edge_attr is None:
            out = torch.cat([x[row], x[col]], dim=-1)
        else:
            out = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_attr_out = self.nn(out)

        return edge_attr_out

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


class MyAttn(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.attn)

    def forward(self, x, edge_index, edge_attr, batch):
        """ """
        query = x.unsqueeze(0)
        key = to_dense_adj(edge_index, batch=batch, edge_attr=edge_attr).squeeze(0)

        adjacency = to_dense_adj(edge_index, batch=batch).squeeze(0)
        key_padding_mask = adjacency == 0
        key_padding_mask[torch.eye(key_padding_mask.size(0)).to(torch.bool)] = 0
        #         attn_mask = torch.zeros_like(key)
        #         attn_mask[mask] = -float("inf")

        x_out, _ = self.attn(query, key, key, key_padding_mask=key_padding_mask)
        #         x_out = torch.where(torch.isnan(x_out), torch.zeros_like(x_out), x_out)
        x_out = x_out.squeeze(0)
        assert (x_out == x_out).all().item()
        assert x.shape == x_out.shape, (x.shape, x_out.shape)
        return x_out

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


class Net(nn.Module):
    def __init__(self, x_input_size, adj_input_size, hidden_size, output_size):
        super().__init__()

        self.embed_x = nn.Sequential(
            nn.Embedding(x_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            # nn.ReLU(),
        )
        self.embed_adj = (
            nn.Sequential(
                nn.Linear(adj_input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                # nn.ELU(),
            )
            if adj_input_size
            else None
        )
        self.graph_conv_0 = get_graph_conv_layer(
            (2 + bool(adj_input_size)) * hidden_size, 2 * hidden_size, hidden_size
        )

        N = 3
        graph_conv = get_graph_conv_layer(3 * hidden_size, 2 * hidden_size, hidden_size)
        self.graph_conv = _get_clones(graph_conv, N)

        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index, edge_attr):
        x = self.forward_without_last_layer(x, edge_index, edge_attr)
        x = self.linear_out(x)
        return x        

    def forward_without_last_layer(self, x, edge_index, edge_attr):
        x = self.embed_x(x)
        # edge_index, _ = add_self_loops(edge_index)  # We should remove self loops in this case!
        edge_attr = self.embed_adj(edge_attr) if edge_attr is not None else None

        x_out, edge_attr_out = self.graph_conv_0(x, edge_index, edge_attr)
        x = x + x_out
        edge_attr = (
            (edge_attr + edge_attr_out) if edge_attr is not None else edge_attr_out
        )

        for i in range(3):
            x = F.relu(x)
            edge_attr = F.relu(edge_attr)
            x_out, edge_attr_out = self.graph_conv[i](x, edge_index, edge_attr)
            x = x + x_out
            edge_attr = edge_attr + edge_attr_out

        return x#, edge_attr


class MyGNN(nn.Module):
    def __init__(self, x_input_size, adj_input_size, hidden_size, output_size):
        super().__init__()

        self.embed_x = nn.Sequential(
            nn.Embedding(x_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            # nn.ReLU(),
        )
        self.embed_adj = (
            nn.Sequential(
                nn.Linear(adj_input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                # nn.ELU(),
            )
            if adj_input_size
            else None
        )
        self.graph_conv_0 = get_graph_conv_layer(
            (2 + bool(adj_input_size)) * hidden_size, 2 * hidden_size, hidden_size
        )

        N = 3
        graph_conv = get_graph_conv_layer(3 * hidden_size, 2 * hidden_size, hidden_size)
        self.graph_conv = _get_clones(graph_conv, N)

        self.linear_out = nn.Linear(hidden_size, output_size)  # re-assign to (hidden_size, 1)
        
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.forward_without_last_layer(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5)
        x = self.linear_out(x)
        return x

    def forward_without_last_layer(self, x, edge_index, edge_attr):
        x = self.embed_x(x)
        # edge_index, _ = add_self_loops(edge_index)  # We should remove self loops in this case!
        edge_attr = self.embed_adj(edge_attr) if edge_attr is not None else None

        x_out, edge_attr_out = self.graph_conv_0(x, edge_index, edge_attr)
        x = x + x_out
        edge_attr = (
            (edge_attr + edge_attr_out) if edge_attr is not None else edge_attr_out
        )

        for i in range(3):
            x = F.relu(x)
            edge_attr = F.relu(edge_attr)
            x_out, edge_attr_out = self.graph_conv[i](x, edge_index, edge_attr)
            x = x + x_out
            edge_attr = edge_attr + edge_attr_out

        return x


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def to_fixed_width(lst, precision=None):
    lst = [round(l, precision) if isinstance(l, float) else l for l in lst]
    return [f"{l: <18}" for l in lst]
