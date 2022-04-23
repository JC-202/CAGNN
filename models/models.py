import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_sparse import SparseTensor

from models.layers import GCNConv,  GINConv,  GATConv

class CAGNN(torch.nn.Module):
    def __init__(self, g, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2,
                 norm_type='None', conv_type='gcn', gate_type='convex'):
        super().__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.conv_type = conv_type
        self.dropout = nn.Dropout(dropout)
        self.gate_type = gate_type

        # encoder, message passing, decoder
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.build_message_passing_layers(num_layers, hidden_channels)
        self.decoder = nn.Linear(hidden_channels, out_channels)

        if gate_type == 'convex':
            self.gate = nn.Linear(hidden_channels * 2, 1)
        elif gate_type == 'convex_MLP_2':
            self.gate = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1)
            )
        elif gate_type == 'convex_MLP_3':
            self.gate = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels * 2),
                nn.ReLU(),
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1)
            )
        elif gate_type == 'concat':
            self.decoder = nn.Linear(hidden_channels*(num_layers+1), out_channels)
        elif gate_type == 'global':
            self.alpha_scalar = nn.Parameter(torch.randn(num_layers+1))
        self.g = self.set_g(g)

    def set_g(self, adj):
        if self.conv_type == 'gat':
            if isinstance(adj, SparseTensor):
                row, col, _ = adj.coo()
                adj = torch.cat([row.unsqueeze(0), col.unsqueeze(0)], dim=0)
            elif isinstance(adj, torch.Tensor):
                if adj.shape[0] == adj.shape[1]:
                    adj = adj.nonzero().t()
        return adj

    def build_message_passing_layers(self, num_layers, hid_channesl):
        self.conv_layers = nn.ModuleList()
        self.learnable_norm_self = nn.ModuleList()
        self.learnable_norm_neighb = nn.ModuleList()
        self.transforms = nn.ModuleList()

        if self.norm_type == 'bn':
            self.learnable_norm_self.append(nn.BatchNorm1d(hid_channesl))
            self.learnable_norm_neighb.append(nn.BatchNorm1d(hid_channesl))
        elif self.norm_type == 'ln':
            self.learnable_norm_self.append(nn.LayerNorm(hid_channesl))
            self.learnable_norm_neighb.append(nn.LayerNorm(hid_channesl))

        for i in range(num_layers):
            if self.conv_type == 'gcn':
                layer = GCNConv(hid_channesl, hid_channesl)
            elif self.conv_type == 'gin':
                layer = GINConv(hid_channesl, hid_channesl)
            elif self.conv_type == 'gat':
                layer = GATConv(hid_channesl, hid_channesl, dropout=0, drop_edge=0, alpha=0.2, nheads=1)
            else:
                raise 'not implement'

            self.conv_layers.append(layer)

            if self.gate_type == 'convex_multi_map':
                self.transforms.append(nn.Linear(hid_channesl, hid_channesl))
            if self.norm_type == 'bn':
                self.learnable_norm_self.append(nn.BatchNorm1d(hid_channesl))
                self.learnable_norm_neighb.append(nn.BatchNorm1d(hid_channesl))
            elif self.norm_type == 'ln':
                self.learnable_norm_self.append(nn.LayerNorm(hid_channesl))
                self.learnable_norm_neighb.append(nn.LayerNorm(hid_channesl))

    def propagate(self, adj, x, layer_index):
        x = self.conv_layers[layer_index](adj, x)
        return x

    def update(self, self_x, conv_x, layer_index=0):
        if self.gate_type in ['convex', 'convex_MLP_2', 'convex_MLP_3']:
            a = self.gate(torch.cat([self_x, conv_x], dim=1)).sigmoid()
            self_x = a * self_x + (1 - a) * conv_x
        elif self.gate_type == 'vector':
            a = self.gate_vector(torch.cat([self_x, conv_x], dim=1)).sigmoid()
            self_x = a * self_x + (1 - a) * conv_x
        elif self.gate_type == 'global':
            a = self.alpha_scalar[layer_index].sigmoid()
            self_x = a * self_x + (1 - a) * conv_x
        elif self.gate_type == 'add':
            self_x = self_x + conv_x
        elif self.gate_type == 'concat':
            self_x = torch.cat([self_x, conv_x], dim=1)
        else:
            self_x = conv_x
        return self_x, conv_x

    def norm(self, x, layer_index=0, is_self=False):
        if is_self and self.gate_type == 'concat':
            return x
        if self.norm_type == 'l2':
            x = F.normalize(x, p=2, dim=1)
        elif self.norm_type in ['bn', 'ln']:
            if is_self:
                x = self.learnable_norm_self[layer_index](x)
            else:
                x = self.learnable_norm_neighb[layer_index](x)
        return x

    def forward(self, x):
        adj = self.g
        x = self.dropout(x)
        init_x = self.encoder(x).relu()
        init_x = self.norm(init_x, 0, is_self=True)
        self_x, conv_x = init_x, init_x
        for i in range(self.num_layers):
            conv_x = self.dropout(conv_x)
            conv_x = self.propagate(adj, conv_x, i)
            self_x, conv_x = self.update(self_x, conv_x, i)
            self_x = self.norm(self_x, i, is_self=True)
            conv_x = self.norm(conv_x, i)
        h = self_x
        h = self.dropout(h)
        o = self.decoder(h)
        return o