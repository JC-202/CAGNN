import torch
import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append('..')
from torch_geometric.nn import MessagePassing
import torch_geometric
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv, GINConv
import torch.nn as nn
from torch_geometric.datasets.planetoid import Planetoid
from torch_geometric.data import Data
import torch.nn.functional as F
import math
import random
import dgl


class MLP_Layer(nn.Module):
    def __init__(self, in_dim, hid_dim, activation, dropout):
        super(MLP_Layer, self).__init__()
        self.linear = nn.Linear(in_dim, hid_dim, bias=False)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))

    def forward(self, adj, x):
        x = self.dropout(x)
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x


class GCNConv(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, hid_dim)

    def forward(self, adj, h):
        h = adj @ h
        h = self.fc(h)
        return h


class SGCConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, adj, h):
        h = adj @ h
        return h



class GINConv(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GINConv, self).__init__()
        self.fc = nn.Linear(nfeat, nhid)
        self.eps = nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, adj, x):
        x = self.eps[0] * x + adj @ x
        x = self.fc(x)
        return x


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer2(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, adj, input):
        dv = input.device

        N = input.size()[0]
        edge = adj

        h = torch.mm(input, self.W)
        # h: N x out
        if torch.isnan(h).any():
            print(self.W)
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        e = self.leakyrelu(self.a.mm(edge_h).squeeze())
        e = e + e.min()
        e = e - e.max()
        edge_e = torch.exp(e)
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        assert not torch.isinf(e_rowsum).any()
        assert not torch.isnan(e_rowsum).any()
        if (e_rowsum == 0).any():
            #e_rowsum[e_rowsum==0
            e_rowsum[e_rowsum==0] = 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum + 1e-8)
        # h_prime: N x out
        if torch.isnan(h_prime).any():
            print(e_rowsum.min(), e_rowsum.max())
            print(e_rowsum)
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.fc = nn.Linear(in_features, out_features)
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    # def special_spmm(self, indices, values, shape, b):
    #     a = torch.sparse_coo_tensor(indices, values, shape)
    #     return a @ b

    def edge_softmax(self, adj, e):
        g = dgl.graph((adj[0, :], adj[1, :]))
        return dgl.nn.functional.edge_softmax(g, e)

    def forward(self, adj, input):
        dv = input.device
        N = input.size()[0]

        #h = torch.mm(input, self.W)
        h = self.fc(input)
        # h: N x out
        if torch.isnan(h).any():
            print(h)
            print(self.W)
            print(self.W.grad)
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[adj[0, :], :], h[adj[1, :], :]), dim=1).t()
        # edge: 2*D x E

        e = self.leakyrelu(self.a.mm(edge_h).squeeze())
        assert not torch.isnan(e).any()
        # edge_e: E

        attention = self.edge_softmax(adj, e)
        attention = self.dropout(attention)

        sparse_attention = SparseTensor(row=adj[0, :], col=adj[1, :], value=attention, sparse_sizes=(N, N)).to(dv)
        h_prime = sparse_attention @ h

        #h_norm = torch.norm(h, p=2, dim=1)
        #h_prime_norm = torch.norm(h, p=2, dim=1)
        #print(h_norm.min(), h_norm.max(), h_prime_norm.min(), h_prime_norm.max())

        #h_prime = self.special_spmm(adj, attention, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GATConv(nn.Module):
    def __init__(self, nfeat, nhid, dropout, drop_edge, alpha, nheads, concat=True):
        """Sparse version of GAT."""
        super().__init__()
        self.dropout = dropout
        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=drop_edge,
                                                 alpha=alpha,
                                                 concat=concat) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, adj, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(adj, x) for att in self.attentions], dim=1)
        x = F.elu(x)
        return x