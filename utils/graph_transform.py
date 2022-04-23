import torch
import dgl
import torch_geometric

def remove_self_loop(edge_index):
    assert edge_index.shape[0] == 2
    edges = dgl.graph((edge_index[0], edge_index[1])).to(edge_index.device)
    edges = edges.remove_self_loop()
    edges = [a.long() for a in edges.edges()]
    edge_index = torch_geometric.data.Data(edge_index=torch.stack(edges)).edge_index
    return edge_index


def normalize_adj(adj):
    d = torch.diag(adj.sum(dim=1)) ** -1
    d[torch.isinf(d)] = 0
    adj = d.mm(adj)
    return adj

def sparse_normalize_adj_left(adj):
    size = adj.size(0)
    ones = torch.ones(size).view(-1,1).to(adj.device())
    d = adj @ ones
    d = d ** -1
    d[torch.isinf(d)] = 0
    return adj * d

# D-1/2 * A * D-1/2
def nomarlizeAdj(adj):
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    return adj

# D-1 * A
def normalizeLelf(adj):
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj
    return adj
