import torch
import torch.nn.functional as F
import torch
from torch_sparse import SparseTensor, spmm
import time
# from torch_geometric.utils import add_remaining_self_loops
from scipy.sparse.linalg import inv
from scipy.sparse import coo_matrix
import numpy as np


def azw(adj, z, w):
    if isinstance(adj, SparseTensor):
        # if adj.is_sparse:
        #     return torch.sparse.mm(adj, z).matmul(w)

        row, col, adj_value = adj.coo()
        edge_index = torch.stack([row, col], dim=0)

        # A*Z then (A*Z)*W
        # # pre_spmm = time.time()
        # temp = spmm(edge_index, adj_value, z.size()[0], z.size()[0], z)
        # # print('time for spmm:', time.time() - pre_spmm)
        # return temp.matmul(w)
        # usage: https://github.com/rusty1s/pytorch_sparse

        # Z*W THEN A*(A*W)  much more efficient!!
        temp = z.matmul(w)
        return spmm(edge_index, adj_value, temp.size()[0], temp.size()[0], temp)
    else:
        return adj.matmul(z.matmul(w))

class GCN(torch.nn.Module):
    def __init__(self,w1,w2):
        super(GCN, self).__init__()
        self.w1=w1
        self.w2=w2
    def forward(self,adj,x):
        z1 = azw(adj, x, self.w1)
        z1 = F.relu(z1)
        z2 = azw(adj, z1, self.w2)
        return z2




