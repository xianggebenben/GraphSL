
import torch.nn as nn
import numpy as np
import networkx as nx
import copy
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
class LPSI(nn.Module):
    def __init__(self, alpha, laplacian, num_node):
        super().__init__()
        self.alpha = alpha
        self.laplacian = laplacian
        self.num_node = num_node

    def forward(self, diff_vec):
        x = (1 - self.alpha) * np.matmul(np.linalg.inv(np.eye(N=self.num_node) - self.alpha * self.laplacian), diff_vec)
        return x

class NetSleuth(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G
    def forward(self, k, diff_vec):
        G = copy.deepcopy(self.G)
        G.remove_nodes_from([n for n in G if n not in np.where(diff_vec == 1)[0]])
        lap = nx.laplacian_matrix(G).toarray()

        seed = []
        while len(seed) < k:
            value, vector = np.linalg.eig(lap)
            index = np.argmax(vector[np.argmin(value)])
            seed_index = list(G.nodes)[index]
            seed.append(seed_index)
            G.remove_node(seed_index)
            if len(G.nodes) == 0:
                break
            lap = nx.laplacian_matrix(G).toarray()
        seed_vec = torch.zeros(diff_vec.shape)
        seed_vec[seed] = 1
        return seed_vec
    
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Multiply with weights
        x = self.lin(x)

        # Step 3: Calculate the normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j


class GCNSI(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(4, 128)
        self.conv2 = GCNConv(128, 128)
        self.fc =torch.nn.Linear(128,2)

    def forward(self, x,edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x
