
import torch.nn as nn
import numpy as np
import networkx as nx
import copy
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
class LPSI(nn.Module):
    def __init__(self, laplacian, num_node):
        super().__init__()
        self.laplacian = laplacian
        self.num_node = num_node

    def forward(self, alpha,diff_vec):
        x = (1 - alpha) * np.matmul(np.linalg.inv(np.eye(N=self.num_node) - alpha * self.laplacian), diff_vec)
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
        super(GCNSI, self).__init__()
        self.conv1 = GCNConv(4, 128)
        self.conv2 = GCNConv(128, 128)
        self.fc =torch.nn.Linear(128,2)

    def forward(self, alpha,laplacian,num_node,threshold,diff_vec,edge_index):
        lpsi =LPSI(laplacian, num_node)
        V3 = copy.deepcopy(diff_vec)
        V4 = copy.deepcopy(diff_vec)
        V3[diff_vec < threshold] =  threshold
        V4[diff_vec >= threshold] =  threshold
        d1 = copy.deepcopy(diff_vec)
        d1 = d1[:, np.newaxis]
        d2 = lpsi(alpha,diff_vec)
        d2 = d2[:, np.newaxis]
        d3 = lpsi(alpha,V3)
        d3 = d3[:, np.newaxis]
        d4 = lpsi(alpha,V4)
        d4 = d4[:, np.newaxis]
        x = np.concatenate((d1, d2, d3, d4), axis=1)
        x = torch.tensor(x,dtype=torch.float)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x

class OJC(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G

    def get_K_list(self, Y, I, target):
        K = list()
        for n1 in self.G.nodes():
            count = 0
            for n2 in self.G[n1]:
                if target[n2] == 1:
                    count += 1
                if count == Y:
                    K.append(n1)
                    break
        K = list(set(K + I))
        return K

    def Candidate(self, Y, I, target):
        K = self.get_K_list(Y, I, target)
        G_prime = self.G.subgraph(K)
        # unforzen
        G_prime = nx.Graph(G_prime)
        if nx.is_connected(G_prime):
            G_bar = G_prime
        else:
            component = nx.connected_components(G_prime)
            R = [list(c)[0] for c in component]
            for n in R[1:]:
                if nx.has_path(self.G, n, R[0]):
                    path = nx.shortest_path(self.G, R[0], n)
                    nx.add_path(G_prime, path)
            G_bar = G_prime
        nx.is_connected(G_bar)
        return K, G_bar

    def forward(self, Y, I, target, num_source):
        K, G_bar = self.Candidate(Y, I, target)
        ecc = list()
        for n in K:
            long_path = 0
            for i in I:
                if nx.has_path(G_bar, n, i):
                    path = nx.shortest_path_length(G_bar, n, i)
                else:
                    path = 0
                if path > long_path:
                    long_path = path
            ecc.append(long_path)
        ecc_arr = np.array(ecc)
        index = np.argsort(ecc_arr).tolist()
        index = index[:num_source]
        x = np.zeros(target.shape)
        x[index] = 1
        return x